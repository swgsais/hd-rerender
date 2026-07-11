#!/usr/bin/env python3
"""
Standalone TRE archive extractor for SWG.

Supports TRE versions 0004, 0005 (NGE-retail) and 0006 (Restoration). Operates
on a single .tre file or a directory of them, in parallel.

Examples
--------
  # Extract everything to e:\\SWGNGE\\extracted in patch-priority order
  python extract_tre.py --src e:\\SWGNGE --out e:\\SWGNGE\\extracted

  # Pull only UI / shader / appearance assets across all .tre files
  python extract_tre.py --src e:\\SWGNGE --out e:\\SWGNGE\\ext-tre \\
      --include "ui/*" --include "shader/*" --include "appearance/*"

  # List entries in one TRE without extracting
  python extract_tre.py --src e:\\SWGNGE\\bottom.tre --list

Priority semantics
------------------
SWG layers TREs: later patches override earlier ones. With the default
--reverse-order flag set, the extractor walks TREs from patch_33 down to
bottom.tre, writing each path the first time it sees it. The result mirrors
what the running client would actually see for each filename.

In parallel mode (--workers > 1), a pre-pass first assigns each inner path
to the single highest-priority TRE that contains it, and each worker then
extracts only its assigned entries. Without that pre-pass, multiple workers
can each see os.path.exists(out_path) == False simultaneously and all write
the same path in nondeterministic order, silently violating patch priority.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

# Compressor IDs (from TreeFile_SearchNode.h)
CT_NONE       = 0
CT_DEPRECATED = 1
CT_ZLIB       = 2

# On disk, Tag fields are stored as little-endian uint32, which byte-reverses
# the 4-char ASCII tag. "TREE0005" lands as b'EERT5000'.
TOKEN_TREE_LE = b'EERT'
SUPPORTED_VERSIONS_LE = (b'4000', b'5000', b'6000')

HEADER_STRUCT       = struct.Struct('<4s4s7I')   # 36 bytes
TOC_ENTRY_V0005     = struct.Struct('<IiiiiI')   # 24 bytes: crc, length, offset, compressor, compLen, fileNameOffset
TOC_ENTRY_V0006     = struct.Struct('<IiiiiIIi') # 32 bytes: crc, length, offset, u1, u2, fileNameOffset, compressor, compLen


class TreError(Exception):
    pass


def _read_name(buf: bytes, offset: int) -> str:
    end = buf.index(b'\x00', offset)
    return buf[offset:end].decode('latin-1').replace('\\', '/')


def _open_tre(path: str):
    f = open(path, 'rb')
    header_buf = f.read(HEADER_STRUCT.size)
    if len(header_buf) != HEADER_STRUCT.size:
        f.close()
        raise TreError(f'{path}: truncated header')
    (token, version, num_files, toc_off, toc_compr, toc_size,
     block_compr, name_size, uncomp_name_size) = HEADER_STRUCT.unpack(header_buf)
    if token != TOKEN_TREE_LE:
        f.close()
        raise TreError(f'{path}: not a TRE file (token={token!r})')
    if version not in SUPPORTED_VERSIONS_LE:
        f.close()
        raise TreError(f'{path}: unsupported TRE version {version!r}')

    is_v0006 = (version == b'6000')
    entry_size = 32 if is_v0006 else TOC_ENTRY_V0005.size
    on_disk_toc_size = entry_size * num_files

    f.seek(toc_off)
    if toc_compr == CT_NONE:
        toc_raw = f.read(on_disk_toc_size)
    else:
        toc_raw = zlib.decompress(f.read(toc_size))
        if len(toc_raw) != on_disk_toc_size:
            f.close()
            raise TreError(f'{path}: TOC decompress size {len(toc_raw)} != {on_disk_toc_size}')

    if block_compr == CT_NONE:
        names_raw = f.read(uncomp_name_size)
    else:
        names_raw = zlib.decompress(f.read(name_size))
        if len(names_raw) != uncomp_name_size:
            f.close()
            raise TreError(f'{path}: name decompress size {len(names_raw)} != {uncomp_name_size}')

    entries = []
    if is_v0006:
        for i in range(num_files):
            (crc, length, offset, _u1, _u2, name_off, compressor, comp_len) = TOC_ENTRY_V0006.unpack_from(toc_raw, i * 32)
            entries.append((_read_name(names_raw, name_off), length, offset, compressor, comp_len, crc))
    else:
        for i in range(num_files):
            (crc, length, offset, compressor, comp_len, name_off) = TOC_ENTRY_V0005.unpack_from(toc_raw, i * 24)
            entries.append((_read_name(names_raw, name_off), length, offset, compressor, comp_len, crc))

    return f, entries, version


def _filter(name: str, include_globs: list[str]) -> bool:
    if not include_globs:
        return True
    nl = name.lower()
    return any(fnmatch.fnmatch(nl, g.lower()) for g in include_globs)


def extract_tre(tre_path: str, out_dir: str, include_globs: list[str], skip_existing: bool,
                allowed_names: set[str] | None = None) -> dict:
    """Extract entries from one TRE.

    If `allowed_names` is provided, only entries whose literal `name` is in the
    set are written and `include_globs` is ignored (already applied during the
    pre-pass). This is how parallel mode enforces patch-priority ownership: the
    pre-pass hands each TRE's worker the exact set of entries it owns, so no
    two workers can race on the same out_path.
    """
    stats = {
        'file': tre_path,
        'files_written': 0,
        'files_skipped': 0,
        'bytes_written': 0,
        'errors': [],
    }
    try:
        f, entries, _version = _open_tre(tre_path)
    except TreError as e:
        stats['errors'].append(str(e))
        return stats
    except Exception as e:  # decompression / IO
        stats['errors'].append(f'{tre_path}: open failed: {e!r}')
        return stats

    try:
        for (name, length, offset, compressor, comp_len, _crc) in entries:
            if allowed_names is None:
                if not _filter(name, include_globs):
                    continue
            elif name not in allowed_names:
                continue
            out_path = os.path.join(out_dir, name)
            if skip_existing and os.path.exists(out_path):
                stats['files_skipped'] += 1
                continue
            try:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                f.seek(offset)
                if compressor == CT_NONE:
                    data = f.read(length)
                else:
                    raw = f.read(comp_len)
                    data = zlib.decompress(raw)
                if len(data) != length:
                    stats['errors'].append(f'{name}: size mismatch {len(data)} != {length}')
                    continue
                with open(out_path, 'wb') as out:
                    out.write(data)
                stats['files_written'] += 1
                stats['bytes_written'] += len(data)
            except Exception as e:
                stats['errors'].append(f'{name}: {e!r}')
    finally:
        f.close()
    return stats


def list_tre(tre_path: str, include_globs: list[str]) -> None:
    try:
        f, entries, version = _open_tre(tre_path)
    except TreError as e:
        print(f'  ERROR: {e}', file=sys.stderr)
        return
    try:
        version_str = bytes(reversed(version)).decode('ascii', errors='replace')
        print(f'{tre_path}  (v{version_str}, {len(entries)} entries)')
        for (name, length, _offset, compressor, comp_len, crc) in entries:
            if not _filter(name, include_globs):
                continue
            print(f'  {name}  len={length}  comp={compressor}  compLen={comp_len}  crc={crc:08x}')
    finally:
        f.close()


def _collect_tre_paths(src: str, reverse_order: bool) -> list[str]:
    if os.path.isfile(src):
        return [src]
    if not os.path.isdir(src):
        raise SystemExit(f'no such path: {src}')
    names = [n for n in os.listdir(src) if n.lower().endswith('.tre')]
    names.sort(reverse=reverse_order)
    return [os.path.join(src, n) for n in names]


def _collect_winners(tre_files: list[str], include_globs: list[str]) -> dict[str, set[str]]:
    """Walk `tre_files` in priority order and claim each inner path to the
    first TRE that contains it; returns {tre_path: {literal_names_owned}}.

    Mirrors the patch-layering collapse in repack_retail.py: the caller has
    already sorted `tre_files` (patch_33 first by default), so first sighting
    wins. Each worker is then told the exact set of entries to extract from
    its TRE, which is what makes parallel extraction deterministic.

    Only the TOC + name table is read here; data blocks are not decoded.
    """
    seen: set[str] = set()  # normalized keys already claimed by an earlier TRE
    result: dict[str, set[str]] = {tre_path: set() for tre_path in tre_files}
    for tre_path in tre_files:
        try:
            f, entries, _ver = _open_tre(tre_path)
        except TreError as e:
            print(f'  pre-pass: skip {os.path.basename(tre_path)}: {e}', file=sys.stderr)
            continue
        except Exception as e:
            print(f'  pre-pass: skip {os.path.basename(tre_path)}: {e!r}', file=sys.stderr)
            continue
        f.close()
        for (name, *_rest) in entries:
            if not _filter(name, include_globs):
                continue
            key = name.lower().replace('\\', '/')
            if key in seen:
                continue
            seen.add(key)
            result[tre_path].add(name)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Extract SWG TRE archives.')
    ap.add_argument('--src', required=True, help='Directory of .tre files or a single .tre file.')
    ap.add_argument('--out', default=None, help='Output directory (required unless --list).')
    ap.add_argument('--include', action='append', default=[],
                    help='Glob filter against internal path; repeatable (e.g. --include "ui/*" --include "shader/*").')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help='Parallel worker processes (default: half of cpu_count).')
    ap.add_argument('--list', action='store_true', help="List TRE contents instead of extracting.")
    ap.add_argument('--overwrite', action='store_true',
                    help='Overwrite files that already exist. Default skips existing (incremental).')
    ap.add_argument('--forward-order', action='store_true',
                    help='Process TREs in alphabetic order (bottom -> patch_33). Default is reverse '
                         '(patch_33 -> bottom) so the highest-priority TRE wins each filename when '
                         '--overwrite is off.')
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.list and not args.out:
        ap.error('--out is required unless --list is given.')

    tre_files = _collect_tre_paths(args.src, reverse_order=not args.forward_order)
    if not tre_files:
        print('no .tre files found', file=sys.stderr)
        return 2

    if args.list:
        for path in tre_files:
            list_tre(path, args.include)
        return 0

    os.makedirs(args.out, exist_ok=True)
    skip_existing = not args.overwrite

    print(f'extracting {len(tre_files)} TREs -> {args.out}  '
          f'(workers={args.workers}, {"overwrite" if args.overwrite else "skip-existing"}, '
          f'{"forward" if args.forward_order else "reverse"} priority order)')
    if args.include:
        print(f'  filters: {args.include}')

    start = time.time()
    totals = {'files_written': 0, 'files_skipped': 0, 'bytes_written': 0, 'errors': 0}

    if args.workers <= 1 or len(tre_files) == 1:
        for path in tre_files:
            _report(extract_tre(path, args.out, args.include, skip_existing), totals)
    else:
        # Pre-pass claims each inner path to its highest-priority TRE so
        # workers cannot race on the same out_path. Without this, multiple
        # workers each see os.path.exists() == False at the same moment and
        # all write the same path in nondeterministic order, which silently
        # breaks patch-priority semantics.
        pre_t0 = time.time()
        winners_per_tre = _collect_winners(tre_files, args.include)
        total_winners = sum(len(v) for v in winners_per_tre.values())
        print(f'  pre-pass: {total_winners:,} unique paths claimed across '
              f'{len(tre_files)} TREs in {time.time() - pre_t0:.1f}s')
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(extract_tre, p, args.out, args.include, skip_existing,
                          winners_per_tre.get(p, set()))
                for p in tre_files
            ]
            for fut in as_completed(futures):
                _report(fut.result(), totals)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s: {totals['files_written']} files, "
          f"{totals['bytes_written']/1e6:.1f} MB extracted, "
          f"{totals['files_skipped']} skipped, {totals['errors']} errors")
    return 0


def _report(stats: dict, totals: dict) -> None:
    base = os.path.basename(stats['file'])
    line = (f"{base}: wrote {stats['files_written']:>6}  "
            f"skipped {stats['files_skipped']:>6}  "
            f"{stats['bytes_written']/1e6:>8.1f} MB")
    if stats['errors']:
        line += f"  errors={len(stats['errors'])}"
    print(line)
    for err in stats['errors'][:3]:
        print(f"    ERR {err}")
    totals['files_written'] += stats['files_written']
    totals['files_skipped'] += stats['files_skipped']
    totals['bytes_written'] += stats['bytes_written']
    totals['errors'] += len(stats['errors'])


if __name__ == '__main__':
    sys.exit(main())
