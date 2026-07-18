#!/usr/bin/env python3
"""
Bisect a TRE archive by entry count, to track down a single bad asset.

Splits one .tre into two smaller .tre files. Splitting is pure pass-through
(build_tre.PassThroughEntry: byte copy of each entry's on-disk bytes, no
decompress/recompress), so it's fast even on multi-GB archives and never
changes an entry's bytes.

Why this finds a single bad file without unpacking to loose files
-------------------------------------------------------------------
SWG's TRE loader is patch-layered: for any given path, the highest-priority
archive that *contains* that path wins; a path missing from that archive
falls through to the next one down. So if you take your rebuilt archive
(say reborn_textures_hd.tre, which is what's causing a green face) and load
only HALF of its entries in its place - higher priority than the original,
unmodified reborn_textures.tre - then:

  - the half you kept is served from the HD archive (new content)
  - the half you dropped falls through to the original archive underneath
    (old, known-good content)

That's exactly a bisection split, for free, with no need to build a second
"other half" archive and no need to touch the original at all.

Workflow
--------
1. Split the suspect archive in half:

       python split_tre.py --src reborn_textures_hd.tre --out-dir bisect/

   This writes bisect/reborn_textures_hd_a.tre and _b.tre (roughly half the
   entries each).

2. In your client's TRE load order, replace reborn_textures_hd.tre with just
   ONE half (say _a.tre), keeping everything else - including the original
   reborn_textures.tre - unchanged. Launch, reproduce the bug.

     - Bug still there  -> the culprit is in that half. Re-run split_tre.py
       with --src pointed at that half-tre to cut it again.
     - Bug gone         -> the culprit is in the OTHER half (the one that
       fell back to original content). Test _b.tre instead, then bisect it.

3. Repeat step 2 on whichever half still reproduces the bug. Each half is
   itself an ordinary loadable TRE, so this recurses cleanly:

       python split_tre.py --src bisect/reborn_textures_hd_a.tre --out-dir bisect/
       -> reborn_textures_hd_a_a.tre / _a_b.tre

   log2(N) rounds gets you from N entries to 1.

Use --include to pre-filter to a glob (e.g. "texture/human/*head*.dds") if
you already have a hunch, so both halves stay small and irrelevant entries
never enter the search.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_tre import _open_tre, _filter, CT_NONE  # type: ignore
from build_tre import TreWriter, PassThroughEntry     # type: ignore


def split_tre(src_path: str, out_dir: str, include_globs: list[str]) -> tuple[str, str]:
    f, entries, _version = _open_tre(src_path)

    filtered = [e for e in entries if _filter(e[0], include_globs)]
    if len(filtered) < 2:
        f.close()
        raise SystemExit(
            f'{src_path}: only {len(filtered)} matching entr{"y" if len(filtered)==1 else "ies"} '
            f'after filtering - nothing left to split. That entry is your culprit.'
            if filtered else
            f'{src_path}: no entries match the given --include filter(s).'
        )

    # Sorted for a deterministic, reproducible split across reruns.
    filtered.sort(key=lambda e: e[0])
    mid = len(filtered) // 2
    halves = {'a': filtered[:mid], 'b': filtered[mid:]}

    stem = os.path.splitext(os.path.basename(src_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []

    try:
        for suffix, group in halves.items():
            out_path = os.path.join(out_dir, f'{stem}_{suffix}.tre')
            w = TreWriter(out_path)
            for (name, length, offset, compressor, comp_len, crc) in group:
                disk_len = length if compressor == CT_NONE else comp_len
                w.add(PassThroughEntry(name, f, offset, disk_len, length, compressor, crc))
            w.write()
            out_paths.append(out_path)
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            print(f'  wrote {out_path}  ({len(group)} entries, {size_mb:.1f} MB)')
    finally:
        f.close()

    return tuple(out_paths)  # type: ignore[return-value]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='Split a TRE into two halves for bisecting a single bad asset.')
    ap.add_argument('--src', required=True, help='Source .tre to split.')
    ap.add_argument('--out-dir', required=True, help='Directory to write the two halves into.')
    ap.add_argument('--include', action='append', default=[],
                    help='Glob filter against internal path; repeatable '
                         '(e.g. --include "texture/*.dds"). Default: all entries.')
    args = ap.parse_args(argv)

    if not os.path.isfile(args.src):
        ap.error(f'--src not found: {args.src}')

    print(f'[split] {args.src}')
    a_path, b_path = split_tre(args.src, args.out_dir, args.include)
    print(f'\nLoad ONE of these in place of {os.path.basename(args.src)} (same priority slot, '
          f'original archive below stays untouched), retest, then re-run split_tre.py on '
          f'whichever half still reproduces the bug:\n'
          f'  still buggy with {os.path.basename(a_path)}  -> split it again\n'
          f'  fixed    with {os.path.basename(a_path)}     -> test {os.path.basename(b_path)} instead')
    return 0


if __name__ == '__main__':
    sys.exit(main())
