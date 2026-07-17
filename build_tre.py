#!/usr/bin/env python3
"""
TRE 0005 writer.

Produces archives in the original retail (TAG_0005) on-disk format:
  - 36-byte header
  - data blocks (each entry's bytes, sequentially, in any order)
  - zlib-compressed table-of-contents (24 bytes per entry, sorted by crc)
  - zlib-compressed name block (concatenated null-terminated paths)
  - MD5 block: 16 bytes per entry, in TOC (crc-sorted) order, uncompressed;
    each digest covers the entry's on-disk bytes (compressed form if the
    entry is stored compressed)

The writer is built around "pass-through" entries: each entry references
bytes already stored inside a *source* TRE (offset + on-disk length), and
we copy those bytes verbatim into the output. No decompress/recompress
cycle, no recomputed CRCs - whatever retail recorded is what we ship.

Usage
-----
    from build_tre import TreWriter, PassThroughEntry

    w = TreWriter('reborn_quest.tre')
    for (name, source_tre_file, src_offset, src_disk_len,
         uncomp_len, compressor, crc) in entries:
        w.add(PassThroughEntry(name, source_tre_file, src_offset,
                               src_disk_len, uncomp_len, compressor, crc))
    w.write()
"""
from __future__ import annotations

import hashlib
import os
import struct
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

# Importing swg_crc triggers its self-test - if the CRC algorithm is wrong
# we fail at module load rather than producing silently-corrupt TREs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swg_crc  # type: ignore  # noqa: E402

HEADER_STRUCT       = struct.Struct('<4s4s7I')   # 36 bytes
TOC_ENTRY_V0005     = struct.Struct('<IiiiiI')   # 24 bytes: crc, length, offset, compressor, compLen, fileNameOffset
TOKEN_TREE_LE       = b'EERT'
VERSION_0005_LE     = b'5000'

CT_NONE = 0
CT_ZLIB = 2

# Copy buffer size for streaming data-block pass-through.
_COPY_CHUNK = 1 << 20  # 1 MiB

# zlib compression level. 9 is slowest+smallest; for ~125K-file builds the
# wall-clock cost vs. level 6 is noticeable - drop this if build time matters
# more than archive size.
_ZLIB_LEVEL = 9


@dataclass
class PassThroughEntry:
    """
    One entry whose data lives in another (already-open) TRE file. The bytes
    at [src_offset, src_offset + src_disk_len) in src_file are copied verbatim
    into the output.

    If compressor == CT_NONE, src_disk_len must equal uncomp_len. Otherwise
    src_disk_len is the on-disk compressed length.
    """
    name: str
    src_file: object         # open binary file-like, seekable
    src_offset: int
    src_disk_len: int        # bytes to copy from src
    uncomp_len: int          # logical (uncompressed) length
    compressor: int          # CT_NONE or CT_ZLIB
    crc: int                 # filename CRC, reused from source TOC


@dataclass
class DiskFileEntry:
    """
    One entry whose data lives in a loose file on disk. The file is read,
    optionally zlib-compressed (kept compressed only if it actually shrinks),
    and written into the output.

    `crc` is computed at write time via swg_crc.calc_path(name) if left as 0.
    """
    name: str
    disk_path: str
    crc: int = 0
    try_compress: bool = True


def _compress_disk_file(disk_path: str, name: str, crc: int, try_compress: bool):
    """Worker-process body: read a loose file, optionally zlib-compress it
    (kept compressed only if it actually shrinks), and md5 the on-disk
    bytes. Pure compute + local I/O, no reference to the output archive, so
    it's safe to run in a separate process. Must stay a module-level
    function so it's picklable for spawn-based multiprocessing.
    """
    with open(disk_path, 'rb') as fh:
        raw = fh.read()
    uncomp_len = len(raw)
    if try_compress and uncomp_len > 0:
        compressed = zlib.compress(raw, level=_ZLIB_LEVEL)
        if len(compressed) < uncomp_len:
            payload, compressor = compressed, CT_ZLIB
        else:
            payload, compressor = raw, CT_NONE
    else:
        payload, compressor = raw, CT_NONE
    resolved_crc = crc if crc else swg_crc.calc_path(name)
    md5 = hashlib.md5(payload).digest()
    return name, resolved_crc, uncomp_len, compressor, payload, md5


@dataclass
class _Prepared:
    """Internal: a staged entry with its final on-disk metadata recorded."""
    name: str
    crc: int
    uncomp_len: int
    compressor: int
    on_disk_len: int
    output_offset: int
    md5: bytes


class TreWriter:
    def __init__(self, output_path: str, workers: int | None = None):
        self.output_path = output_path
        self.entries: list = []   # PassThroughEntry | DiskFileEntry
        # DiskFileEntry prep (read + zlib level 9 + md5) is pure per-file CPU
        # work with no shared state, so it runs in a process pool; defaults
        # to one worker per core.
        self.workers = workers or os.cpu_count() or 4

    def add(self, entry) -> None:
        self.entries.append(entry)

    def write(self) -> None:
        if not self.entries:
            raise ValueError(f'{self.output_path}: no entries to write')

        # Detect collisions early - same path twice would corrupt the
        # name block, and same CRC twice breaks the runtime binary search.
        names_seen: dict[str, int] = {}
        for i, e in enumerate(self.entries):
            n = swg_crc.normalize_path(e.name)
            if n in names_seen:
                raise ValueError(f'duplicate path in entry list: {e.name!r} '
                                 f'(positions {names_seen[n]} and {i})')
            names_seen[n] = i

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)) or '.',
                    exist_ok=True)

        tmp_path = self.output_path + '.tmp'
        with open(tmp_path, 'wb') as out:
            # ---- 1. Header placeholder, filled in at end. ----
            out.write(b'\x00' * HEADER_STRUCT.size)

            # ---- 2. Stream each entry's data block. ----
            # Pass-through entries are sorted by (src_file, src_offset) so
            # reads from each source TRE are sequential. Disk entries are
            # processed in input order afterward.
            prepared: list[_Prepared | None] = [None] * len(self.entries)

            pt_indices = [i for i, e in enumerate(self.entries)
                          if isinstance(e, PassThroughEntry)]
            pt_indices.sort(key=lambda i: (id(self.entries[i].src_file),
                                            self.entries[i].src_offset))
            for i in pt_indices:
                prepared[i] = self._write_passthrough(out, self.entries[i])

            # DiskFileEntry prep is the CPU-heavy part (read + zlib -9 +
            # md5), so it's farmed out to a process pool. The actual writes
            # to `out` still happen here, sequentially in original entry
            # order, so output offsets stay deterministic across reruns
            # regardless of which worker finishes first.
            disk_indices = [i for i, e in enumerate(self.entries)
                            if isinstance(e, DiskFileEntry)]
            if disk_indices:
                max_workers = min(len(disk_indices), self.workers)
                results: dict[int, tuple] = {}
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(_compress_disk_file,
                                    self.entries[i].disk_path,
                                    self.entries[i].name,
                                    self.entries[i].crc,
                                    self.entries[i].try_compress): i
                        for i in disk_indices
                    }
                    for fut in as_completed(futures):
                        results[futures[fut]] = fut.result()
                for i in disk_indices:
                    name, crc, uncomp_len, compressor, payload, md5 = results[i]
                    offset = out.tell()
                    out.write(payload)
                    prepared[i] = _Prepared(name, crc, uncomp_len, compressor,
                                            len(payload), offset, md5)

            # ---- 3. Name block. ----
            name_buf = bytearray()
            name_offsets: list[int] = [0] * len(self.entries)
            for i, e in enumerate(self.entries):
                name_offsets[i] = len(name_buf)
                name_buf.extend(swg_crc.normalize_path(e.name).encode('latin-1'))
                name_buf.append(0)
            uncomp_name_size = len(name_buf)
            comp_name_block  = zlib.compress(bytes(name_buf), level=_ZLIB_LEVEL)

            # ---- 4. TOC, sorted by CRC ascending (required for runtime bsearch). ----
            order = sorted(range(len(self.entries)),
                           key=lambda i: (prepared[i].crc, name_offsets[i]))
            toc_buf = bytearray(TOC_ENTRY_V0005.size * len(self.entries))
            for write_idx, src_idx in enumerate(order):
                p = prepared[src_idx]
                assert p is not None
                comp_len_field = p.on_disk_len if p.compressor != CT_NONE else 0
                TOC_ENTRY_V0005.pack_into(
                    toc_buf, write_idx * TOC_ENTRY_V0005.size,
                    p.crc & 0xFFFFFFFF,
                    p.uncomp_len,
                    p.output_offset,
                    p.compressor,
                    comp_len_field,
                    name_offsets[src_idx],
                )
            comp_toc = zlib.compress(bytes(toc_buf), level=_ZLIB_LEVEL)

            # ---- 5. Append TOC + names + MD5 block. ----
            toc_offset = out.tell()
            out.write(comp_toc)
            out.write(comp_name_block)
            # Readers (SIE, patch tools) expect one 16-byte MD5 of each
            # entry's on-disk bytes after the name block, in TOC order.
            for src_idx in order:
                out.write(prepared[src_idx].md5)

            # ---- 6. Real header. ----
            out.seek(0)
            out.write(HEADER_STRUCT.pack(
                TOKEN_TREE_LE,
                VERSION_0005_LE,
                len(self.entries),
                toc_offset,
                CT_ZLIB,
                len(comp_toc),
                CT_ZLIB,
                len(comp_name_block),
                uncomp_name_size,
            ))

        os.replace(tmp_path, self.output_path)

    @staticmethod
    def _write_passthrough(out, e: PassThroughEntry) -> _Prepared:
        offset = out.tell()
        f = e.src_file
        f.seek(e.src_offset)
        remaining = e.src_disk_len
        digest = hashlib.md5()
        while remaining > 0:
            chunk = f.read(min(_COPY_CHUNK, remaining))
            if not chunk:
                raise IOError(f'{e.name}: source EOF after '
                              f'{e.src_disk_len - remaining}/{e.src_disk_len} bytes')
            out.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        return _Prepared(e.name, e.crc, e.uncomp_len, e.compressor,
                         e.src_disk_len, offset, digest.digest())

if __name__ == '__main__':
    print('build_tre.py is a library; use repack_retail.py or import TreWriter.',
          file=sys.stderr)
    sys.exit(0)
