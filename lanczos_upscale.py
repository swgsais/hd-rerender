#!/usr/bin/env python3
"""
Lanczos 4x upscale for architectural textures.

texconv handles DDS-in / DDS-out directly with LANCZOS3 filter and full
mipmap regen at the new size. No GPU, no AI hallucination. Fast (~10-50
files/sec depending on size).

Architectural textures already look great in source - this just gives them
4x linear resolution so they don't pixelate at close camera distance.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TEXCONV  = THIS_DIR / 'bin' / 'texconv.exe'

# Mirror the format mapping from hd_rerender.py so we re-encode to the original BCx.
TEXCONV_FORMAT = {
    'DXT1':  'BC1_UNORM',
    'DXT3':  'BC2_UNORM',
    'DXT5':  'BC3_UNORM',
    'BC5':   'BC5_UNORM',
    'BC7':   'BC7_UNORM',
    'RGBA8': 'R8G8B8A8_UNORM',
    'RGB8':  'R8G8B8A8_UNORM',
    'L8':    'R8_UNORM',
}


def read_meta(path: Path) -> dict:
    """Mini DDS header parse: width, height, fmt tag."""
    with path.open('rb') as f:
        head = f.read(128)
    if head[:4] != b'DDS ':
        raise ValueError(f'{path}: not a DDS file')
    height = struct.unpack('<I', head[12:16])[0]
    width  = struct.unpack('<I', head[16:20])[0]
    pf_flags = struct.unpack('<I', head[80:84])[0]
    fourcc = head[84:88]
    rgb_bits = struct.unpack('<I', head[88:92])[0]
    a_mask = struct.unpack('<I', head[104:108])[0]
    if pf_flags & 0x4:                            # FOURCC
        if fourcc == b'DXT1':            fmt = 'DXT1'
        elif fourcc == b'DXT3':          fmt = 'DXT3'
        elif fourcc == b'DXT5':          fmt = 'DXT5'
        elif fourcc in (b'ATI2', b'BC5U'): fmt = 'BC5'
        else: fmt = 'DXT5'  # safe default
    elif pf_flags & 0x40:                          # RGB
        if rgb_bits == 32 and a_mask:    fmt = 'RGBA8'
        elif rgb_bits == 24:             fmt = 'RGB8'
        else:                            fmt = 'RGBA8'
    elif pf_flags & 0x20000:                       # LUMINANCE
        fmt = 'L8'
    else:                                          fmt = 'DXT5'
    return {'width': width, 'height': height, 'fmt': fmt}


def lanczos_upscale_one(src: Path, out_dir: Path, scale: int = 4) -> bool:
    """Run texconv on one DDS: read meta, compute 4x dims, invoke texconv."""
    meta = read_meta(src)
    new_w = meta['width']  * scale
    new_h = meta['height'] * scale
    tex_fmt = TEXCONV_FORMAT.get(meta['fmt'], 'BC3_UNORM')
    cmd = [
        str(TEXCONV),
        '-nologo',
        '-y',
        '-singleproc',           # one file per invocation; parallelism comes
                                 # from running many invocations concurrently,
                                 # not from texconv's own internal threading
        '-w', str(new_w),
        '-h', str(new_h),
        '-if', 'CUBIC',         # Bicubic - standard choice for UPSCALE.
                                # FANT was producing artifacts (rings/oscillation)
                                # because it's optimized for DOWNSCALE only.
        '-m', '0',                          # full mip chain at new size
        '-f', tex_fmt,
        '-ft', 'dds',
        '-o', str(out_dir),
        str(src),
    ]
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc == 0 and (out_dir / src.name).exists()


def main() -> int:
    ap = argparse.ArgumentParser(description='Lanczos 4x DDS upscale via texconv')
    ap.add_argument('--src-dir',  required=True, help='dir of source DDS files')
    ap.add_argument('--out-dir',  required=True, help='dir for upscaled DDS output')
    ap.add_argument('--names',    required=False, help='optional JSON file with a list of filenames to process')
    ap.add_argument('--scale',    type=int, default=4)
    ap.add_argument('--workers',  type=int, default=os.cpu_count() or 4,
                    help='concurrent texconv processes (default: cpu_count)')
    args = ap.parse_args()

    if not TEXCONV.exists():
        raise SystemExit(f'texconv not found at {TEXCONV}')

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.names:
        with open(args.names, 'r', encoding='utf-8') as f:
            wanted = set(json.load(f))
        todo = [src_dir / n for n in wanted if (src_dir / n).exists()]
    else:
        todo = sorted(src_dir.glob('*.dds'))

    print(f'lanczos {args.scale}x: {len(todo)} files {src_dir} -> {out_dir}  '
          f'(workers={args.workers})')
    t0 = time.time()
    ok = bad = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(lanczos_upscale_one, p, out_dir, args.scale) for p in todo]
        for fut in as_completed(futures):
            if fut.result():
                ok += 1
            else:
                bad += 1
            done += 1
            if done % 50 == 0 or done == len(todo):
                rate = done / max(0.001, time.time() - t0)
                eta = (len(todo) - done) / max(0.001, rate)
                print(f'  {done}/{len(todo)}  {rate:.1f} files/s  eta {eta:.0f}s')
    print(f'done: {ok} ok, {bad} failed, {time.time()-t0:.0f}s')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
