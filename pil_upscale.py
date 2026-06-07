#!/usr/bin/env python3
"""
Clean upscale pipeline: DDS -> PNG -> PIL Lanczos 4x -> PNG -> DDS.

Bypasses texconv's internal resize (which produced artifacts in our v3 run).
PIL Lanczos is the standard mathematical Lanczos resampling, pure Python,
deterministic, no surprises.

Pipeline per file:
  1. texconv decode: DDS -> PNG (preserves source RGBA exactly)
  2. PIL.Image.resize with LANCZOS filter -> 4x PNG
  3. texconv encode: PNG -> DDS with original BCx format + full mip chain
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
TEXCONV  = THIS_DIR / 'bin' / 'texconv.exe'

TEXCONV_FORMAT = {
    'DXT1':  'BC1_UNORM',
    'DXT3':  'BC2_UNORM',
    'DXT5':  'BC3_UNORM',
    'BC5':   'BC5_UNORM',
    'BC7':   'BC7_UNORM',
    'BGRA8': 'B8G8R8A8_UNORM',  # legacy DDS BGRA mask order (SWG retail default)
    'RGBA8': 'R8G8B8A8_UNORM',  # modern DDS RGBA mask order
    'BGR8':  'B8G8R8A8_UNORM',  # promote BGR to BGRA, preserves channel order
    'RGB8':  'R8G8B8A8_UNORM',
    'L8':    'R8_UNORM',
}


def read_dds_format(path: Path) -> str:
    """Return DDS format tag preserving channel order.
    BGRA8 = legacy "ARGB" mask convention (R in high bits, B in low bits)
    RGBA8 = modern mask convention (R in low bits, B in high bits)
    The bug: mixing these makes red/blue render swapped.
    """
    with path.open('rb') as f:
        head = f.read(128)
    pf_flags = struct.unpack('<I', head[80:84])[0]
    fourcc = head[84:88]
    rgb_bits = struct.unpack('<I', head[88:92])[0]
    r_mask = struct.unpack('<I', head[92:96])[0]
    b_mask = struct.unpack('<I', head[100:104])[0]
    a_mask = struct.unpack('<I', head[104:108])[0]
    if pf_flags & 0x4:
        if fourcc == b'DXT1': return 'DXT1'
        if fourcc == b'DXT3': return 'DXT3'
        if fourcc == b'DXT5': return 'DXT5'
        if fourcc in (b'ATI2', b'BC5U'): return 'BC5'
        return 'DXT5'
    if pf_flags & 0x40:
        # Detect channel order from masks. SWG/DX9-era DDS files use BGRA
        # (R mask in 0x00FF0000, B mask in 0x000000FF) — the "legacy" layout.
        # Modern DDS uses RGBA. Output format MUST match source's order, or
        # red and blue channels render swapped.
        if rgb_bits == 32:
            if r_mask == 0x00FF0000 and b_mask == 0x000000FF:
                return 'BGRA8'
            return 'RGBA8'
        if rgb_bits == 24:
            if r_mask == 0xFF0000:
                return 'BGR8'
            return 'RGB8'
        return 'RGBA8'
    if pf_flags & 0x20000:
        return 'L8'
    return 'DXT5'


FILTERS = {
    'lanczos':  Image.LANCZOS,
    'bicubic':  Image.BICUBIC,
    'hamming':  Image.HAMMING,
    'box':      Image.BOX,
}


def upscale_one(src_dds: Path, png_tmp_dir: Path, out_dds_dir: Path, scale: int = 4, filt=Image.LANCZOS) -> bool:
    """Full pipeline: DDS -> PNG -> 4x PNG -> DDS. Returns True on success."""
    name_stem = src_dds.stem
    src_fmt = read_dds_format(src_dds)
    tex_fmt = TEXCONV_FORMAT.get(src_fmt, 'BC3_UNORM')

    # 1) Decode source DDS to PNG
    rc = subprocess.call(
        [str(TEXCONV), '-nologo', '-y', '-ft', 'png', '-m', '1',
         '-o', str(png_tmp_dir), str(src_dds)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    src_png = png_tmp_dir / f'{name_stem}.png'
    if rc != 0 or not src_png.exists():
        return False

    # 2) PIL Lanczos resize to 4x
    img = Image.open(src_png)
    new_size = (img.size[0] * scale, img.size[1] * scale)
    out_img = img.resize(new_size, filt)
    # Write to a 4x PNG (overwrite the source PNG with the upscaled one)
    out_png = png_tmp_dir / f'{name_stem}_4x.png'
    out_img.save(out_png, optimize=False)

    # 3) Encode upscaled PNG back to DDS
    rc = subprocess.call(
        [str(TEXCONV), '-nologo', '-y',
         '-f', tex_fmt, '-m', '0', '-ft', 'dds',
         '-o', str(out_dds_dir), str(out_png)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # texconv outputs <png_stem>.dds; rename so it matches the original DDS name.
    produced = out_dds_dir / f'{name_stem}_4x.dds'
    target   = out_dds_dir / src_dds.name
    if produced.exists():
        if target.exists(): target.unlink()
        produced.rename(target)

    # Cleanup intermediate PNGs (optional - keep for inspection during debug)
    # src_png.unlink(missing_ok=True)
    # out_png.unlink(missing_ok=True)

    return rc == 0 and target.exists()


def main() -> int:
    ap = argparse.ArgumentParser(description='PIL-Lanczos 4x DDS upscale via PNG intermediates')
    ap.add_argument('--src-dir',  required=True)
    ap.add_argument('--out-dir',  required=True)
    ap.add_argument('--png-tmp',  required=True, help='intermediate PNG directory')
    ap.add_argument('--names',    required=False, help='optional JSON list of filenames')
    ap.add_argument('--scale',    type=int, default=4)
    ap.add_argument('--filter',   default='lanczos',
                    choices=list(FILTERS.keys()),
                    help='resampling filter (lanczos=sharp+ringing; bicubic=smoother)')
    args = ap.parse_args()
    filt = FILTERS[args.filter]

    if not TEXCONV.exists():
        raise SystemExit(f'texconv not found at {TEXCONV}')

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    png_tmp = Path(args.png_tmp)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_tmp.mkdir(parents=True, exist_ok=True)

    if args.names:
        with open(args.names, 'r', encoding='utf-8') as f:
            wanted = set(json.load(f))
        todo = [src_dir / n for n in sorted(wanted) if (src_dir / n).exists()]
    else:
        todo = sorted(src_dir.glob('*.dds'))

    print(f'PIL {args.filter} {args.scale}x: {len(todo)} files {src_dir} -> {out_dir}')
    t0 = time.time()
    ok = bad = 0
    for i, p in enumerate(todo, 1):
        # Skip if already done (idempotent restart support)
        target = out_dir / p.name
        if target.exists():
            ok += 1
            continue
        if upscale_one(p, png_tmp, out_dir, args.scale, filt):
            ok += 1
        else:
            bad += 1
        if i % 25 == 0 or i == len(todo):
            rate = i / max(0.001, time.time() - t0)
            eta = (len(todo) - i) / max(0.001, rate)
            print(f'  {i}/{len(todo)}  {rate:.1f} files/s  eta {eta:.0f}s')
    print(f'done: {ok} ok, {bad} failed, {time.time()-t0:.0f}s')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
