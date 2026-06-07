#!/usr/bin/env python3
"""
Build side-by-side comparison montages for the pilot A/B.

For each file in a category, lay out [source (nearest-upscaled) | variant1 |
variant2 | ...] as a labeled row, plus a 2x center-crop zoom row underneath so
ringing / hallucination (e.g. AI inventing grass on stone) is obvious.

Usage:
  python compare_pilot.py --category hardsurface --variants hs_lanczos,hs_span,hs_dat2 --n 12
  python compare_pilot.py --category arch        --variants pilot,arch_span        --n 12
        (variant "pilot" = the arch Lanczos@3x render; "main" = default routing)

Reads source PNGs from staging/png_in and each variant's upscaled PNGs from
staging/render/<variant>/png_out. Writes montages to staging/compare/<category>/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

THIS_DIR = Path(__file__).resolve().parent
STAGING  = THIS_DIR / 'staging'
PNG_IN   = STAGING / 'png_in'
RENDER   = STAGING / 'render'
CATS     = STAGING / 'restoration_categories.json'

PANEL_H  = 384     # display height of each full panel
CROP     = 192     # center-crop size (at source res) for the zoom row
LABEL_H  = 22


def label(img: Image.Image, text: str) -> Image.Image:
    """Add a label bar above an image."""
    out = Image.new('RGB', (img.width, img.height + LABEL_H), (24, 24, 24))
    out.paste(img.convert('RGB'), (0, LABEL_H))
    d = ImageDraw.Draw(out)
    d.text((4, 4), text, fill=(0, 255, 128))
    return out


def fit_h(img: Image.Image, h: int) -> Image.Image:
    w = max(1, round(img.width * h / img.height))
    return img.resize((w, h), Image.NEAREST)


def center_crop(img: Image.Image, box: int) -> Image.Image:
    box = min(box, img.width, img.height)
    l = (img.width - box) // 2
    t = (img.height - box) // 2
    return img.crop((l, t, l + box, t + box))


def hcat(imgs: list[Image.Image], gap: int = 6) -> Image.Image:
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new('RGB', (w, h), (40, 40, 40))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0)); x += im.width + gap
    return out


def vcat(imgs: list[Image.Image], gap: int = 6) -> Image.Image:
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new('RGB', (w, h), (40, 40, 40))
    y = 0
    for im in imgs:
        out.paste(im, (0, y)); y += im.height + gap
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--category', required=True)
    ap.add_argument('--variants', required=True, help='comma list of render variant tags')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cats = json.loads(CATS.read_text(encoding='utf-8'))
    names = sorted(cats.get(args.category, []))[:args.n]
    variants = args.variants.split(',')
    out_dir = Path(args.out) if args.out else (STAGING / 'compare' / args.category)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for base in names:
        stem = Path(base).stem
        src_p = PNG_IN / f'{stem}.png'
        if not src_p.exists():
            continue
        var_imgs = {}
        for v in variants:
            vp = RENDER / v / 'png_out' / f'{stem}.png'
            if vp.exists():
                var_imgs[v] = Image.open(vp)
        if not var_imgs:
            continue

        # output resolution = first available variant's size
        out_size = next(iter(var_imgs.values())).size
        src = Image.open(src_p)
        src_disp = src.resize(out_size, Image.NEAREST)   # blocky reference @ target res

        full_row = [label(fit_h(src_disp, PANEL_H), f'SOURCE {src.width}x{src.height} (nearest)')]
        for v, im in var_imgs.items():
            full_row.append(label(fit_h(im, PANEL_H), f'{v}  {im.width}x{im.height}'))

        # zoom row: center crop at source res, then nearest-scale to a fixed display
        zsrc = center_crop(src, CROP).resize((PANEL_H, PANEL_H), Image.NEAREST)
        zoom_row = [label(zsrc, 'src crop')]
        for v, im in var_imgs.items():
            scale = im.width / src.width
            zb = int(CROP * scale)
            zoom_row.append(label(center_crop(im, zb).resize((PANEL_H, PANEL_H), Image.NEAREST), f'{v} crop'))

        montage = vcat([hcat(full_row), hcat(zoom_row)])
        montage.save(out_dir / f'{stem}.png')
        made += 1

    print(f'wrote {made} montages -> {out_dir}')
    print(f'variants compared: {variants}')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
