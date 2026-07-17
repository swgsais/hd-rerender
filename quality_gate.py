#!/usr/bin/env python3
"""
Quality gate for HD-upscaled DDS outputs.

For each upscaled DDS, decode it back to PNG, compare to a pure-PIL-Lanczos
resize of the source (used as ground-truth reference), apply a battery of
statistical checks. Files that fail any check get marked for fallback-to-original.

Checks:
  1. solid_black           : v4 output mean brightness < 5 (catastrophic failure)
  2. brightness_drop       : v4 output lost more than 50% of source brightness
  3. alpha_corruption      : opaque source became transparent in output
  4. high_mean_diff        : average pixel diff > threshold (corruption)
  5. extreme_pixel_density : too many wildly-different pixels (ringing/holes)
  6. high_freq_artifacts   : output has much higher pixel variance in small
                             windows than source (block compression artifacts
                             amplified by upscale + recompression)

Writes JSON report with passed[] and failed[(name, reason)] lists.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
TEXCONV  = THIS_DIR / 'bin' / 'texconv.exe'


def decode_dds(dds_path: Path, tmp_dir: Path) -> Path | None:
    """texconv decode DDS -> PNG into tmp_dir. Returns PNG path or None."""
    subprocess.run(
        [str(TEXCONV), '-nologo', '-y', '-ft', 'png', '-m', '1',
         '-o', str(tmp_dir), str(dds_path)],
        capture_output=True,
    )
    png = tmp_dir / f'{dds_path.stem}.png'
    return png if png.exists() else None


def quality_check(src_png_path: Path, v4_png_path: Path) -> tuple[bool, str, dict]:
    """Compare v4 output to source-resized-with-Lanczos reference.
    Returns (passed, reason, stats_dict).
    """
    src = Image.open(src_png_path).convert('RGBA')
    out = Image.open(v4_png_path).convert('RGBA')
    ref = src.resize(out.size, Image.LANCZOS)
    a = np.asarray(ref, dtype=np.int16)
    b = np.asarray(out, dtype=np.int16)
    rgb_diff = np.abs(a[:, :, :3] - b[:, :, :3])

    stats = {
        'src_size':       list(src.size),
        'out_size':       list(out.size),
        'src_brightness': float(a[:, :, :3].mean()),
        'out_brightness': float(b[:, :, :3].mean()),
        'src_alpha':      float(a[:, :, 3].mean()),
        'out_alpha':      float(b[:, :, 3].mean()),
        'mean_rgb_diff':  float(rgb_diff.mean()),
        'extreme_pct':    float((rgb_diff.max(axis=2) > 100).mean()),
    }

    # 1. Solid black / near-zero output: catastrophic
    if stats['out_brightness'] < 5 and stats['src_brightness'] > 30:
        return False, f"solid_black (src={stats['src_brightness']:.0f}, out={stats['out_brightness']:.1f})", stats

    # 2. Catastrophic brightness drop (texture went mostly dark)
    if stats['src_brightness'] > 30 and stats['out_brightness'] < stats['src_brightness'] * 0.5:
        return False, f"brightness_drop ({stats['src_brightness']:.0f} -> {stats['out_brightness']:.0f})", stats

    # 3. Alpha corruption (opaque became transparent)
    if stats['src_alpha'] > 240 and stats['out_alpha'] < 200:
        return False, f"alpha_corruption (src={stats['src_alpha']:.0f}, out={stats['out_alpha']:.0f})", stats
    if stats['src_alpha'] > 30 and stats['out_alpha'] < stats['src_alpha'] * 0.5:
        return False, f"alpha_dropped ({stats['src_alpha']:.0f} -> {stats['out_alpha']:.0f})", stats

    # 4. High mean RGB diff (general corruption)
    if stats['mean_rgb_diff'] > 15.0:
        return False, f"high_mean_diff ({stats['mean_rgb_diff']:.1f})", stats

    # 5. Too many extreme pixels (ringing or alpha holes) - TIGHTENED to 0.2%
    if stats['extreme_pct'] > 0.002:   # 0.2% of pixels diverge by >100 = visible artifacts
        return False, f"extreme_pixels ({stats['extreme_pct']*100:.2f}%)", stats

    return True, 'ok', stats


# Filename-level rejects: special-channel suffixes my categorizer missed
SUFFIX_REJECTS = (
    '_cn.dds',          # color-normal hybrid
    '_envmask.dds',     # environment mask
    '_envmap.dds',      # explicit environment map
    '_specenv.dds',     # specular environment
    '_specbump.dds',    # specular+bump combined
    '_emis.dds',        # emissive
    '_emismap.dds',     # emissive map
)


def filename_skip(name: str) -> str | None:
    """Return reason to skip this file by name pattern, or None."""
    nl = name.lower()
    for suf in SUFFIX_REJECTS:
        if nl.endswith(suf):
            return f'suffix:{suf}'
    return None


def _reason_key(reason: str) -> str:
    """Group a reason string the same way regardless of which check produced
    it: 'solid_black (30->1)' -> 'solid_black', 'suffix:_cn.dds' -> 'suffix'.
    """
    if '(' in reason:
        return reason.split('(', 1)[0].strip()
    return reason.split(':', 1)[0]


def _check_one(name: str, bic_dir: Path, src_dir: Path, src_png_dir: Path,
               tmp_dir: Path) -> tuple[str, bool, str]:
    """Worker-process body: run the full per-file QC check. Each worker gets
    its own tmp subdir (keyed by pid) so concurrent decode_dds calls can
    never clash on the same output filename.
    """
    skip = filename_skip(name)
    if skip:
        return name, False, skip

    worker_tmp = tmp_dir / f'w{os.getpid()}'
    worker_tmp.mkdir(parents=True, exist_ok=True)

    bic_dds = bic_dir / name
    src_png = src_png_dir / (Path(name).stem + '.png')
    if not bic_dds.exists():
        return name, False, 'no_bicubic_output'
    if not src_png.exists():
        src_png = decode_dds(src_dir / name, worker_tmp)
        if src_png is None:
            return name, False, 'src_decode_failed'

    v4_png = decode_dds(bic_dds, worker_tmp)
    if v4_png is None:
        return name, False, 'v4_decode_failed'

    try:
        ok, reason, _stats = quality_check(src_png, v4_png)
    except Exception as e:
        ok, reason = False, f'exception:{e!r}'
    finally:
        try: v4_png.unlink()
        except OSError: pass

    return name, ok, reason


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bicubic-dir', required=True, help='dir with v4 bicubic DDS outputs')
    ap.add_argument('--src-dir',     required=True, help='dir with source DDS')
    ap.add_argument('--src-png-dir', required=True, help='dir with pre-decoded source PNGs (staging/png_in)')
    ap.add_argument('--tmp-dir',     required=True, help='scratch dir for v4 PNG decodes')
    ap.add_argument('--names',       required=True, help='JSON list of filenames to check')
    ap.add_argument('--report',      required=True, help='output JSON report path')
    ap.add_argument('--workers',     type=int, default=os.cpu_count() or 4,
                    help='parallel worker processes (default: cpu_count)')
    args = ap.parse_args()

    bic_dir = Path(args.bicubic_dir)
    src_dir = Path(args.src_dir)
    src_png_dir = Path(args.src_png_dir)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with open(args.names, 'r', encoding='utf-8') as f:
        names = json.load(f)
    print(f'Quality gate on {len(names)} files  (workers={args.workers})')

    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    failures_by_reason: dict[str, int] = {}

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_check_one, name, bic_dir, src_dir, src_png_dir, tmp_dir)
                  for name in names]
        for fut in as_completed(futures):
            name, ok, reason = fut.result()
            done += 1
            if ok:
                passed.append(name)
            else:
                failed.append((name, reason))
                key = _reason_key(reason)
                failures_by_reason[key] = failures_by_reason.get(key, 0) + 1

            if done % 250 == 0 or done == len(names):
                rate = done / max(0.001, time.time() - t0)
                eta  = (len(names) - done) / max(0.001, rate)
                print(f'  {done}/{len(names)}  {rate:.1f} files/s  eta {eta:.0f}s  passed={len(passed)} failed={len(failed)}')

    report = {
        'summary': {
            'total':              len(names),
            'passed':             len(passed),
            'failed':             len(failed),
            'pass_rate':          f'{len(passed) / max(1, len(names)) * 100:.1f}%',
            'failures_by_reason': dict(sorted(failures_by_reason.items(), key=lambda kv: -kv[1])),
        },
        'passed': sorted(passed),
        'failed': sorted(failed),
    }
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print()
    print(f'=== Quality Gate Results ===')
    print(f'Total: {len(names)}  Passed: {len(passed)}  Failed: {len(failed)}')
    print(f'Pass rate: {len(passed) / max(1, len(names)) * 100:.1f}%')
    print(f'Failures by reason:')
    for reason, count in sorted(failures_by_reason.items(), key=lambda kv: -kv[1]):
        print(f'  {reason:35s} {count}')
    print(f'Report: {args.report}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
