#!/usr/bin/env python3
"""
Dump a TRE archive's table of contents to CSV: tre,path,length.

TOC-only — reads headers and the name/TOC tables, never touches payload
bytes, so this works even against encrypted archives (e.g. SWGRestoration's
AES-encrypted TREs) where the data itself can't be extracted.

This is the format mirror_restoration.py's `manifest` command expects for
restoration-toc/{restoration,reborn,retail_swgnge}_index.csv.

Usage:
  python dump_toc_csv.py --src E:/SWGRestoration --out restoration-toc/restoration_index.csv
  python dump_toc_csv.py --src E:/SWGReborn      --out restoration-toc/reborn_index.csv
  python dump_toc_csv.py --src E:/SWGNGE_retail  --out restoration-toc/retail_swgnge_index.csv
  python dump_toc_csv.py --src bottom.tre        --out bottom_index.csv

--src may be a single .tre file or a directory of them (same patch-priority
walk order as extract_tre.py); every entry from every TRE is written,
duplicates across archives included, so downstream tools can see the full
per-archive picture rather than a pre-collapsed one.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from extract_tre import _collect_tre_paths, _open_tre, TreError   # noqa: E402


def dump_toc_csv(src: str, out_csv: str) -> int:
    tre_files = _collect_tre_paths(src, reverse_order=True)
    if not tre_files:
        print('no .tre files found', file=sys.stderr)
        return 2

    n_rows = 0
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['tre', 'path', 'length'])
        for tre_path in tre_files:
            base = Path(tre_path).name
            try:
                fh, entries, _ver = _open_tre(tre_path)
            except TreError as e:
                print(f'  WARN: skip {base}: {e}', file=sys.stderr)
                continue
            except Exception as e:
                print(f'  WARN: skip {base}: {e!r}', file=sys.stderr)
                continue
            fh.close()
            for (name, length, *_rest) in entries:
                w.writerow([base, name, length])
                n_rows += 1
            print(f'  {base}: {len(entries)} entries')

    print(f'wrote {n_rows} rows across {len(tre_files)} TRE(s) -> {out_csv}')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='.tre file or directory of them')
    ap.add_argument('--out', required=True, help='output CSV path')
    args = ap.parse_args(argv)
    return dump_toc_csv(args.src, args.out)


if __name__ == '__main__':
    sys.exit(main())
