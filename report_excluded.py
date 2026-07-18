#!/usr/bin/env python3
"""
Report on the textures categorize.py routed into the "never upscaled or
shipped" buckets (cube, special, ui, sky — see EXCLUDED_CATEGORIES in
hd_rerender.py), with whatever metadata the pipeline already has lying
around:

  category      - which bucket categorize.py put it in, and why (see
                  categorize.py's module docstring for the routing rules)
  width/height  - original DDS dimensions (from manifest.json)
  fmt/mips      - original BC format + mip count (from manifest.json)
  filesize      - on-disk size of the extracted DDS, in bytes
  origin_tre    - which source archive it was pulled from under
                  patch-priority layering (only if --tre is given)

By default this reports the four buckets the pipeline actually excludes
(cube/special/ui/sky); pass --category to narrow it or widen it to any
bucket categorize.py produces (arch/organic/hardsurface included).

Usage:
  python report_excluded.py --staging staging/reborn_textures
  python report_excluded.py --staging staging/reborn_textures --category ui --category special
  python report_excluded.py --staging staging/reborn_textures \\
      --tre E:/path/reborn_textures.tre --format csv --out excluded.csv

Reads:
  <staging>/categories.json       - category -> [names]  (categorize.py / phase_extract)
  <staging>/manifest.json         - name -> {width, height, fmt, mips}
  <staging>/dds_in/texture/*.dds  - on-disk file size

Writes: stdout (or --out) in --format {table, json, csv, md} (default table).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from categorize import categorize as categorize_one       # noqa: E402
from extract_tre import _collect_tre_paths, _open_tre     # noqa: E402

# Same set hd_rerender.py never upscales or ships; kept in sync manually
# since this is a read-only reporting tool, not a pipeline phase.
EXCLUDED_CATEGORIES = ('cube', 'special', 'ui', 'sky')
ALL_CATEGORIES = ('cube', 'special', 'ui', 'sky', 'arch', 'organic', 'hardsurface')


def load_categories(staging: Path) -> dict[str, list[str]]:
    """categories.json if phase_extract already wrote one; otherwise
    categorize on the fly from manifest.json (mirrors
    hd_rerender.load_excluded_names's fallback for older staging dirs)."""
    cat_path = staging / 'categories.json'
    if cat_path.exists():
        return json.loads(cat_path.read_text(encoding='utf-8'))
    src_dir = staging / 'dds_in' / 'texture'
    manifest = load_manifest(staging)
    cats: dict[str, list[str]] = {k: [] for k in ALL_CATEGORIES}
    for name in manifest:
        cats[categorize_one(name, src_dir / name)].append(name)
    return cats


def load_manifest(staging: Path) -> dict[str, dict]:
    manifest_path = staging / 'manifest.json'
    if not manifest_path.exists():
        return {}
    with manifest_path.open('r', encoding='utf-8') as f:
        return json.load(f).get('entries', {})


def scan_origin_tres(tre_src: Path, include_globs: list[str]) -> dict[str, tuple[str, int]]:
    """Walk tre_src (single .tre or a directory of them) in patch-priority
    order and return {basename: (origin_tre_basename, declared_length)} for
    the first (highest-priority) TRE that contains each matching entry.

    Mirrors extract_tre._collect_winners, but also keeps the declared
    on-disk length so callers get a file size even when staging/dds_in
    hasn't been populated (report can run straight off a --tre source).
    """
    tre_files = _collect_tre_paths(str(tre_src), reverse_order=True)
    seen: set[str] = set()
    result: dict[str, tuple[str, int]] = {}
    for tre_path in tre_files:
        try:
            f, entries, _ver = _open_tre(tre_path)
        except Exception as e:
            print(f'  WARN: skip {tre_path}: {e}', file=sys.stderr)
            continue
        f.close()
        base = Path(tre_path).name
        for (name, length, *_rest) in entries:
            nl = name.lower()
            if include_globs and not any(_fnmatch(nl, g.lower()) for g in include_globs):
                continue
            key = nl
            if key in seen:
                continue
            seen.add(key)
            result[Path(name).name] = (base, length)
    return result


def _fnmatch(name: str, glob: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(name, glob)


def build_records(staging: Path, categories: list[str], tre_src: Path | None) -> dict[str, list[dict]]:
    cats = load_categories(staging)
    manifest = load_manifest(staging)
    dds_dir = staging / 'dds_in' / 'texture'

    origins: dict[str, tuple[str, int]] = {}
    if tre_src is not None:
        origins = scan_origin_tres(tre_src, ['texture/*.dds'])

    out: dict[str, list[dict]] = {}
    for cat in categories:
        rows = []
        for name in cats.get(cat, []):
            meta = manifest.get(name, {})
            on_disk = dds_dir / name
            filesize = on_disk.stat().st_size if on_disk.exists() else None
            origin_tre, declared_len = origins.get(name, (None, None))
            if filesize is None:
                filesize = declared_len
            rows.append({
                'name': name,
                'category': cat,
                'width': meta.get('width'),
                'height': meta.get('height'),
                'fmt': meta.get('fmt'),
                'mips': meta.get('mips'),
                'filesize': filesize,
                'origin_tre': origin_tre,
            })
        out[cat] = rows
    return out


def write_table(records: dict[str, list[dict]], out) -> None:
    have_origin = any(r['origin_tre'] for rows in records.values() for r in rows)
    cols = ['name', 'width', 'height', 'fmt', 'mips', 'filesize']
    if have_origin:
        cols.append('origin_tre')
    for cat, rows in records.items():
        total_bytes = sum(r['filesize'] or 0 for r in rows)
        print(f'\n== {cat}  ({len(rows)} files, {total_bytes/1e6:.1f} MB) ==', file=out)
        if not rows:
            continue
        widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
        print('  ' + '  '.join(c.ljust(widths[c]) for c in cols), file=out)
        for r in rows:
            print('  ' + '  '.join(str(r[c]).ljust(widths[c]) for c in cols), file=out)


def write_json(records: dict[str, list[dict]], out) -> None:
    totals = {cat: len(rows) for cat, rows in records.items()}
    json.dump({'totals': totals, 'categories': records}, out, indent=2)
    out.write('\n')


def write_csv(records: dict[str, list[dict]], out) -> None:
    fieldnames = ['category', 'name', 'width', 'height', 'fmt', 'mips', 'filesize', 'origin_tre']
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()
    for rows in records.values():
        for r in rows:
            w.writerow(r)


def write_md(records: dict[str, list[dict]], out) -> None:
    have_origin = any(r['origin_tre'] for rows in records.values() for r in rows)
    cols = ['name', 'width', 'height', 'fmt', 'mips', 'filesize']
    if have_origin:
        cols.append('origin_tre')
    for cat, rows in records.items():
        print(f'\n### {cat} ({len(rows)} files)\n', file=out)
        if not rows:
            print('_none_', file=out)
            continue
        print('| ' + ' | '.join(cols) + ' |', file=out)
        print('| ' + ' | '.join('---' for _ in cols) + ' |', file=out)
        for r in rows:
            print('| ' + ' | '.join(str(r[c]) for c in cols) + ' |', file=out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--staging', required=True, help='staging dir, e.g. staging/<tre stem>')
    ap.add_argument('--tre', default=None,
                    help='source .tre (or directory of them) to resolve which archive each '
                         'file was pulled from under patch-priority layering; also used as a '
                         'file-size fallback when staging/dds_in has not been populated')
    ap.add_argument('--category', action='append', dest='categories', default=None,
                    choices=ALL_CATEGORIES,
                    help='repeatable; restrict to specific bucket(s). default: the four '
                         'buckets the pipeline never upscales or ships '
                         f'({", ".join(EXCLUDED_CATEGORIES)})')
    ap.add_argument('--format', choices=['table', 'json', 'csv', 'md'], default='table')
    ap.add_argument('--out', default=None, help='write to this path instead of stdout')
    args = ap.parse_args(argv)

    staging = Path(args.staging).resolve()
    if not staging.exists():
        ap.error(f'staging dir does not exist: {staging}')

    categories = args.categories or list(EXCLUDED_CATEGORIES)
    tre_src = Path(args.tre).resolve() if args.tre else None
    if tre_src is not None and not tre_src.exists():
        ap.error(f'--tre path does not exist: {tre_src}')

    records = build_records(staging, categories, tre_src)

    writer = {'table': write_table, 'json': write_json, 'csv': write_csv, 'md': write_md}[args.format]
    if args.out:
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            writer(records, f)
        total = sum(len(rows) for rows in records.values())
        print(f'wrote {total} records across {len(records)} categories -> {args.out}')
    else:
        writer(records, sys.stdout)

    return 0


if __name__ == '__main__':
    sys.exit(main())
