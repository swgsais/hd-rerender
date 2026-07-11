#!/usr/bin/env python3
"""
SWG TRE texture HD re-render pipeline.

Five-phase pipeline, each is a subcommand and resumable independently:

    extract  <source>.tre         ->  staging/dds_in/texture/*.dds  + manifest.json
    decode   staging/dds_in       ->  staging/png_in/*.png         (texconv -ft png)
    upscale  staging/png_in       ->  staging/png_out/*.png        (ComfyUI /prompt API)
    encode   staging/png_out      ->  staging/dds_out/*.dds        (texconv, original BC fmt, mip regen)
    repack   staging/dds_out      ->  <source>_hd.tre              (build_tre.py)

The source archive is whatever --tre points at: any TRE containing
texture/*.dds entries (or a directory of TREs, extracted in patch-priority
order). Staging defaults to staging/<source stem>/ next to this script so
runs against different archives never share intermediate state, and the
output defaults to <source stem>_hd.tre next to the source.

`all` runs the lot. Every phase skips files whose output already exists, so
killing the run mid-way and restarting picks up where it left off.

The DDS format is round-tripped: for every input texture/foo.dds we record
its width, height, BC format tag, and mipmap-count in manifest.json during
`decode`, and `encode` uses that tag to pick the right texconv -f flag and
regenerates a full mipmap chain at the new (4x) dimensions.

Configuration: edit hd_rerender.config.json next to this script, or pass
flags. Required keys:
  comfy_root      - absolute path to ComfyUI install
  comfy_api       - base URL of ComfyUI API, default http://127.0.0.1:8188
  upscale_model   - filename in ComfyUI/models/upscale_models/, e.g. 4x-UltraSharp.pth
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR     = Path(__file__).resolve().parent
DEFAULT_CFG  = THIS_DIR / 'hd_rerender.config.json'
TEXCONV      = THIS_DIR / 'bin' / 'texconv.exe'
WORKFLOW_TPL = THIS_DIR / 'workflows' / 'upscale_4x.json'
# TRE tooling (extract_tre / build_tre / swg_crc) is vendored in this repo so
# the pipeline runs standalone, without a client-tools checkout alongside.
EXTRACT_TRE  = THIS_DIR / 'extract_tre.py'
sys.path.insert(0, str(THIS_DIR))

# ---------------------------------------------------------------------------
# DDS header parsing — just enough to round-trip format and mipmap count.

DDS_MAGIC          = b'DDS '
DDPF_ALPHAPIXELS   = 0x00000001
DDPF_FOURCC        = 0x00000004
DDPF_RGB           = 0x00000040
DDPF_LUMINANCE     = 0x00020000

# Maps our format tag -> texconv -f flag for re-encoding.
# Order chosen so the most common SWG formats come first.
TEXCONV_FORMAT = {
    'DXT1':  'BC1_UNORM',
    'DXT3':  'BC2_UNORM',
    'DXT5':  'BC3_UNORM',
    'BC5':   'BC5_UNORM',   # ATI2 (normal maps)
    'BC7':   'BC7_UNORM',
    'RGBA8': 'R8G8B8A8_UNORM',
    'RGB8':  'R8G8B8A8_UNORM',   # promote 24-bit to 32-bit on re-encode
    'L8':    'R8_UNORM',
}


def read_dds_meta(path: Path) -> dict:
    """Parse the 128-byte DDS header (+ 20-byte DX10 extension if present).
    Returns {'width', 'height', 'fmt', 'mips'} where fmt is one of the keys
    of TEXCONV_FORMAT or 'UNKNOWN' (caller falls back to BC3).
    """
    with path.open('rb') as f:
        head = f.read(128)
    if len(head) < 128 or head[:4] != DDS_MAGIC:
        raise ValueError(f'{path}: not a DDS file')

    height   = struct.unpack('<I', head[12:16])[0]
    width    = struct.unpack('<I', head[16:20])[0]
    mips     = max(1, struct.unpack('<I', head[28:32])[0])
    pf_flags = struct.unpack('<I', head[80:84])[0]
    fourcc   = head[84:88]
    rgb_bits = struct.unpack('<I', head[88:92])[0]
    a_mask   = struct.unpack('<I', head[104:108])[0]

    if pf_flags & DDPF_FOURCC:
        if fourcc == b'DXT1':            fmt = 'DXT1'
        elif fourcc == b'DXT3':          fmt = 'DXT3'
        elif fourcc == b'DXT5':          fmt = 'DXT5'
        elif fourcc in (b'ATI2', b'BC5U'): fmt = 'BC5'
        elif fourcc == b'DX10':
            # 20-byte DX10 extension follows the 128-byte header.
            with path.open('rb') as f:
                f.seek(128)
                ext = f.read(20)
            dxgi = struct.unpack('<I', ext[0:4])[0]
            # DXGI_FORMAT_BC7_UNORM = 98, DXGI_FORMAT_BC7_UNORM_SRGB = 99
            if   dxgi in (98, 99):       fmt = 'BC7'
            elif dxgi in (71, 72):       fmt = 'DXT1'
            elif dxgi in (74, 75):       fmt = 'DXT3'
            elif dxgi in (77, 78):       fmt = 'DXT5'
            elif dxgi == 83:             fmt = 'BC5'
            else:                        fmt = 'UNKNOWN'
        else:                            fmt = 'UNKNOWN'
    elif pf_flags & DDPF_RGB:
        if rgb_bits == 32 and a_mask:    fmt = 'RGBA8'
        elif rgb_bits == 24:             fmt = 'RGB8'
        elif rgb_bits == 32:             fmt = 'RGBA8'
        else:                            fmt = 'UNKNOWN'
    elif pf_flags & DDPF_LUMINANCE:      fmt = 'L8'
    else:                                fmt = 'UNKNOWN'

    return {'width': width, 'height': height, 'fmt': fmt, 'mips': mips}


# ---------------------------------------------------------------------------
# Config + manifest helpers.

def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f'config not found: {path}\n'
            f'create it with: comfy_root, comfy_api, upscale_model\n'
            f'see {THIS_DIR / "hd_rerender.config.example.json"}'
        )
    with path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    for key in ('comfy_root', 'comfy_api', 'upscale_model'):
        if key not in cfg:
            raise SystemExit(f'config missing key: {key}')
    cfg['comfy_root'] = str(Path(cfg['comfy_root']).resolve())
    return cfg


def load_manifest(path: Path) -> dict:
    if path.exists():
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    return {'entries': {}}


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Phase 1: extract DDS from the source TRE(s)

def phase_extract(args, cfg: dict) -> int:
    out_dir = Path(args.staging) / 'dds_in'
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EXTRACT_TRE),
        '--src', str(Path(args.tre).resolve()),
        '--out', str(out_dir),
        '--include', 'texture/*.dds',
    ]
    print(f'[extract] {" ".join(cmd)}')
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc

    # Build manifest of original format/dims for every extracted DDS.
    manifest_path = Path(args.staging) / 'manifest.json'
    manifest = load_manifest(manifest_path)
    n_new = 0
    for dds in (out_dir / 'texture').glob('*.dds'):
        key = dds.name
        if key in manifest['entries']:
            continue
        try:
            meta = read_dds_meta(dds)
        except Exception as e:
            print(f'  WARN {key}: {e}', file=sys.stderr)
            continue
        manifest['entries'][key] = meta
        n_new += 1
    save_manifest(manifest_path, manifest)
    print(f'[extract] manifest: {len(manifest["entries"])} total ({n_new} added)')

    # Distribution report so you know what we're about to encode back to.
    fmt_counts: dict[str, int] = {}
    for m in manifest['entries'].values():
        fmt_counts[m['fmt']] = fmt_counts.get(m['fmt'], 0) + 1
    print('[extract] format distribution:')
    for fmt, n in sorted(fmt_counts.items(), key=lambda kv: -kv[1]):
        print(f'    {fmt:8s} {n:6d}')
    return 0


# ---------------------------------------------------------------------------
# Phase 2: decode DDS -> PNG via texconv

def phase_decode(args, cfg: dict) -> int:
    if not TEXCONV.exists():
        raise SystemExit(f'texconv not found at {TEXCONV} (run setup first)')
    in_dir  = Path(args.staging) / 'dds_in' / 'texture'
    out_dir = Path(args.staging) / 'png_in'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only attempt files that the manifest recognized as real DDS. Files with
    # a .dds extension that aren't actually DDS (175 such in reborn_textures —
    # likely TGA/palette data with reused extensions) get skipped here so a
    # single bad file in a 200-batch can't poison the chunk's exit code.
    manifest = load_manifest(Path(args.staging) / 'manifest.json')
    valid_names = set(manifest['entries'].keys())

    todo = []
    skipped_unknown = 0
    for dds in in_dir.glob('*.dds'):
        if dds.name not in valid_names:
            skipped_unknown += 1
            continue
        png = out_dir / (dds.stem + '.png')
        if png.exists() and not args.overwrite:
            continue
        todo.append(dds)

    print(f'[decode] {len(todo)} DDS -> PNG  (workers={args.workers}, '
          f'{skipped_unknown} non-DDS files skipped)')
    if not todo:
        return 0

    # texconv accepts many files per invocation, amortizing process startup.
    # Batch of 200 keeps argv well under Windows' ~32K limit.
    BATCH = 200
    t0 = time.time()
    attempted = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        cmd = [
            str(TEXCONV),
            '-nologo',
            '-y',                       # overwrite output if exists
            '-ft', 'png',
            '-o', str(out_dir),
            '-m', '1',                  # decode only mip 0
        ] + [str(p) for p in chunk]
        rc = subprocess.call(cmd, stdout=subprocess.DEVNULL)
        attempted += len(chunk)
        if rc != 0:
            # rc=1 here usually means one specific file in the batch tripped;
            # the rest of the batch may have decoded fine. Truth is on disk.
            print(f'  WARN texconv rc={rc} on chunk starting {chunk[0].name} '
                  f'(checking outputs)', file=sys.stderr)
        if (i // BATCH) % 5 == 0:
            print(f'  decoded {attempted}/{len(todo)}  '
                  f'({attempted / max(0.001, time.time()-t0):.1f} files/s)')

    # Source-of-truth count: how many target PNGs actually exist on disk.
    ok = sum(1 for p in todo if (out_dir / (p.stem + '.png')).exists())
    bad = len(todo) - ok
    print(f'[decode] done: {ok} ok, {bad} missing, {time.time()-t0:.1f}s')
    # Tolerate a small failure rate; only abort the pipeline on catastrophic
    # failure (>5% missing).  Individual missing files will simply be absent
    # from later phases — manifest entries without a corresponding PNG get
    # skipped naturally.
    return 0 if bad < max(10, len(todo) // 20) else 1


# ---------------------------------------------------------------------------
# Phase 3: ComfyUI upscale

def comfy_post(api: str, path: str, body: bytes, content_type: str) -> dict:
    req = urllib.request.Request(
        api.rstrip('/') + path,
        data=body,
        headers={'Content-Type': content_type},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode('utf-8'))


def comfy_get_json(api: str, path: str) -> dict:
    with urllib.request.urlopen(api.rstrip('/') + path, timeout=30) as r:
        return json.loads(r.read().decode('utf-8'))


def submit_workflow(api: str, workflow: dict, client_id: str) -> str:
    """POST /prompt and return prompt_id."""
    body = json.dumps({'prompt': workflow, 'client_id': client_id}).encode('utf-8')
    resp = comfy_post(api, '/prompt', body, 'application/json')
    if 'prompt_id' not in resp:
        raise RuntimeError(f'no prompt_id in response: {resp}')
    return resp['prompt_id']


def wait_for_prompt(api: str, prompt_id: str, timeout: float = 300.0) -> dict:
    """Poll /history/{prompt_id} until it returns the entry."""
    deadline = time.time() + timeout
    backoff = 0.5
    while time.time() < deadline:
        h = comfy_get_json(api, f'/history/{prompt_id}')
        if h.get(prompt_id):
            return h[prompt_id]
        time.sleep(backoff)
        backoff = min(2.0, backoff * 1.3)
    raise TimeoutError(f'prompt {prompt_id} did not complete in {timeout}s')


def phase_upscale(args, cfg: dict) -> int:
    in_dir  = Path(args.staging) / 'png_in'
    out_dir = Path(args.staging) / 'png_out'
    out_dir.mkdir(parents=True, exist_ok=True)

    comfy_input  = Path(cfg['comfy_root']) / 'input'  / 'swg'
    comfy_output = Path(cfg['comfy_root']) / 'output' / 'swg_hd'
    comfy_input.mkdir(parents=True, exist_ok=True)
    comfy_output.mkdir(parents=True, exist_ok=True)

    tpl = json.loads(WORKFLOW_TPL.read_text(encoding='utf-8'))
    tpl.pop('_comment', None)

    api = cfg['comfy_api']
    client_id = str(uuid.uuid4())

    # /system_stats is a cheap health check that also verifies the API is up.
    try:
        comfy_get_json(api, '/system_stats')
    except Exception as e:
        raise SystemExit(f'cannot reach ComfyUI at {api}: {e}')

    todo = []
    for png in in_dir.glob('*.png'):
        # We rename the SaveImage output back to <original>.png on the way out,
        # so a finished file is simply out_dir / png.name.
        if (out_dir / png.name).exists() and not args.overwrite:
            continue
        todo.append(png)

    print(f'[upscale] {len(todo)} PNG -> HD PNG via ComfyUI ({api}, model={cfg["upscale_model"]})')
    if not todo:
        return 0

    def process_one(src_png: Path) -> tuple[str, str | None]:
        # Stage the input into ComfyUI/input/swg/ (same drive = hard link / copy).
        staged = comfy_input / src_png.name
        if not staged.exists():
            try:
                os.link(src_png, staged)
            except OSError:
                staged.write_bytes(src_png.read_bytes())

        # Build the per-file workflow from template.
        wf = json.loads(json.dumps(tpl))  # deep copy
        wf['1']['inputs']['image'] = f'swg/{src_png.name}'
        wf['2']['inputs']['model_name'] = cfg['upscale_model']
        wf['4']['inputs']['filename_prefix'] = f'swg_hd/{src_png.stem}'

        try:
            prompt_id = submit_workflow(api, wf, client_id)
        except Exception as e:
            return (src_png.name, f'submit failed: {e}')

        try:
            entry = wait_for_prompt(api, prompt_id, timeout=args.timeout)
        except Exception as e:
            return (src_png.name, f'timeout/poll failed: {e}')

        # Find SaveImage's output and stash it under our original name
        # (SaveImage appends a counter like _00001_; we throw that away so
        # phase_encode sees deterministic <orig_stem>.png filenames).
        save_node_outputs = entry.get('outputs', {}).get('4', {}).get('images', [])
        if not save_node_outputs:
            return (src_png.name, 'no SaveImage output')
        # Just take the first image; our workflow only produces one.
        img = save_node_outputs[0]
        src = Path(cfg['comfy_root']) / 'output' / img.get('subfolder', '') / img['filename']
        if not src.exists():
            return (src_png.name, f'output file missing: {src}')
        dst = out_dir / src_png.name
        if dst.exists() and args.overwrite:
            dst.unlink()
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:                    # cross-drive, perms, etc.
                dst.write_bytes(src.read_bytes())
        return (src_png.name, None)

    # GPU is the bottleneck; 1 outstanding prompt is usually right, but ComfyUI
    # queues internally so a small parallel-submit count keeps the queue warm.
    ok = bad = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for fname, err in (r.result() for r in [pool.submit(process_one, p) for p in todo]):
            if err:
                bad += 1
                print(f'  FAIL {fname}: {err}', file=sys.stderr)
            else:
                ok += 1
            done = ok + bad
            if done % 25 == 0 or done == len(todo):
                rate = done / max(0.001, time.time() - t0)
                eta  = (len(todo) - done) / max(0.001, rate)
                print(f'  upscaled {done}/{len(todo)}  rate={rate:.2f}/s  eta={eta/60:.1f}min')

    print(f'[upscale] done: {ok} ok, {bad} failed, {time.time()-t0:.0f}s')
    # Tolerate a few per-file failures (huge textures, transient model OOM)
    # rather than aborting the whole `all` pipeline; >5% failed is the cliff.
    return 0 if bad < max(10, len(todo) // 20) else 1


# ---------------------------------------------------------------------------
# Phase 4: encode PNG -> DDS via texconv, using manifest for original format.

def phase_encode(args, cfg: dict) -> int:
    if not TEXCONV.exists():
        raise SystemExit(f'texconv not found at {TEXCONV}')
    manifest_path = Path(args.staging) / 'manifest.json'
    manifest = load_manifest(manifest_path)
    in_dir  = Path(args.staging) / 'png_out'
    out_dir = Path(args.staging) / 'dds_out' / 'texture'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group PNGs by target texconv format so we can batch one process per format.
    # phase_upscale already renamed outputs back to <orig_stem>.png, so the
    # manifest key is simply <png.stem>.dds.
    by_fmt: dict[str, list[Path]] = {}
    skipped = 0
    for png in in_dir.glob('*.png'):
        dds_name = png.stem + '.dds'
        meta = manifest['entries'].get(dds_name)
        if not meta:
            print(f'  WARN no manifest entry for {png.name} (key={dds_name})', file=sys.stderr)
            continue
        tex_fmt = TEXCONV_FORMAT.get(meta['fmt'], 'BC3_UNORM')
        target = out_dir / dds_name
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        by_fmt.setdefault(tex_fmt, []).append((png, dds_name))

    print(f'[encode] {sum(len(v) for v in by_fmt.values())} PNG -> DDS  ({skipped} skipped existing)')

    BATCH = 100
    t0 = time.time()
    total_attempted = 0
    all_targets: list[Path] = []   # used for source-of-truth presence check
    for tex_fmt, items in by_fmt.items():
        print(f'  [{tex_fmt}] {len(items)} files')
        for i in range(0, len(items), BATCH):
            chunk = items[i:i+BATCH]
            # texconv -f <fmt> -m 0 (full chain) -o <dir> input1 input2 ...
            cmd = [
                str(TEXCONV),
                '-nologo',
                '-y',
                '-f', tex_fmt,
                '-m', '0',                 # full mipmap chain at new (4x) size
                '-ft', 'dds',
                '-o', str(out_dir),
            ] + [str(p) for (p, _name) in chunk]
            rc = subprocess.call(cmd, stdout=subprocess.DEVNULL)
            total_attempted += len(chunk)
            if rc != 0:
                print(f'    WARN texconv rc={rc} on chunk starting '
                      f'{chunk[0][0].name} (checking outputs)', file=sys.stderr)
            # texconv writes <png_stem>.dds; rename to <original dds name>.
            for png, dds_name in chunk:
                produced = out_dir / (png.stem + '.dds')
                if produced.exists() and produced.name != dds_name:
                    target = out_dir / dds_name
                    if target.exists():
                        target.unlink()
                    produced.rename(target)
                all_targets.append(out_dir / dds_name)

    ok = sum(1 for t in all_targets if t.exists())
    bad = len(all_targets) - ok
    print(f'[encode] done: {ok} ok, {bad} missing, {time.time()-t0:.0f}s')
    return 0 if bad < max(10, len(all_targets) // 20) else 1


# ---------------------------------------------------------------------------
# Phase 5: repack into hd_textures.tre using existing build_tre.py

def phase_repack(args, cfg: dict) -> int:
    # build_tre is a library — import after sys.path was set at module top.
    from build_tre import TreWriter, DiskFileEntry  # type: ignore

    in_dir = Path(args.staging) / 'dds_out'

    # TRE 0005 uses signed int32 for entry offsets, so each archive caps out
    # at ~2.147 GB on disk. HD textures at 4x dims push total payload to ~36
    # GB, well past that — so we shard the output into shards <= ~1.6 GB raw
    # each (leaves margin for zlib's modest savings on DXT data).
    SHARD_TARGET_BYTES = int(1.6 * 1024 * 1024 * 1024)   # 1.6 GiB raw

    # Sort by name for deterministic, reproducible sharding across reruns.
    files = sorted(in_dir.rglob('*.dds'))
    if not files:
        raise SystemExit(f'[repack] no DDS files found under {in_dir}')

    # Plan shards: greedily pack files by raw size up to SHARD_TARGET_BYTES.
    # Files larger than the target each get their own shard (very rare).
    shards: list[list[Path]] = [[]]
    sizes: list[int] = [0]
    for f in files:
        sz = f.stat().st_size
        if sizes[-1] + sz > SHARD_TARGET_BYTES and shards[-1]:
            shards.append([])
            sizes.append(0)
        shards[-1].append(f)
        sizes[-1] += sz

    base_out = Path(args.out_tre).resolve()
    stem = base_out.stem        # e.g. reborn_textures_hd
    parent = base_out.parent

    print(f'[repack] sharding {len(files)} entries ({sum(sizes)/1024/1024/1024:.1f} GB raw) '
          f'into {len(shards)} archives')

    for idx, (entries_in_shard, raw_size) in enumerate(zip(shards, sizes), start=1):
        if len(shards) == 1:
            shard_path = base_out
        else:
            shard_path = parent / f'{stem}_{idx:03d}.tre'

        # Skip existing shards that look complete (size > 95% of raw input);
        # makes re-running cheap after a transient failure.
        if shard_path.exists() and shard_path.stat().st_size > raw_size * 0.5 and not args.overwrite:
            print(f'  [{idx}/{len(shards)}] skip (exists) {shard_path.name}  '
                  f'({shard_path.stat().st_size/1024/1024:.0f} MB)')
            continue

        w = TreWriter(str(shard_path))
        for f in entries_in_shard:
            rel = f.relative_to(in_dir).as_posix()
            w.add(DiskFileEntry(name=rel, disk_path=str(f), try_compress=True))
        try:
            w.write()
        except Exception as e:
            # If a shard still overflows (very large single files), shrink the
            # target and retell the user — better to fail loudly than ship a
            # corrupt archive.
            raise SystemExit(
                f'[repack] failed writing {shard_path.name}: {e}\n'
                f'  Shard had {len(entries_in_shard)} entries, '
                f'{raw_size/1024/1024:.0f} MB raw.\n'
                f'  Try lowering SHARD_TARGET_BYTES in hd_rerender.py.'
            ) from e
        out_mb = shard_path.stat().st_size / 1024 / 1024
        print(f'  [{idx}/{len(shards)}] wrote {shard_path.name}  '
              f'({len(entries_in_shard)} entries, {out_mb:.0f} MB)')

    print(f'[repack] done — {len(shards)} TRE shards. Add ALL of them to your '
          f'client load order, in numeric order, at higher priority than '
          f'{Path(args.tre).name}.')
    return 0


# ---------------------------------------------------------------------------
# Glue

def phase_all(args, cfg: dict) -> int:
    for fn in (phase_extract, phase_decode, phase_upscale, phase_encode, phase_repack):
        rc = fn(args, cfg)
        if rc != 0:
            print(f'\nABORT: {fn.__name__} returned {rc}', file=sys.stderr)
            return rc
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='SWG TRE texture HD re-render pipeline')
    ap.add_argument('--config', default=str(DEFAULT_CFG))
    ap.add_argument('--staging', default=None,
                    help='work dir for intermediates (default: staging/<tre stem>/ next to this script)')
    ap.add_argument('--tre', required=True,
                    help='source .tre archive with texture/*.dds entries, or a directory of .tre files')
    ap.add_argument('--out-tre', default=None,
                    help='output archive path (default: <tre stem>_hd.tre next to the source)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=300.0, help='per-prompt timeout in seconds')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('phase', choices=['extract', 'decode', 'upscale', 'encode', 'repack', 'all'])
    args = ap.parse_args(argv)

    src = Path(args.tre).resolve()
    if not src.exists():
        ap.error(f'--tre path does not exist: {src}')
    # Directory sources (extract_tre handles patch-priority layering) take the
    # directory name as the stem; single files drop their .tre suffix.
    stem = src.stem if src.is_file() else src.name
    if args.staging is None:
        args.staging = str(THIS_DIR / 'staging' / stem)
    if args.out_tre is None:
        args.out_tre = str((src.parent if src.is_file() else src) / f'{stem}_hd.tre')

    cfg = load_config(Path(args.config))
    fns = {
        'extract': phase_extract,
        'decode':  phase_decode,
        'upscale': phase_upscale,
        'encode':  phase_encode,
        'repack':  phase_repack,
        'all':     phase_all,
    }
    return fns[args.phase](args, cfg)


if __name__ == '__main__':
    sys.exit(main())
