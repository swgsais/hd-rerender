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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR     = Path(__file__).resolve().parent
DEFAULT_CFG  = THIS_DIR / 'hd_rerender.config.json'
TEXCONV      = THIS_DIR / 'bin' / 'texconv.exe'
WORKFLOW_TPL = THIS_DIR / 'workflows' / 'upscale_4x_batch.json'
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

# texconv formats whose block compressor already spreads a *single* file's
# work across all available cores (DirectXTex's BC7 encoder in particular).
# Running many of these concurrently oversubscribes the machine, so they get
# a small, no-singleproc process pool. Every other format compresses fast
# per file with little/no internal threading, so those run -singleproc,
# workers-wide, trading one-file-per-core for one-invocation-uses-all-cores.
SELF_THREADED_TEXCONV_FORMATS = {'BC7_UNORM'}


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


# Texture buckets that must never be AI-upscaled (see categorize.py): the
# engine reads them as structured data, not imagery. Upscaled versions of
# these are what corrupted load screens, character face tinting, and sky
# gradients in-game.
EXCLUDED_CATEGORIES = ('cube', 'special', 'ui', 'sky')


def load_excluded_names(staging: Path) -> set[str]:
    """DDS basenames the pipeline must not upscale or ship. Reads
    categories.json written by phase_extract; for older staging dirs that
    predate it, categorizes on the fly from the manifest."""
    cat_path = staging / 'categories.json'
    if cat_path.exists():
        cats = json.loads(cat_path.read_text(encoding='utf-8'))
    else:
        from categorize import categorize
        src_dir = staging / 'dds_in' / 'texture'
        cats = {}
        for name in load_manifest(staging / 'manifest.json')['entries']:
            cats.setdefault(categorize(name, src_dir / name), []).append(name)
    return {n for k in EXCLUDED_CATEGORIES for n in cats.get(k, [])}


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

    # Category routing. Cube maps, UI atlases, and channel-data textures
    # (normals/masks, gradient LUTs, customization patterns) are data the
    # engine reads structurally, not imagery — AI-upscaling them corrupts
    # load screens, character face tinting, and sky gradients in-game.
    # Later phases skip these buckets entirely; the client falls back to the
    # original archive for those entries.
    from categorize import categorize
    cats: dict[str, list[str]] = {k: [] for k in
                                  ('cube', 'special', 'ui', 'sky', 'arch', 'organic', 'hardsurface')}
    for name in manifest['entries']:
        cats[categorize(name, out_dir / 'texture' / name)].append(name)
    for k in cats:
        cats[k].sort()
    (Path(args.staging) / 'categories.json').write_text(
        json.dumps(cats, indent=0), encoding='utf-8')
    print('[extract] category routing:')
    for k, v in cats.items():
        skip = '   -> skipped (engine data, ships as original)' if k in EXCLUDED_CATEGORIES else ''
        print(f'    {k:12s} {len(v):6d}{skip}')
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
    excluded = load_excluded_names(Path(args.staging))

    todo = []
    skipped_unknown = 0
    skipped_excluded = 0
    skipped_toobig = 0
    for dds in in_dir.glob('*.dds'):
        if dds.name not in valid_names:
            skipped_unknown += 1
            continue
        if dds.name in excluded:
            skipped_excluded += 1
            continue
        # Sources already at/above --max-source-dim never reach the upscale
        # phase (see phase_upscale) - skip decoding them here too so a huge
        # texture doesn't burn texconv time/disk turning into a PNG we're
        # just going to discard a moment later. Keeps the cap enforced as
        # early in the pipeline as possible, same as encode/repack already do.
        meta = manifest['entries'].get(dds.name)
        if meta is not None and max(meta['width'], meta['height']) > args.max_source_dim:
            skipped_toobig += 1
            continue
        png = out_dir / (dds.stem + '.png')
        if png.exists() and not args.overwrite:
            continue
        todo.append(dds)

    print(f'[decode] {len(todo)} DDS -> PNG  (workers={args.workers}, '
          f'{skipped_unknown} non-DDS + {skipped_excluded} engine-data + '
          f'{skipped_toobig} >{args.max_source_dim}px files skipped)')
    if not todo:
        return 0

    # texconv accepts many files per invocation, amortizing process startup.
    # Batch of 200 keeps argv well under Windows' ~32K limit. Batches are
    # dispatched --workers-wide as concurrent texconv child processes (the
    # decode direction has no BC compressor to self-thread, so a serial
    # invocation-per-batch loop was leaving every core but one idle).
    # -singleproc keeps each child from also spinning up its own internal
    # thread pool and fighting the others for cores.
    BATCH = 200
    chunks = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    base_cmd = [
        str(TEXCONV),
        '-nologo',
        '-y',                       # overwrite output if exists
        '-singleproc',
        '-ft', 'png',
        '-o', str(out_dir),
        '-m', '1',                  # decode only mip 0
    ]

    t0 = time.time()
    attempted = 0
    completed_chunks = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(subprocess.call, base_cmd + [str(p) for p in c],
                                stdout=subprocess.DEVNULL): c for c in chunks}
        for fut in as_completed(futures):
            chunk = futures[fut]
            rc = fut.result()
            attempted += len(chunk)
            completed_chunks += 1
            if rc != 0:
                # rc=1 here usually means one specific file in the batch tripped;
                # the rest of the batch may have decoded fine. Truth is on disk.
                print(f'  WARN texconv rc={rc} on chunk starting {chunk[0].name} '
                      f'(checking outputs)', file=sys.stderr)
            if completed_chunks % 5 == 0 or completed_chunks == len(chunks):
                print(f'  decoded ~{attempted}/{len(todo)}  '
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


def comfy_interrupt(api: str) -> None:
    """Best-effort: cancel whatever ComfyUI is currently executing.

    Giving up on a poll (wait_for_prompt raising TimeoutError) does NOT
    cancel the job server-side - it just stops us waiting for it. Without
    this, a caller that resubmits after a timeout piles a new job in behind
    one that's still running/queued, which is what actually causes ComfyUI
    to look permanently stuck under repeated retries: each timeout adds net
    new backlog instead of clearing anything. Only unambiguous when at most
    one batch is in flight at a time (see --workers on the upscale phase).
    """
    try:
        req = urllib.request.Request(api.rstrip('/') + '/interrupt', data=b'', method='POST')
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # best-effort only - don't let a failed cancel mask the real error


# Pixel budget per batch: a (256,256) batch gets ~128 images, (512,512) ~32,
# (1024,1024) ~8, (2048,2048) ~2. Doubled from the original conservative
# starting point after a real run showed peak VRAM (Dedicated Memory) never
# crossing ~6.3GB / 38% on a 16GB card - retry_shrinking() below is the
# safety net if this still overshoots on some batch, so keep raising this
# (and re-checking peak VRAM) as long as there's clear headroom.
DEFAULT_BATCH_PIXEL_BUDGET = 256 * 256 * 512


def read_png_dims(path: Path) -> tuple[int, int]:
    """Width/height straight out of the PNG signature + IHDR chunk (first 24
    bytes) - same struct-unpack approach as read_dds_meta above, and much
    cheaper across thousands of files than routing through PIL's full
    Image.open plugin dispatch just to read two integers.
    """
    with path.open('rb') as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'{path}: not a PNG file')
    return struct.unpack('>II', head[16:24])


def load_png_dims(staging: Path, todo: list[Path]) -> dict[Path, tuple[int, int]]:
    """Real on-disk (width, height) for each PNG that has a manifest entry.

    Batches must be grouped by what's actually in the PNG file, not by the
    manifest's recorded width/height for its source DDS: phase_extract only
    ever writes a manifest entry once (`if key in manifest['entries']:
    continue`) and phase_decode skips re-decoding an existing PNG, so a
    leftover/stale png_in file can silently disagree with a since-updated
    manifest entry. Grouping by the stale manifest number put a
    differently-sized image in a same-size batch and crashed
    SWGLoadImageBatch, which validates the real image bytes. Files with no
    manifest entry at all are still excluded - phase_encode has nothing to
    re-encode them to.
    """
    manifest = load_manifest(staging / 'manifest.json')
    entries = manifest['entries']
    dims: dict[Path, tuple[int, int]] = {}
    for p in todo:
        if entries.get(p.stem + '.dds') is None:
            continue
        try:
            dims[p] = read_png_dims(p)
        except Exception as e:
            print(f'  WARN {p.name}: cannot read PNG dims: {e}', file=sys.stderr)
    return dims


def build_upscale_batches(dims: dict[Path, tuple[int, int]], pixel_budget: int) -> list[list[Path]]:
    """Group same-size files into VRAM-budgeted batches. One /prompt
    submission per batch instead of per file is the whole point - see
    phase_upscale for why.
    """
    by_size: dict[tuple[int, int], list[Path]] = {}
    for p, wh in dims.items():
        by_size.setdefault(wh, []).append(p)

    batches: list[list[Path]] = []
    for (w, h), files in by_size.items():
        files.sort()  # deterministic batch membership across reruns
        cap = max(1, pixel_budget // (w * h))
        for i in range(0, len(files), cap):
            batches.append(files[i:i + cap])
    return batches


def retry_shrinking(batch: list[Path], attempt) -> list[tuple[str, str | None]]:
    """Run attempt(batch) and, on any failure, retry just the failed subset
    split into two smaller batches - halving bottoms out at size 1, so this
    is bounded and terminates. A batch-level failure is most likely VRAM
    exhaustion at that batch size (the upscale workflow is all-or-nothing
    per graph execution), so a too-large --batch-pixel-budget self-corrects
    at runtime instead of just failing outright.

    Timeouts (error strings tagged 'TIMEOUT: ' by attempt()) are excluded
    from this - resubmitting a timed-out batch doesn't address anything
    (the batch size likely wasn't the problem, and ComfyUI may still be
    working through the original, now-interrupted job), it just piles more
    load behind whatever's actually stuck. Report those once and let a
    later `upscale` re-run pick them up via the normal skip-existing
    resumability, instead of auto-splitting into a retry storm.

    attempt(batch) -> list[(name, err|None)] in the same order as batch.
    """
    outcomes = attempt(batch)
    by_name = {p.name: p for p in batch}
    results = [(n, e) for n, e in outcomes if e is None]
    failed = [(n, e) for n, e in outcomes if e is not None]
    # Timeouts are always final, even when mixed with other failures in the
    # same batch — only the non-timeout subset is worth re-splitting.
    results += [(n, e) for n, e in failed if e.startswith('TIMEOUT: ')]
    failed = [(n, e) for n, e in failed if not e.startswith('TIMEOUT: ')]
    if not failed:
        return results
    if len(failed) == 1:
        n, e = failed[0]
        return results + [(n, f'{e} (size-1, no further retry possible)')]

    failed_batch = [by_name[n] for n, _e in failed]
    mid = len(failed_batch) // 2
    print(f'  [upscale] {len(failed_batch)}/{len(batch)} failed in a batch '
          f'({failed[0][1]}), retrying as batches of {mid} and {len(failed_batch)-mid}',
          file=sys.stderr, flush=True)
    return (results
            + retry_shrinking(failed_batch[:mid], attempt)
            + retry_shrinking(failed_batch[mid:], attempt))


def phase_upscale(args, cfg: dict) -> int:
    in_dir  = Path(args.staging) / 'png_in'
    out_dir = Path(args.staging) / 'png_out'
    out_dir.mkdir(parents=True, exist_ok=True)

    comfy_input      = Path(cfg['comfy_root']) / 'input'  / 'swg'
    comfy_output_root = Path(cfg['comfy_root']) / 'output'
    batch_subfolder  = 'swg_hd_batch'
    comfy_input.mkdir(parents=True, exist_ok=True)
    (comfy_output_root / batch_subfolder).mkdir(parents=True, exist_ok=True)

    tpl = json.loads(WORKFLOW_TPL.read_text(encoding='utf-8'))
    tpl.pop('_comment', None)

    api = cfg['comfy_api']
    client_id = str(uuid.uuid4())

    # /system_stats is a cheap health check that also verifies the API is up.
    try:
        comfy_get_json(api, '/system_stats')
    except Exception as e:
        raise SystemExit(f'cannot reach ComfyUI at {api}: {e}')

    # The batch workflow depends on our custom nodes; fail once with install
    # instructions instead of letting every /prompt bounce on an unknown
    # class_type.
    try:
        node_info = comfy_get_json(api, '/object_info/SWGLoadImageBatch')
    except Exception:
        node_info = {}
    if 'SWGLoadImageBatch' not in node_info:
        raise SystemExit(
            'ComfyUI is running but the SWG batch nodes are not loaded.\n'
            f'Copy {THIS_DIR / "comfyui_custom_nodes" / "swg_batch_io.py"} into\n'
            f'{Path(cfg["comfy_root"]) / "custom_nodes"}\\ and restart ComfyUI.'
        )

    excluded = load_excluded_names(Path(args.staging))
    todo = []
    skipped_excluded = 0
    for png in in_dir.glob('*.png'):
        # A finished file is simply out_dir / png.name - our save node
        # writes under the original filename, no counter/prefix guessing.
        if (png.stem + '.dds') in excluded:      # stale decode from an older run
            skipped_excluded += 1
            continue
        if (out_dir / png.name).exists() and not args.overwrite:
            continue
        todo.append(png)

    print(f'[upscale] {len(todo)} PNG -> HD PNG via ComfyUI ({api}, model={cfg["upscale_model"]}, '
          f'{skipped_excluded} engine-data files skipped)')
    if not todo:
        return 0

    dims = load_png_dims(Path(args.staging), todo)
    missing_dims = [p for p in todo if p not in dims]
    if missing_dims:
        print(f'  WARN {len(missing_dims)} files have no manifest entry, skipping '
              f'(e.g. {missing_dims[0].name})', file=sys.stderr)

    # Sources already at/above --max-source-dim gain almost nothing from AI
    # upscaling but dominate render time and archive size (a 2048 source at
    # 4x is a 128+ MB DDS). They ship as originals via client fallback.
    too_big = [p for p in list(dims) if max(dims[p]) > args.max_source_dim]
    for p in too_big:
        del dims[p]
    if too_big:
        print(f'  {len(too_big)} sources > {args.max_source_dim}px skipped '
              f'(already high-res; ship as originals)')

    batches = build_upscale_batches(dims, args.batch_pixel_budget)
    sizes_seen = sorted({dims[b[0]] for b in batches})
    print(f'  {len(batches)} batches across {len(sizes_seen)} distinct sizes '
          f'(pixel budget={args.batch_pixel_budget:,})')

    def attempt(batch: list[Path]) -> list[tuple[str, str | None]]:
        """Submit one same-size batch and check its outputs. Called directly
        by process_batch below, and again (on smaller sub-batches) by
        retry_shrinking if this batch fails - most likely a VRAM OOM at
        this size, since the upscale workflow is all-or-nothing per graph
        execution.
        """
        names = [p.name for p in batch]

        # Stage inputs into ComfyUI/input/swg/ (same drive = hard link / copy).
        # comfy_input lives inside the ComfyUI install, not our own staging/
        # or output/ dirs, so it survives across unrelated runs (different
        # archive, older version of this one). A same-named leftover there
        # from a prior run would otherwise never get refreshed, so we can't
        # just trust staged.exists() like before. But re-staging is cheap
        # only when os.link succeeds (same-drive hardlink, an O(1) metadata
        # op) - on a cross-drive ComfyUI install it falls back to a full
        # read+write copy, and unconditionally redoing that for every batch
        # attempt (including retries and resumed runs where the file was
        # already staged correctly) adds real I/O for nothing. A size check
        # is a single cheap stat() and still catches the actual failure mode
        # (a differently-sized leftover) without repeating the expensive path
        # when nothing has changed.
        for p in batch:
            staged = comfy_input / p.name
            if staged.exists():
                if staged.stat().st_size == p.stat().st_size:
                    continue
                staged.unlink()
            try:
                os.link(p, staged)
            except OSError:
                staged.write_bytes(p.read_bytes())

        wf = json.loads(json.dumps(tpl))  # deep copy
        wf['1']['inputs']['filenames'] = '\n'.join(f'swg/{n}' for n in names)
        wf['2']['inputs']['model_name'] = cfg['upscale_model']
        wf['4']['inputs']['filenames'] = '\n'.join(names)
        wf['4']['inputs']['subfolder'] = batch_subfolder

        try:
            prompt_id = submit_workflow(api, wf, client_id)
            wait_for_prompt(api, prompt_id, timeout=args.timeout)
        except TimeoutError as e:
            # Giving up on the poll doesn't cancel the job server-side -
            # interrupt it so it isn't still running/queued when we (or a
            # split retry) submit the next thing.
            comfy_interrupt(api)
            return [(n, f'TIMEOUT: {e}') for n in names]
        except Exception as e:
            return [(n, f'batch submit/poll failed: {e}') for n in names]

        # We control the exact output filenames ourselves (SWGSaveImageBatch
        # writes each slot under its real name), so there's no SaveImage
        # prefix+counter metadata to parse - just check disk directly, same
        # "truth is on disk" approach as decode/encode.
        results = []
        for n in names:
            src = comfy_output_root / batch_subfolder / n
            results.append((n, None) if src.exists() else (n, f'batch output missing: {src}'))
        return results

    def process_batch(batch: list[Path]) -> list[tuple[str, str | None]]:
        return retry_shrinking(batch, attempt)

    # ONE batch in flight at a time. comfy_interrupt (fired on timeout)
    # cancels whatever ComfyUI is currently executing — with concurrent
    # submissions that is most likely some OTHER submission's healthy batch,
    # so a single timeout would cascade into killing good work. Batches are
    # already sized to saturate the GPU on their own; --workers still drives
    # the CPU-bound phases.
    total = sum(len(b) for b in batches)
    ok = bad = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(process_batch, b) for b in batches]
        for fut in as_completed(futures):
            batch_ok = batch_bad = 0
            for name, err in fut.result():
                if err:
                    bad += 1
                    batch_bad += 1
                    print(f'  FAIL {name}: {err}', file=sys.stderr, flush=True)
                    continue
                src = comfy_output_root / batch_subfolder / name
                dst = out_dir / name
                if dst.exists() and args.overwrite:
                    dst.unlink()
                if not dst.exists():
                    try:
                        os.link(src, dst)
                    except OSError:                # cross-drive, perms, etc.
                        dst.write_bytes(src.read_bytes())
                ok += 1
                batch_ok += 1
            # One line per completed batch, regardless of --batch-pixel-budget
            # (a large batch might otherwise go minutes with no output; a
            # small one would spam under the old every-250-files gate).
            done = ok + bad
            rate = done / max(0.001, time.time() - t0)
            eta  = (total - done) / max(0.001, rate)
            print(f'  batch done: {batch_ok} ok, {batch_bad} failed  |  '
                  f'overall {done}/{total}  rate={rate:.2f}/s  eta={eta/60:.1f}min',
                  flush=True)

    print(f'[upscale] done: {ok} ok, {bad} failed, {time.time()-t0:.0f}s')
    # Tolerate a few per-file failures (huge textures, transient model OOM)
    # rather than aborting the whole `all` pipeline; >5% failed is the cliff.
    return 0 if bad < max(10, len(todo) // 20) else 1


# ---------------------------------------------------------------------------
# Phase 4: encode PNG -> DDS via texconv, using manifest for original format.

def prepare_ship_png(png_out: Path, png_in_dir: Path, ship_png: Path,
                     target_wh: tuple[int, int]) -> tuple[str, bool, str | None]:
    """Turn a raw 4x render into the PNG we actually encode and ship:

    - Lanczos-downscale to target_wh (source dims * --ship-scale, capped at
      --max-dim). Shipping the raw 4x quadrupled archive size for detail the
      engine never resolves on screen; render-4x-then-ship-2x is the recipe
      the composite-armor pilot validated against SWGRestoration.
    - Re-attach the source alpha channel (ComfyUI's save path is RGB-only;
      lost alpha turns alpha-cut foliage/glass/decals into opaque quads).

    Runs in a worker process; exceptions are returned, not raised, so one
    corrupt file can't abort the whole encode phase.
    Returns (name, alpha_attached, error|None).
    """
    from PIL import Image
    # Our own generated textures, not untrusted uploads - the decompression
    # bomb guard false-positives on legitimately large 4x renders.
    Image.MAX_IMAGE_PIXELS = None
    try:
        img = Image.open(png_out)
        tw, th = target_wh
        # Never enlarge beyond what the model actually rendered.
        if img.width < tw or img.height < th:
            tw, th = img.width, img.height
        rgb = img.convert('RGB')
        if (rgb.width, rgb.height) != (tw, th):
            rgb = rgb.resize((tw, th), Image.LANCZOS)

        # Alpha: prefer a real alpha already present on the render, else
        # lift the source's alpha; fully-opaque channels are dropped.
        alpha = None
        if img.mode == 'RGBA' and img.split()[3].getextrema()[0] < 255:
            alpha = img.split()[3].resize((tw, th), Image.LANCZOS)
        else:
            src_png = png_in_dir / png_out.name
            if src_png.exists():
                src = Image.open(src_png)
                if src.mode in ('RGBA', 'LA') or 'transparency' in src.info:
                    a = src.convert('RGBA').split()[3]
                    if a.getextrema()[0] < 255:
                        alpha = a.resize((tw, th), Image.LANCZOS)

        out = Image.merge('RGBA', (*rgb.split(), alpha)) if alpha else rgb
        out.save(ship_png)
        return (png_out.name, alpha is not None, None)
    except Exception as e:
        return (png_out.name, False, repr(e))


def phase_encode(args, cfg: dict) -> int:
    if not TEXCONV.exists():
        raise SystemExit(f'texconv not found at {TEXCONV}')
    manifest_path = Path(args.staging) / 'manifest.json'
    manifest = load_manifest(manifest_path)
    in_dir  = Path(args.staging) / 'png_out'
    png_in_dir = Path(args.staging) / 'png_in'
    out_dir = Path(args.staging) / 'dds_out' / 'texture'
    out_dir.mkdir(parents=True, exist_ok=True)

    ship_dir = Path(args.staging) / 'png_ship'
    ship_dir.mkdir(parents=True, exist_ok=True)

    # First pass: figure out which files actually need work (manifest lookup
    # + skip-existing), without doing any CPU work yet.
    excluded = load_excluded_names(Path(args.staging))
    pending: list[tuple[Path, str, str, tuple[int, int]]] = []   # (png, dds_name, tex_fmt, target_wh)
    skipped = 0
    skipped_excluded = 0
    for png in in_dir.glob('*.png'):
        dds_name = png.stem + '.dds'
        if dds_name in excluded:                 # stale upscale from an older run
            skipped_excluded += 1
            continue
        meta = manifest['entries'].get(dds_name)
        if not meta:
            print(f'  WARN no manifest entry for {png.name} (key={dds_name})', file=sys.stderr)
            continue
        if max(meta['width'], meta['height']) > args.max_source_dim:
            skipped_excluded += 1                # stale render of a high-res source
            continue
        tex_fmt = TEXCONV_FORMAT.get(meta['fmt'], 'BC3_UNORM')
        target = out_dir / dds_name
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        target_wh = (min(meta['width'] * args.ship_scale, args.max_dim),
                     min(meta['height'] * args.ship_scale, args.max_dim))
        pending.append((png, dds_name, tex_fmt, target_wh))

    # Ship-prep (downscale to ship size + alpha re-attach) is pure per-file
    # CPU work with no shared state, so fan it out across processes.
    alpha_fixed = 0
    prep_errors = 0
    prep_failed: set[str] = set()
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = pool.map(prepare_ship_png,
                               [p for p, _dn, _f, _t in pending],
                               [png_in_dir] * len(pending),
                               [ship_dir / p.name for p, _dn, _f, _t in pending],
                               [t for _p, _dn, _f, t in pending])
            for name, fixed, err in results:
                if err:
                    prep_errors += 1
                    prep_failed.add(name)
                    print(f'  WARN ship-prep failed for {name}: {err}', file=sys.stderr)
                elif fixed:
                    alpha_fixed += 1
        if prep_errors:
            print(f'  {prep_errors} files failed ship-prep and are skipped this run',
                  file=sys.stderr)

    # Group by target texconv format so we can batch one process per format.
    # Encode reads the prepared PNGs in png_ship/, not the raw 4x renders.
    by_fmt: dict[str, list[tuple[Path, str]]] = {}
    for png, dds_name, tex_fmt, _t in pending:
        if png.name in prep_failed:
            continue
        by_fmt.setdefault(tex_fmt, []).append((ship_dir / png.name, dds_name))

    print(f'[encode] {sum(len(v) for v in by_fmt.values())} PNG -> DDS  '
          f'({skipped} skipped existing, {skipped_excluded} engine-data/high-res skipped, '
          f'{alpha_fixed} source alphas re-attached)')

    BATCH = 100
    t0 = time.time()
    all_targets: list[Path] = []   # used for source-of-truth presence check
    for tex_fmt, items in by_fmt.items():
        print(f'  [{tex_fmt}] {len(items)} files')
        chunks = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]

        # BC7's compressor already spreads a single file's block compression
        # across every core (DirectXTex uses hardware_concurrency() by
        # default), so running many BC7 texconv processes at once just makes
        # them fight over cores — cap concurrency low and let each one use
        # the whole machine. Every other format compresses fast per file
        # with little internal threading, so run -singleproc, workers-wide:
        # one file's worth of work per core instead of per texconv
        # invocation, which is what was leaving most of a 9950X idle before.
        heavy = tex_fmt in SELF_THREADED_TEXCONV_FORMATS
        chunk_workers = min(len(chunks), 3) if heavy else max(1, args.workers)

        # texconv -f <fmt> -m 0 (full chain) -o <dir> input1 input2 ...
        # -sepalpha: filter alpha independently of color when generating
        # mips; texconv's default alpha-weighted filter darkens color in
        # low-alpha regions on lower mips (dark-at-distance bug).
        base_cmd = [
            str(TEXCONV),
            '-nologo',
            '-y',
            '-f', tex_fmt,
            '-m', '0',                 # full mipmap chain at new (4x) size
            '-sepalpha',
            '-ft', 'dds',
            '-o', str(out_dir),
        ]
        if not heavy:
            base_cmd.append('-singleproc')

        def run_chunk(chunk):
            cmd = base_cmd + [str(p) for (p, _name) in chunk]
            rc = subprocess.call(cmd, stdout=subprocess.DEVNULL)
            return rc, chunk

        with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
            chunk_results = list(pool.map(run_chunk, chunks))

        for rc, chunk in chunk_results:
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
    # Engine-data textures (cube/special/ui/sky buckets) and high-res
    # sources never ship in the HD archive — drop any strays left in
    # dds_out by runs that predate the routing/caps, so a rebuild also
    # repairs old staging in place.
    excluded = load_excluded_names(Path(args.staging))
    entries = load_manifest(Path(args.staging) / 'manifest.json')['entries']

    def ships(f: Path) -> bool:
        if f.name in excluded:
            return False
        if not entries:            # no manifest to judge by — name filter only
            return True
        meta = entries.get(f.name)
        if meta is None:           # not produced by this pipeline run
            print(f'[repack] WARN dropping {f.name}: no manifest entry', file=sys.stderr)
            return False
        return max(meta['width'], meta['height']) <= args.max_source_dim

    all_dds = list(in_dir.rglob('*.dds'))
    files = sorted(f for f in all_dds if ships(f))
    n_dropped = len(all_dds) - len(files)
    if n_dropped:
        print(f'[repack] dropping {n_dropped} engine-data/high-res files from the '
              f'archive (client falls back to originals)')
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

        w = TreWriter(str(shard_path), workers=args.workers)
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
    ap.add_argument('--ship-scale', type=int, default=2,
                    help='shipped size = source dims x this (default 2). The model still '
                         'renders at its native 4x; encode Lanczos-downscales to this — '
                         'the render-4x-ship-2x recipe validated in the composite-armor '
                         'pilot. Shipping raw 4x quadruples archive size for detail the '
                         'engine never resolves. Changing this between runs requires '
                         'encode --overwrite (or deleting dds_out) — already-encoded DDS '
                         'at the old size otherwise pass the skip-existing check.')
    ap.add_argument('--max-dim', type=int, default=2048,
                    help='hard cap on shipped texture width/height (default 2048; some '
                         'SWG shaders cap at 2048 and larger textures balloon archives)')
    ap.add_argument('--max-source-dim', type=int, default=512,
                    help='skip sources larger than this on their longest side (default '
                         '512). High-res sources gain little from AI upscaling but '
                         'dominate render time and archive size; they ship as originals.')
    ap.add_argument('--batch-pixel-budget', type=int, default=DEFAULT_BATCH_PIXEL_BUDGET,
                    help='upscale phase: max width*height*count per ComfyUI batch '
                         f'(default {DEFAULT_BATCH_PIXEL_BUDGET:,} = 128 files at 256x256, '
                         'scaled down for larger textures) - raise/lower to fit your VRAM')
    ap.add_argument('--timeout', type=float, default=900.0,
                    help='per-batch prompt timeout in seconds (default 900 = 15min; '
                         'batches now carry many files per prompt, not one, so this needs '
                         'more headroom than the old single-file default of 300)')
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
