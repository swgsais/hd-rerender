#!/usr/bin/env python3
"""
Mirror SWGRestoration's HD asset *selection*, re-rendered from our own sources.

SWGRestoration encrypts their TRE payloads (AES), so we cannot use their bytes.
But their TOC (restoration-toc/restoration_index.csv) is a readable
manifest of WHICH textures they enhanced. This tool:

  manifest : restoration HD list  ->  intersect with our sources  -> targets.json
             + per-category routing plan (categories.json)
  render   : per-category re-render of the targets, reusing cached staging/
             - arch        -> PIL Lanczos          (AI hallucinates grass on stone)
             - organic     -> ComfyUI 4x + downscale (DAT2 validated for foliage)
             - hardsurface -> ComfyUI 4x + downscale (model chosen in pilot)
             - special/cube/ui -> copy original (channel data / engine constraints)
             encode is BGRA-aware (pil_upscale.read_dds_format) so R/B never swap.
  pack     : build TRE shards from a render variant (build_tre.py)

Reuses the proven building blocks rather than duplicating them:
  pil_upscale.read_dds_format / TEXCONV_FORMAT  (BGRA-correct encode)
  hd_rerender.submit_workflow / wait_for_prompt  (ComfyUI /prompt API)
  categorize.categorize                           (bucketing)
  build_tre.TreWriter / DiskFileEntry             (TRE 0005 writer)

Staging layout (cached from prior runs, reused here) - resolved from --tre
the same way hd_rerender.py resolves it (staging/<tre stem>/, or --staging
to override), so running both tools against the same --tre always agrees on
where things are. No manual copying between the two tools' staging dirs.
  <staging>/dds_in/texture/*.dds   source DDS (for format detection)
  <staging>/png_in/*.png           source decoded PNG (texconv -ft png -m 1)
  <staging>/render/<variant>/png_out/*.png   upscaled PNG (intermediate)
  <staging>/render/<variant>/dds_out/texture/*.dds   final DDS for packing
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from pil_upscale import read_dds_format, TEXCONV_FORMAT        # BGRA-aware
from categorize import categorize                              # bucketing
import hd_rerender as hr                                       # ComfyUI helpers + load_config + resolve_staging

TEXCONV = THIS_DIR / 'bin' / 'texconv.exe'
TOC_DIR = THIS_DIR / 'restoration-toc'


@dataclass(frozen=True)
class Paths:
    staging: Path
    dds_in: Path
    png_in: Path
    render: Path
    targets_json: Path
    categories_json: Path

    @classmethod
    def for_staging(cls, staging: Path) -> 'Paths':
        return cls(
            staging=staging,
            dds_in=staging / 'dds_in' / 'texture',
            png_in=staging / 'png_in',
            render=staging / 'render',
            targets_json=staging / 'restoration_targets.json',
            categories_json=staging / 'restoration_categories.json',
        )

# Per-category routing plan. method: lanczos | comfy | copy.  scale: linear factor.
# model only used for comfy. These are the plan-approved defaults; --model and
# --scale on `render` override for A/B pilots.
SPAN = '4x-PBRify_UpscalerSPANV4.pth'
DAT2 = '4x-PBRify_UpscalerDAT2_V1.pth'
DEVIANCE = '4x_BS_DevianceMIP.pth'
CATEGORY_PLAN = {
    'arch':        {'method': 'lanczos', 'scale': 3, 'model': None},
    'organic':     {'method': 'comfy',   'scale': 3, 'model': DAT2},
    # Composite-armor pilot vs SWGRestoration reference: DevianceMIP matched
    # resto's smooth-bright look (PSNR 25.6 on reborn source, texture energy
    # 29.4 vs resto 29.0); DAT2 amplified dither into dot-grid artifacts.
    'hardsurface': {'method': 'comfy',   'scale': 2, 'model': DEVIANCE},
    'special':     {'method': 'copy',    'scale': 1, 'model': None},
    'cube':        {'method': 'copy',    'scale': 1, 'model': None},
    'ui':          {'method': 'copy',    'scale': 1, 'model': None},
    'sky':         {'method': 'copy',    'scale': 1, 'model': None},  # AI hallucinates into gradients
}


# ---------------------------------------------------------------------------
# manifest: build the scoped target list + category plan

def load_index_paths(csv_path: Path, hd_only: bool = False) -> dict[str, int]:
    """Return {lower_path: max_length} for .dds rows. hd_only keeps only the
    Restoration HD* archives (their enhanced set)."""
    import csv
    out: dict[str, int] = {}
    with csv_path.open(newline='') as fh:
        for r in csv.DictReader(fh):
            if hd_only and '_hd' not in r['tre'].lower():
                continue
            p = r['path'].lower()
            if not p.endswith('.dds'):
                continue
            n = int(r['length'])
            if p not in out or n > out[p]:
                out[p] = n
    return out


def cmd_manifest(args, paths: Paths) -> int:
    hd  = load_index_paths(TOC_DIR / 'restoration_index.csv', hd_only=True)
    reb = load_index_paths(TOC_DIR / 'reborn_index.csv')
    ret = load_index_paths(TOC_DIR / 'retail_swgnge_index.csv')
    ours = set(reb) | set(ret)

    targets = sorted(p for p in hd if p in ours)
    missing = sorted(p for p in hd if p not in ours)

    # "in ours" only proves the name appears in a TOC scan (dump_toc_csv.py
    # reads TRE headers only, which always succeeds). It says nothing about
    # whether the actual payload made it onto disk - real extraction reads
    # and decompresses the payload, which can fail per-file (corrupt data,
    # size mismatch, etc.) without that ever showing up in a TOC scan. Cross-
    # check against dds_in directly so "sourced" isn't a false promise -
    # render_one already handles a name-only target gracefully ('no source
    # dds'), but the count here should say so up front instead of claiming
    # 100% coverage.
    extracted: list[str] = []
    not_extracted: list[str] = []
    for p in targets:
        base = p.split('/')[-1]
        (extracted if (paths.dds_in / base).exists() else not_extracted).append(p)

    paths.staging.mkdir(parents=True, exist_ok=True)
    paths.targets_json.write_text(json.dumps(
        {'targets': targets, 'missing': missing, 'not_extracted': not_extracted,
         'restoration_hd_total': len(hd), 'sourced': len(targets),
         'actually_extracted': len(extracted)},
        indent=0), encoding='utf-8')

    # Categorize each target. categorize() wants the source DDS path for the
    # cube-header check; we have it in dds_in/texture/.
    cats: dict[str, list[str]] = {k: [] for k in CATEGORY_PLAN}
    no_png = []
    for p in targets:
        base = p.split('/')[-1]                       # texture/foo.dds -> foo.dds
        src_dds = paths.dds_in / base
        cat = categorize(base, src_dds)
        cats[cat].append(base)
        if not (paths.png_in / (Path(base).stem + '.png')).exists():
            no_png.append(base)

    paths.categories_json.write_text(json.dumps(cats, indent=0), encoding='utf-8')

    print(f'staging                     : {paths.staging}')
    print(f'Restoration HD .dds         : {len(hd)}')
    print(f'sourced (name match in TOC) : {len(targets)}')
    print(f'  of which extracted to disk: {len(extracted)}')
    print(f'  name-only (NOT on disk)   : {len(not_extracted)}  <- will fail render with "no source dds"')
    print(f'unsourceable (no name match): {len(missing)}')
    print(f'targets without cached PNG  : {len(no_png)}  (need extract+decode for full run)')
    print('--- category plan ---')
    for k, plan in CATEGORY_PLAN.items():
        tag = f"{plan['method']}@{plan['scale']}x" + (f" [{plan['model'].split('_')[-1]}]" if plan['model'] else '')
        print(f'  {k:12s} {len(cats[k]):6d}   -> {tag}')
    enh = sum(len(cats[k]) for k in ('arch', 'organic', 'hardsurface'))
    print(f'  {"ENHANCED":12s} {enh:6d}   (arch+organic+hardsurface)')
    print(f'wrote {paths.targets_json.name} + {paths.categories_json.name}')
    return 0


# ---------------------------------------------------------------------------
# render helpers

def encode_png_to_dds(out_png: Path, src_dds: Path, dds_out_dir: Path, dds_name: str) -> bool:
    """Encode PNG -> DDS preserving the source's BCx format AND channel order
    (BGRA8 -> B8G8R8A8_UNORM, etc.), full mip chain. dds_name is the bare
    'foo.dds'; written into dds_out_dir."""
    tex_fmt = TEXCONV_FORMAT.get(read_dds_format(src_dds), 'BC3_UNORM')
    # -sepalpha: filter the alpha channel INDEPENDENTLY of color when generating
    # mips. texconv's default is alpha-weighted, which darkens color in low-alpha
    # regions on lower mips -> surfaces go dark at distance. SWG's DXT5 alpha is a
    # shading/spec mask (not transparency), so color must filter separately.
    rc = subprocess.call(
        [str(TEXCONV), '-nologo', '-y', '-f', tex_fmt, '-m', '0', '-sepalpha', '-ft', 'dds',
         '-o', str(dds_out_dir), str(out_png)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    produced = dds_out_dir / (out_png.stem + '.dds')
    target = dds_out_dir / dds_name
    if produced.exists() and produced != target:
        if target.exists():
            target.unlink()
        produced.rename(target)
    return rc == 0 and target.exists()


def comfy_upscale_4x(src_png: Path, cfg: dict, model: str, work_subdir: str) -> Path | None:
    """Run one image through ComfyUI's 4x model via the batch API (a
    single-item batch) - the shared workflow template
    (workflows/upscale_4x_batch.json) only speaks the SWGLoadImageBatch /
    SWGSaveImageBatch custom nodes (see comfyui_custom_nodes/swg_batch_io.py),
    not the stock LoadImage/SaveImage nodes. Returns path to the 4x PNG.
    """
    comfy_root = Path(cfg['comfy_root'])
    comfy_input = comfy_root / 'input' / 'swg'
    comfy_input.mkdir(parents=True, exist_ok=True)
    staged = comfy_input / src_png.name
    if not staged.exists():
        try:
            os.link(src_png, staged)
        except OSError:
            staged.write_bytes(src_png.read_bytes())

    tpl = json.loads(hr.WORKFLOW_TPL.read_text(encoding='utf-8'))
    tpl.pop('_comment', None)
    tpl['1']['inputs']['filenames'] = f'swg/{src_png.name}'
    tpl['2']['inputs']['model_name'] = model
    tpl['4']['inputs']['filenames'] = src_png.name
    tpl['4']['inputs']['subfolder'] = work_subdir

    api = cfg['comfy_api']
    prompt_id = hr.submit_workflow(api, tpl, str(uuid.uuid4()))
    hr.wait_for_prompt(api, prompt_id, timeout=300.0)
    # SWGSaveImageBatch returns no UI/outputs metadata in the history entry
    # (unlike stock SaveImage) - it writes the exact filename given, so check
    # disk directly, same as hd_rerender.phase_upscale does for this node.
    out = comfy_root / 'output' / work_subdir / src_png.name
    return out if out.exists() else None


def render_one(base: str, plan: dict, cfg: dict | None, dds_in: Path, png_in: Path,
               png_out_dir: Path, dds_out_dir: Path, overwrite: bool) -> tuple[str, str | None]:
    """Render a single texture per its category plan. Returns (base, error|None)."""
    src_dds = dds_in / base
    dds_target = dds_out_dir / base
    if dds_target.exists() and not overwrite:
        return (base, None)
    if not src_dds.exists():
        return (base, 'no source dds')

    method, scale = plan['method'], plan['scale']

    # copy categories: ship the original untouched.
    if method == 'copy':
        try:
            dds_target.write_bytes(src_dds.read_bytes())
            return (base, None)
        except OSError as e:
            return (base, f'copy failed: {e}')

    src_png = png_in / (Path(base).stem + '.png')
    if not src_png.exists():
        return (base, 'no cached png')
    try:
        with Image.open(src_png) as im:
            ow, oh = im.size
        tw, th = ow * scale, oh * scale

        if method == 'lanczos':
            with Image.open(src_png) as im:
                # Pillow's resize() raises for palette ('P') and other non-
                # RGB(A) modes unless converted first - some customization/
                # index-map textures decode to exactly that. RGBA is a safe
                # universal target; encode_png_to_dds re-derives the real
                # output format from the source DDS header regardless, so an
                # RGBA intermediate PNG for a non-alpha target just has its
                # alpha discarded by texconv, same as the comfy path below.
                up = im.convert('RGBA').resize((tw, th), Image.LANCZOS)
        elif method == 'comfy':
            ai4x = comfy_upscale_4x(src_png, cfg, plan['model'], 'swg_mirror')
            if ai4x is None:
                return (base, 'comfy: no output')
            with Image.open(ai4x) as im:
                rgb = im.convert('RGB').resize((tw, th), Image.LANCZOS)
            # ComfyUI's SaveImage writes RGB only -> the source alpha is lost,
            # which turns alpha-cut foliage/glass/decals into opaque quads
            # (the teal "broken bush" bug). Re-attach the source alpha channel,
            # Lanczos-upscaled, whenever the source has real transparency.
            with Image.open(src_png) as sim:
                sim = sim.convert('RGBA')
                if sim.split()[3].getextrema()[0] < 255:
                    a = sim.split()[3].resize((tw, th), Image.LANCZOS)
                    up = Image.merge('RGBA', (*rgb.split(), a))
                else:
                    up = rgb
        else:
            return (base, f'unknown method {method}')

        out_png = png_out_dir / (Path(base).stem + '.png')
        up.save(out_png, optimize=False)
    except Exception as e:
        return (base, f'resize failed: {e!r}')

    if not encode_png_to_dds(out_png, src_dds, dds_out_dir, base):
        return (base, 'encode failed')
    return (base, None)


def cmd_render(args, paths: Paths) -> int:
    cats = json.loads(paths.categories_json.read_text(encoding='utf-8'))

    # Resolve which (base, plan) pairs to render.
    if args.names:
        names = json.loads(Path(args.names).read_text(encoding='utf-8'))
        # build a base->category lookup so each named file gets its plan
        base2cat = {b: c for c, lst in cats.items() for b in lst}
        work = [(b, base2cat.get(b, 'hardsurface')) for b in names]
    else:
        wanted = set(args.category.split(',')) if args.category else set(CATEGORY_PLAN)
        work = [(b, c) for c in wanted for b in cats.get(c, [])]

    # Build the effective plan, applying --method/--scale/--model overrides.
    def eff_plan(cat: str) -> dict:
        p = dict(CATEGORY_PLAN[cat])
        if args.method: p['method'] = args.method
        if args.scale:  p['scale'] = args.scale
        if args.model:  p['model'] = args.model
        return p

    # pilot: take the first N of each requested category (deterministic, sorted)
    if args.pilot:
        bycat: dict[str, list[str]] = {}
        for b, c in work:
            bycat.setdefault(c, []).append(b)
        work = [(b, c) for c in sorted(bycat) for b in sorted(bycat[c])[:args.pilot]]

    needs_comfy = any(eff_plan(c)['method'] == 'comfy' for _, c in work)
    cfg = None
    if needs_comfy:
        cfg = hr.load_config(Path(args.config))
        try:
            hr.comfy_get_json(cfg['comfy_api'], '/system_stats')
        except Exception as e:
            raise SystemExit(f'cannot reach ComfyUI at {cfg["comfy_api"]}: {e}\n'
                             f'Start ComfyUI (the {", ".join(sorted({eff_plan(c)["method"] for _,c in work}))} '
                             f'route needs it), or render only --category arch,special,cube,ui.')

    variant = args.variant
    png_out_dir = paths.render / variant / 'png_out'
    dds_out_dir = paths.render / variant / 'dds_out' / 'texture'
    png_out_dir.mkdir(parents=True, exist_ok=True)
    dds_out_dir.mkdir(parents=True, exist_ok=True)

    from collections import Counter
    plan_summary = Counter()
    for _, c in work:
        p = eff_plan(c)
        plan_summary[f"{c}:{p['method']}@{p['scale']}x"] += 1
    print(f'[render] variant={variant}  {len(work)} files')
    for k, n in sorted(plan_summary.items()):
        print(f'   {k:28s} {n}')

    ok = bad = 0
    t0 = time.time()
    fails: list[tuple[str, str]] = []
    reasons: Counter = Counter()

    def work_one(item):
        b, c = item
        return render_one(b, eff_plan(c), cfg, paths.dds_in, paths.png_in,
                           png_out_dir, dds_out_dir, args.overwrite)

    # arch/copy are CPU/disk bound -> threads help; comfy is GPU-serialized but
    # ComfyUI queues internally, so a few outstanding submits keep it warm.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for b, err in pool.map(work_one, work):
            if err:
                bad += 1
                fails.append((b, err))
                reasons[err.split(':')[0].strip()] += 1
            else:
                ok += 1
            done = ok + bad
            if done % 25 == 0 or done == len(work):
                rate = done / max(0.001, time.time() - t0)
                eta = (len(work) - done) / max(0.001, rate)
                top = ', '.join(f'{r}={n}' for r, n in reasons.most_common(3))
                print(f'  {done}/{len(work)}  {rate:.2f}/s  eta {eta/60:.1f}min  ok={ok} bad={bad}'
                      + (f'  [{top}]' if top else ''))

    print(f'[render] done: {ok} ok, {bad} failed, {time.time()-t0:.0f}s')
    for reason, n in reasons.most_common():
        print(f'   {reason:24s} {n}')
    for b, e in fails[:20]:
        print(f'   FAIL {b}: {e}', file=sys.stderr)
    if len(fails) > 20:
        print(f'   ... +{len(fails)-20} more (full list in render_report.json)', file=sys.stderr)

    report = {'variant': variant, 'total': len(work), 'ok': ok, 'failed': bad,
              'reasons': dict(reasons.most_common()), 'failed_files': sorted(fails)}
    report_path = paths.render / variant / 'render_report.json'
    report_path.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'   report: {report_path}')
    return 0 if bad < max(5, len(work)//20) else 1


# ---------------------------------------------------------------------------
# qc: method-aware quality gate; failures fall back to the original DDS.
#
# The black-splotch / color-corruption checks (solid_black, brightness_drop,
# alpha_corruption) are method-AGNOSTIC and are the ones we actually need.
# The "deviation from a Lanczos reference" checks (mean_diff, extreme_pixels)
# are only meaningful for the Lanczos path (arch) — an AI upscale is SUPPOSED
# to deviate from Lanczos, so applying them to AI outputs flags good results.

def _img_stats(arr):
    import numpy as np
    return float(arr[:, :, :3].mean()), float(arr[:, :, 3].mean())


def _qc_one(dds_path: Path, cat: str, method_override: str | None, tmp_dir: Path,
           fallback: bool, dds_in: Path, png_in: Path) -> tuple[str, bool, str | None]:
    """Worker-process body for one texture's QC check. Runs in a separate
    process so the texconv decode subprocess + numpy diff work actually
    spreads across cores. Each worker gets its own tmp subdir (keyed by pid)
    so concurrent decode_dds calls can't clash on the same output filename.

    dds_in/png_in are passed explicitly rather than read from module globals
    because ProcessPoolExecutor workers (spawned fresh, not forked, on
    Windows) re-import this module from scratch and would otherwise only
    ever see hardcoded defaults, not whatever --staging/--tre resolved to in
    the parent process.
    """
    import numpy as np
    from quality_gate import decode_dds          # reuse the texconv decode

    base = dds_path.name
    plan = CATEGORY_PLAN.get(cat, CATEGORY_PLAN['hardsurface'])
    method = method_override or plan['method']
    if method == 'copy':                       # untouched originals always pass
        return base, True, None

    src_png = png_in / (Path(base).stem + '.png')
    worker_tmp = tmp_dir / f'w{os.getpid()}'
    worker_tmp.mkdir(parents=True, exist_ok=True)
    out_png = decode_dds(dds_path, worker_tmp)
    if out_png is None or not src_png.exists():
        return base, True, None                # can't judge -> keep

    reason = None
    try:
        src = Image.open(src_png).convert('RGBA')
        out = Image.open(out_png).convert('RGBA')
        a = np.asarray(src.resize(out.size, Image.LANCZOS), dtype=np.int16)
        b = np.asarray(out, dtype=np.int16)
        sb, sa = _img_stats(a)
        ob, oa = _img_stats(b)
        # method-agnostic corruption checks (the splotch guards)
        if ob < 5 and sb > 30:
            reason = f'solid_black ({sb:.0f}->{ob:.1f})'
        elif sb > 30 and ob < sb * 0.5:
            reason = f'brightness_drop ({sb:.0f}->{ob:.0f})'
        elif sa > 240 and oa < 200:
            reason = f'alpha_corruption ({sa:.0f}->{oa:.0f})'
        elif sa > 30 and oa < sa * 0.5:
            reason = f'alpha_dropped ({sa:.0f}->{oa:.0f})'
        elif sa < 235 and oa > 250:                # transparent source -> opaque output
            reason = f'alpha_filled ({sa:.0f}->{oa:.0f})'
        # Lanczos-deviation checks only for the Lanczos path
        elif method == 'lanczos':
            diff = np.abs(a[:, :, :3] - b[:, :, :3])
            if diff.mean() > 15.0:
                reason = f'high_mean_diff ({diff.mean():.1f})'
            elif (diff.max(axis=2) > 100).mean() > 0.002:
                reason = f'extreme_pixels ({(diff.max(axis=2)>100).mean()*100:.2f}%)'
    except Exception as e:
        reason = f'exception:{e!r}'
    finally:
        try: out_png.unlink()
        except OSError: pass

    if reason is not None and fallback:            # overwrite with the untouched original
        src_dds = dds_in / base
        if src_dds.exists():
            dds_path.write_bytes(src_dds.read_bytes())

    return base, reason is None, reason


def cmd_qc(args, paths: Paths) -> int:
    cats = json.loads(paths.categories_json.read_text(encoding='utf-8'))
    base2cat = {b: c for c, lst in cats.items() for b in lst}

    dds_out_dir = paths.render / args.variant / 'dds_out' / 'texture'
    tmp = paths.render / args.variant / 'qc_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    files = sorted(dds_out_dir.glob('*.dds'))
    if not files:
        raise SystemExit(f'[qc] no rendered DDS under {dds_out_dir}')

    passed = 0
    failed: list[tuple[str, str]] = []
    from collections import Counter
    reasons = Counter()

    done = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(_qc_one, dds, base2cat.get(dds.name, 'hardsurface'),
                        args.method, tmp, args.fallback, paths.dds_in, paths.png_in)
            for dds in files
        ]
        for fut in as_completed(futures):
            base, ok, reason = fut.result()
            done += 1
            if ok:
                passed += 1
            else:
                failed.append((base, reason))
                reasons[reason.split(' ')[0]] += 1

            if done % 500 == 0 or done == len(files):
                print(f'  qc {done}/{len(files)}  pass={passed} fail={len(failed)}')

    report = {'variant': args.variant, 'total': len(files), 'passed': passed,
              'failed': len(failed), 'reasons': dict(reasons.most_common()),
              'fallback_applied': bool(args.fallback),
              'failed_files': sorted(failed)}
    rep_path = paths.render / args.variant / 'qc_report.json'
    rep_path.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'[qc] {passed}/{len(files)} pass, {len(failed)} fail'
          + (' (fell back to originals)' if args.fallback else ' (report only)'))
    for r, n in reasons.most_common():
        print(f'   {r:18s} {n}')
    print(f'   report: {rep_path}')
    return 0


# ---------------------------------------------------------------------------
# pack: build TRE shards from a render variant

def cmd_pack(args, paths: Paths) -> int:
    from build_tre import TreWriter, DiskFileEntry
    in_dir = paths.render / args.variant / 'dds_out'
    files = sorted(in_dir.rglob('*.dds'))
    if not files:
        raise SystemExit(f'[pack] no DDS under {in_dir}')

    SHARD = int(1.6 * 1024**3)
    shards: list[list[Path]] = [[]]
    sizes = [0]
    for f in files:
        sz = f.stat().st_size
        if sizes[-1] + sz > SHARD and shards[-1]:
            shards.append([]); sizes.append(0)
        shards[-1].append(f); sizes[-1] += sz

    base = Path(args.out).resolve()
    print(f'[pack] {len(files)} entries ({sum(sizes)/1024**3:.2f} GB raw) -> {len(shards)} shard(s)')
    for idx, (group, raw) in enumerate(zip(shards, sizes), 1):
        path = base if len(shards) == 1 else base.parent / f'{base.stem}_{idx:03d}.tre'
        w = TreWriter(str(path))
        for f in group:
            rel = f.relative_to(in_dir).as_posix()
            w.add(DiskFileEntry(name=rel, disk_path=str(f), try_compress=True))
        w.write()
        print(f'  [{idx}/{len(shards)}] {path.name}  {len(group)} entries  {path.stat().st_size/1024**2:.0f} MB')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Mirror SWGRestoration HD selection, re-rendered locally.')

    # Shared across every subcommand so staging always resolves the same way
    # hd_rerender.py resolves it - pass the identical --tre you used for
    # `hd_rerender.py extract`/`decode` and this lands on the same dir
    # automatically. --staging still works to override it directly.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--tre', required=True,
                        help='same --tre you passed to hd_rerender.py extract/decode; '
                             'resolves to staging/<tre stem>/ so nothing needs to be '
                             'copied between the two tools')
    common.add_argument('--staging', default=None,
                        help='override the staging dir instead of deriving it from --tre')

    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('manifest', parents=[common], help='build scoped target list + category plan')

    r = sub.add_parser('render', parents=[common], help='per-category re-render')
    r.add_argument('--config', default=str(THIS_DIR / 'hd_rerender.config.json'))
    r.add_argument('--category', help='comma list: arch,organic,hardsurface,special,cube,ui')
    r.add_argument('--names', help='JSON list of bare dds names to render (overrides --category)')
    r.add_argument('--variant', default='main', help='output variant tag (for A/B)')
    r.add_argument('--pilot', type=int, default=0, help='take first N of each category')
    r.add_argument('--method', choices=['lanczos', 'comfy', 'copy'], help='override category method')
    r.add_argument('--scale', type=int, help='override linear scale factor')
    r.add_argument('--model', help='override ComfyUI upscale model')
    r.add_argument('--workers', type=int, default=4)
    r.add_argument('--overwrite', action='store_true')

    q = sub.add_parser('qc', parents=[common],
                       help='method-aware quality gate; --fallback reverts failures to original')
    q.add_argument('--variant', default='main')
    q.add_argument('--method', choices=['lanczos', 'comfy', 'copy'], help='force a check mode (default: per-category)')
    q.add_argument('--fallback', action='store_true', help='overwrite failed outputs with the original DDS')
    q.add_argument('--workers', type=int, default=os.cpu_count() or 4,
                   help='parallel QC worker processes (default: cpu_count)')

    p = sub.add_parser('pack', parents=[common], help='build TRE shard(s) from a render variant')
    p.add_argument('--variant', default='main')
    p.add_argument('--out', default=r'E:\SWGNGE\reborn_restoration_hd.tre')

    args = ap.parse_args(argv)
    paths = Paths.for_staging(hr.resolve_staging(args.tre, args.staging))
    return {'manifest': cmd_manifest, 'render': cmd_render,
            'qc': cmd_qc, 'pack': cmd_pack}[args.cmd](args, paths)


if __name__ == '__main__':
    sys.exit(main())
