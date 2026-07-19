# SWG TRE HD Re-Render Pipeline

End-to-end pipeline that takes textures out of any TRE archive (`--tre`),
upscales them through a local ComfyUI ESRGAN pass, re-encodes back to the
original DDS/BCx format with regenerated mipmaps, and ships them as
`<source>_hd.tre` for the client to load at higher patch priority. The source
can be a single `.tre` or a directory of them (extracted in patch-priority
order).

## What it does

```
<source>.tre  (e.g. reborn_textures.tre)                               client loads
        │                                                                    ▲
        │ extract  (extract_tre.py, filter texture/*.dds)                    │
        ▼                                                                    │
   staging/dds_in/texture/*.dds  +  manifest.json (orig fmt + dims)          │
        │                                                                    │
        │ decode   (texconv -ft png  -m 1)                                   │
        ▼                                                                    │
   staging/png_in/*.png                                                      │
        │                                                                    │
        │ upscale  (ComfyUI /prompt API, one model for everything - see      │
        │           Category routing / --quality below)                     │
        ▼                                                                    │
   staging/png_out/*.png                                                     │
        │                                                                    │
        │ encode   (texconv -f <orig BCx> -m 0 = full mip chain)             │
        ▼                                                                    │
   staging/dds_out/texture/*.dds                                             │
        │                                                                    │
        │ repack   (build_tre.TreWriter, TRE 0005)                           │
        ▼                                                                    │
   <source>_hd.tre  (sharded _001, _002, … if > 2 GiB)  ─────────────────────┘
```

Every phase is **resumable**. Re-running skips outputs that already exist.
Run individual phases for diagnosis, or `all` for the full pipeline.

**Category routing:** `extract` buckets every texture via `categorize.py` and
writes `staging/categories.json`. The `cube`, `special`, `ui`, and `sky`
buckets (cube maps, normal/spec/mask channel data, gradient LUTs,
customization index patterns, UI atlases, load screens, skydomes) are
**never upscaled or shipped** — the engine reads those as structured data,
not imagery, and upscaled versions corrupt load screens, character face
tinting, and sky gradients in-game. The client falls back to the original
archive for them. Every other texture (`arch`/`organic`/`hardsurface`) gets
identical treatment — one ComfyUI model, chosen by `--quality` (see below),
one shipped scale (`--ship-scale`). `repack` also drops any strays left in
`dds_out` by runs that predate the routing, so re-running `repack` alone
repairs an old staging dir in place.

**Quality tiers:** `--quality {low,med,high}` (default `med`) selects the
ComfyUI model, via `QUALITY_MODELS` in `hd_rerender.py`. All three tiers
currently point at the same model (SPAN) — direct testing found it faster
than the alternatives tried, with no vegetation-hallucination issue observed
on architecture either, so there's no validated reason yet to run a
different model for any tier. `QUALITY_MODELS` has commented-out example
lines showing how to point `low`/`high` at a different model once one is
validated for that purpose. `--model NAME` overrides `--quality` entirely
for one-off experiments.

## One-time setup

### 1. texconv.exe

Download into `bin/`:

```powershell
Invoke-WebRequest `
  -Uri https://github.com/microsoft/DirectXTex/releases/download/may2026/texconv.exe `
  -OutFile bin/texconv.exe
```

Verify it runs: `./bin/texconv.exe -help | Select-Object -First 5`.

### 2. ComfyUI upscale model

Drop your upscale `.pth` into `<ComfyUI>/models/upscale_models/`. The
default for every `--quality` tier is `4x-PBRify_UpscalerSPANV4.pth` — the
PBRify model family is trained on old-game-texture-to-PBR restoration, and
the SPAN variant is fast (~0.35s/file on RTX 4090). Drop in from your
preferred PBRify release.

This is just the current default, not a hardcoded requirement — `model` in
the config (or `--model` on the CLI) can point at any `.pth` in
`upscale_models/` instead, e.g. the generic-purpose `4x-UltraSharp.pth`
(<https://openmodeldb.info/models/4x-UltraSharp>).

### 3. Comfy Custom Node File

Copy `hd-rerender/comfyui_custom_nodes/swg_batch_io.py` to `<ComfyUI>/custom_nodes/`.

### 4. config

Copy `hd_rerender.config.example.json` → `hd_rerender.config.json` and edit:

```jsonc
{
  "comfy_root": "C:/ComfyUI_windows_portable/ComfyUI",  // dir with main.py
  "comfy_api":  "http://127.0.0.1:8188"                 // ComfyUI server URL
}
```

That's the whole config — `model` is optional (omit it to use `--quality`'s
default). Add it only if you want every run to default to a specific `.pth`
without passing `--model` each time.

### 5. Start ComfyUI

```powershell
& "$ComfyRoot/python_embeded/python.exe" "$ComfyRoot/ComfyUI/main.py" --listen 127.0.0.1
```

Confirm it's up with `curl http://127.0.0.1:8188/system_stats`.

## Run

### Full pipeline

```powershell
python hd_rerender.py --tre E:\path\to\reborn_textures.tre all
```

### Phase by phase (recommended on first run)

```powershell
$src = "E:\path\to\reborn_textures.tre"
python hd_rerender.py --tre $src extract   # ~1 min,  ~3 GB to staging/<stem>/dds_in/
python hd_rerender.py --tre $src decode    # ~5 min,  staging/<stem>/png_in/
python hd_rerender.py --tre $src upscale   # hours on a single GPU — this is the long one
python hd_rerender.py --tre $src encode    # ~15 min, staging/<stem>/dds_out/
python hd_rerender.py --tre $src repack    # ~1 min,  <stem>_hd.tre next to the source
```

### Useful flags

```
--tre PATH           REQUIRED. Source .tre with texture/*.dds entries, or a
                     directory of .tre files (patch-priority extraction).
--out-tre PATH       output TRE; default <tre stem>_hd.tre next to the source
--staging DIR        work dir; default ./staging/<tre stem>/ — each source
                     archive gets its own staging so runs never mix
--workers N          decode/encode parallelism (default 4). upscale uses N as
                     parallel-submit count; ComfyUI queues internally so 2-4
                     keeps the GPU saturated.
--quality {low,med,high}
                     which ComfyUI model to use (default med) — see Quality
                     tiers above. All three currently resolve to the same
                     model.
--model NAME         override --quality entirely with a specific .pth for
                     this run
--ship-scale N       shipped size = source dims x this (default 4 = ship the
                     model's full native render, no downscale). Changing
                     this between runs requires upscale/encode --overwrite
                     (or deleting png_out/dds_out) — already-rendered files
                     at the old size otherwise pass the skip-existing check
                     silently.
--batch N            upscale batch size, in "files at 256x256" terms (e.g.
                     --batch 512) — simpler alternative to
                     --batch-pixel-budget below; raise/lower to fit your VRAM
--batch-pixel-budget N
                     upscale batch size directly in max width*height*count
                     per ComfyUI batch (default 33,554,432)
--timeout SECS       per-batch prompt timeout; raise on slow GPUs (default 900)
--overwrite          re-do already-completed files
--config PATH        config json; default ./hd_rerender.config.json
--max-dim N          hard cap on shipped texture width/height (default 2048)
--max-source-dim N   skip sources above this size on their longest side; they
                     ship as originals (default 512)
```

## Upgrading a staging dir from an older revision

Every phase's resumability check is "does the output file already exist" —
it can't detect "exists, but was rendered under a different scale/model
than this run would produce." If you're resuming a staging dir that was
last touched by an older revision of this pipeline (different default
scale, different model, or the old per-category Lanczos/DAT2/DevianceMIP
routing), its `png_out`/`dds_out` entries may silently be stale rather than
wrong-looking — same file size, different actual content, no error raised.

Safest path: delete `staging/<stem>/png_out/`, `staging/<stem>/png_ship/`,
and `staging/<stem>/dds_out/` (or pass `--overwrite` to `upscale`/`encode`)
before running against an old staging dir. `dds_in/`, `manifest.json`, and
`categories.json` never need touching — `extract`/`decode` don't need a
re-run.

## How re-encoding preserves the original format

The `extract` phase reads each DDS header and records `{width, height, fmt,
mips}` into `staging/manifest.json`. `fmt` is one of:

| Tag    | DDS source                 | Re-encode flag         |
| ------ | -------------------------- | ---------------------- |
| DXT1   | FOURCC `DXT1`              | `-f BC1_UNORM`         |
| DXT3   | FOURCC `DXT3`              | `-f BC2_UNORM`         |
| DXT5   | FOURCC `DXT5`              | `-f BC3_UNORM`         |
| BC5    | FOURCC `ATI2` or `BC5U`    | `-f BC5_UNORM`         |
| BC7    | DX10 ext, DXGI 98/99       | `-f BC7_UNORM`         |
| RGBA8  | DDPF_RGB, 32 bits, alpha   | `-f R8G8B8A8_UNORM`    |
| RGB8   | DDPF_RGB, 24 bits          | `-f R8G8B8A8_UNORM` *  |
| L8     | DDPF_LUMINANCE             | `-f R8_UNORM`          |

\* RGB8 is promoted to RGBA8 on re-encode — clean way to handle the 24-bit edge
case without writing format-aware texconv plumbing. Unknown formats default to
`BC3_UNORM` (safe — supports alpha).

The `encode` phase passes `-m 0` to texconv, which generates a full mipmap
chain at the new (`--ship-scale`, default 4×) dimensions, so each upscaled
texture ends with the same mip pyramid the runtime expects.

## Deploying (sharded output)

A full reborn_textures HD pass produces a large volume of raw DDS — at the
default `--ship-scale 4` that's 16× the pixels of the original per texture,
plus regenerated mipmap chains. TRE 0005's signed-int32 offset field caps
single archives at 2 GiB, so the repack phase shards output into multiple
files: `reborn_textures_hd_001.tre`, `_002.tre`, …

**Add ALL shards to the client load order**, in numeric order, at higher
priority than `reborn_textures.tre`. For Reborn's patch manifest:

```
reborn_textures_hd_001.tre
reborn_textures_hd_002.tre
...
reborn_textures_hd_023.tre
reborn_textures.tre              # original, now overridden per-entry by HD shards
```

The runtime walks TREs in priority order and uses the first match per inner
path, so any texture covered by the HD shards will load from there; anything
not covered (animations, sounds, etc.) falls through to the originals.

Verify in-game with a known landmark texture (e.g. a Tatooine wall) and
check that DDS dimensions on disk are `--ship-scale`× the original. The
pipeline preserves the original BCx format and regenerates a full mipmap
chain at the new size, so the runtime doesn't need any other config
changes.

## Performance recorded for a 20,056-texture full pass (RTX 4090, SPAN)

| Phase     | Wall-clock  | Throughput               |
|-----------|-------------|--------------------------|
| extract   | <1 min      | 20,056 valid DDS + 173 non-DDS skipped |
| decode    | ~2 min      | ~170 files/s via texconv |
| upscale   | **1h 58m**  | **2.82 files/s sustained on RTX 4090** |
| encode    | ~21 min     | ~16 files/s via texconv  |
| repack    | ~25 min     | 23 shards, zlib level 9  |
| **total** | **~3h**     | **35.1 GB raw → 14.5 GB compressed in 23 TRE shards** |

## Troubleshooting

| Symptom                                        | Cause / fix                                          |
| ---------------------------------------------- | ---------------------------------------------------- |
| `cannot reach ComfyUI at …`                    | ComfyUI not running, or `comfy_api` URL wrong        |
| `no SaveImage output` for some files          | model OOMed on a huge texture — lower `--workers` to 1, or lower `--batch` |
| upscale phase hangs on one file               | raise `--timeout`, check GPU temp                    |
| textures look wrong for their content type (e.g. hallucinated detail where there shouldn't be any) | try a different `--model` — see Quality tiers; not every model suits every content type equally |
| a specific texture is corrupted/discolored (e.g. character face, eyes) | it's probably index/customization data that `categorize.py` didn't catch — see `categorize.py`'s `SPECIAL_CONTAINS`/`SPECIAL_SUFFIX_RE` and add a targeted rule for it |
| black/magenta blocks in-game                  | encode picked wrong format — inspect `manifest.json` |
| client doesn't load HD TRE                    | not in load order, or load-order priority too low    |
| game crashes loading a specific texture        | dimensions exceed engine cap (some shaders cap at 2048) — exclude that file |

To exclude problem files, drop them from `staging/png_out/` and re-run encode
and repack; the manifest stays intact so re-running extract won't reset it.

## Files in this directory

```
hd_rerender.py                       main driver (subcommands: extract/decode/upscale/encode/repack/all)
categorize.py                        per-texture category routing (arch/organic/hardsurface excluded from special/cube/ui/sky)
hd_rerender.config.example.json      copy to hd_rerender.config.json
workflows/upscale_4x_batch.json      ComfyUI /prompt API workflow template (batch load/save custom nodes)
comfyui_custom_nodes/swg_batch_io.py SWGLoadImageBatch/SWGSaveImageBatch - copy into ComfyUI's custom_nodes/
bin/texconv.exe                      DirectXTex tool (you download — see setup)
staging/                             scratch; everything here is reproducible
README.md                            this file
```
