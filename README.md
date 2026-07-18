# SWG TRE HD Re-Render Pipeline

End-to-end pipeline that takes textures out of any TRE archive (`--tre`),
upscales each one with the method validated for its content type (Lanczos for
architecture, ComfyUI ESRGAN with a different model per remaining category —
see **Category routing** below), re-encodes back to the original DDS/BCx
format with regenerated mipmaps, and ships them as `<source>_hd.tre` for the
client to load at higher patch priority. The source can be a single `.tre` or
a directory of them (extracted in patch-priority order).

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
        │ upscale  — routed per category (categories.json from extract):     │
        │   arch          ─▶ PIL Lanczos direct to 3x (no ComfyUI)          │
        │   organic       ─▶ ComfyUI (DAT2 model)     ─▶ Lanczos to 3x      │
        │   hardsurface   ─▶ ComfyUI (DevianceMIP model) ─▶ Lanczos to 2x   │
        │   cube/special/ui/sky ─▶ never upscaled, skip straight to repack   │
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
writes `staging/categories.json`. Routing (`CATEGORY_PLAN` in
`hd_rerender.py`), validated against SWGRestoration's shipped output:

| Category      | Method                    | Scale | Why                                                                 |
| -------------- | ------------------------- | :---: | -------------------------------------------------------------------- |
| `arch`         | PIL Lanczos (no AI)       | 3×    | AI upscalers hallucinate vegetation onto stone                       |
| `organic`      | ComfyUI, DAT2 model       | 3×    | DAT2's training data aligns well with foliage                        |
| `hardsurface`  | ComfyUI, DevianceMIP model | 2×   | matched the reference's smooth-bright look; DAT2 dithered into a dot-grid artifact on this bucket |
| `special`, `cube`, `ui`, `sky` | never upscaled | — | cube maps, normal/spec/mask channel data, gradient LUTs, customization patterns, UI atlases, load screens — the engine reads these as structured data, and upscaled versions corrupt load screens, character face tinting, and sky gradients in-game |

The client falls back to the original archive for the never-upscaled
buckets. `--ship-scale N` overrides every category to one uniform scale, for
quick experiments. `repack` also drops any strays left in `dds_out` by runs
that predate this routing, so re-running `repack` alone repairs an old
staging dir in place.

## One-time setup

### 1. texconv.exe

Download into `bin/`:

```powershell
Invoke-WebRequest `
  -Uri https://github.com/microsoft/DirectXTex/releases/download/may2026/texconv.exe `
  -OutFile bin/texconv.exe
```

Verify it runs: `./bin/texconv.exe -help | Select-Object -First 5`.

### 2. ComfyUI upscale models

`arch` never touches ComfyUI (pure Lanczos), but `organic` and `hardsurface`
each use their own validated model — drop **both** into
`<ComfyUI>/models/upscale_models/`:

- `4x-PBRify_UpscalerDAT2_V1.pth` — used for `organic` (foliage/plants)
- `4x_BS_DevianceMIP.pth` — used for `hardsurface` (ships/droids/weapons/props/characters)

Both are from the PBRify model family, drop in from your preferred PBRify
release. These are just the validated defaults, not hardcoded requirements —
`organic_model`/`hardsurface_model` in the config (or `--organic-model`/
`--hardsurface-model` on the CLI) can point at any `.pth` in
`upscale_models/` instead, e.g. `4x-PBRify_UpscalerSPANV4.pth` (fast,
~0.35s/file on RTX 4090) or the generic-purpose `4x-UltraSharp.pth`
(<https://openmodeldb.info/models/4x-UltraSharp>) if you'd rather not use the
pilot-validated pair.


### 3. Comfy Custom Node File

Copy `hd-rerender/comfyui_custom_nodes/swg_batch_io.py` to `<ComfyUI>/custom_nodes/`.

### 4. config

Copy `hd_rerender.config.example.json` → `hd_rerender.config.json` and edit:

```jsonc
{
  "comfy_root":         "C:/ComfyUI_windows_portable/ComfyUI",  // dir with main.py
  "comfy_api":          "http://127.0.0.1:8188",                // ComfyUI server URL
  "organic_model":      "4x-PBRify_UpscalerDAT2_V1.pth",        // filename in upscale_models/
  "hardsurface_model":  "4x_BS_DevianceMIP.pth"                 // filename in upscale_models/
}
```

`organic_model`/`hardsurface_model` are optional — omit either (or the whole
config beyond `comfy_root`/`comfy_api`) to fall back to the validated
defaults shown above, baked into `hd_rerender.py` itself. A pre-routing
config's `upscale_model` key still works too, as a legacy alias that only
feeds `hardsurface_model` (never `organic_model`) — see **Upgrading an
existing config or staging dir** below if you have one.

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
--timeout SECS       per-prompt timeout; raise on slow GPUs (default 300)
--overwrite          re-do already-completed files
--config PATH        config json; default ./hd_rerender.config.json
--ship-scale N       override EVERY category to this scale uniformly (default:
                     validated per-category scales — see Category routing)
--organic-model NAME override the organic category's ComfyUI model for this run
--hardsurface-model NAME
                     override the hardsurface category's ComfyUI model for this run
--max-dim N          hard cap on shipped texture width/height (default 2048)
--max-source-dim N   skip sources above this size on their longest side; they
                     ship as originals (default 512)
```

## Upgrading an existing config or staging dir

If your `hd_rerender.config.json` predates per-category routing, it likely
sets `upscale_model` to whatever single model you were using for everything
(e.g. SPAN). After upgrading, that key still works, but only as an alias for
`hardsurface_model` — `organic_model` will silently fall back to the DAT2
default regardless. To get the exact validated routing, either delete
`upscale_model` from your config or add `hardsurface_model` explicitly
pointing at DevianceMIP.

**Stale renders in a resumed staging dir are a real risk, not just a config
nit** — every phase's resumability check is "does the output file already
exist," which can't detect "exists, but was rendered under the old uniform
scale/model." Concretely:

- An old `png_out` for an `arch` file is a leftover ComfyUI render, not the
  new Lanczos render — it'll be silently kept forever unless removed.
- `organic`/`hardsurface` renders may be the *right size* but the *wrong
  model* (e.g. everything rendered with SPAN before) — same silent-keep
  problem, and this one produces no error, just the old model's output.
- `dds_out` at the old ship-scale won't be regenerated at the new
  per-category scale without a re-run.

Before running the new routing against a staging dir from a pre-routing run:
delete `staging/<stem>/png_out/`, `staging/<stem>/png_ship/`, and
`staging/<stem>/dds_out/` (or pass `--overwrite` to `upscale`/`encode`).
`dds_in/`, `manifest.json`, and `categories.json` don't need touching —
`extract`/`decode` never need a re-run.

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
chain at the new (per-category scale — see Category routing) dimensions, so
each upscaled texture ends with the same mip pyramid the runtime expects.

## Deploying (sharded output)

A full reborn_textures HD pass produces a large volume of raw DDS — actual
size depends on the category mix (`arch`/`organic` at 3× per side = 9×
pixels; `hardsurface` at 2× per side = 4× pixels; all regenerating full
mipmap chains). TRE 0005's signed-int32 offset field caps single archives at
2 GiB, so the repack phase shards output into multiple files:
`reborn_textures_hd_001.tre`, `_002.tre`, …

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

Verify in-game with a known landmark texture (e.g. a Tatooine wall — `arch`
category, so 3× the original) and check DDS dimensions on disk match its
category's scale (see Category routing above). The pipeline preserves the
original BCx format and regenerates a full mipmap chain at the new size, so
the runtime doesn't need any other config changes.

## Performance recorded for a 20,056-texture full pass (RTX 4090 + SPAN v4)

⚠ Recorded before per-category routing existed — a single uniform ComfyUI
pass at one model, not today's Lanczos-direct `arch` + two separate ComfyUI
passes (`organic`, `hardsurface`). Wall-clock will differ: `upscale` no
longer touches ComfyUI at all for `arch` (usually the fastest phase now),
and `organic`/`hardsurface` run as two sequential batch groups instead of
one. Kept here as a rough order-of-magnitude reference, not a current
benchmark.

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
| `cannot reach ComfyUI at …`                    | ComfyUI not running, or `comfy_api` URL wrong — only raised if `organic`/`hardsurface` have work pending; an `arch`-only run never needs ComfyUI up at all |
| `no SaveImage output` for some files          | model OOMed on a huge texture — lower `--workers` to 1 |
| upscale phase hangs on one file               | raise `--timeout`, check GPU temp                    |
| `hardsurface` textures look dithered / dot-grid pattern | wrong model resolved — confirm `hardsurface_model` (or legacy `upscale_model`) actually points at DevianceMIP, not DAT2 |
| `arch` textures come out the wrong size, or unchanged | check the log for a `[upscale:arch]` block — if it's missing, `png_out` was probably a stale render from before per-category routing (see Upgrading, above) |
| black/magenta blocks in-game                  | encode picked wrong format — inspect `manifest.json` |
| client doesn't load HD TRE                    | not in load order, or load-order priority too low    |
| game crashes loading a specific texture        | dimensions exceed engine cap (some shaders cap at 2048) — exclude that file |

To exclude problem files, drop them from `staging/png_out/` and re-run encode
and repack; the manifest stays intact so re-running extract won't reset it.

## Files in this directory

```
hd_rerender.py                       main driver (subcommands: extract/decode/upscale/encode/repack/all)
categorize.py                        per-texture category routing (arch/organic/hardsurface/special/cube/ui/sky)
hd_rerender.config.example.json      copy to hd_rerender.config.json
workflows/upscale_4x_batch.json      ComfyUI /prompt API workflow template (batch load/save custom nodes)
comfyui_custom_nodes/swg_batch_io.py SWGLoadImageBatch/SWGSaveImageBatch - copy into ComfyUI's custom_nodes/
bin/texconv.exe                      DirectXTex tool (you download — see setup)
staging/                             scratch; everything here is reproducible
README.md                            this file
```
