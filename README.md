# SWG TRE HD Re-Render Pipeline

End-to-end pipeline that takes textures out of `reborn_textures.tre`, runs them
through a local ComfyUI ESRGAN upscale (4×), re-encodes back to the original
DDS/BCx format with regenerated mipmaps, and ships them as
`reborn_textures_hd.tre` for the client to load at higher patch priority.

## What it does

```
reborn_textures.tre                                                    client loads
        │                                                                    ▲
        │ extract  (extract_tre.py, filter texture/*.dds)                    │
        ▼                                                                    │
   staging/dds_in/texture/*.dds  +  manifest.json (orig fmt + dims)          │
        │                                                                    │
        │ decode   (texconv -ft png  -m 1)                                   │
        ▼                                                                    │
   staging/png_in/*.png                                                      │
        │                                                                    │
        │ upscale  (ComfyUI /prompt API, LoadImage → ESRGAN → SaveImage)     │
        ▼                                                                    │
   staging/png_out/*.png   ←   ComfyUI/output/swg_hd/*.png                   │
        │                                                                    │
        │ encode   (texconv -f <orig BCx> -m 0 = full mip chain)             │
        ▼                                                                    │
   staging/dds_out/texture/*.dds                                             │
        │                                                                    │
        │ repack   (build_tre.TreWriter, TRE 0005)                           │
        ▼                                                                    │
   client/tre/reborn_textures_hd.tre  ──────────────────────────────────────┘
```

Every phase is **resumable**. Re-running skips outputs that already exist.
Run individual phases for diagnosis, or `all` for the full pipeline.

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

Drop your upscale `.pth` into `<ComfyUI>/models/upscale_models/`.

`4x-PBRify_UpscalerSPANV4.pth` recommended for SWG-style assets — the PBRify
model family is specifically trained on old-game-texture-to-PBR restoration,
and the SPAN variant is fast (~0.35s/file on RTX 4090) without quality loss
vs. the DAT2 variant. Drop in from your preferred PBRify release.

If you have a generic-purpose model, `4x-UltraSharp.pth` is the standard
fallback — download from <https://openmodeldb.info/models/4x-UltraSharp>.

### 3. config

Copy `hd_rerender.config.example.json` → `hd_rerender.config.json` and edit:

```jsonc
{
  "comfy_root":     "C:/ComfyUI_windows_portable/ComfyUI",  // dir with main.py
  "comfy_api":      "http://127.0.0.1:8188",                // ComfyUI server URL
  "upscale_model":  "4x-UltraSharp.pth"                     // filename in upscale_models/
}
```

### 4. Start ComfyUI

```powershell
& "$ComfyRoot/python_embeded/python.exe" "$ComfyRoot/ComfyUI/main.py" --listen 127.0.0.1
```

Confirm it's up with `curl http://127.0.0.1:8188/system_stats`.

## Run

### Full pipeline

```powershell
python hd_rerender.py all
```

### Phase by phase (recommended on first run)

```powershell
python hd_rerender.py extract   # ~1 min,  ~3 GB to staging/dds_in/
python hd_rerender.py decode    # ~5 min,  staging/png_in/
python hd_rerender.py upscale   # hours on a single GPU — this is the long one
python hd_rerender.py encode    # ~15 min, staging/dds_out/
python hd_rerender.py repack    # ~1 min,  reborn_textures_hd.tre
```

### Useful flags

```
--workers N          decode/encode parallelism (default 4). upscale uses N as
                     parallel-submit count; ComfyUI queues internally so 2-4
                     keeps the GPU saturated.
--timeout SECS       per-prompt timeout; raise on slow GPUs (default 300)
--overwrite          re-do already-completed files
--tre PATH           input TRE; default client/tre/reborn_textures.tre
--out-tre PATH       output TRE; default client/tre/reborn_textures_hd.tre
--staging DIR        staging root; default ./staging/
--config PATH        config json; default ./hd_rerender.config.json
```

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
chain at the new (4×) dimensions, so each upscaled texture ends with the same
mip pyramid the runtime expects.

## Deploying (sharded output)

A full reborn_textures HD pass produces **~35 GB of raw DDS** (16× the
original — 4× pixels per side × regenerated mipmap chains). TRE 0005's signed-
int32 offset field caps single archives at 2 GiB, so the repack phase shards
output into multiple files: `reborn_textures_hd_001.tre`, `_002.tre`, …,
typically 20-25 shards at ~600-800 MB compressed each.

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

Verify in-game with a known landmark texture (e.g. a Tatooine wall) and check
that DDS dimensions on disk are 4× the original. The pipeline preserves the
original BCx format and regenerates a full mipmap chain at the new size, so
the runtime doesn't need any other config changes.

## Performance recorded for a 20,056-texture full pass (RTX 4090 + SPAN v4)

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
| `no SaveImage output` for some files          | model OOMed on a huge texture — lower `--workers` to 1 |
| upscale phase hangs on one file               | raise `--timeout`, check GPU temp                    |
| black/magenta blocks in-game                  | encode picked wrong format — inspect `manifest.json` |
| client doesn't load HD TRE                    | not in load order, or load-order priority too low    |
| game crashes loading a specific texture        | dimensions exceed engine cap (some shaders cap at 2048) — exclude that file |

To exclude problem files, drop them from `staging/png_out/` and re-run encode
and repack; the manifest stays intact so re-running extract won't reset it.

## Files in this directory

```
hd_rerender.py                       main driver (subcommands: extract/decode/upscale/encode/repack/all)
hd_rerender.config.example.json      copy to hd_rerender.config.json
workflows/upscale_4x.json            ComfyUI /prompt API workflow template
bin/texconv.exe                      DirectXTex tool (you download — see setup)
staging/                             scratch; everything here is reproducible
README.md                            this file
```
