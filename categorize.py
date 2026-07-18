#!/usr/bin/env python3
"""
Categorize every texture in the manifest into one of these buckets:

  cube        - cube map (DDS_CAPS2_CUBEMAP flag set) - DON'T TOUCH
                upscale pipeline destroys the 6-face structure
  special     - normal/spec/alpha/blend/etc (non-color channel data) - DON'T TOUCH
                AI upscalers destroy encoded vector/intensity data
  ui          - UI panels, cursors, particles - DON'T TOUCH
                engine has fixed pixel coords / D3D9 cursor constraints
  arch        - architectural / environmental / hard surfaces - USE LANCZOS
                AI upscalers (SPAN, DAT2) hallucinate vegetation patterns on stone
  organic     - foliage, plants, decd_, flora_ - USE DAT2
                AI upscaler's training data ALIGNS for these (validated in pilot)
  hardsurface - ships, droids, weapons, armor, props, characters - USE DAT2
                no organic patterns to hallucinate; AI adds useful detail
                (NOT independently validated yet - needs pilot)

Writes:
  staging/categories.json - { 'cube': [...], 'special': [...], ... }
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

DDS_CUBEMAP_FLAG = 0x00000200  # DDS_CAPS2_CUBEMAP

# ---------------------------------------------------------------------------
# Patterns for each category, evaluated in order. First match wins.

ARCHITECTURAL_PREFIXES = (
    'thm_', 'pak_thm_',                         # theme/architectural assets
    'asteroid_hutt_',                           # space station walls
    'conc_',                                    # concrete/paving
    'rock_',                                    # rocks/cliffs
    'imprv_',                                   # building improvements
    'bldg_',                                    # buildings
    'dirt_', 'mud_', 'sand_', 'snow_',          # ground surfaces
    'door_', 'gate_',                           # architectural elements
    'ptch_',                                    # terrain patches
    'tatt_',                                    # tatooine biome
    'tato_',                                    # tatooine architecture
    'dant_',                                    # dantooine
    'endr_',                                    # endor architecture
    'lok_',                                     # lok
    'nboo_', 'naboo_',                          # naboo
    'kash_', 'kashyyyk_',                       # kashyyyk
    'talus_',                                   # talus
    'yavin_',                                   # yavin
    'corl_', 'corellia_',                       # corellia
    'rori_',                                    # rori
    'must_', 'mustafar_',                       # mustafar
    'mvalley_',                                 # mvalley
    'thed_',                                    # theed
    'hoth_',                                    # hoth
)

ARCHITECTURAL_CONTAINS = (
    '_wall', '_floor', '_ceiling', '_pillar', '_arch_',
    '_dome', '_roof', '_window_', '_stair', '_brick',
    '_adobe', '_stucco', '_pueblo', '_hovel', '_house_',
    '_concrete', '_pavement', '_cobblestone',
)

ORGANIC_PREFIXES = (
    'flora_', 'flow_', 'radl_', 'grss_',
    'decd_',                                    # decorative (mostly plants)
)
ORGANIC_CONTAINS = (
    '_grass', '_bush', '_shrub', '_fern', '_leaf', '_leaves',
    '_branch', '_flower', '_blossom', '_petal', '_palm',
    '_cactus', '_lilly', '_funnel', '_thorn', '_vine', '_moss',
    '_sprig', '_sprout', '_frond', '_fruit', '_orchard', '_tree',
)

UI_PREFIXES = ('ui_', 'cursor', 'pt_', 'fx_', 'lod_')
UI_CONTAINS = ('cursor',)

SPECIAL_PREFIXES = (
    'grad_',     # gradient LUT strips (sky color ramps, etc.) - the engine
                 # samples these as lookup data; upscaling shifts the ramp
    'skybox_',   # skybox faces authored without the cube-map header flag
    'hum_',      # human character textures - manually excluded after the
                 # hum_f_freckles_s01.dds green-face corruption
)
SPECIAL_CONTAINS = (
    '_pattern',  # character face/body customization index patterns - the
                 # palette system reads these as indices, not imagery
    'gradient',  # gradient LUTs (gradient_sky1, sw_gradient_*, ...)
    '_grad.',    # gradient LUTs named as a suffix (glass_grad.dds etc.)
    '_face',     # species face/head diffuse - tinted through the palette
    '_head',     # system at runtime; resampling shifts the index colors.
                 # Costs a few safe skips (weapon "_head" parts etc.) but
                 # face corruption is far worse than a non-HD vibroblade.
)

# Skydome imagery: not palette data, but the arch bucket's Lanczos-only
# warning applies double here - AI upscalers hallucinate texture into what
# must stay a smooth atmospheric gradient. Shipped at original resolution.
SKY_PREFIXES  = ('sky_', 'cloudtile_', 'env_')
SKY_CONTAINS  = ('_sky_', '_sky.')
SKY_NOT       = ('skyskraper', 'skyscraper', 'skyhook')   # buildings, not sky

SPECIAL_SUFFIX_RE = re.compile(
    r'_(n[0-9]*|s[0-9]*|norm|normal|spec|spc|det|a|b|d|e|g|h|m|hue)\.dds$',
    re.IGNORECASE,
)


def is_cube(path: Path) -> bool:
    """Check DDS_CAPS2_CUBEMAP flag in 128-byte header."""
    try:
        with path.open('rb') as f:
            head = f.read(128)
        if head[:4] != b'DDS ':
            return False
        caps2 = struct.unpack('<I', head[112:116])[0]
        return bool(caps2 & DDS_CUBEMAP_FLAG)
    except OSError:
        return False


def categorize(name: str, src_path: Path) -> str:
    nl = name.lower()

    # 1) Cube map (header-derived) - hard skip
    if src_path.exists() and is_cube(src_path):
        return 'cube'

    # 2) UI / cursor / particle - hard skip
    if any(nl.startswith(p) for p in UI_PREFIXES):
        return 'ui'
    if any(c in nl for c in UI_CONTAINS):
        return 'ui'
    if 'mask' in nl or 'facenormal' in nl:
        return 'special'

    # 3) Special channel data (normal/spec/alpha mask/etc.) - hard skip
    if SPECIAL_SUFFIX_RE.search(nl):
        return 'special'
    if any(nl.startswith(p) for p in SPECIAL_PREFIXES):
        return 'special'
    if any(c in nl for c in SPECIAL_CONTAINS):
        return 'special'

    # 3b) Skydome / atmosphere - hard skip (AI hallucinates into gradients)
    if not any(g in nl for g in SKY_NOT):
        if any(nl.startswith(p) for p in SKY_PREFIXES):
            return 'sky'
        if any(c in nl for c in SKY_CONTAINS):
            return 'sky'

    # 4) Organic - DAT2
    if any(nl.startswith(p) for p in ORGANIC_PREFIXES):
        return 'organic'
    if any(c in nl for c in ORGANIC_CONTAINS):
        return 'organic'

    # 5) Architectural - LANCZOS
    if any(nl.startswith(p) for p in ARCHITECTURAL_PREFIXES):
        return 'arch'
    if any(c in nl for c in ARCHITECTURAL_CONTAINS):
        return 'arch'

    # 6) Default: hard surface (ships, droids, characters, props) - DAT2
    return 'hardsurface'


def main() -> int:
    here = Path(__file__).resolve().parent
    manifest_path = here / 'staging' / 'manifest.json'
    src_dir       = here / 'staging' / 'dds_in' / 'texture'
    out_path      = here / 'staging' / 'categories.json'

    if not manifest_path.exists():
        raise SystemExit(f'manifest not found: {manifest_path}')

    with manifest_path.open('r', encoding='utf-8') as f:
        manifest = json.load(f)

    cats: dict[str, list[str]] = {
        'cube': [], 'special': [], 'ui': [], 'sky': [],
        'arch': [], 'organic': [], 'hardsurface': [],
    }
    for name in manifest['entries']:
        cat = categorize(name, src_dir / name)
        cats[cat].append(name)

    for k in cats:
        cats[k].sort()

    with out_path.open('w', encoding='utf-8') as f:
        json.dump(cats, f, indent=2)

    print(f'Categorized {sum(len(v) for v in cats.values())} textures:')
    for k, v in cats.items():
        print(f'  {k:12s} {len(v):>5}')
    print(f'\nwrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
