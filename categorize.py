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
    'cels_',     # samples these as lookup data; upscaling shifts the ramp
    'skybox_',   # skybox faces authored without the cube-map header flag
    'dirt_',     #
    'frst_',
    'watr_',
    'nebula_',
    'nebula2_',
    'nebulon',
    'pirate_asteroid',
    'neutral',
    'shuttle_',
    'slave1_',
    'spc_',
    'star_destroyer',
    'sw_',
    'spacestation',
    'sd_',
    'science_transport',
    'smuggler_',
    'planet',
    'transport_',
    'tradefed_',
    'wpm_',
    'sorosub3000',
    'space_station',
    'space_destroyer',
    'star',
)
SPECIAL_CONTAINS = (
    '_pattern',  # character face/body customization index patterns - the
    '_spc_',     # palette system reads these as indices, not imagery
    'gradient',  # gradient LUTs (gradient_sky1, sw_gradient_*, ...)
    '_grad.',    # gradient LUTs named as a suffix (glass_grad.dds etc.)
    '_face',     # species face/head diffuse - tinted through the palette
    '_head',     # system at runtime; resampling shifts the index colors.
                 # Costs a few safe skips (weapon "_head" parts etc.) but
                 # face corruption is far worse than a non-HD vibroblade.
    '_eye',      # eye customization set (hum_b_eye.dds/_m.dds/eyespec.dds -
    '_n.',       # missing-eyes corruption case). Catches the whole set
                 # regardless of suffix convention (the bare _m and the
                 # underscore-less eyespec companions don't match
    'rock_',     # SPECIAL_SUFFIX_RE, so this needs to stand on its own).
    'hum_f_',    # freckles customization set (hum_f_freckles_s01..s05.dds -
    'hum_m_',    # the original green-face corruption case). Matches every
    'light_',    # numbered variant via substring, without relying on the
    'water_',    # number itself meaning anything (see SPECIAL_SUFFIX_RE's
                 # comment on why numbered suffixes aren't used as a rule).
)

# Skydome imagery: not palette data, but the arch bucket's Lanczos-only
# warning applies double here - AI upscalers hallucinate texture into what
# must stay a smooth atmospheric gradient. Shipped at original resolution.
SKY_PREFIXES  = ('sky_', 'cloudtile_', 'env_')
SKY_CONTAINS  = ('_sky_', '_sky.')
SKY_NOT       = ('skyskraper', 'skyscraper', 'skyhook')   # buildings, not sky

# Explicit channel-type name suffixes only - NOT numbered variants (_s01,
# _n01, etc). Numbered suffixes turned out to be a common plain style/variant
# naming convention across many ordinary texture types in these TREs (e.g.
# wall_s01.dds), not a reliable signal of index/channel data on their own -
# using them here would silently exclude a large number of legitimate
# textures from upscaling. norm/normal/spec/spc/det/hue name an actual
# channel type and are far more specific, so they stay. Known customization-
# overlay families that DO need protection (freckles, eye - both palette/
# index data, not real color imagery) are matched by name instead, in
# SPECIAL_CONTAINS below - surgical per-family, not a suffix heuristic.
SPECIAL_SUFFIX_RE = re.compile(
    r'_(norm|normal|spec|spc|det|hue)\.dds$',
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


def categorize_with_reason(name: str, src_path: Path) -> tuple[str, str]:
    """Same routing as categorize(), but also returns which specific rule
    fired - e.g. 'special'/'contains:mask' vs 'arch'/'prefix:thm_'. Buckets
    can hide a dominant rule behind a single count; this is what
    report_excluded.py's --format reasons uses to break one open."""
    nl = name.lower()

    # 1) Cube map (header-derived) - hard skip
    if src_path.exists() and is_cube(src_path):
        return 'cube', 'dds_cubemap_flag'

    # 2) UI / cursor / particle - hard skip
    for p in UI_PREFIXES:
        if nl.startswith(p):
            return 'ui', f'prefix:{p}'
    for c in UI_CONTAINS:
        if c in nl:
            return 'ui', f'contains:{c}'
    if 'mask' in nl:
        return 'special', 'contains:mask'
    if 'facenormal' in nl:
        return 'special', 'contains:facenormal'

    # 3) Special channel data (normal/spec/alpha mask/etc.) - hard skip
    m = SPECIAL_SUFFIX_RE.search(nl)
    if m:
        return 'special', f'suffix_regex:{m.group(0)}'
    for p in SPECIAL_PREFIXES:
        if nl.startswith(p):
            return 'special', f'prefix:{p}'
    for c in SPECIAL_CONTAINS:
        if c in nl:
            return 'special', f'contains:{c}'

    # 3b) Skydome / atmosphere - hard skip (AI hallucinates into gradients)
    if not any(g in nl for g in SKY_NOT):
        for p in SKY_PREFIXES:
            if nl.startswith(p):
                return 'sky', f'prefix:{p}'
        for c in SKY_CONTAINS:
            if c in nl:
                return 'sky', f'contains:{c}'

    # 4) Organic - DAT2
    for p in ORGANIC_PREFIXES:
        if nl.startswith(p):
            return 'organic', f'prefix:{p}'
    for c in ORGANIC_CONTAINS:
        if c in nl:
            return 'organic', f'contains:{c}'

    # 5) Architectural - LANCZOS
    for p in ARCHITECTURAL_PREFIXES:
        if nl.startswith(p):
            return 'arch', f'prefix:{p}'
    for c in ARCHITECTURAL_CONTAINS:
        if c in nl:
            return 'arch', f'contains:{c}'

    # 6) Default: hard surface (ships, droids, characters, props) - DAT2
    return 'hardsurface', 'default'


def categorize(name: str, src_path: Path) -> str:
    return categorize_with_reason(name, src_path)[0]


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
