#!/usr/bin/env python3
"""One-off: re-encode the main-run upscaled PNGs to DDS with -sepalpha mip
filtering (fixes the dark-at-distance / dark lower-mip bug from texconv's default
alpha-weighted mip downsampling), then re-apply the QC fallbacks. Encode-only,
no GPU, no re-render. mip 0 is identical to before, so QC verdicts still hold."""
import sys, json, subprocess, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pil_upscale import read_dds_format, TEXCONV_FORMAT

THIS=Path(__file__).resolve().parent
TEX=THIS/'bin'/'texconv.exe'
PNGOUT=THIS/'staging'/'render'/'main'/'png_out'
DDS_IN=THIS/'staging'/'dds_in'/'texture'
DDS_OUT=THIS/'staging'/'render'/'main'/'dds_out'/'texture'
QCREP=THIS/'staging'/'render'/'main'/'qc_report.json'

# group upscaled PNGs by their source DDS target format
groups=defaultdict(list)
missing=0
for p in PNGOUT.glob('*.png'):
    src=DDS_IN/(p.stem+'.dds')
    if not src.exists():
        missing+=1; continue
    fmt=TEXCONV_FORMAT.get(read_dds_format(src),'BC3_UNORM')
    groups[fmt].append(p)
total=sum(len(v) for v in groups.values())
print(f're-encoding {total} PNG -> DDS with -sepalpha  (fmts: {{ {", ".join(f"{k}:{len(v)}" for k,v in groups.items())} }}, {missing} no-src skipped)', flush=True)

BATCH=80; done=0; t0=time.time()
for fmt, plist in groups.items():
    for i in range(0,len(plist),BATCH):
        chunk=plist[i:i+BATCH]
        subprocess.run([str(TEX),'-nologo','-y','-f',fmt,'-m','0','-sepalpha','-ft','dds','-o',str(DDS_OUT)]
                       +[str(p) for p in chunk], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        done+=len(chunk)
        if done % 2000 < BATCH:
            print(f'  {done}/{total}  ({done/max(0.001,time.time()-t0):.0f}/s)', flush=True)
print(f're-encode done: {done} files, {time.time()-t0:.0f}s', flush=True)

# re-apply QC fallbacks (mip0 unchanged -> same failures should stay on originals)
fb=0
if QCREP.exists():
    rep=json.loads(QCREP.read_text())
    for entry in rep.get('failed_files',[]):
        name=entry[0] if isinstance(entry,(list,tuple)) else entry
        src=DDS_IN/name
        if src.exists():
            (DDS_OUT/name).write_bytes(src.read_bytes()); fb+=1
print(f're-applied {fb} QC fallbacks (reverted to originals)', flush=True)
print('SEPALPHA RE-ENCODE COMPLETE', flush=True)
