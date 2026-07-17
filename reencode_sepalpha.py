#!/usr/bin/env python3
"""One-off: re-encode the main-run upscaled PNGs to DDS with -sepalpha mip
filtering (fixes the dark-at-distance / dark lower-mip bug from texconv's default
alpha-weighted mip downsampling), then re-apply the QC fallbacks. Encode-only,
no GPU, no re-render. mip 0 is identical to before, so QC verdicts still hold."""
import sys, json, os, subprocess, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pil_upscale import read_dds_format, TEXCONV_FORMAT

THIS=Path(__file__).resolve().parent
TEX=THIS/'bin'/'texconv.exe'
WORKERS=os.cpu_count() or 4
# BC7's compressor already spreads a single file's block compression across
# every core, so BC7 batches run with just a couple of concurrent processes
# and no -singleproc; every other format compresses fast per file with
# little internal threading, so those run -singleproc, WORKERS-wide.
SELF_THREADED_FORMATS={'BC7_UNORM'}
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
    chunks=[plist[i:i+BATCH] for i in range(0,len(plist),BATCH)]
    heavy=fmt in SELF_THREADED_FORMATS
    chunk_workers=min(len(chunks),3) if heavy else max(1,WORKERS)
    base_cmd=[str(TEX),'-nologo','-y','-f',fmt,'-m','0','-sepalpha','-ft','dds','-o',str(DDS_OUT)]
    if not heavy:
        base_cmd.append('-singleproc')

    def run_chunk(chunk,base_cmd=base_cmd):
        subprocess.run(base_cmd+[str(p) for p in chunk], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return len(chunk)

    with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
        for n in pool.map(run_chunk, chunks):
            done+=n
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
