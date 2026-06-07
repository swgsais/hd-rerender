"""Sanity-check that all asset references in the shader_fix override TRE
shaders resolve to files that exist in some TRE."""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extract_tre import _open_tre

# Tokens that begin a valid SWG asset path inside a .sht binary
PATH_PREFIXES = ('texture/', 'texture\\',
                 'vertex_program/', 'vertex_program\\',
                 'pixel_program/', 'pixel_program\\',
                 'effect/', 'effect\\',
                 'shader/', 'shader\\')
ASSET_EXTS = ('.dds', '.vsh', '.psh', '.eft', '.sht', '.tga')
NAME_RE = re.compile(r'[A-Za-z0-9_/\\.\-]+')

def gather_refs(staging_dir):
    refs = set()
    for fn in os.listdir(staging_dir):
        if not fn.endswith('.sht'):
            continue
        data = open(os.path.join(staging_dir, fn), 'rb').read()
        for m in re.findall(rb'[\x20-\x7E]{8,}', data):
            s = m.decode('latin-1')
            for token in PATH_PREFIXES:
                idx = s.find(token)
                if idx < 0:
                    continue
                tail = s[idx:].replace('\\', '/')
                mm = NAME_RE.match(tail)
                if mm and mm.group(0).lower().endswith(ASSET_EXTS):
                    refs.add(mm.group(0))
    return refs

def all_tre_paths(swg_root):
    paths = set()
    for tre in sorted(os.listdir(swg_root)):
        if not tre.endswith('.tre'):
            continue
        p = os.path.join(swg_root, tre)
        try:
            f, entries, _v = _open_tre(p)
            f.close()
            for (n, *_r) in entries:
                paths.add(n.lower())
        except Exception:
            pass
    return paths

if __name__ == '__main__':
    refs = gather_refs('staging/shader_fix/shader')
    paths = all_tre_paths('E:/SWGNGE')
    present = sorted(r for r in refs if r.lower() in paths)
    missing = sorted(r for r in refs if r.lower() not in paths)
    print(f'{len(refs)} unique asset refs across shader_fix .sht files')
    print(f'  present in some TRE: {len(present)}')
    print(f'  MISSING:             {len(missing)}')
    for m in missing:
        print(f'    {m}')
