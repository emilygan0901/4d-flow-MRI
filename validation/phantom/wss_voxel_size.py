import numpy as np, json, warnings, slicerheart_4dflow as f
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"
from slicerheart_4dflow import phantom as ph
warnings.filterwarnings('ignore')
n=(40,60,60); sp=0.5
o=tuple(-(m-1)/2*sp for m in n[::-1])
A=ph.voxel_to_world_matrix(spacing=(sp,)*3, origin=o)
out={}
for name,kw in [('pulsatile', dict(nt=20)), ('steady', dict(nt=20, harmonics=((0,800.0),)))]:
    W=ph.womersley(shape=n, voxel_to_world=A, radius_mm=12.0, **kw)
    tp=int(np.argmax(np.abs(W['wss_pa']))); ta=np.mean(np.abs(W['wss_pa']))
    rows=[]
    for ds in (1,2,4,5):
        Ac=ph.voxel_to_world_matrix(spacing=(sp*ds,)*3, origin=tuple(x+(ds-1)/2*sp for x in o))
        sh=(n[0]//ds,ds,n[1]//ds,ds,n[2]//ds,ds)
        mc=W['mask'].reshape(sh).mean(axis=(1,3,5))>=0.5
        a=ph.acquire(W['velocity'], 2000.0, mask=W['mask'], snr=1e6, downsample=ds, rng=np.random.default_rng(0))
        r=f.wall_shear_stress(a['velocity_true'], mc, Ac)
        pk=np.mean(r['wss_magnitude'][tp]); tw=np.mean(r['tawss'])
        rows.append((sp*ds, 100*(pk-abs(W['wss_pa'][tp]))/abs(W['wss_pa'][tp]), 100*(tw-ta)/ta))
        print(name, 'voxel %.1f mm  peak WSS err %+.1f%%  TAWSS err %+.1f%%' % rows[-1])
    out[name]=rows
json.dump(out, open(OUT/'wss_voxel_size.json','w'))
