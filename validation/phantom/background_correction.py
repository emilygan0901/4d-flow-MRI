import numpy as np, time, json, slicerheart_4dflow as f
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"
from slicerheart_4dflow import phantom as ph
res={}
for seed in range(5):
    P=ph.chest(rng=np.random.default_rng(seed))
    vt, vm, mag, lumen, static = P['velocity_true'], P['velocity'], P['magnitude'], P['lumen'], P['static']
    err=lambda v,m: float(np.mean(np.linalg.norm((v-vt)[:,m],axis=-1)))
    res.setdefault('none',[]).append((err(vm,static), err(vm,lumen), 0))
    for meth in ['ols','msac','wrls','war+msac','arto']:
        t=time.time()
        out=f.correct_background_phase(vm, mag, method=meth)
        vc=out[0] if isinstance(out,tuple) else (out['velocity'] if isinstance(out,dict) else out)
        res.setdefault(meth,[]).append((err(vc,static), err(vc,lumen), time.time()-t))
S={k:{'static_mean':float(np.mean(np.array(v)[:,0])),'static_sd':float(np.std(np.array(v)[:,0])),'lumen_mean':float(np.mean(np.array(v)[:,1])),'time_s':float(np.mean(np.array(v)[:,2]))} for k,v in res.items()}
for k,s in S.items(): print(f"{k:9s} static {s['static_mean']:.2f}±{s['static_sd']:.2f} mm/s  lumen {s['lumen_mean']:.2f}  {s['time_s']:.1f}s")
json.dump(S, open(OUT/'background_correction.json','w'), indent=1)
