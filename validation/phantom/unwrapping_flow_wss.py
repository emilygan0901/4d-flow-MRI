import numpy as np, json, warnings, slicerheart_4dflow as f
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results"
from slicerheart_4dflow import phantom as ph
warnings.filterwarnings('ignore')
n=(40,60,60); sp=0.5; ds=5
o=tuple(-(m-1)/2*sp for m in n[::-1])
A=ph.voxel_to_world_matrix(spacing=(sp,)*3, origin=o)
W=ph.womersley(shape=n, voxel_to_world=A, radius_mm=12.0, nt=20)
Ac=ph.voxel_to_world_matrix(spacing=(sp*ds,)*3, origin=tuple(x+(ds-1)/2*sp for x in o))
mc=W['mask'].reshape(n[0]//ds,ds,n[1]//ds,ds,n[2]//ds,ds).mean(axis=(1,3,5))>=0.5
dt=np.diff(W['t_s']).mean()
Qtrue=W['flow_mm3_s']; Vtrue=Qtrue.sum()*dt/1000   # mL per cycle
tp=int(np.argmax(np.abs(W['wss_pa']))); tawss_true=np.mean(np.abs(W['wss_pa']))
def flow(v, mask): return f.plane_flow(v, Ac, centre=(0,0,0), normal=(0,0,1), radius=14.0, lumen_mask=mask)['flow']
def wss(v, mask):
    r=f.wall_shear_stress(v, mask, Ac); return float(np.mean(r['wss_magnitude'][tp])), float(np.mean(r['tawss']))
R={}
for venc in (1000.0, 600.0):
    for seed in range(5):
        a=ph.acquire(W['velocity'], venc, mask=W['mask'], snr=1.8, downsample=ds, rng=np.random.default_rng(seed))
        vm, vt, mag = a['velocity'], a['velocity_true'], a['magnitude']
        lum=(np.abs(a['wrap_count']).sum(-1)>0)[:, mc]
        uv=f.unwrap_velocity(vm, venc, Ac, method='laplacian_then_temporal')
        resid=(np.abs(uv-vt).max(-1)>venc/2)[:, mc]
        row=dict(
          wrapped_pct=100*lum.mean(), residual_pct=100*resid.mean(),
          vol_raw_err=100*(flow(vm,mc).sum()*dt/1000-Vtrue)/Vtrue,
          vol_unw_err=100*(flow(uv,mc).sum()*dt/1000-Vtrue)/Vtrue,
          noise=a['velocity_noise_mm_s'][0])
        pk,ta=wss(uv,mc); pkt,tat=wss(vt,mc)
        row.update(wss_peak_err=100*(pk-abs(W['wss_pa'][tp]))/abs(W['wss_pa'][tp]), tawss_err=100*(ta-tawss_true)/tawss_true,
                   wss_peak_err_noisefree=100*(pkt-abs(W['wss_pa'][tp]))/abs(W['wss_pa'][tp]), tawss_err_noisefree=100*(tat-tawss_true)/tawss_true,
                   vol_noisefree_err=100*(flow(vt,mc).sum()*dt/1000-Vtrue)/Vtrue)
        R.setdefault(venc,[]).append(row)
print('true stroke volume mL', round(Vtrue,1), ' peak |WSS| Pa', round(abs(W['wss_pa'][tp]),3), ' TAWSS Pa', round(tawss_true,3))
S={}
for venc,rows in R.items():
    S[venc]={k:(float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0]}
    print(f'\nVENC {venc:.0f} mm/s'); [print(f'  {k:18s} {m:8.2f} ± {s:.2f}') for k,(m,s) in S[venc].items()]
json.dump({'Vtrue':Vtrue,'S':{str(k):v for k,v in S.items()}}, open(OUT/'unwrapping_flow_wss.json','w'), indent=1)
