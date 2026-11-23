"""Test 7: does divergence-free filtering improve WSS?

The library recommends divergence-free filtering before quantities that
depend on velocity gradients. Womersley tube, 2.5 mm voxels, VENC 1 m/s,
SNR 1.8, 5 seeds. Each filter ("fdm", "relaxed", "dfw") is applied and
compared with no filter on: velocity error in the lumen (noise removed),
net flow, and peak WSS and TAWSS against the analytic values.

The noise-free case is included to separate the two sources of WSS error:
random noise (which filtering can reduce) and the voxel-size bias found
earlier (which it cannot).
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import slicerheart_4dflow as f
from slicerheart_4dflow import phantom as ph
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results"

n = (40, 60, 60); sp = 0.5; ds = 5
o = tuple(-(m - 1) / 2 * sp for m in n[::-1])
A = ph.voxel_to_world_matrix(spacing=(sp,) * 3, origin=o)
W = ph.womersley(shape=n, voxel_to_world=A, radius_mm=12.0, nt=20)
Ac = ph.voxel_to_world_matrix(spacing=(sp * ds,) * 3, origin=tuple(x + (ds - 1) / 2 * sp for x in o))
mc = W["mask"].reshape(n[0] // ds, ds, n[1] // ds, ds, n[2] // ds, ds).mean(axis=(1, 3, 5)) >= 0.5
tp = int(np.argmax(np.abs(W["wss_pa"])))
wss_true, tawss_true, Qtrue = abs(W["wss_pa"][tp]), float(np.mean(np.abs(W["wss_pa"]))), W["flow_mm3_s"].sum()

def score(v, vt):
    w = f.wall_shear_stress(v, mc, Ac)
    q = f.plane_flow(v, Ac, centre=(0, 0, 0), normal=(0, 0, 1), radius=20.0, lumen_mask=mc)["flow"].sum()
    return dict(vel_rmse=float(np.sqrt(np.mean(np.sum((v - vt)[:, mc] ** 2, -1)))),
                flow_err=100 * (q - Qtrue) / Qtrue,
                wss_peak_err=100 * (np.mean(w["wss_magnitude"][tp]) - wss_true) / wss_true,
                tawss_err=100 * (np.mean(w["tawss"]) - tawss_true) / tawss_true)

methods = ["none", "fdm", "relaxed", "dfw"]
R = {m: [] for m in methods}; tm = {m: [] for m in methods}
for seed in range(5):
    a = ph.acquire(W["velocity"], 1000.0, mask=W["mask"], snr=1.8, downsample=ds, rng=np.random.default_rng(seed))
    vm, vt = a["velocity"], a["velocity_true"]
    if seed == 0:
        nf = score(vt, vt)
    for m in methods:
        t0 = time.time()
        v = vm if m == "none" else f.divergence_free_sequence(vm, mc, Ac, method=m)
        v = v[0] if isinstance(v, tuple) else v
        tm[m].append(time.time() - t0)
        R[m].append(score(v, vt))
res = {"noise_free_no_filter": nf}
print(f"noise free, no filter: peak WSS {nf['wss_peak_err']:+.1f}%  TAWSS {nf['tawss_err']:+.1f}%   (the voxel-size bias)")
print(f"{'filter':8s} {'vel RMSE mm/s':>14s} {'flow':>7s} {'peak WSS':>9s} {'TAWSS':>8s} {'time s':>7s}")
for m in methods:
    s = {k: (float(np.mean([r[k] for r in R[m]])), float(np.std([r[k] for r in R[m]]))) for k in R[m][0]}
    s["time_s"] = float(np.mean(tm[m]))
    res[m] = s
    print(f"{m:8s} {s['vel_rmse'][0]:9.1f} ± {s['vel_rmse'][1]:3.1f} {s['flow_err'][0]:+6.1f}% {s['wss_peak_err'][0]:+8.1f}% {s['tawss_err'][0]:+7.1f}% {s['time_s']:7.1f}")
json.dump(res, open(OUT / "test7_divergence_free_wss.json", "w"), indent=1)
