"""Test 6: sensitivity to the lumen segmentation.

Automatic and manual segmentations are rarely exact. The Womersley tube
(radius 12 mm, 2.5 mm voxels, VENC 1 m/s, SNR 1.8, as in the earlier tests)
is analysed with the true mask, a mask eroded by one voxel (2.5 mm) and a
mask dilated by one voxel. Outcomes: net flow,
lumen radius from the mask, flow from the 16 x 32 polar grid scaled to that
radius, and WSS. 5 noise seeds.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import slicerheart_4dflow as f
from slicerheart_4dflow import phantom as ph
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.planes import sample_plane, grid_flow
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results"

n = (40, 60, 60); sp = 0.5; ds = 5
o = tuple(-(m - 1) / 2 * sp for m in n[::-1])
A = ph.voxel_to_world_matrix(spacing=(sp,) * 3, origin=o)
W = ph.womersley(shape=n, voxel_to_world=A, radius_mm=12.0, nt=20)
Ac = ph.voxel_to_world_matrix(spacing=(sp * ds,) * 3, origin=tuple(x + (ds - 1) / 2 * sp for x in o))
mc = W["mask"].reshape(n[0] // ds, ds, n[1] // ds, ds, n[2] // ds, ds).mean(axis=(1, 3, 5)) >= 0.5
spc = (sp * ds,) * 3
masks = {"true": mc, "eroded 1 voxel": f.erode_mm(mc, spc, 2.5), "dilated 1 voxel": f.dilate_mm(mc, spc, 2.5)}
tp = int(np.argmax(np.abs(W["wss_pa"])))
Qtrue = W["flow_mm3_s"]
T, N, B = np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])

def outcomes(v, m):
    pf = f.plane_flow(v, Ac, centre=(0, 0, 0), normal=(0, 0, 1), radius=20.0, lumen_mask=m)
    q = pf["flow"]
    area = float(np.atleast_1d(pf["area"])[0]) if "area" in pf else float(np.atleast_1d(pf["lumen_area"])[0])
    R = np.sqrt(area / np.pi)
    S = sample_plane(v, Ac, (0, 0, 0), T, N, B, R)
    qg = grid_flow(S["v_local"][..., 0], S["r_mm"], R)
    w = f.wall_shear_stress(v, m, Ac)
    return dict(net_flow=q.sum(), radius=R,
                grid_flow=qg.sum(), wss_peak=float(np.mean(w["wss_magnitude"][tp])), tawss=float(np.mean(w["tawss"])))

ref = dict(net_flow=Qtrue.sum(), radius=12.0,
           grid_flow=Qtrue.sum(), wss_peak=abs(W["wss_pa"][tp]), tawss=float(np.mean(np.abs(W["wss_pa"]))))
R = {k: [] for k in masks}
for seed in range(5):
    a = ph.acquire(W["velocity"], 1000.0, mask=W["mask"], snr=1.8, downsample=ds, rng=np.random.default_rng(seed))
    for k, m in masks.items():
        R[k].append(outcomes(a["velocity"], m))
res = {"truth": {k: float(v) for k, v in ref.items()}}
print(f"{'mask':16s} {'net flow':>9s} {'radius':>9s} {'grid flow':>10s} {'peak WSS':>9s} {'TAWSS':>8s}   (% error vs truth)")
for k in masks:
    mean = {q: float(np.mean([r[q] for r in R[k]])) for q in ref}
    err = {q: 100 * (mean[q] - ref[q]) / ref[q] for q in ref}
    res[k] = dict(mean=mean, error_pct=err)
    print(f"{k:16s} {err['net_flow']:+8.1f}% {err['radius']:+8.1f}% {err['grid_flow']:+9.1f}% {err['wss_peak']:+8.1f}% {err['tawss']:+7.1f}%")
json.dump(res, open(OUT / "test6_mask_sensitivity.json", "w"), indent=1)
