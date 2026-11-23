"""Test 2: flow is conserved along the vessel and at a branch.

(a) Stenosis tube (radius 12 mm, 30% constriction), generated at 0.5 mm and
    acquired at 2.5 mm with noise. Planes every 5 mm along the centreline.
    The same flow passes every plane, so measured flow should not change
    with plane position, including through the narrowing.
(b) Bifurcation (60/40 split). Parent flow should equal the sum of the
    daughter flows (mass_conservation).
"""
import json, warnings
from pathlib import Path
import numpy as np
import slicerheart_4dflow as f
from slicerheart_4dflow import phantom as ph
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results"
res = {}

# (a) stenosis ---------------------------------------------------------
sp, ds = 0.5, 5
n = (160, 70, 70)
o = tuple(-(m - 1) / 2 * sp for m in n[::-1])
A = ph.voxel_to_world_matrix(spacing=(sp,) * 3, origin=o)
S = ph.stenosis(shape=n, voxel_to_world=A, radius_mm=12.0, severity=0.3, length_mm=30.0, nt=10)
Ac = ph.voxel_to_world_matrix(spacing=(sp * ds,) * 3, origin=tuple(x + (ds - 1) / 2 * sp for x in o))
mc = S["mask"].reshape(n[0] // ds, ds, n[1] // ds, ds, n[2] // ds, ds).mean(axis=(1, 3, 5)) >= 0.5
Qtrue = S["flow_mm3_s"]
cl = np.c_[np.zeros(15), np.zeros(15), np.linspace(-35, 35, 15)]
planes = f.centerline_planes(cl, step_mm=5.0, radius_mm=16.0)
per_seed = []
for seed in range(5):
    a = ph.acquire(S["velocity"], 2500.0, mask=S["mask"], snr=12.0, downsample=ds, rng=np.random.default_rng(seed))
    v = a["velocity"]
    errs = []
    for p in planes:
        q = f.plane_flow(v, Ac, centre=p["centre"], normal=p["normal"], radius=p["radius"], lumen_mask=mc)["flow"]
        errs.append(100 * (q.sum() - Qtrue.sum()) / Qtrue.sum())
    per_seed.append(errs)
E = np.array(per_seed)
z = [float(p["centre"][2]) for p in planes]
res["stenosis"] = dict(z_mm=z, net_flow_error_pct_mean=E.mean(0).tolist(), net_flow_error_pct_sd=E.std(0).tolist(),
                       max_abs_error_pct=float(np.abs(E).max()), cv_between_planes_pct=float(np.mean(E.std(1))))
print("stenosis: plane z (mm) and net-flow error %")
for zi, m, s in zip(z, E.mean(0), E.std(0)):
    print(f"  z={zi:+5.1f}  {m:+5.2f} ± {s:.2f}")
print(f"  max |error| {np.abs(E).max():.2f}%   spread between planes (SD) {np.mean(E.std(1)):.2f}%")

# (b) bifurcation --------------------------------------------------------
B = ph.bifurcation(nt=10)
rows = []
for seed in range(5):
    a = ph.acquire(B["velocity"], 1500.0, mask=B["mask"], snr=12.0, rng=np.random.default_rng(seed))
    flows = {k: f.plane_flow(a["velocity"], B["voxel_to_world"], lumen_mask=B["mask"], **pl)["flow"] for k, pl in B["planes"].items()}
    mcon = f.mass_conservation(flows, [("bifurcation", ["parent"], ["daughter1", "daughter2"])], rr_ms=800.0)[0]
    split = flows["daughter1"].sum() / (flows["daughter1"].sum() + flows["daughter2"].sum())
    rows.append((mcon.get("percent_error", mcon.get("percent")), split))
R = np.array(rows, float)
res["bifurcation"] = dict(mass_error_pct_mean=float(R[:, 0].mean()), mass_error_pct_sd=float(R[:, 0].std()),
                          measured_split=float(R[:, 1].mean()), true_split=0.6)
print(f"bifurcation: parent vs daughters {R[:,0].mean():+.2f} ± {R[:,0].std():.2f}%  split {R[:,1].mean():.3f} (true 0.600)")
json.dump(res, open(OUT / "test2_flow_conservation.json", "w"), indent=1)
