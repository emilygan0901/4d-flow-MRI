"""Test 3: polar plane sampling and the local frame.

(a) Profile test. Poiseuille flow in a tube whose axis is oblique to the
    image axes (so nothing lines up with the voxel grid), at 1 mm (noise
    free, isolates the sampler) and at 2.5 mm with noise (clinical case).
    On the 16 x 32 polar grid the axial velocity must follow the analytic
    parabola, radial and circumferential velocity must be ~0, and flow
    integrated on the grid must match the analytic flow.
(b) Frame test. An aorta-like centreline (ascending, arch, descending, with
    an out-of-plane twist and 0.3 mm jitter, as from VMTK). The parallel-
    transport frame must stay orthonormal and turn smoothly, with no flips.
    The Frenet frame is computed for comparison.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from slicerheart_4dflow import phantom as ph
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.planes import parallel_transport_frames, sample_plane, grid_flow
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results"
res = {}

# (a) profile -------------------------------------------------------------
R, vmax = 12.0, 800.0
axis = np.array([0.35, 0.25, 0.90]); axis /= np.linalg.norm(axis)
Qtrue = np.pi * R**2 * vmax / 2
N0 = np.cross(axis, [1, 0, 0]); N0 /= np.linalg.norm(N0); B0 = np.cross(axis, N0)

def run(sp, ds, snr, seeds):
    n = (40 * ds, 40 * ds, 40 * ds) if ds > 1 else (60, 60, 60)
    o = tuple(-(m - 1) / 2 * sp for m in n[::-1])
    A = ph.voxel_to_world_matrix(spacing=(sp,) * 3, origin=o)
    P = ph.poiseuille(shape=n, voxel_to_world=A, radius_mm=R, v_max_mm_s=vmax, axis_world=tuple(axis))
    Ac = ph.voxel_to_world_matrix(spacing=(sp * ds,) * 3, origin=tuple(x + (ds - 1) / 2 * sp for x in o))
    out = []
    for s in seeds:
        if snr is None:
            v = P["velocity"][:, ::1] if ds == 1 else ph.acquire(P["velocity"], 2000.0, mask=P["mask"], snr=1e6, downsample=ds, rng=np.random.default_rng(s))["velocity"]
        else:
            v = ph.acquire(P["velocity"], 1500.0, mask=P["mask"], snr=snr, downsample=ds, rng=np.random.default_rng(s))["velocity"]
        S = sample_plane(v, Ac, (0, 0, 0), axis, N0, B0, R)
        vl = S["v_local"][0]
        r = S["r_mm"]
        prof = vl[..., 0].mean(1)
        true = vmax * (1 - r**2 / R**2)
        out.append(dict(
            profile_rmse_pct=float(100 * np.sqrt(np.mean((prof - true) ** 2)) / vmax),
            profile_rmse_inner_pct=float(100 * np.sqrt(np.mean((prof[:12] - true[:12]) ** 2)) / vmax),
            radial_rms_pct=float(100 * np.sqrt(np.mean(vl[..., 1] ** 2)) / vmax),
            circ_rms_pct=float(100 * np.sqrt(np.mean(vl[..., 2] ** 2)) / vmax),
            flow_err_pct=float(100 * (grid_flow(vl[..., 0], r, R) - Qtrue) / Qtrue),
            profile=prof.tolist()))
    keys = [k for k in out[0] if k != "profile"]
    return {k: (float(np.mean([o[k] for o in out])), float(np.std([o[k] for o in out]))) for k in keys} | {"profile_seed0": out[0]["profile"]}

res["profile_1mm_noise_free"] = run(1.0, 1, None, [0])
res["profile_2.5mm_snr12"] = run(0.5, 5, 12.0, range(5))
for name in ("profile_1mm_noise_free", "profile_2.5mm_snr12"):
    r = res[name]
    print(f"{name}: axial profile RMSE {r['profile_rmse_pct'][0]:.2f}% of vmax (inner 75% of radius {r['profile_rmse_inner_pct'][0]:.2f}%), "
          f"radial {r['radial_rms_pct'][0]:.2f}%, circumferential {r['circ_rms_pct'][0]:.2f}%, flow error {r['flow_err_pct'][0]:+.2f}%")

# (b) frame ---------------------------------------------------------------
rng = np.random.default_rng(0)
asc = np.c_[np.zeros(60), np.zeros(60), np.linspace(0, 59, 60)]
th = np.linspace(0, np.pi, 95)[1:]
arch = np.c_[30 - 30 * np.cos(th), 8 * np.sin(th), 60 + 30 * np.sin(th)]          # out-of-plane bulge
desc = np.c_[np.full(150, 60.0), np.zeros(150), np.linspace(59, -90, 150)]
cl = np.vstack([asc, arch, desc]) + rng.normal(0, 0.3, (60 + 94 + 150, 3))
p, T, N, B = parallel_transport_frames(cl, reference=(0, 1, 0))
ortho = max(np.abs((T * N).sum(1)).max(), np.abs((T * B).sum(1)).max(), np.abs(np.linalg.norm(N, axis=1) - 1).max())
step = np.degrees(np.arccos(np.clip((N[1:] * N[:-1]).sum(1), -1, 1)))
# Frenet normal for comparison
dT = np.gradient(T, axis=0); k = np.linalg.norm(dT, axis=1)
Nf = dT / np.where(k[:, None] > 1e-9, k[:, None], 1)
stepf = np.degrees(np.arccos(np.clip((Nf[1:] * Nf[:-1]).sum(1), -1, 1)))
s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
res["frame"] = dict(centreline_length_mm=float(s[-1]), max_orthonormality_error=float(ortho),
                    parallel_transport_max_step_deg=float(step.max()), parallel_transport_flips_over_45deg=int((step > 45).sum()),
                    frenet_max_step_deg=float(stepf.max()), frenet_flips_over_45deg=int((stepf > 45).sum()))
f_ = res["frame"]
print(f"frame: length {f_['centreline_length_mm']:.0f} mm, orthonormality error {f_['max_orthonormality_error']:.1e}, "
      f"parallel transport max turn {f_['parallel_transport_max_step_deg']:.1f} deg/step ({f_['parallel_transport_flips_over_45deg']} flips), "
      f"Frenet max turn {f_['frenet_max_step_deg']:.1f} deg/step ({f_['frenet_flips_over_45deg']} flips)")
json.dump(res, open(OUT / "test3_plane_sampling.json", "w"), indent=1)
