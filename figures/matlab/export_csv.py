"""Export the saved results to plain numeric CSV files for the MATLAB figure scripts.

Run from the repository root:  python figures/matlab/export_csv.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "figures" / "matlab" / "data"
OUT.mkdir(exist_ok=True)
R = ROOT / "validation" / "results"

# phantom A: background correction (order: none, OLS, ARTO, MSAC, WRLS, WAR+MSAC)
bg = json.load(open(R / "background_correction.json"))
keys = ["none", "ols", "arto", "msac", "wrls", "war+msac"]
np.savetxt(OUT / "phantom_background.csv", [[i, bg[k]["static_mean"], bg[k]["static_sd"]] for i, k in enumerate(keys)],
           delimiter=",", header="method_index,residual_mm_s,sd_mm_s", comments="")
# phantom B: unwrapping curves (recomputed, fixed seed)
sys.path.insert(0, str(ROOT / "figures"))
from make_phantom_figures import unwrap_curves, pwv_distributions
t, qt, qa, qu = unwrap_curves()
np.savetxt(OUT / "phantom_unwrap.csv", np.c_[t, qt, qa, qu], delimiter=",", header="time_ms,true,aliased,unwrapped", comments="")
# phantom C: WSS vs voxel
ws = json.load(open(R / "wss_voxel_size.json"))
P, S = np.array(ws["pulsatile"]), np.array(ws["steady"])
np.savetxt(OUT / "phantom_wss.csv", np.c_[P[:, 0], S[:, 1], P[:, 2], P[:, 1]], delimiter=",",
           header="voxel_mm,steady,pulsatile_mean,pulsatile_peak", comments="")
# phantom D: PWV distributions (columns: true_pwv, method 1=two-plane 2=regression, estimate)
D = pwv_distributions()["pwv"]
rows = [[float(c), m, v] for c, d in D.items() for m, key in ((1, "two_plane"), (2, "regression")) for v in d[key] if np.isfinite(v)]
np.savetxt(OUT / "phantom_pwv.csv", rows, delimiter=",", header="true_pwv,method,estimate", comments="")

# clinical: Bland-Altman points
rows = json.load(open(R / "clinical_dry_run.json"))["rows"]
P = [p for r in rows if "error" not in r for k, p in r["planes"].items()
     if not k.endswith(("_prox", "_dist")) and "cvi42_net_mL" in p and "lumen_radius_mm" in p
     and not p.get("cvi42_known_difference") and abs(p["cvi42_net_mL"]) > 1.0]
np.savetxt(OUT / "clinical_agreement.csv",
           [[p["cvi42_net_mL"], p["net_mL"], p["net_mL_none"], p["lumen_radius_mm"]] for p in P],
           delimiter=",", header="cvi42_mL,pipeline_mL,no_background_mL,radius_mm", comments="")

# synthetic: recovery and cohort power
E = json.load(open(ROOT / "synthetic" / "results" / "evaluation.json"))
fac = ["mean_velocity_peak_m_s", "helicity", "eccentricity", "reversal", "dean", "radius_mm",
       "voxel_mm", "frames", "venc_m_s", "snr", "background_m_s"]
rec = []
for k in fac:
    ae, pc = E["recovery_R2"]["ae"][k], E["recovery_R2"]["pca"][k]
    rec.append([ae["nonlinear"], pc["nonlinear"], ae.get("beyond_geometry_nonlinear", np.nan)])
np.savetxt(OUT / "synthetic_recovery.csv", rec, delimiter=",", header="ae_R2,pca_R2,ae_beyond_geometry_R2", comments="")
coh = E["planted_cohort"]
np.savetxt(OUT / "synthetic_power.csv",
           [[coh["ae"][k]["helicity_d"], coh["ae"][k]["detection_rate_adjusted"], coh["pca"][k]["detection_rate_adjusted"]] for k in coh["ae"]],
           delimiter=",", header="helicity_d,ae_detection,pca_detection", comments="")
print(sorted(p.name for p in OUT.iterdir()))
