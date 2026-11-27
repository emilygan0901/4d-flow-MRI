"""Evaluate the synthetic feature-recovery test.

Uses synthetic/data/features.npz (cross-fitted, Procrustes-aligned features)
and the factor tables. Reports, for the autoencoder and the PCA baseline:

1. Recovery: how well each known factor is predicted from the 8 features of
   a plane (cross-validated R^2, folds by case), with a linear probe (ridge)
   and a non-linear probe (k-nearest neighbours).
2. Beyond geometry: the same after removing local geometry (lumen radius and
   Dean strength) from both features and factor by linear regression.
3. Acquisition invariance: in the invariance set (same flow, 4 acquisitions),
   the share of each feature's variance due to acquisition alone.
4. Reproducibility: correlation of matched features between seeds after
   Procrustes alignment on the reference set.
5. Planted cohort test: 21 "LDS" and 12 "volunteer" cases resampled from the
   300 with a planted dilation difference (always) and a planted helicity
   difference (0, moderate, large). Subject-level features (mean over planes)
   are compared by rank regression adjusted for mean lumen radius, with
   Benjamini-Hochberg FDR over the 8 features. Detection rate is power when
   helicity differs and the false-positive rate when only dilation differs;
   the unadjusted analysis is shown for comparison.

Writes synthetic/results/evaluation.json.  Run: python synthetic/evaluate.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "synthetic" / "data", ROOT / "synthetic" / "results"
RES.mkdir(exist_ok=True)
F = np.load(DATA / "features.npz")
main = json.load(open(DATA / "main.json")); inv = json.load(open(DATA / "invariance.json"))
n, p = len(main["factors"]), len(main["planes"][0])
case, plane = F["case"], F["plane"]


def table(js):
    rows = []
    for fct, planes in zip(js["factors"], js["planes"]):
        for m in planes:
            r = dict(fct); r.update(m)
            r["mean_velocity_peak_m_s"] = fct["peak_flow_ml_s"] / (np.pi * m["radius_mm"] ** 2)   # mL/s / mm^2 = m/s
            rows.append(r)
    return rows


T = table(main)
col = lambda k, rows=T: np.array([r[k] for r in rows], float)
FACTORS = {
    "geometry": ["radius_mm", "dean", "arch_bulge_mm"],
    "haemodynamics": ["mean_velocity_peak_m_s", "peak_flow_ml_s", "helicity", "eccentricity", "reversal"],
    "acquisition": ["voxel_mm", "frames", "venc_m_s", "snr", "background_m_s"],
}


def probe_r2(Z, y, groups, kind, mask=None):
    if mask is not None:
        Z, y, groups = Z[mask], y[mask], groups[mask]
    est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))) if kind == "linear" \
        else make_pipeline(StandardScaler(), KNeighborsRegressor(15, weights="distance"))
    pred = cross_val_predict(est, Z, y, groups=groups, cv=GroupKFold(5))
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def residualise(A, G):
    return A - LinearRegression().fit(G, A).predict(G)


res = {"settings": dict(cases=n, planes_per_case=p, latent=int(F["ae"].shape[2]), seeds=int(F["ae"].shape[0]))}
res["reconstruction_mse"] = {m: float(F[f"err_{m}"].mean()) for m in ("ae", "pca")}

# 1-2. recovery
geo = np.c_[col("radius_mm"), col("dean")]
rec = {}
for model in ("ae", "pca"):
    Z = F[model][0]
    rec[model] = {}
    for grp, keys in FACTORS.items():
        for k in keys:
            y = col(k)
            r = dict(linear=probe_r2(Z, y, case, "linear"), nonlinear=probe_r2(Z, y, case, "knn"))
            if grp == "haemodynamics":
                r["beyond_geometry_nonlinear"] = probe_r2(residualise(Z, geo), residualise(y[:, None], geo)[:, 0], case, "knn")
            rec[model][k] = {a: round(b, 3) for a, b in r.items()}
    # reversal timing and extent, only where reversal is strong enough to see
    strong = col("reversal") > 0.25
    for k in ("reversal_phase", "reversal_width_deg"):
        rec[model][k + " (reversal>0.25)"] = {"nonlinear": round(probe_r2(Z, col(k), case, "knn", strong), 3)}
res["recovery_R2"] = rec

# 3. acquisition invariance
TI = table(inv)
cfg = np.repeat([f["config"] for f in inv["factors"]], p)
pl_i = np.tile(np.arange(p), len(inv["factors"]))
inv_share = {}
for model in ("ae", "pca"):
    Zi = F[f"inv_{model}"][0, 0]
    share = []
    for d in range(Zi.shape[1]):
        z = Zi[:, d]
        within = np.mean([z[(cfg == c) & (pl_i == q)].var() for c in np.unique(cfg) for q in range(p)])
        share.append(within / z.var())
    inv_share[model] = dict(mean=float(np.mean(share)), per_feature=[round(s, 3) for s in share])
res["acquisition_share_of_variance"] = inv_share

# 4. reproducibility across seeds
rep = {}
for model in ("ae", "pca"):
    S = F[model]; Si = F[f"inv_{model}"]
    rs = []
    for s in range(1, S.shape[0]):
        R, _ = orthogonal_procrustes(Si[s, 0], Si[0, 0])
        A, B = S[s] @ R, S[0]
        rs += [float(np.corrcoef(A[:, d], B[:, d])[0, 1]) for d in range(A.shape[1])]
    rep[model] = dict(median_r=float(np.median(rs)), min_r=float(np.min(rs)), share_r_ge_0_8=float(np.mean(np.array(rs) >= 0.8)))
    # recovery stability: helicity R^2 per seed
    rep[model]["helicity_R2_per_seed"] = [round(probe_r2(S[s], col("helicity"), case, "knn"), 3) for s in range(S.shape[0])]
res["reproducibility"] = rep

# 5. planted cohort test
fac = main["factors"]
dil = np.array([f["dilation"] for f in fac]); hel = np.array([f["helicity"] for f in fac])
mean_r = np.array([np.mean([m["radius_mm"] for m in pl]) for pl in main["planes"]])
z = lambda a: (a - a.mean()) / a.std()


def bh_any(pv, q=0.05):
    ps = np.sort(pv); m = len(ps)
    return bool(np.any(ps <= q * np.arange(1, m + 1) / m))


def group_p(feat, g, cov=None):
    y = stats.rankdata(feat)
    Xd = np.c_[np.ones_like(g), g] if cov is None else np.c_[np.ones_like(g), g, stats.rankdata(cov)]
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = len(y) - Xd.shape[1]
    se = np.sqrt(resid @ resid / dof * np.linalg.inv(Xd.T @ Xd)[1, 1])
    return 2 * stats.t.sf(abs(beta[1] / se), dof)


def cohort(rng, beta_h, gamma_d=1.5):
    wl = np.exp(beta_h * z(hel) + gamma_d * z(dil)); wl /= wl.sum()
    lds = rng.choice(n, 21, replace=False, p=wl)
    rest = np.setdiff1d(np.arange(n), lds)
    wv = np.exp(-gamma_d * z(dil[rest])); wv /= wv.sum()
    vol = rng.choice(rest, 12, replace=False, p=wv)
    return lds, vol


cres = {}
for model in ("ae", "pca"):
    Zs = F[model][0].reshape(n, p, -1).mean(1)               # subject-level: mean over planes
    cres[model] = {}
    for label, bh in (("dilation only (null)", 0.0), ("moderate helicity difference", 0.75), ("large helicity difference", 1.5)):
        rng = np.random.default_rng(7)
        det_adj = det_raw = 0; dh, dd = [], []
        for _ in range(1000):
            lds, vol = cohort(rng, bh)
            idx = np.r_[lds, vol]; g = np.r_[np.ones(21), np.zeros(12)]
            pv_adj = [group_p(Zs[idx, d], g, mean_r[idx]) for d in range(Zs.shape[1])]
            pv_raw = [group_p(Zs[idx, d], g) for d in range(Zs.shape[1])]
            det_adj += bh_any(np.array(pv_adj)); det_raw += bh_any(np.array(pv_raw))
            sd = lambda a: np.sqrt((a[lds].var(ddof=1) * 20 + a[vol].var(ddof=1) * 11) / 31)
            dh.append((hel[lds].mean() - hel[vol].mean()) / sd(hel)); dd.append((dil[lds].mean() - dil[vol].mean()) / sd(dil))
        cres[model][label] = dict(detection_rate_adjusted=det_adj / 1000, detection_rate_unadjusted=det_raw / 1000,
                                  helicity_d=round(float(np.mean(dh)), 2), dilation_d=round(float(np.mean(dd)), 2))
res["planted_cohort"] = cres
json.dump(res, open(RES / "evaluation.json", "w"), indent=1)
print(json.dumps(res, indent=1))
