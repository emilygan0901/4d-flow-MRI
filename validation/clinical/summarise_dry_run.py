"""Summarise validation/results/clinical_dry_run.json.

Agreement of aortic net flow (main planes) with cvi42 for the locked
background correction and the alternatives, as bias and 95% limits of
agreement (Bland-Altman), plus conservation along each vessel, encoding,
gating coverage and timing. Writes clinical_dry_run_summary.json.
"""
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[2] / "validation" / "results"
rows = [r for r in json.load(open(RES / "clinical_dry_run.json"))["rows"]]
ok = [r for r in rows if "error" not in r]


def ba(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    return dict(n=int(len(d)), bias_mL=float(d.mean()), loa_mL=[float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
                bias_pct=float(np.mean(100 * d / b)), mean_abs_pct=float(np.mean(np.abs(100 * d / b))))


mains = [(r["case"], k, p) for r in ok for k, p in r["planes"].items()
         if not k.endswith(("_prox", "_dist")) and "cvi42_net_mL" in p and not p.get("cvi42_known_difference")
         and abs(p["cvi42_net_mL"]) > 1.0]
S = {"cases_run": len(rows), "cases_failed": [r["case"] for r in rows if "error" in r], "planes_vs_cvi42": len(mains)}
for key, label in (("net_mL", "locked (war+msac)"), ("net_mL_msac", "msac"), ("net_mL_ols", "ols"),
                   ("net_mL_none", "no background correction"), ("library_net_mL", "library outputs")):
    pairs = [(p[key], p["cvi42_net_mL"]) for _, _, p in mains if key in p]
    if len(pairs) > 2:
        S[f"vs_cvi42: {label}"] = ba(*zip(*pairs))
pairs = [(p["net_mL"], p["library_net_mL"]) for _, _, p in mains if "library_net_mL" in p]
S["locked vs library outputs"] = ba(*zip(*pairs))
pairs = [(p["grid_net_mL"], p["net_mL"]) for _, _, p in mains if "grid_net_mL" in p]
S["polar grid vs plane_flow"] = ba(*zip(*pairs))
spreads = [v for r in ok for v in r["conservation_spread_pct"].values() if v is not None]
S["conservation_spread_pct"] = dict(median=float(np.median(spreads)), max=float(np.max(spreads)), n=len(spreads))
S["encoding_consistent"] = f"{sum(r['encoding_consistent'] for r in ok)}/{len(ok)}"
S["gating_coverage_pct"] = sorted(round(100 * r["gating_coverage"]) for r in ok)
S["frames"] = sorted(r["frames"] for r in ok)
S["frame_ms"] = [round(min(r["frame_ms"] for r in ok), 1), round(max(r["frame_ms"] for r in ok), 1)]
S["process_s"] = dict(median=float(np.median([r["process_s"] for r in ok])), max=float(max(r["process_s"] for r in ok)))
S["lumen_radius_mm"] = sorted(round(p["lumen_radius_mm"], 1) for _, _, p in mains if "lumen_radius_mm" in p)
json.dump(S, open(RES / "clinical_dry_run_summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
