"""Bland-Altman figure of the clinical dry run (aortic net flow against cvi42).

(A) locked pipeline (WAR+MSAC background correction), (B) no background
correction. Points are the 29 aortic planes of the 15 public SlicerHeart
cases, shaded by lumen radius. Reads validation/results/clinical_dry_run.json.

Run from the repository root:  python figures/make_dry_run_figure.py
"""
import json, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
INK, Q, EDGE, BAND = "#1F2328", "#5A6270", "#BFC4C9", "#EEF1F3"
SHADES = {"< 5 mm": "#A9D3D8", "5–8 mm": "#3E9BA6", "> 8 mm": "#1D5A62"}     # one hue, light to dark
plt.rcParams.update({"font.family": ["Inter", "Arial", "DejaVu Sans"], "font.size": 8.5, "axes.edgecolor": Q,
                     "axes.labelcolor": INK, "xtick.color": Q, "ytick.color": Q, "axes.linewidth": 0.8})

rows = json.load(open(ROOT / "validation" / "results" / "clinical_dry_run.json"))["rows"]
P = [p for r in rows if "error" not in r for k, p in r["planes"].items()
     if not k.endswith(("_prox", "_dist")) and "cvi42_net_mL" in p and "lumen_radius_mm" in p
     and not p.get("cvi42_known_difference") and abs(p["cvi42_net_mL"]) > 1.0]
ref = np.array([p["cvi42_net_mL"] for p in P])
rad = np.array([p["lumen_radius_mm"] for p in P])
group = np.where(rad < 5, "< 5 mm", np.where(rad <= 8, "5–8 mm", "> 8 mm"))


def panel(ax, key, letter, title):
    m = np.array([p[key] for p in P])
    mean, diff = (m + ref) / 2, m - ref
    bias, sd = diff.mean(), diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    ax.axhspan(lo, hi, color=BAND, zorder=0)
    ax.axhline(0, color=EDGE, lw=0.7, zorder=1)
    ax.axhline(bias, color=INK, lw=1.2, zorder=2)
    for y in (lo, hi):
        ax.axhline(y, color=INK, lw=0.9, ls="--", zorder=2)
    for g, col in SHADES.items():
        s = group == g
        ax.scatter(mean[s], diff[s], s=34, color=col, edgecolor="white", linewidth=0.8, zorder=3)
    x1 = 81
    for y, txt in ((hi, f"+1.96 SD  {hi:+.1f}"), (bias, f"Bias  {bias:+.1f}"), (lo, f"−1.96 SD  {lo:+.1f}")):
        ax.text(x1, y + 0.35, txt.replace("-", "−"), ha="right", va="bottom", fontsize=7.5, color=INK)
    ax.set_xlim(-82, 82); ax.set_ylim(-12, 17.5)
    ax.set_xlabel("Mean of pipeline and cvi42 net flow (mL per beat)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="bottom")
    ax.text(-0.04, 1.04, title, transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom")
    return bias, lo, hi


fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
panel(a, "net_mL", "A", "Locked pipeline (WAR+MSAC)")
panel(b, "net_mL_none", "B", "Without background correction")
a.set_ylabel("Pipeline − cvi42 (mL per beat)")
handles = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="white", ms=7, label=g) for g, c in SHADES.items()]
fig.legend(handles=handles, title="Lumen radius", loc="lower center", ncol=3, frameon=False, fontsize=7.5,
           title_fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
for ext in ("png", "pdf", "svg"):
    fig.savefig(ROOT / "figures" / f"Figure_clinical_agreement.{ext}", dpi=300, bbox_inches="tight")
print("n =", len(P))
