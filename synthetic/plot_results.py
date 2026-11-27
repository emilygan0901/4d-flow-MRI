"""Figure of the synthetic feature-recovery test (reads synthetic/results/evaluation.json)."""
import json, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "synthetic" / "results" / "evaluation.json"))
TEAL, GREY, INK, Q, EDGE, PINK = "#3E9BA6", "#9AA1A9", "#1F2328", "#5A6270", "#BFC4C9", "#E8436B"
plt.rcParams.update({"font.family": ["Inter", "Arial", "DejaVu Sans"], "font.size": 8, "axes.edgecolor": Q,
                     "xtick.color": Q, "ytick.color": Q, "axes.linewidth": 0.8})

ROWS = [("Haemodynamics", "mean_velocity_peak_m_s", "Mean velocity"), ("Haemodynamics", "helicity", "Helicity"),
        ("Haemodynamics", "eccentricity", "Jet eccentricity"), ("Haemodynamics", "reversal", "Flow reversal"),
        ("Geometry", "dean", "Dean (curvature)"), ("Geometry", "radius_mm", "Lumen radius"),
        ("Acquisition", "voxel_mm", "Voxel size"), ("Acquisition", "frames", "Frames"), ("Acquisition", "venc_m_s", "VENC"),
        ("Acquisition", "snr", "SNR"), ("Acquisition", "background_m_s", "Background offset")]

fig = plt.figure(figsize=(7.2, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.45)
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(ROWS))[::-1].astype(float)
for off, m, c, lab in ((0.18, "ae", TEAL, "Autoencoder"), (-0.18, "pca", GREY, "PCA")):
    v = [max(R["recovery_R2"][m][k]["nonlinear"], 0) for _, k, _ in ROWS]
    ax.barh(y + off, v, 0.34, color=c, label=lab)
for i, (g, k, lab) in enumerate(ROWS):
    if g == "Haemodynamics":
        b = R["recovery_R2"]["ae"][k]["beyond_geometry_nonlinear"]
        ax.plot(max(b, 0), y[i] + 0.18, marker="|", ms=9, mew=2, color=INK)
ax.set_yticks(y, [lab for _, _, lab in ROWS])
for yy in (y[3] - 0.5, y[5] - 0.5):
    ax.axhline(yy, color=EDGE, lw=0.6)
ax.set_ylim(y[-1] - 0.6, y[0] + 1.1)
for yy, g in ((y[0] + 0.75, "Haemodynamics"), (y[4] + 0.45, "Geometry"), (y[6] + 0.45, "Acquisition")):
    ax.text(0.995, yy, g, ha="right", va="center", fontsize=7.5, color=Q, style="italic")
ax.text(0.03, (y[6] + y[10]) / 2, "All ≈ 0, so acquisition is not encoded", va="center", fontsize=7.5, color=INK)
ax.set_xlim(0, 1); ax.set_xlabel("Variance explained by the 8 features (cross-validated R²)")
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
ax.text(0.995, 0.17, "| autoencoder after removing local geometry", transform=ax.transAxes, ha="right", fontsize=7, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(-0.45, 1.04, "A", transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK)
ax.text(-0.36, 1.04, "What the features capture", transform=ax.transAxes, fontsize=9.5, color=INK)

ax = fig.add_subplot(gs[0, 1])
labels = list(R["planted_cohort"]["ae"].keys())
d = [R["planted_cohort"]["ae"][k]["helicity_d"] for k in labels]
for m, c, mk, lab in (("ae", TEAL, "o", "Autoencoder"), ("pca", GREY, "s", "PCA")):
    ax.plot(d, [R["planted_cohort"][m][k]["detection_rate_adjusted"] for k in labels], marker=mk, color=c, lw=1.6, ms=5, label=lab)
ax.axhline(0.05, color=PINK, lw=0.9, ls="--"); ax.text(1.55, 0.065, "5% false-positive level", ha="right", fontsize=7, color=PINK)
ax.axhline(0.8, color=EDGE, lw=0.9, ls=":"); ax.text(0.0, 0.82, "80% target power", fontsize=7, color=Q)
ax.set_ylim(0, 1); ax.set_xlim(-0.05, 1.65)
ax.set_xlabel("Planted helicity difference, LDS vs volunteers (Cohen's d)")
ax.set_ylabel("Cohorts with a significant feature")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.75), fontsize=7.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(-0.25, 1.04, "B", transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK)
ax.text(-0.13, 1.04, "Planted 21 vs 12 cohorts", transform=ax.transAxes, fontsize=9.5, color=INK)
fig.savefig(ROOT / "synthetic" / "results" / "Figure_synthetic_recovery.png", dpi=300, bbox_inches="tight")
fig.savefig(ROOT / "synthetic" / "results" / "Figure_synthetic_recovery.pdf", bbox_inches="tight")
