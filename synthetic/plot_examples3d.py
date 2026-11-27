"""Check figure for the 3D synthetic aortas.

Top: the geometry (centreline, dilated lumen outline, the 20 analysis planes)
for a baseline case. Below: cross-sections at an ascending, arch and
descending plane after the full pipeline, for single-mechanism cases
(Dean flow only, helicity, eccentric jet, localised reversal, uniform
reversal). Colour is axial velocity, arrows in-plane velocity. Peak systole,
except the reversal rows, which show the reversal phase.

Run from the repository root:  python synthetic/plot_examples3d.py
"""
import sys, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.generator3d import simulate, BASELINE, local_geometry
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
INK, Q, TEAL = "#1F2328", "#5A6270", "#3E9BA6"
plt.rcParams.update({"font.family": ["Inter", "Arial", "DejaVu Sans"], "font.size": 8})

QUIET = dict(helicity=0.0, eccentricity=0.0, reversal=0.0, snr=40.0, background_m_s=0.0)
CASES = [("Dean flow only\n(from 3D curvature)", {}),
         ("Helicity 0.25", dict(helicity=0.25)),
         ("Eccentric jet 0.4\n(anterior)", dict(eccentricity=0.4, jet_angle_deg=0.0)),
         ("Localised reversal 0.5, 120° sector", dict(reversal=0.5, reversal_width_deg=120.0, reversal_angle_deg=90.0)),
         ("Uniform reversal 0.5", dict(reversal=0.5, reversal_width_deg=360.0))]
PLANES = [(2, "ascending"), (10, "mid-arch"), (17, "descending")]


def disc(ax, x, k):
    n_r, n_t = x.shape[1:3]
    r_e = np.linspace(0, 1, n_r + 1); t_e = 2 * np.pi * (np.arange(n_t + 1) - 0.5) / n_t
    TE, RE = np.meshgrid(t_e, r_e)
    pc = ax.pcolormesh(RE * np.cos(TE), RE * np.sin(TE), x[k, ..., 0], cmap="RdBu_r", vmin=-1.2, vmax=1.2,
                       shading="flat", rasterized=True)
    rc = (np.arange(n_r) + 0.5) / n_r; tc = 2 * np.pi * np.arange(n_t) / n_t
    TT, RR = np.meshgrid(tc, rc)
    ur, ut = x[k, ..., 1], x[k, ..., 2]
    U, V = ur * np.cos(TT) - ut * np.sin(TT), ur * np.sin(TT) + ut * np.cos(TT)
    sel = (slice(2, None, 3), slice(None, None, 2))
    ax.quiver((RR * np.cos(TT))[sel], (RR * np.sin(TT))[sel], U[sel], V[sel], angles="xy", scale_units="xy",
              scale=1.4, width=0.012, color=INK)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=Q, lw=0.8))
    ax.set_aspect("equal"); ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08); ax.axis("off")
    return pc


fig = plt.figure(figsize=(7.2, 12.0))
gs = fig.add_gridspec(len(CASES) + 1, 4, height_ratios=[1.5] + [1] * len(CASES), width_ratios=[1.3, 1, 1, 1],
                      hspace=0.45, wspace=0.35)
# geometry panel
G = local_geometry(BASELINE)
X0, M0, meta0, vol = simulate(dict(BASELINE, **QUIET), seed=1, return_volume=True)
ax = fig.add_subplot(gs[0, :2])
p, r, N = G["p"], G["r"], G["N"]
side = np.cross(G["T"], [0, 1.0, 0]); side /= np.linalg.norm(side, axis=1, keepdims=True)
ax.fill(np.r_[p[:, 0] + r * side[:, 0], (p[:, 0] - r * side[:, 0])[::-1]],
        np.r_[p[:, 2] + r * side[:, 2], (p[:, 2] - r * side[:, 2])[::-1]], color="#E7F1F2", ec=TEAL, lw=0.8)
ax.plot(p[:, 0], p[:, 2], color=INK, lw=0.8, ls="--")
for i, m in enumerate(meta0):
    g = int(np.argmin(np.abs(G["s"] - m["s_mm"])))
    a, b = p[g] + 1.1 * r[g] * side[g], p[g] - 1.1 * r[g] * side[g]
    ax.plot([a[0], b[0]], [a[2], b[2]], color="#E8436B" if i in [q for q, _ in PLANES] else Q, lw=1.4 if i in [q for q, _ in PLANES] else 0.6)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("A  Geometry and the 20 analysis planes", loc="left", fontsize=9, color=INK, fontweight="bold")
ax2 = fig.add_subplot(gs[0, 2:])
s = [m["s_mm"] for m in meta0]
ax2.plot(s, [m["net_flow_ml"] for m in meta0], "o-", color=TEAL, ms=3, label="Net flow per beat (mL)")
ax2.set_ylim(0, 110); ax2.set_xlabel("Distance along centreline (mm)"); ax2.set_ylabel("Net flow (mL)")
ax2b = ax2.twinx()
ax2b.plot(s, [m["dean"] for m in meta0], "s--", color="#E8436B", ms=3)
ax2b.set_ylabel("Dean amplitude", color="#E8436B"); ax2b.set_ylim(0, 1)
for a_ in (ax2, ax2b):
    a_.spines["top"].set_visible(False)
ax2.set_title("B  Net flow and Dean strength", loc="left", fontsize=9, color=INK, fontweight="bold")
for i, (label, extra) in enumerate(CASES):
    fct = dict(BASELINE, **QUIET); fct.update(extra)
    X, M, meta = simulate(fct, seed=1)
    print(label, "net flow ascending/arch/descending (mL):", [round(meta[q]["net_flow_ml"], 1) for q, _ in PLANES])
    for j, (pi, pname) in enumerate(PLANES):
        rev = "reversal" in extra and extra.get("reversal", 0) > 0
        k = int(round(fct["reversal_phase"] * 20)) % 20 if rev else int(np.argmax(X[pi, ..., 0].sum((1, 2))))
        a_ = fig.add_subplot(gs[i + 1, j + 1]); pc = disc(a_, X[pi], k)
        if i == 0:
            a_.set_title(pname, fontsize=8.5, color=Q)
    t = fig.add_subplot(gs[i + 1, 0]); t.axis("off")
    t.text(0.0, 0.5, label, fontsize=8.5, color=INK, fontweight="bold", va="center", wrap=True)
cax = fig.add_axes([0.35, 0.04, 0.5, 0.01]); cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
cb.set_label("Axial velocity (m/s). Arrows show in-plane velocity. N (anterior) points right", color=INK)
fig.subplots_adjust(left=0.03, right=0.93, top=0.97, bottom=0.08)
out = ROOT / "synthetic" / "examples3d.png"
fig.savefig(out, dpi=200)
print(out)
