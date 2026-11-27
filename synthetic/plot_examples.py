"""Gallery of synthetic planes: one factor varied per row, all others at baseline.

Each panel is the 16 x 32 polar grid as the autoencoder will see it, after
rendering, partial volume, noise and the study's own sampling code. Colour is
axial (through-plane) velocity, arrows the in-plane velocity. Rows show peak
systole, except flow reversal, which is shown in late systole.

Run from the repository root:  python synthetic/plot_examples.py
"""
import sys, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.generator import render_case, BASELINE
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
INK, Q = "#1F2328", "#5A6270"
plt.rcParams.update({"font.family": ["Inter", "Arial", "DejaVu Sans"], "font.size": 8})

ROWS = [("helicity", "Rotational flow (helicity)", [0.0, 0.1, 0.2, 0.3], "{:.1f}"),
        ("eccentricity", "Jet eccentricity (jet towards right)", [0.0, 0.15, 0.3, 0.45], "{:.2f}"),
        ("reversal", "Late-systolic near-wall flow reversal", [0.0, 0.17, 0.33, 0.5], "{:.2f}"),
        ("curvature", "Curvature (outer wall at right)", [0.0, 0.33, 0.67, 1.0], "{:.2f}"),
        ("diameter_mm", "Diameter (normalised to the lumen, so only partial volume differs)", [20, 28, 36, 45], "{:.0f} mm"),
        ("voxel_mm", "Voxel size (acquisition)", [1.5, 2.0, 2.5, 3.0], "{:.1f} mm")]

fig, axs = plt.subplots(len(ROWS), 4, figsize=(7.2, 11.6))
for i, (key, label, levels, fmt) in enumerate(ROWS):
    for j, lev in enumerate(levels):
        f = dict(BASELINE); f[key] = lev
        f["noise_m_s"] = 0.02                                  # light noise so the pattern is visible
        x, m, R = render_case(f, np.random.default_rng(3))
        k = 7 if key == "reversal" else int(np.argmax(x[..., 0].sum((1, 2))))
        n_r, n_t = x.shape[1:3]
        r_e = np.linspace(0, 1, n_r + 1); t_e = 2 * np.pi * (np.arange(n_t + 1) - 0.5) / n_t
        TE, RE = np.meshgrid(t_e, r_e)
        ax = axs[i, j]
        pc = ax.pcolormesh(RE * np.cos(TE), RE * np.sin(TE), x[k, ..., 0], cmap="RdBu_r", vmin=-1.4, vmax=1.4,
                           shading="flat", rasterized=True)
        rc = (np.arange(n_r) + 0.5) / n_r; tc = 2 * np.pi * np.arange(n_t) / n_t
        TT, RR = np.meshgrid(tc, rc)
        ur, ut = x[k, ..., 1], x[k, ..., 2]
        U, V = ur * np.cos(TT) - ut * np.sin(TT), ur * np.sin(TT) + ut * np.cos(TT)
        sel = (slice(2, None, 3), slice(None, None, 2))
        ax.quiver((RR * np.cos(TT))[sel], (RR * np.sin(TT))[sel], U[sel], V[sel], angles="xy", scale_units="xy",
                  scale=1.6, width=0.012, color=INK)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=Q, lw=0.8))
        ax.set_aspect("equal"); ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08); ax.axis("off")
        ax.set_title(fmt.format(lev), fontsize=8, color=Q, pad=1)
    axs[i, 0].text(-0.05, 1.2, label, transform=axs[i, 0].transAxes, fontsize=9, color=INK, fontweight="bold")
cax = fig.add_axes([0.2, 0.045, 0.6, 0.012]); cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
cb.set_label("Axial velocity (m/s). Arrows show in-plane velocity", color=INK)
fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.1, hspace=0.42, wspace=0.08)
out = ROOT / "synthetic" / "examples.png"
fig.savefig(out, dpi=200)
print(out)
