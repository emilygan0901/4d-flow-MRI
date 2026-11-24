"""Figures of the digital phantom validation.

Figure_phantom_validation  (A) background correction methods, (B) phase
                           unwrapping, (C) WSS against voxel size,
                           (D) PWV precision against aortic stiffness.
Figure_frame_comparison    parallel-transport against Frenet frames on an
                           aorta-like centreline.

Panels A and C read the saved results in validation/results. Panels B and D
and the frame figure recompute small simulations with fixed seeds (about
one minute in total). Colours follow the study schematic.

Run from the repository root:  python figures/make_phantom_figures.py
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import slicerheart_4dflow as f
from slicerheart_4dflow import phantom as ph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.timing import arrival_time, pwv_regression
from pipeline.planes import parallel_transport_frames
warnings.filterwarnings("ignore")
RES, OUT = ROOT / "validation" / "results", ROOT / "figures"

PINK, TEAL, GREEN, PUR = "#E8436B", "#3E9BA6", "#7DB343", "#7B3FA0"
INK, Q, EDGE, GREY, BAND = "#1F2328", "#5A6270", "#BFC4C9", "#9AA1A9", "#EEF1F3"
plt.rcParams.update({"font.family": ["Inter", "Arial", "DejaVu Sans"], "font.size": 8.5, "axes.edgecolor": Q, "axes.labelcolor": INK,
                     "xtick.color": Q, "ytick.color": Q, "axes.linewidth": 0.8, "axes.titlesize": 9.5,
                     "legend.frameon": False, "legend.fontsize": 7.5})


def tidy(ax, letter, title):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", color=INK, pad=8)
    ax.text(-0.16, 1.07, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="bottom")


# ---------------------------------------------------------------- data
def unwrap_curves():
    n, sp, ds = (40, 60, 60), 0.5, 5
    o = tuple(-(m - 1) / 2 * sp for m in n[::-1])
    A = ph.voxel_to_world_matrix(spacing=(sp,) * 3, origin=o)
    W = ph.womersley(shape=n, voxel_to_world=A, radius_mm=12.0, nt=20)
    Ac = ph.voxel_to_world_matrix(spacing=(sp * ds,) * 3, origin=tuple(x + (ds - 1) / 2 * sp for x in o))
    mc = W["mask"].reshape(n[0] // ds, ds, n[1] // ds, ds, n[2] // ds, ds).mean(axis=(1, 3, 5)) >= 0.5
    a = ph.acquire(W["velocity"], 600.0, mask=W["mask"], snr=1.8, downsample=ds, rng=np.random.default_rng(0))
    uv = f.unwrap_velocity(a["velocity"], 600.0, Ac, method="laplacian_then_temporal")
    fl = lambda v: f.plane_flow(v, Ac, centre=(0, 0, 0), normal=(0, 0, 1), radius=14.0, lumen_mask=mc)["flow"] / 1000
    return W["t_s"] * 1000, W["flow_mm3_s"] / 1000, fl(a["velocity"]), fl(uv)


def pwv_distributions(cache=RES / "test5_pwv_distributions.json"):
    if cache.exists():
        return json.load(open(cache))
    RR, L, NPL, NOISE, REP, NT = 800.0, 250.0, 20, 0.03, 200, 25
    NF = 8000
    W = np.fft.rfft(ph.aortic_waveform(NF)); kk = np.arange(len(W))
    w = np.full(len(kk), 2.0); w[0] = 1.0; w[-1] = 1.0
    def delayed(dly, t):
        return np.real(np.exp(2j * np.pi * np.outer(t / RR, kk)) * w @ (W / NF * np.exp(-2j * np.pi * kk * dly / RR)))
    d = np.linspace(0, L, NPL); rng = np.random.default_rng(1); out = {}
    for c in (4.0, 6.0, 8.0, 12.0):
        reg, two = [], []
        for _ in range(REP):
            t = rng.uniform(0, RR / NT) + np.arange(NT) / NT * RR
            ta = np.array([arrival_time(delayed(di / c, t) + rng.normal(0, NOISE, NT), RR) + t[0] for di in d])
            reg.append(pwv_regression(ta, d)); dt = ta[-1] - ta[0]; two.append(L / dt if dt > 0 else np.nan)
        out[str(c)] = dict(regression=reg, two_plane=two)
    json.dump(dict(settings=dict(rr_ms=RR, length_mm=L, planes=NPL, frames=NT, noise_frac=NOISE, repeats=REP), pwv=out),
              open(cache, "w"))
    return json.load(open(cache))


# ---------------------------------------------------------------- figure 1
def figure_validation():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))
    (a, b), (c, d) = axs

    # A background correction
    bg = json.load(open(RES / "background_correction.json"))
    order = ["none", "ols", "arto", "msac", "wrls", "war+msac"]
    labels = ["None", "OLS", "ARTO", "MSAC", "WRLS", "WAR+MSAC"]
    vals = [bg[k]["static_mean"] for k in order]; sds = [bg[k]["static_sd"] for k in order]
    cols = [PINK, EDGE, EDGE, EDGE, EDGE, TEAL]
    a.barh(range(len(order)), vals, xerr=sds, color=cols, height=0.62, error_kw=dict(ecolor=Q, lw=0.8, capsize=2))
    a.set_xscale("log"); a.set_xlim(0.1, 40)
    a.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    a.set_yticks(range(len(order)), labels); a.invert_yaxis()
    for i, v in enumerate(vals):
        a.text(v * 1.25, i, f"{v:.2f}", va="center", fontsize=7.5, color=INK, fontweight="bold" if i == 5 else "normal")
    a.set_xlabel("Residual velocity in static tissue (mm/s, log scale)")
    tidy(a, "A", "Background phase correction")

    # B unwrapping
    t, qt, qa, qu = unwrap_curves()
    b.plot(t, qt, color=INK, lw=2.4, label="True flow", zorder=1)
    b.plot(t, qa, color=PINK, lw=1.5, marker="o", ms=3, label="Aliased (VENC 0.6 m/s)", zorder=2)
    b.plot(t, qu, color=TEAL, lw=1.5, ls="--", marker="s", ms=2.8, label="Unwrapped", zorder=3)
    b.set_xlabel("Time in cardiac cycle (ms)"); b.set_ylabel("Flow (mL/s)")
    b.legend(loc="upper right", handlelength=1.8)
    e2e = json.load(open(RES / "unwrapping_flow_wss.json"))["S"]["600.0"]
    b.set_ylim(-75, 235)
    b.text(0.97, 0.04, f"Stroke volume error\naliased {e2e['vol_raw_err'][0]:+.1f}%\nunwrapped {e2e['vol_unw_err'][0]:+.1f}%",
           transform=b.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK, linespacing=1.5)
    tidy(b, "B", "Phase unwrapping")

    # C WSS vs voxel size
    ws = json.load(open(RES / "wss_voxel_size.json"))
    P, S = np.array(ws["pulsatile"]), np.array(ws["steady"])
    c.axvspan(2.25, 2.75, color=BAND, zorder=0)
    c.text(2.5, 12, "Clinical\nvoxel size", ha="center", va="top", fontsize=7, color=Q)
    c.axhline(0, color=EDGE, lw=0.7)
    c.plot(S[:, 0], S[:, 1], color=TEAL, lw=1.6, marker="o", ms=4, label="Steady flow")
    c.plot(P[:, 0], P[:, 2], color=PINK, lw=1.6, marker="o", ms=4, label="Pulsatile, time-averaged")
    c.plot(P[:, 0], P[:, 1], color=PINK, lw=1.6, ls="--", marker="v", ms=4, label="Pulsatile, peak systole")
    c.set_xlim(0.3, 2.8); c.set_ylim(-80, 14); c.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    c.set_xlabel("Voxel size (mm)"); c.set_ylabel("Error in WSS (%)")
    c.legend(loc="lower left")
    c.annotate(f"{P[-1,1]:.0f}%", (P[-1, 0], P[-1, 1]), xytext=(0, -8), textcoords="offset points", fontsize=7.5, color=PINK, ha="center", va="top")
    tidy(c, "C", "Wall shear stress")

    # D PWV
    D = pwv_distributions()["pwv"]
    cs = [4.0, 6.0, 8.0, 12.0]; pos = np.arange(len(cs))
    for i, cc in enumerate(cs):
        for off, key, col in ((-0.18, "two_plane", GREY), (0.18, "regression", TEAL)):
            v = np.array(D[str(cc)][key], float); v = v[np.isfinite(v)]
            bp = d.boxplot(v, positions=[i + off], widths=0.3, patch_artist=True, showfliers=False,
                           whis=(5, 95), medianprops=dict(color=INK, lw=1.2))
            for p in bp["boxes"]:
                p.set_facecolor(col); p.set_edgecolor(col); p.set_alpha(0.9)
            for k in ("whiskers", "caps"):
                for p in bp[k]:
                    p.set_color(col); p.set_lw(1.0)
        d.plot([i - 0.38, i + 0.38], [cc, cc], color=INK, lw=1.0, ls=":")
    d.set_xticks(pos, [f"{c:.0f}" for c in cs]); d.set_xlim(-0.55, 3.55); d.set_ylim(0, 30)
    d.set_xlabel("True PWV (m/s)"); d.set_ylabel("Estimated PWV (m/s)")
    d.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Rectangle((0, 0), 1, 1, color=TEAL),
                      Line2D([], [], color=INK, ls=":")],
             labels=["Two planes", "20-plane regression", "True value"], loc="upper left")
    tidy(d, "D", "Pulse wave velocity")

    fig.tight_layout(h_pad=2.6, w_pad=2.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Figure_phantom_validation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def figure_frames():
    rng = np.random.default_rng(0)
    asc = np.c_[np.zeros(60), np.zeros(60), np.linspace(0, 59, 60)]
    th = np.linspace(0, np.pi, 95)[1:]
    arch = np.c_[30 - 30 * np.cos(th), 8 * np.sin(th), 60 + 30 * np.sin(th)]
    desc = np.c_[np.full(150, 60.0), np.zeros(150), np.linspace(59, -90, 150)]
    cl = np.vstack([asc, arch, desc]) + rng.normal(0, 0.3, (304, 3))
    p, T, N, B = parallel_transport_frames(cl, reference=(0, 1, 0))
    dT = np.gradient(T, axis=0); k = np.linalg.norm(dT, axis=1)
    Nf = dT / np.where(k[:, None] > 1e-9, k[:, None], 1)
    step = np.degrees(np.arccos(np.clip((N[1:] * N[:-1]).sum(1), -1, 1)))
    stepf = np.degrees(np.arccos(np.clip((Nf[1:] * Nf[:-1]).sum(1), -1, 1)))

    fig = plt.figure(figsize=(7.2, 3.9))
    idx = np.arange(4, len(p) - 4, 9)
    for j, (NN, title, col, st, letter) in enumerate(((N, "Parallel transport (used)", TEAL, step, "A"),
                                                     (Nf, "Frenet frame (rejected)", PINK, stepf, "B"))):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        ax.plot(*p.T, color=INK, lw=2.2, alpha=0.85)
        ax.quiver(*p[idx].T, *(NN[idx] * 14).T, color=col, lw=1.3, arrow_length_ratio=0.25)
        ax.view_init(elev=14, azim=-62); ax.set_box_aspect((70, 30, 160))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_visible(False); a.line.set_color((1, 1, 1, 0))
        x0 = 0.25 + 0.5 * j
        fig.text(x0, 0.975, title, ha="center", va="top", color=INK, fontsize=10)
        fig.text(0.02 + 0.5 * j, 0.975, letter, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
        fig.text(x0, 0.02, f"Largest turn {st.max():.0f}° between points, {int((st > 45).sum())} flips over 45°",
                 ha="center", va="bottom", fontsize=8, color=col)
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0.07, wspace=-0.15)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Figure_frame_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_validation()
    figure_frames()
    print("written to", OUT)
