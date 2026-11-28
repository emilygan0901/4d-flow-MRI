"""Project timeline (Figure 3A of the planning report). Run: python figures/make_gantt.py"""
import logging
from datetime import date
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9.5,
                     "hatch.linewidth": 0.8})
INK, Q, EDGE = "#1F2328", "#5A6270", "#BFC4C9"
DONE, PLAN, PAR, WRITE, RED = "#9AA1A9", "#1D5A62", "#A9D3D8", "#D9822B", "#C8102E"
d = lambda m, day, y=None: date(y or (2026 if m >= 10 else 2027), m, day)

# (label, start, end, style)  style: done | progress | plan | parallel | write | buffer
TASKS = [
    ("Background reading and scoping", d(10, 1), d(10, 31), "done"),
    ("Literature review", d(10, 15), d(11, 20), "done"),
    ("Pipeline verification", d(11, 1), d(11, 30), "done"),
    ("Planning report", d(11, 10), d(1, 8), "progress"),
    ("WP0 Data and tool readiness", d(12, 1), d(1, 17), "plan"),
    ("WP1 Pre-processing (volunteers)", d(1, 17), d(1, 31), "plan"),
    ("WP2 Established parameters", d(2, 1), d(2, 28), "plan"),
    ("Deep learning input (volunteers)", d(2, 1), d(3, 15), "parallel"),
    ("WP3 Full cohort and baseline", d(3, 1), d(3, 31), "plan"),
    ("WP4 Deep learning", d(4, 1), d(4, 30), "plan"),
    ("WP5 Interpretation", d(5, 1), d(5, 24), "plan"),
    ("Final report writing", d(5, 3), d(5, 30), "write"),
    ("Buffer", d(5, 31), d(6, 13), "buffer"),
]
MILESTONES = [("M0", d(1, 17), 4), ("M1", d(1, 31), 5), ("M2", d(2, 28), 6), ("M3", d(3, 31), 8),
              ("M4", d(4, 30), 9), ("M5", d(6, 13), 12)]
DECISIONS = [("D1, D4", d(1, 31)), ("D2", d(2, 28)), ("D3", d(4, 30))]

fig, ax = plt.subplots(figsize=(6.4, 4.4))
n = len(TASKS)
for i, (lab, s, e, st) in enumerate(TASKS):
    y = n - 1 - i
    kw = dict(height=0.62, left=mdates.date2num(s), align="center", zorder=3)
    w = mdates.date2num(e) - mdates.date2num(s)
    if st == "done":
        ax.barh(y, w, color=DONE, **kw)
    elif st == "progress":
        ax.barh(y, w, color="white", edgecolor=DONE, hatch="////", lw=0.8, **kw)
    elif st == "plan":
        ax.barh(y, w, color=PLAN, **kw)
    elif st == "parallel":
        ax.barh(y, w, color=PAR, **kw)
    elif st == "write":
        ax.barh(y, w, color=WRITE, **kw)
    else:
        ax.barh(y, w, color="white", edgecolor=WRITE, hatch="////", lw=0.8, **kw)
for lab, dt, row in MILESTONES:
    y = n - 1 - row
    ax.plot(mdates.date2num(dt), y, "D", ms=7, color=INK, mec="white", mew=0.8, zorder=5)
    left = lab == "M0"   # keep the label clear of the D1, D4 line
    ax.text(mdates.date2num(dt) + (-2.5 if left else 2.5), y + 0.38, lab, fontsize=9, fontweight="bold", color=INK,
            va="bottom", ha="right" if left else "left", zorder=6)
# decision points: thin dashed red lines behind the bars, labels in boxes above the plot
for lab, dt in DECISIONS:
    x = mdates.date2num(dt)
    ax.axvline(x, color=RED, lw=1.4, ls=(0, (4, 3)), zorder=2, alpha=0.9)
    ax.text(x, n - 0.2, lab, ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=RED,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=RED, lw=0.8), zorder=7, clip_on=False)
ax.set_yticks(range(n), [t[0] for t in TASKS][::-1], fontsize=9.5)
ax.set_ylim(-0.7, n + 0.6)
ax.set_xlim(mdates.date2num(d(10, 1)), mdates.date2num(d(6, 30)))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.tick_params(axis="x", labelsize=9, colors=Q)
ax.tick_params(axis="y", length=0, colors=INK)
for mon in [d(mm, 1) for mm in (11, 12)] + [d(mm, 1) for mm in (1, 2, 3, 4, 5, 6)]:
    ax.axvline(mdates.date2num(mon), color="#EEF1F3", lw=0.8, zorder=1)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(Q)
handles = [Patch(fc=DONE, label="Completed"), Patch(fc="white", ec=DONE, hatch="////", label="In progress"),
           Patch(fc=PLAN, label="Planned work package"), Patch(fc=PAR, label="Parallel development"),
           Patch(fc=WRITE, label="Write-up"), Patch(fc="white", ec=WRITE, hatch="////", label="Buffer"),
           Line2D([], [], marker="D", ls="", color=INK, label="Milestone"),
           Line2D([], [], color=RED, ls=(0, (4, 3)), label="Decision point")]
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, handlelength=1.4, columnspacing=1.0)
out = Path(__file__).resolve().parent / "Figure_gantt"
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{out}.pdf", bbox_inches="tight")
print(out)
