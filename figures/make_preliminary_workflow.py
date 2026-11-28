"""Overview of the preliminary work: three tests, each answering what the previous one could not.

Run from the repository root:  python figures/make_preliminary_workflow.py
"""
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
INK, Q, TEAL, PINK, GREEN = "#1F2328", "#5A6270", "#3E9BA6", "#E8436B", "#7DB343"
plt.rcParams.update({"font.family": ["Inter", "Arial", "DejaVu Sans"], "font.size": 8})

fig = plt.figure(figsize=(7.2, 4.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, c=Q):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=c, lw=1.3))


cols = [
    (2, "1  Digital phantoms", TEAL,
     "Computer-generated scans with\nexactly known velocity\n(tubes, stenosis, bifurcation, thorax)",
     "Does each processing step\nrecover the true flow?",
     "Yes for correction and sampling.\nWSS and PWV have limits."),
    (35, "2  Public clinical scans", PINK,
     "15 real 4D Flow scans with\nindependent measurements\nfrom clinical software (cvi42)",
     "Does the whole pipeline work\non real scans?",
     "Yes, agrees with cvi42.\nNeighbouring planes vary."),
    (68, "3  Synthetic 3D aortas", GREEN,
     "300 aortas with known geometry,\nflow and scanner settings,\nvaried independently",
     "Do learned features capture flow\nbeyond diameter? Can H2 detect it?",
     "Yes, but only large effects.\nDiameter leaks into features."),
]
for x, title, c, data, q, a in cols:
    w = 30
    box(x, 46, w, 8, c, c)
    ax.text(x + w / 2, 50, title, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    for y, h, lab, txt in ((31.5, 11, "Data", data), (20, 9.5, "Question", q), (8.5, 9.5, "Answer", a)):
        box(x, y, w, h, "white", "#D0D5DA")
        ax.text(x + 1.2, y + h - 1.4, lab.upper(), fontsize=6.5, color=c, fontweight="bold", va="top")
        ax.text(x + w / 2, y + h / 2 - 0.9, txt, ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.35)
for x in (32.9, 65.9):
    arrow(x, 50, x + 2.3, 50)
ax.text(50, 57.2, "Each test answers a question the previous one could not", ha="center", fontsize=8.5, color=Q, style="italic")
box(2, 0.6, 96, 5.2, "#F3F5F7", "#D0D5DA")
ax.text(50, 3.2, "Outputs   Locked processing settings  ·  design decisions in Section 4  ·  December set-up tasks",
        ha="center", va="center", fontsize=8.2, color=INK)
for x in (17, 50, 83):
    arrow(x, 8.0, x, 6.4)
out = ROOT / "figures" / "Figure_preliminary_workflow"
fig.savefig(f"{out}.png", dpi=300); fig.savefig(f"{out}.pdf")
print(out)
