"""Save report figures of a public test case shown in 3D Slicer.

Run in the Slicer Python console after open_case_in_slicer.py and
show_4d_flow.py (same CASE):
    exec(open("validation/clinical/save_case_figure.py").read())

Writes to figures/clinical_<CASE>/:
  pathlines_3d.png      3D view: lumen surface and pathlines (white background)
  speed_peak_systole.png  slice view with the speed overlay at peak systole
  flow_curves.csv       flow (mL/s) per frame for each aortic plane, this pipeline and cvi42
  flow_curves.png       the same as a plot
  Figure_clinical_<CASE>.png / .pdf   the three panels combined (A, B, C)

Public anonymised data from the SlicerHeart project (CC BY 4.0): credit it in
the caption. Never run this on study data outside the Secure Enclave.
"""
import os
import numpy as np
import slicer, vtk, ScreenCapture
import slicerheart_4dflow as f

CASE = globals().get("CASE", "Patient0007")
OUTDIR = os.path.join(os.getcwd(), "figures", f"clinical_{CASE}")
os.makedirs(OUTDIR, exist_ok=True)
try:
    import matplotlib
except ImportError:
    slicer.util.pip_install("matplotlib")
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
PINK, TEAL, INK, Q, EDGE = "#E8436B", "#3E9BA6", "#1F2328", "#5A6270", "#BFC4C9"
cap = ScreenCapture.ScreenCaptureLogic()
lm = slicer.app.layoutManager()
slicer.util.setViewControllersVisible(False)

# ---- A. 3D pathlines -------------------------------------------------------------
lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
slicer.app.processEvents()
vn = slicer.util.getNodesByClass("vtkMRMLViewNode")[0]
vn.SetBackgroundColor(1, 1, 1); vn.SetBackgroundColor2(1, 1, 1)
vn.SetBoxVisible(False); vn.SetAxisLabelsVisible(False)
for sn in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
    sn.SetSliceVisible(False)                           # hide slice planes so the camera frames the vessels
view3d = lm.threeDWidget(0).threeDView()
view3d.resetFocalPoint(); view3d.resetCamera(); slicer.app.processEvents()
p3d = os.path.join(OUTDIR, "pathlines_3d.png")
cap.captureImageFromView(view3d, p3d)

# ---- B. speed at peak systole ------------------------------------------------------
lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
browser = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")[0]
seq = slicer.util.getFirstNodeByClassByName("vtkMRMLSequenceNode", "speed")
peak, best = 0, -1.0
for i in range(seq.GetNumberOfDataNodes()):
    a = slicer.util.arrayFromVolume(seq.GetNthDataNode(i))
    if a.sum() > best:
        best, peak = float(a.sum()), i
browser.SetSelectedItemNumber(peak)
red = lm.sliceWidget("Red")
red.sliceLogic().FitSliceToAll(); slicer.app.processEvents()
pspd = os.path.join(OUTDIR, "speed_peak_systole.png")
cap.captureImageFromView(red.sliceView(), pspd)
lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
slicer.util.setViewControllersVisible(True)

# ---- C. flow curves ---------------------------------------------------------------------
data = f.clinical_test_data(cases=[CASE])
C = f.load_case(data, CASE)
v = f.preprocess(C["velocity_raw_encoded_t"], C["magnitude_t"], np.asarray(C["venc_encoded_mm_s"]),
                 C["voxel_to_world"], lumen_mask=C["lumen"], background_method="war+msac",
                 unwrap_method="laplacian_then_temporal", encoding_to_world=np.asarray(C["encoding_to_world"]),
                 verbose=False)["velocity"]
t = np.asarray(C["time_ms"], float); dt = C["rr_ms"] / len(t)
cvi = (C.get("references") or {}).get("cvi42", {}).get("planes", {})
cols, header, curves = [t], ["time_ms"], []
for name, pl in C["planes"].items():
    if name.endswith(("_prox", "_dist")) or not name.lower().startswith(("ao", "dao", "aortic")):
        continue
    q = np.asarray(f.plane_flow(v, C["voxel_to_world"], centre=pl["centre"], normal=pl["normal"], radius=pl["radius"],
                                lumen_mask=C.get("plane_lumen", {}).get(name, C["lumen"]))["flow"]) / 1000
    cols.append(q); header.append(f"{name}_pipeline_mL_s")
    cq = None
    if name in cvi:
        c = cvi[name]; s = c.get("plane_normal_sign", 1) or 1
        ct, cv = np.asarray(c.get("time_ms", []), float), np.asarray(c.get("flow_rate_ml_s", []), float) * s
        if len(ct) == len(cv) > 1:
            cq = (ct, cv); cols.append(np.interp(t, ct, cv)); header.append(f"{name}_cvi42_mL_s")
    curves.append((name, q, q.sum() * dt / 1000, cq, cvi.get(name, {}).get("net_volume_mL")))
np.savetxt(os.path.join(OUTDIR, "flow_curves.csv"), np.column_stack(cols), delimiter=",",
           header=",".join(header), comments="", fmt="%.3f")

plt.rcParams.update({"font.family": ["Inter", "Arial", "DejaVu Sans"], "font.size": 8.5, "axes.edgecolor": Q, "xtick.color": Q, "ytick.color": Q})
def flow_axes(ax):
    for i, (name, q, net, cq, cnet) in enumerate(curves):
        col = TEAL if i == 0 else PINK
        ax.plot(t, q, color=col, lw=1.8, marker="o", ms=3, label=f"{name}, this pipeline ({net:.1f} mL)")
        if cq is not None:
            s = cvi[name].get("plane_normal_sign", 1) or 1
            ax.plot(*cq, color=col, lw=1.2, ls="--", label=f"{name}, cvi42 ({cnet * s:.1f} mL)")
    ax.axhline(0, color=EDGE, lw=0.7)
    ax.set_xlabel("Time in cardiac cycle (ms)"); ax.set_ylabel("Flow (mL/s)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

fig, ax = plt.subplots(figsize=(3.6, 2.8)); flow_axes(ax); fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "flow_curves.png"), dpi=300); plt.close(fig)

# ---- combined figure ------------------------------------------------------------------------
def crop(path, background, pad=12):
    """Cut a screenshot down to its content (rows and columns that are not background)."""
    img = plt.imread(path)[..., :3]
    diff = np.abs(img - np.asarray(background, float)).max(-1) > 0.08
    rows = np.where(diff.mean(1) > 0.02)[0]; cols = np.where(diff.mean(0) > 0.02)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, img.shape[1])
    return img[r0:r1, c0:c1]

fig = plt.figure(figsize=(7.2, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 0.9, 1.35], wspace=0.38)
for j, (path, bg, title) in enumerate(((p3d, (1, 1, 1), "Pathlines"), (pspd, (0, 0, 0), "Speed, peak systole"))):
    ax = fig.add_subplot(gs[0, j]); ax.imshow(crop(path, bg)); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.text(-0.04, 1.1, "AB"[j], transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK)
ax = fig.add_subplot(gs[0, 2]); flow_axes(ax)
ax.set_title("Aortic flow", loc="left", fontsize=9, color=INK)
ax.text(-0.22, 1.08, "C", transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUTDIR, f"Figure_clinical_{CASE}.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved to", OUTDIR)
