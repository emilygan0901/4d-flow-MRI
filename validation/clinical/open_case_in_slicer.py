"""Open one public SlicerHeart test case in 3D Slicer and plot its aortic flow.

Run in the Slicer Python console, from the repository root:
    CASE = "Patient0007"          # optional, any of Patient0001-Patient0015
    exec(open("validation/clinical/open_case_in_slicer.py").read())

Downloads only that case (about 100-250 MB) to ~/.SlicerHeart, loads the
magnitude and processed velocity sequences, the PC-MRA, the lumen
segmentation and the measurement planes, then computes the flow through the
aortic planes with the locked settings and plots the flow curves against
cvi42. Public anonymised data (CC BY 4.0), not study data.
"""
import glob, os
import numpy as np
import slicer
import slicerheart_4dflow as f

CASE = globals().get("CASE", "Patient0007")
data = f.clinical_test_data(cases=[CASE])
inp, out = f.case_directories(data, CASE)
print("Case folder:", inp)

# ---- load into the Slicer scene ---------------------------------------------
slicer.mrmlScene.Clear(0)
mag = slicer.util.loadNodeFromFile(os.path.join(inp, "acquisition", "magnitude.seq.nrrd"), "SequenceFile")
vel = slicer.util.loadNodeFromFile(os.path.join(out, "velocity.seq.nrrd"), "SequenceFile")
pcmra = slicer.util.loadVolume(os.path.join(out, "pcmra.nrrd"))
seg = slicer.util.loadSegmentation(os.path.join(inp, "annotations", "Segmentation.seg.nrrd"))
seg.CreateClosedSurfaceRepresentation()
for p in sorted(glob.glob(os.path.join(inp, "annotations", "*.mrk.json"))):
    if not p.endswith(("_prox.mrk.json", "_dist.mrk.json")):
        slicer.util.loadMarkups(p)
slicer.util.setSliceViewerLayers(background=pcmra, fit=True)
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()

# ---- flow with the locked settings --------------------------------------------
C = f.load_case(data, CASE)
P = f.preprocess(C["velocity_raw_encoded_t"], C["magnitude_t"], np.asarray(C["venc_encoded_mm_s"]),
                 C["voxel_to_world"], lumen_mask=C["lumen"], background_method="war+msac",
                 unwrap_method="laplacian_then_temporal", encoding_to_world=np.asarray(C["encoding_to_world"]),
                 verbose=False)
t = np.asarray(C["time_ms"], float)
dt = C["rr_ms"] / len(t)
cvi = (C.get("references") or {}).get("cvi42", {}).get("planes", {})
cols, names = [t], ["Time (ms)"]
for name, pl in C["planes"].items():
    base = name.lower()
    if name.endswith(("_prox", "_dist")) or not base.startswith(("ao", "dao", "aortic")):
        continue
    q = np.asarray(f.plane_flow(P["velocity"], C["voxel_to_world"], centre=pl["centre"], normal=pl["normal"],
                                radius=pl["radius"], lumen_mask=C.get("plane_lumen", {}).get(name, C["lumen"]))["flow"]) / 1000
    cols.append(q); names.append(f"{name} (this pipeline)")
    msg = f"{name}: net {q.sum() * dt / 1000:.1f} mL per beat"
    if name in cvi:
        c = cvi[name]; sgn = c.get("plane_normal_sign", 1) or 1
        msg += f", cvi42 {c['net_volume_mL'] * sgn:.1f} mL"
        ct = np.asarray(c.get("time_ms", []), float); cq = np.asarray(c.get("flow_rate_ml_s", []), float) * sgn
        if len(ct) == len(cq) and len(ct) > 1:
            cols.append(np.interp(t, ct, cq)); names.append(f"{name} (cvi42)")
    print(msg)
slicer.util.plot(np.column_stack(cols), xColumnIndex=0, columnNames=names,
                 title=f"{CASE} aortic flow (mL/s)", show=True)
print("Done. Press play on the Sequence toolbar to watch the cardiac cycle.")
