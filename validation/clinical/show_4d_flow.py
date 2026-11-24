"""Show the 4D flow of a public SlicerHeart test case in 3D Slicer.

Run in the Slicer Python console after open_case_in_slicer.py (same CASE):
    exec(open("validation/clinical/show_4d_flow.py").read())

Adds two things to the scene:
  1. a speed sequence (|v| per frame, mm/s) shown as a colour overlay on the
     slices, synchronised with the magnitude sequence, so pressing play on the
     Sequence toolbar shows the blood speed through the heartbeat;
  2. pathlines: virtual particles released from the ascending aorta plane at
     the start of the cycle and followed through one heartbeat, shown in the
     3D view as lines coloured by time.
Velocity is processed with the locked settings (war+msac, laplacian then
temporal unwrapping). Public anonymised data (CC BY 4.0), not study data.
"""
import numpy as np
import slicer, vtk
import slicerheart_4dflow as f

CASE = globals().get("CASE", "Patient0007")
SEED_PLANE = globals().get("SEED_PLANE", None)          # default: first aortic plane found
data = f.clinical_test_data(cases=[CASE])
C = f.load_case(data, CASE, world="RAS")                # RAS: Slicer's scene coordinates
A = np.asarray(C["voxel_to_world"], float)
print("Processing velocity ...")
v = f.preprocess(C["velocity_raw_encoded_t"], C["magnitude_t"], np.asarray(C["venc_encoded_mm_s"]), A,
                 lumen_mask=C["lumen"], background_method="war+msac", unwrap_method="laplacian_then_temporal",
                 encoding_to_world=np.asarray(C["encoding_to_world"]), verbose=False)["velocity"]
t_ms = np.asarray(C["time_ms"], float)

# ---- 1. speed sequence ------------------------------------------------------------
ijk2ras = vtk.vtkMatrix4x4()
for r in range(4):
    for c in range(4):
        ijk2ras.SetElement(r, c, A[r, c])
speed = np.linalg.norm(v, axis=-1).astype(np.float32)
speed[:, ~C["lumen"]] = 0                                # show blood only
seq = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "speed")
seq.SetIndexName("time"); seq.SetIndexUnit("ms")
for k, t in enumerate(t_ms):
    tmp = slicer.util.addVolumeFromArray(speed[k], ijkToRAS=ijk2ras, name="speed_tmp")
    seq.SetDataNodeAtValue(tmp, f"{t:.2f}")
    slicer.mrmlScene.RemoveNode(tmp)
browser = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
browser = browser[0] if browser else slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode")
browser.AddSynchronizedSequenceNode(seq)
browser.SetIndexDisplayFormat("%.0f ms")
speed_node = browser.GetProxyNode(seq)
speed_node.SetName("speed (mm/s)")
disp = speed_node.GetDisplayNode() or (speed_node.CreateDefaultDisplayNodes() or speed_node.GetDisplayNode())
disp.SetAndObserveColorNodeID(slicer.util.getNode("ColdToHotRainbow").GetID())
disp.AutoWindowLevelOff(); disp.SetWindowLevelMinMax(0, float(np.percentile(speed[speed > 0], 99)))
disp.SetApplyThreshold(True); disp.SetLowerThreshold(50.0)
for name in ("Red", "Yellow", "Green"):
    comp = slicer.app.layoutManager().sliceWidget(name).sliceLogic().GetSliceCompositeNode()
    comp.SetForegroundVolumeID(speed_node.GetID()); comp.SetForegroundOpacity(0.6)

# ---- 2. pathlines -------------------------------------------------------------------
names = [n for n in C["planes"] if not n.endswith(("_prox", "_dist")) and n.lower().startswith(("ao", "aortic"))]
name = SEED_PLANE or (names[0] if names else list(C["planes"])[0])
pl = C["planes"][name]
seeds = f.seeds_from_plane(C["plane_lumen"].get(name, C["lumen"]) if C.get("plane_lumen") else C["lumen"],
                           A, pl["centre"], pl["normal"], pl["radius"], spacing_mm=1.0)
print(f"Tracing {len(seeds)} particles from plane {name} through one heartbeat ...")
traces = f.advect_pathlines(v, A, C["rr_ms"], seeds, t0_ms=float(t_ms[0]), mask=C["lumen"])
model = slicer.modules.models.logic().AddModel(f.to_vtk_polydata(traces))
model.SetName(f"pathlines from {name}")
md = model.GetDisplayNode()
md.SetActiveScalarName("time_ms"); md.SetScalarVisibility(True)
md.SetAndObserveColorNodeID(slicer.util.getNode("Viridis").GetID() if slicer.mrmlScene.GetFirstNodeByName("Viridis")
                            else slicer.util.getNode("ColdToHotRainbow").GetID())
md.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDataScalarRange)
md.SetLineWidth(2); md.SetOpacity(0.9)
seg = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
if seg:
    seg[0].GetDisplayNode().SetOpacity3D(0.15)          # see the lines through the vessel wall
slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
print("Done. Press play on the Sequence toolbar for the speed overlay; rotate the 3D view to see the pathlines.")
