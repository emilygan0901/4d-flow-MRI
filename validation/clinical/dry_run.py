"""Dry run of the locked pipeline on the public SlicerHeart 4D Flow test cases.

The 15 cases are anonymised clinical 4D Flow scans released by the
SlicerHeart project under CC BY 4.0 (slicerheart_4dflow.clinical_test_data).
They are not study data. For each case the raw velocity is processed with the
settings of config/pipeline_config.yaml, and the aortic plane flows are
compared with three references:

  cvi42     flows measured independently in Circle cvi42 (clinical software)
  library   flows in the data set's own outputs (SlicerHeart Import 4D Flow)
  prox/dist planes 3-5 mm up and downstream, where the flow must be the same

It also checks the velocity encoding, the gating (how much of the RR interval
the frames cover, which decides linear or Fourier phase normalisation), runs
the 16 x 32 polar sampler on each aortic plane, and records processing time.

Usage (data downloaded to the default cache, about 4 GB):
    python validation/clinical/dry_run.py [--data DIR] [--cases Patient0001 ...]
Writes validation/results/clinical_dry_run.json (summary numbers only, no images).
"""
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
import slicerheart_4dflow as f

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.planes import sample_plane, grid_flow
warnings.filterwarnings("ignore")
OUT = ROOT / "validation" / "results"
AORTIC = ("ao", "dao", "aortic valve", "aov")
ALT_BACKGROUND = ("ols", "msac", None)


def is_aortic(name):
    base = name.lower().replace("_prox", "").replace("_dist", "")
    return any(base == a or base.startswith(a + " ") or base.startswith(a + "-") for a in AORTIC)


def main_plane(name):
    return name.replace("_prox", "").replace("_dist", "")


def run_case(data, cid):
    t0 = time.time()
    C = f.load_case(data, cid)
    t_load = time.time() - t0
    A, rr, tms = C["voxel_to_world"], float(C["rr_ms"]), np.asarray(C["time_ms"], float)
    nt = len(tms)
    dt = rr / nt
    coverage = float((tms[-1] - tms[0] + np.median(np.diff(tms))) / rr)
    lumen = C["lumen"]
    t1 = time.time()
    P = f.preprocess(C["velocity_raw_encoded_t"], C["magnitude_t"], np.asarray(C["venc_encoded_mm_s"]), A,
                     lumen_mask=lumen, background_method="war+msac", unwrap_method="laplacian_then_temporal",
                     encoding_to_world=np.asarray(C["encoding_to_world"]), verbose=False)
    t_proc = time.time() - t1
    v = P["velocity"]
    enc = f.encoding_check(v, lumen, A)
    # sensitivity: the same unwrapping with other background corrections
    alt = {}
    for bm in ALT_BACKGROUND:
        try:
            alt[bm] = f.preprocess(C["velocity_raw_encoded_t"], C["magnitude_t"], np.asarray(C["venc_encoded_mm_s"]), A,
                                   lumen_mask=lumen, background_method=bm, unwrap_method="laplacian_then_temporal",
                                   encoding_to_world=np.asarray(C["encoding_to_world"]), verbose=False)["velocity"]
        except Exception as e:
            alt[bm] = None
    planes = {k: p for k, p in C.get("planes", {}).items() if is_aortic(k)}
    regions = C.get("plane_lumen", {}) or {}
    cvi = (C.get("references") or {}).get("cvi42", {})
    lib = (C.get("flows") or {}).get("planes", {})
    res_planes = {}
    for name, pl in planes.items():
        region = regions.get(name, lumen)
        pf = f.plane_flow(v, A, centre=pl["centre"], normal=pl["normal"], radius=pl["radius"], lumen_mask=region)
        q = np.asarray(pf["flow"]) / 1000.0                       # mL/s
        net = float(q.sum() * dt / 1000.0)                        # mL per beat
        row = dict(net_mL=net, peak_flow_mL_s=float(np.abs(q).max()))
        for bm, va in alt.items():
            if va is not None:
                qa = np.asarray(f.plane_flow(va, A, centre=pl["centre"], normal=pl["normal"], radius=pl["radius"],
                                             lumen_mask=region)["flow"])
                row[f"net_mL_{bm or 'none'}"] = float(qa.sum() * dt / 1e6)
        if name in lib:
            row["library_net_mL"] = float(lib[name]["net_volume_mL"])
        cp = (cvi.get("planes") or {}).get(name)
        if cp is not None:
            sgn = cp.get("plane_normal_sign", 1) or 1
            row["cvi42_net_mL"] = float(cp["net_volume_mL"]) * sgn
            if cp.get("known_difference"):
                row["cvi42_known_difference"] = cp["known_difference"]
        # polar sampler on the main planes
        if not name.endswith(("_prox", "_dist")):
            area = float(np.atleast_1d(pf["area"]).mean())
            R = np.sqrt(area / np.pi)
            n = np.asarray(pl["normal"], float); n /= np.linalg.norm(n)
            ref = np.array([0.0, 1.0, 0.0]) - n[1] * n
            if np.linalg.norm(ref) < 1e-6:
                ref = np.cross(n, [1.0, 0, 0])
            N = ref / np.linalg.norm(ref); B = np.cross(n, N)
            S = sample_plane(v, A, pl["centre"], n, N, B, R, mask=region)
            vl = S["v_local"]
            ksys = int(np.argmax(np.abs(q)))
            ax = np.abs(vl[ksys, ..., 0]).mean()
            inpl = np.sqrt(vl[ksys, ..., 1] ** 2 + vl[ksys, ..., 2] ** 2).mean()
            qg = grid_flow(vl[..., 0], S["r_mm"], R) / 1000.0
            row.update(lumen_radius_mm=R, grid_net_mL=float(qg.sum() * dt / 1000.0),
                       inplane_to_axial_peak=float(inpl / ax), grid_outside_fov=S["outside_fov"])
        res_planes[name] = row
    # conservation along each vessel (prox / plane / dist)
    cons = {}
    for name in res_planes:
        if name.endswith(("_prox", "_dist")):
            continue
        group = [res_planes[k]["net_mL"] for k in res_planes if main_plane(k) == name]
        if len(group) > 1:
            m = np.mean(group)
            cons[name] = float(100 * (max(group) - min(group)) / abs(m)) if m else None
    return dict(case=cid, vendor=C["acquisition"].get("vendor"), scanner=C["acquisition"].get("scanner"),
                frames=nt, rr_ms=rr, frame_ms=dt, gating_coverage=coverage,
                venc_mm_s=list(map(float, C["venc_encoded_mm_s"])),
                voxel_mm=[float(x) for x in np.linalg.norm(np.asarray(A)[:3, :3], axis=0)],
                encoding_consistent=bool(enc["consistent"]), encoding_ratio=float(enc["ratio"]),
                load_s=t_load, process_s=t_proc, planes=res_planes, conservation_spread_pct=cons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder of the data set (default: download to the library cache)")
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    data = a.data or f.clinical_test_data()
    cases = a.cases or f.dataset_cases(data)
    rows = []
    for cid in cases:
        try:
            r = run_case(data, cid)
        except Exception as e:                                   # record and continue
            r = dict(case=cid, error=f"{type(e).__name__}: {e}")
        rows.append(r)
        if "error" in r:
            print(cid, "ERROR", r["error"]); continue
        ao = {k: p for k, p in r["planes"].items() if not k.endswith(("_prox", "_dist"))}
        msg = "  ".join(f"{k}: {p['net_mL']:.1f} mL" + (f" (cvi42 {p['cvi42_net_mL']:.1f})" if "cvi42_net_mL" in p else "") for k, p in ao.items())
        print(f"{cid} {r['frames']} frames, coverage {100*r['gating_coverage']:.0f}%, encoding {'ok' if r['encoding_consistent'] else 'CHECK'}, "
              f"{r['process_s']:.0f} s | {msg}")
    json.dump(dict(dataset="SlicerHeart clinical test data rev002 (CC BY 4.0)", rows=rows),
              open(OUT / "clinical_dry_run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
