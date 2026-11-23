"""Test 1: velocity component order and sign.

A scanner or DICOM import can swap velocity components or flip their sign.
The stenosis phantom (exactly divergence-free) is acquired with noise, then
corrupted by known swaps and sign flips. encoding_check must identify the
mapping that undoes the corruption, and expected_flow_sign must fix the
remaining global sign (which divergence cannot detect, since -v is also
divergence-free).
"""
import json, warnings, itertools
from pathlib import Path
import numpy as np
import slicerheart_4dflow as f
from slicerheart_4dflow import phantom as ph
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results"

S = ph.stenosis(radius_mm=12.0, severity=0.3, nt=8, shape=(64, 40, 40))
A, mask, vt = S["voxel_to_world"], S["mask"], S["velocity"]
centre = np.array([19.5, 19.5, 31.5]) - np.array([0, 0, 22.0])   # inlet-side plane, flow along +z

corruptions = {
    "none": ((0, 1, 2), (1, 1, 1)),
    "flip z": ((0, 1, 2), (1, 1, -1)),
    "flip x": ((0, 1, 2), (-1, 1, 1)),
    "swap x,y": ((1, 0, 2), (1, 1, 1)),
    "swap x,z": ((2, 1, 0), (1, 1, 1)),
    "swap y,z + flip x": ((0, 2, 1), (-1, 1, 1)),
    "flip all": ((0, 1, 2), (-1, -1, -1)),
}
rows = []
for seed in range(3):
    a = ph.acquire(vt, 2500.0, mask=mask, snr=12.0, rng=np.random.default_rng(seed))
    vm = a["velocity"]
    for name, (perm, sign) in corruptions.items():
        vc = vm[..., list(perm)] * np.array(sign)
        e = f.encoding_check(vc, mask, A)
        bp, bs = e["best"]
        fixed = vc[..., list(bp)] * np.array(bs)
        s = f.expected_flow_sign(fixed, A, centre=tuple(centre), normal=(0, 0, 1), radius=16.0, lumen_mask=mask)
        if not s["sign_ok"]:
            fixed = -fixed
        err = np.linalg.norm((fixed - vt)[:, mask], axis=-1).mean()
        noise = np.linalg.norm((vm - vt)[:, mask], axis=-1).mean()
        ok = bool(abs(err - noise) < 1e-6 * max(noise, 1))
        rows.append(dict(seed=seed, corruption=name, detected=bool(not e["consistent"]) if name != "none" else bool(e["consistent"]),
                         ratio=float(e["ratio"]), recovered=ok))
summary = {}
for name in corruptions:
    r = [x for x in rows if x["corruption"] == name]
    summary[name] = dict(recovered=f"{sum(x['recovered'] for x in r)}/{len(r)}",
                         flagged_correctly=f"{sum(x['detected'] for x in r)}/{len(r)}",
                         ratio_mean=round(float(np.mean([x['ratio'] for x in r])), 2))
    print(f"{name:20s} recovered {summary[name]['recovered']}  flagged {summary[name]['flagged_correctly']}  ratio {summary[name]['ratio_mean']}")
json.dump(dict(summary=summary, rows=rows), open(OUT / "test1_encoding_sign.json", "w"), indent=1)
