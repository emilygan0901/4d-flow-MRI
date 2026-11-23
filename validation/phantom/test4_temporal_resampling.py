"""Test 4: resampling every subject to 20 phases of the RR interval.

A reference aortic flow waveform is sampled as a scanner would, with
different frame counts (15-40) and heart rates (RR 600-1000 ms), then
resampled to 20 phases with normalise_cycle. The result is compared with
the true waveform at the same 20 phases. Noise free, so the error is due to
resampling alone.
"""
import json, sys
from pathlib import Path
import numpy as np
from slicerheart_4dflow import phantom as ph
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.timing import normalise_cycle
OUT = Path(__file__).resolve().parents[1] / "results"

NF = 4000
true = ph.aortic_waveform(NF)                     # one cycle, peak ~1
def truth_at(phase):                              # phase in [0, 1)
    return np.interp(phase * NF, np.arange(NF + 1), np.r_[true, true[0]])

ref20 = truth_at(np.arange(20) / 20)
sv_true = true.mean()                             # stroke volume per unit RR (mean flow)
tpk_true = np.argmax(true) / NF
rows = []
rng = np.random.default_rng(0)
for method in ("linear", "fourier"):
    for rr in (600.0, 800.0, 1000.0):
        for nt in (15, 20, 25, 30, 40):
            m = []
            for _ in range(50):                   # random trigger offset: frames do not line up with the 20 phases
                t = rng.uniform(0, rr / nt) + np.arange(nt) / nt * rr
                native = truth_at((t % rr) / rr)
                out = normalise_cycle(native, t, rr, 20, method)
                m.append((100 * np.sqrt(np.mean((out - ref20) ** 2)) / true.max(),
                          100 * (out.mean() - sv_true) / sv_true,
                          100 * (out.max() - ref20.max()) / ref20.max(),
                          (np.argmax(out) - np.argmax(ref20)) / 20 * rr))
            m = np.array(m)
            rows.append(dict(method=method, rr_ms=rr, native_frames=nt, frame_ms=rr / nt,
                             nrmse_pct=float(m[:, 0].mean()), mean_flow_err_pct=float(m[:, 1].mean()),
                             peak_err_pct=float(m[:, 2].mean()), peak_err_pct_worst=float(m[:, 2].min()),
                             peak_phase_err_ms_max=float(np.abs(m[:, 3]).max())))
print("method   RR  frames frame(ms) NRMSE%  meanflow%  peak%(mean/worst)  peak-time max(ms)")
for r in rows:
    print(f"{r['method']:7s} {r['rr_ms']:5.0f} {r['native_frames']:5d} {r['frame_ms']:8.1f} {r['nrmse_pct']:6.2f} {r['mean_flow_err_pct']:+9.2f} "
          f"{r['peak_err_pct']:+7.2f}/{r['peak_err_pct_worst']:+6.2f} {r['peak_phase_err_ms_max']:10.1f}")
# what the 20-phase grid itself costs: true waveform sampled at 20 phases vs full
res = dict(rows=rows, grid_limit=dict(
    mean_flow_err_pct_20_phases=float(100 * (ref20.mean() - sv_true) / sv_true),
    peak_err_pct_20_phases=float(100 * (ref20.max() - true.max()) / true.max())))
print("20-phase grid itself: mean flow %+.2f%%, peak %+.2f%% vs continuous waveform" %
      (res["grid_limit"]["mean_flow_err_pct_20_phases"], res["grid_limit"]["peak_err_pct_20_phases"]))
json.dump(res, open(OUT / "test4_temporal_resampling.json", "w"), indent=1)
