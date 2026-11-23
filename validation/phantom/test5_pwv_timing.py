"""Test 5: pulse wave velocity from transit time.

A rigid-tube phantom has no wave travel, so the pulse delay is added by
hand. The reference aortic flow waveform is delayed at each of 20 planes by
distance / PWV, sampled at clinical frame rates with 3% noise, and the foot
of each curve is found by the intersecting-tangent method. PWV is the
inverse slope of arrival time against distance (all 20 planes), compared
with the two-plane estimate (first and last plane only).

Aortic length 250 mm (ascending to diaphragm), RR 800 ms. True PWV 4-12 m/s
covers young healthy aortas to stiff aortas. 200 noise repeats.
"""
import json, sys
from pathlib import Path
import numpy as np
from slicerheart_4dflow import phantom as ph
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.timing import arrival_time, pwv_regression
OUT = Path(__file__).resolve().parents[1] / "results"

RR, L, NPL, NOISE, REP = 800.0, 250.0, 20, 0.03, 200
NF = 8000
wave = ph.aortic_waveform(NF)
W = np.fft.rfft(wave)
kk = np.arange(len(W))
def delayed(delay_ms, t_ms):
    """Waveform delayed by delay_ms, evaluated at times t_ms (exact Fourier shift)."""
    C = W / NF * np.exp(-2j * np.pi * kk * delay_ms / RR)
    w = np.full(len(kk), 2.0); w[0] = 1.0; w[-1] = 1.0
    return np.real(np.exp(2j * np.pi * np.outer(t_ms / RR, kk)) * w @ C)

d = np.linspace(0, L, NPL)
rng = np.random.default_rng(0)
rows = []
for nt in (20, 25, 30, 40):
    for c in (4.0, 6.0, 8.0, 12.0):                      # m/s = mm/ms
        reg, two = [], []
        for _ in range(REP):
            t = rng.uniform(0, RR / nt) + np.arange(nt) / nt * RR
            ta = []
            for di in d:
                q = delayed(di / c, t) + rng.normal(0, NOISE, nt)
                ta.append(arrival_time(q, RR) + t[0])     # samples start at t[0]
            ta = np.array(ta)
            reg.append(pwv_regression(ta, d))
            dt = ta[-1] - ta[0]
            two.append(L / dt if dt > 0 else np.nan)
        reg, two = np.array(reg), np.array(two)
        rows.append(dict(frames=nt, frame_ms=RR / nt, true_pwv=c,
                         regression_bias_pct=float(100 * (np.nanmedian(reg) - c) / c),
                         regression_iqr_pct=float(100 * np.subtract(*np.nanpercentile(reg, [75, 25])) / c),
                         regression_within_20pct=float(np.mean(np.abs(reg - c) / c < 0.2) * 100),
                         twoplane_bias_pct=float(100 * (np.nanmedian(two) - c) / c),
                         twoplane_iqr_pct=float(100 * np.subtract(*np.nanpercentile(two, [75, 25])) / c),
                         twoplane_within_20pct=float(np.mean(np.abs(two - c) / c < 0.2) * 100),
                         transit_ms=L / c))
print("frames frame(ms) PWV transit(ms) | 20-plane regression: bias% IQR% within20% | two-plane: bias% IQR% within20%")
for r in rows:
    print(f"{r['frames']:5d} {r['frame_ms']:8.1f} {r['true_pwv']:4.0f} {r['transit_ms']:8.1f} | "
          f"{r['regression_bias_pct']:+7.1f} {r['regression_iqr_pct']:6.1f} {r['regression_within_20pct']:6.0f} | "
          f"{r['twoplane_bias_pct']:+7.1f} {r['twoplane_iqr_pct']:6.1f} {r['twoplane_within_20pct']:6.0f}")
json.dump(dict(settings=dict(rr_ms=RR, length_mm=L, planes=NPL, noise_frac=NOISE, repeats=REP), rows=rows),
          open(OUT / "test5_pwv_timing.json", "w"), indent=1)
