"""Cardiac-cycle normalisation and pulse-wave-velocity timing.

normalise_cycle  resample a gated series to n_out phases spanning one RR
                 interval, so subjects with different heart rates and frame
                 counts are comparable phase by phase.
arrival_time     foot of the systolic upstroke of a flow curve, by the
                 intersecting-tangent method on a periodic (FFT)
                 interpolation of the curve.
pwv_regression   PWV (m/s) as the inverse slope of arrival time against
                 centreline distance over all planes.
"""
import numpy as np
import slicerheart_4dflow as f


def normalise_cycle(series, times_ms, rr_ms, n_out=20, method="linear"):
    """series (nt, ...) at frame times times_ms -> (n_out, ...) at
    phases k / n_out of the RR interval (k = 0 .. n_out - 1).

    method "linear": periodic linear interpolation (resample_frames).
    method "fourier": band-limited periodic interpolation; needs frames
    uniformly spaced over the whole RR interval (retrospective gating).
    """
    dst = np.arange(n_out) / n_out * rr_ms
    series, times_ms = np.asarray(series, float), np.asarray(times_ms, float)
    if method == "linear":
        return f.resample_frames(series, times_ms, dst, periodic=True, period=rr_ms)
    nt = len(times_ms)
    k = np.arange(nt // 2 + 1)                               # harmonic numbers
    Y = np.fft.rfft(series.reshape(nt, -1), axis=0) / nt
    C = Y * np.exp(-2j * np.pi * k * times_ms[0] / rr_ms)[:, None]   # first frame may not be at t = 0
    w = np.full(len(k), 2.0)
    w[0] = 1.0
    if nt % 2 == 0:
        w[-1] = 1.0                                          # Nyquist term counted once
    basis = np.exp(2j * np.pi * np.outer(dst / rr_ms, k)) * w
    return np.real(basis @ C).reshape((n_out,) + series.shape[1:])


def fourier_upsample(y, n_fine):
    """Band-limited periodic interpolation of a uniformly sampled cycle."""
    y = np.asarray(y, float)
    Y = np.fft.rfft(y)
    return np.fft.irfft(Y, n_fine) * n_fine / len(y)


def arrival_time(flow, rr_ms, n_fine=2000):
    """Foot time (ms) of the systolic upstroke, intersecting-tangent method.

    The tangent at the point of maximum upslope before the systolic peak is
    intersected with the horizontal line through the preceding minimum.
    """
    q = fourier_upsample(flow, n_fine)
    dt = rr_ms / n_fine
    ipk = int(np.argmax(q))
    # search window: from 40% of the cycle before the peak up to the peak
    w = np.arange(ipk - int(0.4 * n_fine), ipk + 1)
    qw = q[w % n_fine]
    dq = np.gradient(qw, dt)
    imax = int(np.argmax(dq))
    qmin = qw[: imax + 1].min()
    t_tan = w[imax] * dt
    return t_tan - (qw[imax] - qmin) / dq[imax]


def pwv_regression(arrival_ms, distance_mm):
    """PWV in m/s (mm/ms) from a straight-line fit of arrival vs distance."""
    slope = np.polyfit(np.asarray(distance_mm, float), np.asarray(arrival_ms, float), 1)[0]
    return 1.0 / slope if slope > 0 else np.nan
