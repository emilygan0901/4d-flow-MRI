"""Synthetic aortic flow planes with known geometry, haemodynamics and acquisition.

Each case is one analysis plane through a (locally) straight or curved aorta.
Velocity is defined analytically in physical units, rendered onto an MRI voxel
grid with partial volume, corrupted by noise and background offset, sampled
at the scanner's native frames, and then passed through the SAME code the
study uses: pipeline.planes.sample_plane (16 x 32 polar grid in the local
axial / radial / circumferential frame, radius from the lumen area) and
pipeline.timing.normalise_cycle (20 phases of the RR interval).

Factors (all recorded per case):
  geometry       diameter_mm, curvature (0-1, Dean-type secondary flow and
                 skew to the outer wall), ellipticity (0-0.2, non-circular lumen)
  haemodynamics  helicity (circumferential velocity, fraction of peak),
                 eccentricity (jet offset, fraction of radius),
                 reversal (late-systolic near-wall backflow, fraction of peak),
                 peak_velocity_m_s
  acquisition    voxel_mm, noise_m_s, frames (native), background_m_s
  nuisance       jet_angle_deg (direction of the eccentric jet)

mode="independent": every factor drawn independently.
mode="coupled": physics links them as in real aortas -- velocity falls with
diameter for a fixed stroke volume, and curvature adds helicity and
eccentricity (Dean flow).

These are simplified analytic fields with LDS-informed parameter ranges, not
CFD and not real LDS data.
"""
import sys
from pathlib import Path
import numpy as np
from slicerheart_4dflow import phantom as ph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.planes import sample_plane
from pipeline.timing import normalise_cycle

RANGES = {
    "diameter_mm": (20.0, 45.0), "curvature": (0.0, 1.0), "ellipticity": (0.0, 0.2),
    "helicity": (0.0, 0.3), "eccentricity": (0.0, 0.45), "reversal": (0.0, 0.5), "peak_velocity_m_s": (0.7, 1.8),
    "voxel_mm": (1.5, 3.0), "noise_m_s": (0.02, 0.08), "frames": (15, 40), "background_m_s": (0.0, 0.02),
}
GEOMETRY = ["diameter_mm", "curvature", "ellipticity"]
HAEMODYNAMICS = ["helicity", "eccentricity", "reversal", "peak_velocity_m_s"]
ACQUISITION = ["voxel_mm", "noise_m_s", "frames", "background_m_s"]
BASELINE = {"diameter_mm": 30.0, "curvature": 0.3, "ellipticity": 0.05, "helicity": 0.1, "eccentricity": 0.1,
            "reversal": 0.1, "peak_velocity_m_s": 1.2, "voxel_mm": 2.0, "noise_m_s": 0.04, "frames": 25,
            "background_m_s": 0.01, "jet_angle_deg": 0.0}

_NF = 2000
_WAVE = ph.aortic_waveform(_NF)
_WAVE = _WAVE / _WAVE.max()


def _wave(phase):
    """Systolic flow waveform (peak 1) at cycle phase in [0, 1)."""
    return np.interp((phase % 1.0) * _NF, np.arange(_NF + 1), np.r_[_WAVE, _WAVE[0]])


def _late_systole(phase):
    """Timing of late-systolic flow reversal."""
    d = (phase % 1.0) - 0.36
    return np.exp(-0.5 * (d / 0.05) ** 2)


def velocity_field(x_mm, y_mm, phase, f):
    """Velocity (vx, vy, vz in m/s) and lumen indicator at in-plane points.

    x points to the outer wall of the curve (theta = 0, frame axis N), y = B,
    z = through-plane (axial).
    """
    R = f["diameter_mm"] / 2
    x, y = x_mm / R, y_mm / R
    rho = np.hypot(x, y); th = np.arctan2(y, x)
    rb = 1 + f["ellipticity"] * np.cos(2 * th)                   # lumen boundary (normalised)
    re = rho / rb                                                # 0 centre, 1 wall
    inside = re < 1.0
    a = np.deg2rad(f["jet_angle_deg"])
    cx, cy = f["eccentricity"] * np.cos(a), f["eccentricity"] * np.sin(a)
    d = np.hypot(x / rb - cx, y / rb - cy) / (1 + f["eccentricity"])
    prof = np.clip(1 - d ** 2.5, 0, None) * np.clip(1 - re ** 8, 0, None)        # blunt, no slip
    prof *= 1 + 0.6 * f["curvature"] * re * np.cos(th)                          # skew to outer wall
    s, vp = _wave(phase), f["peak_velocity_m_s"]
    wall = np.clip((re - 0.4) / 0.6, 0, 1) ** 1.5 * np.clip(1 - re ** 8, 0, None)
    vz = vp * (s * prof - f["reversal"] * _late_systole(phase) * wall)
    # in-plane: helicity (solid-body-like swirl) + Dean vortex pair from curvature
    taper = np.clip(1 - re, 0, None)
    u_th = vp * s * (f["helicity"] * 4 * re * taper
                     - 0.15 * f["curvature"] * np.sin(th) * (1 - re ** 2) * (1 - 5 * re ** 2))
    u_r = vp * s * 0.15 * f["curvature"] * np.cos(th) * (1 - re ** 2) ** 2
    vx = u_r * np.cos(th) - u_th * np.sin(th)
    vy = u_r * np.sin(th) + u_th * np.cos(th)
    return np.stack([vx, vy, vz], -1) * inside[..., None], inside


def render_case(f, rng, rr_ms=800.0, supersample=4):
    """One synthetic plane -> (20, 16, 32, 3) axial/radial/circumferential m/s, lumen fraction (16, 32)."""
    vox, R = float(f["voxel_mm"]), f["diameter_mm"] / 2
    half = int(np.ceil((R * 1.35) / vox)) + 2
    n = 2 * half + 1
    c = (np.arange(n) - half) * vox                              # voxel centres, mm
    off = ((np.arange(supersample) + 0.5) / supersample - 0.5) * vox
    xs = (c[:, None] + off[None, :]).reshape(-1)                 # n*s sub-samples per axis
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    nt = int(f["frames"])
    t0 = rng.uniform(0, rr_ms / nt)
    t_ms = t0 + np.arange(nt) / nt * rr_ms
    vol = np.zeros((nt, 3, n, n, 3), np.float32)                 # (t, k, j, i, xyz), 3 identical slices
    _, ins = velocity_field(XX, YY, 0.0, f)
    frac = ins.reshape(n, supersample, n, supersample).mean(axis=(1, 3))
    mask2d = frac >= 0.5
    # background: smooth linear offset across the field of view (residual after correction)
    g = rng.normal(size=(3, 2)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    Xc, Yc = np.meshgrid(c, c, indexing="xy")
    bg = f["background_m_s"] * (g[:, 0, None, None] * Xc / (R * 1.35) + g[:, 1, None, None] * Yc / (R * 1.35))
    for k, t in enumerate(t_ms):
        v, _ = velocity_field(XX, YY, t / rr_ms, f)
        v = v.reshape(n, supersample, n, supersample, 3).mean(axis=(1, 3))        # partial volume
        v = v + rng.normal(0, f["noise_m_s"], v.shape) + np.moveaxis(bg, 0, -1)
        vol[k, :] = v[None]
    A = np.eye(4); A[0, 0] = A[1, 1] = A[2, 2] = vox
    A[:3, 3] = (-half * vox, -half * vox, -vox)                  # plane z = 0 is the middle slice
    area = mask2d.sum() * vox ** 2
    Req = np.sqrt(area / np.pi)
    mask3 = np.repeat(mask2d[None], 3, 0)
    S = sample_plane(vol, A, (0, 0, 0), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                     Req, mask=mask3)
    out = normalise_cycle(S["v_local"], t_ms, rr_ms, 20, method="fourier")
    return out.astype(np.float32), S["inside"].astype(np.float32), float(Req)


def draw_factors(rng, mode="independent", fixed=None):
    f = {k: (rng.integers(lo, hi + 1) if k == "frames" else rng.uniform(lo, hi)) for k, (lo, hi) in RANGES.items()}
    f["jet_angle_deg"] = rng.uniform(-45, 45)
    if mode == "coupled":
        f["peak_velocity_m_s"] = float(np.clip(f["peak_velocity_m_s"] * (30.0 / f["diameter_mm"]) ** 2, 0.4, 2.5))
        f["helicity"] += 0.2 * f["curvature"]
        f["eccentricity"] += 0.2 * f["curvature"]
    if fixed:
        f.update(fixed)
    return f


def make_dataset(n, seed=0, mode="independent", fixed=None):
    """n cases -> X (n, 20, 16, 32, 3), lumen fraction (n, 16, 32), factor table (list of dicts)."""
    rng = np.random.default_rng(seed)
    X, M, F = [], [], []
    for _ in range(n):
        f = draw_factors(rng, mode, fixed)
        x, m, req = render_case(f, rng)
        f["radius_eq_mm"] = req
        X.append(x); M.append(m); F.append(f)
    return np.stack(X), np.stack(M), F
