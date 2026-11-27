"""Synthetic 3D aortas with known geometry, haemodynamics and acquisition.

Each case is a full 4D Flow volume of an aorta-like vessel (ascending segment,
arch, descending segment) that goes through the SAME processing as study data:

  1. geometry     candy-cane centreline: ascending length, arch radius,
                  out-of-plane arch bulge; lumen radius with optional
                  ascending dilation (aneurysm-like bulge, as in LDS)
  2. 3D flow      analytic velocity in the local frame of the vessel.
                  Flow rate Q(t) is the same through every cross-section, so
                  mean velocity follows the local area (mass conservation),
                  and jet eccentricity, skew and reversal never change Q.
                  Haemodynamic factors: helicity, jet eccentricity and its
                  direction, flow reversal (amplitude, angular extent and
                  location, timing in the cycle), peak flow.
                  Curvature effects come from the 3D geometry itself: axial
                  skew towards the outer wall and a Dean pair of counter-
                  rotating vortices, scaled by sqrt(radius / arch radius),
                  developing through the arch and decaying downstream.
  3. acquisition  voxel size with partial volume (2x2x2 sub-sampling), number
                  of frames with a random trigger offset, VENC with phase
                  wrapping, velocity noise from VENC and SNR, a smooth
                  second-order background phase offset over static tissue,
                  and a magnitude image
  4. processing   slicerheart_4dflow.preprocess with the locked settings
                  (WAR+MSAC background correction, Laplacian-then-temporal
                  unwrapping), then 20 planes along the centreline with
                  parallel-transport frames, equal-area radius from
                  plane_flow, 16 x 32 polar sampling and Fourier
                  normalisation to 20 phases -- the study's own code.

Every input factor and the local geometry at every plane are recorded, so
the autoencoder's features can be checked against what was put in.
Simplified analytic flow with LDS-informed ranges, not CFD, not real data.
"""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import slicerheart_4dflow as f
from slicerheart_4dflow import phantom as ph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.planes import parallel_transport_frames, select_planes, sample_plane
from pipeline.timing import normalise_cycle
warnings.filterwarnings("ignore")

RANGES = {
    # geometry
    "diameter_mm": (22.0, 34.0), "dilation": (0.0, 0.6), "arch_radius_mm": (32.0, 48.0), "arch_bulge_mm": (0.0, 12.0),
    # haemodynamics
    "peak_flow_ml_s": (250.0, 550.0), "helicity": (0.0, 0.3), "eccentricity": (0.0, 0.45),
    "reversal": (0.0, 0.5), "reversal_width_deg": (90.0, 360.0), "reversal_phase": (0.30, 0.50),
    # acquisition
    "voxel_mm": (1.8, 3.0), "frames": (15, 40), "venc_m_s": (1.0, 2.0), "snr": (10.0, 30.0), "background_m_s": (0.0, 0.05),
}
GEOMETRY = ["diameter_mm", "dilation", "arch_radius_mm", "arch_bulge_mm"]
HAEMODYNAMICS = ["peak_flow_ml_s", "helicity", "eccentricity", "reversal", "reversal_width_deg", "reversal_phase"]
ACQUISITION = ["voxel_mm", "frames", "venc_m_s", "snr", "background_m_s"]
NUISANCE = ["jet_angle_deg", "reversal_angle_deg"]
BASELINE = {"diameter_mm": 28.0, "dilation": 0.2, "arch_radius_mm": 40.0, "arch_bulge_mm": 5.0, "peak_flow_ml_s": 400.0,
            "helicity": 0.1, "eccentricity": 0.1, "reversal": 0.15, "reversal_width_deg": 360.0, "reversal_phase": 0.38,
            "voxel_mm": 2.2, "frames": 25, "venc_m_s": 1.5, "snr": 20.0, "background_m_s": 0.02,
            "jet_angle_deg": 0.0, "reversal_angle_deg": 0.0}
RR_MS, ASC_MM, DESC_MM, DEAN_DECAY_MM = 800.0, 55.0, 90.0, 40.0
_NF = 2000
_W = ph.aortic_waveform(_NF)
_W = _W / _W.max()


def wave(phase):
    return np.interp((np.asarray(phase) % 1.0) * _NF, np.arange(_NF + 1), np.r_[_W, _W[0]])


# ---------------------------------------------------------------- geometry
def centreline(fct, step=0.5):
    """Ordered centreline (n, 3) mm, arc length s, and region label per point."""
    Rc, b = fct["arch_radius_mm"], fct["arch_bulge_mm"]
    asc = np.c_[np.zeros(int(ASC_MM / step)), np.zeros(int(ASC_MM / step)), np.arange(int(ASC_MM / step)) * step]
    th = np.arange(0, np.pi, step / Rc)[1:]
    arch = np.c_[Rc - Rc * np.cos(th), b * np.sin(th), ASC_MM + Rc * np.sin(th)]
    desc = np.c_[np.full(int(DESC_MM / step), 2 * Rc), np.zeros(int(DESC_MM / step)),
                 ASC_MM - np.arange(int(DESC_MM / step)) * step]
    p = np.vstack([asc, arch, desc])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    region = np.r_[np.zeros(len(asc)), np.ones(len(arch)), np.full(len(desc), 2)].astype(int)
    return p, s, region


def radius_along(fct, s):
    """Lumen radius (mm): ascending dilation as a smooth bulge centred 30 mm above the root."""
    r0 = fct["diameter_mm"] / 2
    return r0 * (1 + fct["dilation"] * np.exp(-0.5 * ((s - 30.0) / 14.0) ** 2))


def local_geometry(fct):
    """Centreline, frames, radius, curvature ratio and Dean amplitude along s."""
    p, s, region = centreline(fct)
    _, T, N, B = parallel_transport_frames(p, reference=(0, 1, 0), smooth=0)
    r = radius_along(fct, s)
    dT = np.gradient(T, s, axis=0)
    kappa = np.linalg.norm(dT, axis=1)
    outer = -dT / np.where(kappa[:, None] > 1e-6, kappa[:, None], 1)         # away from the centre of curvature
    last = None
    for i in range(len(p)):                                                 # carry the last outer direction through straight parts
        if kappa[i] > 1e-4:
            last = outer[i]
        elif last is not None:
            outer[i] = last - last.dot(T[i]) * T[i]
            outer[i] /= np.linalg.norm(outer[i])
    first = np.argmax(kappa > 1e-4)
    outer[:first] = outer[first]
    delta = np.sqrt(np.clip(r * kappa, 0, 1))                               # sqrt(radius / radius of curvature)
    dean = np.zeros_like(s)                                                 # develops in the bend, decays after it
    for i in range(1, len(s)):
        ds = s[i] - s[i - 1]
        dean[i] = dean[i - 1] + (delta[i] - dean[i - 1]) * min(1.0, ds / (DEAN_DECAY_MM if delta[i] < dean[i - 1] else 15.0))
    return dict(p=p, s=s, region=region, T=T, N=N, B=B, r=r, kappa=kappa, outer=outer, dean=dean)


# ---------------------------------------------------------------- 3D flow (separable in time)
def spatial_fields(pts, G, fct):
    """Velocity patterns at points (m, 3): each multiplied by a time course later.

    Returns inside (m,), and vector fields (m, 3) in m/s for unit time course:
      fwd  axial flow at peak, with eccentricity and curvature skew, scaled so Q is the same at every s
      rev  reversal pattern (negative axial, near the wall, over an angular sector)
      sec  in-plane flow at peak: helicity swirl + Dean vortex pair
    """
    tree = cKDTree(G["p"])
    dist, idx = tree.query(pts)
    rel = pts - G["p"][idx]
    T, N, B, O = G["T"][idx], G["N"][idx], G["B"][idx], G["outer"][idx]
    a_ax = np.einsum("ij,ij->i", rel, T)
    rel = rel - a_ax[:, None] * T                                          # in-plane offset
    r = G["r"][idx]
    rho = np.linalg.norm(rel, axis=1) / r
    inside = rho < 1.0
    th = np.arctan2(np.einsum("ij,ij->i", rel, B), np.einsum("ij,ij->i", rel, N))   # angle from N towards B
    th_o = np.arctan2(np.einsum("ij,ij->i", O, B), np.einsum("ij,ij->i", O, N))     # outer-wall angle
    dean = G["dean"][idx]
    # axial profile: blunt, eccentric jet, skewed to the outer wall, normalised to unit mean over the disc
    a = np.deg2rad(fct["jet_angle_deg"]); e = fct["eccentricity"]
    x, y = rho * np.cos(th), rho * np.sin(th)
    d = np.hypot(x - e * np.cos(a), y - e * np.sin(a)) / (1 + e)
    prof = np.clip(1 - d ** 2.5, 0, None) * np.clip(1 - rho ** 8, 0, None) * (1 + 0.9 * dean * rho * np.cos(th - th_o))
    norm = _profile_mean(e, a, dean, th_o)                                 # mean of prof over the disc, per point
    vmean = fct["peak_flow_ml_s"] * 1e3 / (np.pi * r ** 2) / 1000.0        # m/s at peak flow
    fwd_ax = vmean * prof / norm
    # reversal: near-wall band over an angular sector centred on reversal_angle
    w = np.deg2rad(fct["reversal_width_deg"]); c0 = np.deg2rad(fct["reversal_angle_deg"])
    dth = np.angle(np.exp(1j * (th - c0)))
    sector = np.ones_like(th) if w >= 2 * np.pi - 1e-6 else np.clip(1 - (np.abs(dth) / (w / 2)) ** 4, 0, 1)
    wall = np.clip((rho - 0.4) / 0.6, 0, 1) ** 1.5 * np.clip(1 - rho ** 8, 0, None)
    rev_ax = -fct["reversal"] * vmean * 1.8 * wall * sector
    # in-plane: helicity swirl and a Dean pair (stream function f(rho) sin(theta - theta_outer))
    taper = np.clip(1 - rho, 0, None)
    u_th = vmean * (fct["helicity"] * 6 * rho * taper)
    tho = th - th_o
    ud = 0.35 * vmean * dean
    u_r = ud * np.cos(tho) * (1 - rho ** 2) ** 2
    u_th = u_th - ud * np.sin(tho) * (1 - rho ** 2) * (1 - 5 * rho ** 2)
    e_r = np.cos(th)[:, None] * N + np.sin(th)[:, None] * B
    e_t = -np.sin(th)[:, None] * N + np.cos(th)[:, None] * B
    m = inside[:, None]
    return inside, fwd_ax[:, None] * T * m, rev_ax[:, None] * T * m, (u_r[:, None] * e_r + u_th[:, None] * e_t) * m


_RG, _TG = np.meshgrid((np.arange(24) + 0.5) / 24, 2 * np.pi * np.arange(48) / 48, indexing="ij")
_DEAN_GRID = np.linspace(0, 1, 11)
_REL_GRID = np.linspace(-np.pi, np.pi, 25)


def _profile_mean(e, a, dean, th_o):
    """Disc-average of the axial profile, so that Q is unchanged by eccentricity and skew.

    The average depends on the Dean amplitude and on the angle between the
    jet and the outer wall, so it is tabulated on both and interpolated.
    """
    x, y = _RG * np.cos(_TG), _RG * np.sin(_TG)
    d = np.hypot(x - e * np.cos(a), y - e * np.sin(a)) / (1 + e)
    base = np.clip(1 - d ** 2.5, 0, None) * np.clip(1 - _RG ** 8, 0, None)
    w = _RG / _RG.sum()
    tab = np.array([[(base * (1 + 0.9 * g * _RG * np.cos(_TG - (a + rel))) * w).sum() for rel in _REL_GRID]
                    for g in _DEAN_GRID])
    rel_pts = np.angle(np.exp(1j * (th_o - a)))
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator((_DEAN_GRID, _REL_GRID), tab)(np.c_[np.clip(dean, 0, 1), rel_pts])


# ---------------------------------------------------------------- acquisition + processing
def simulate(fct, seed=0, n_planes=20, return_volume=False):
    """One case -> planes (n_planes, 20, 16, 32, 3) m/s, lumen fraction (n_planes, 16, 32), per-plane geometry."""
    rng = np.random.default_rng(seed)
    G = local_geometry(fct)
    vox = float(fct["voxel_mm"])
    rmax = G["r"].max()
    lo = G["p"].min(0) - rmax - 3 * vox
    hi = G["p"].max(0) + rmax + 3 * vox
    n = np.ceil((hi - lo) / vox).astype(int) + 1                            # (ni, nj, nk) along x, y, z
    ii, jj, kk = [np.arange(m) for m in n]
    centres = np.stack(np.meshgrid(ii, jj, kk, indexing="ij"), -1).reshape(-1, 3) * vox + lo
    dist, _ = cKDTree(G["p"]).query(centres)
    near = np.where(dist < (rmax + vox))[0]                                # only voxels that can touch the lumen
    sub = (np.stack(np.meshgrid(*[np.array([-0.25, 0.25])] * 3, indexing="ij"), -1).reshape(-1, 3)) * vox
    acc = {k: np.zeros((len(near), 3)) for k in ("fwd", "rev", "sec")}
    frac = np.zeros(len(near))
    for o in sub:                                                          # partial volume, 2x2x2
        ins, fw, rv, sc = spatial_fields(centres[near] + o, G, fct)
        acc["fwd"] += fw / 8; acc["rev"] += rv / 8; acc["sec"] += sc / 8; frac += ins / 8
    shape = (n[2], n[1], n[0])                                             # (nk, nj, ni)
    def to_grid(vals, dim=3):
        out = np.zeros((np.prod(n), dim) if dim > 1 else np.prod(n))
        out[near] = vals
        out = out.reshape(n[0], n[1], n[2], *(() if dim == 1 else (dim,)))
        return np.ascontiguousarray(np.moveaxis(out, (0, 1, 2), (2, 1, 0)))
    F, Rv, S = to_grid(acc["fwd"]), to_grid(acc["rev"]), to_grid(acc["sec"])
    lumen = to_grid(frac, 1) >= 0.5
    nt = int(fct["frames"])
    t_ms = rng.uniform(0, RR_MS / nt) + np.arange(nt) / nt * RR_MS
    ph_ = t_ms / RR_MS
    rev_t = np.exp(-0.5 * (((ph_ - fct["reversal_phase"] + 0.5) % 1 - 0.5) / 0.05) ** 2)
    # background: smooth 2nd-order polynomial per component, scaled to background_m_s at the FOV edge
    Z, Y, X = np.meshgrid(*[np.linspace(-1, 1, m) for m in shape], indexing="ij")
    terms = np.stack([X, Y, Z, X * Y, X * Z, Y * Z, X ** 2 - 1 / 3, Y ** 2 - 1 / 3, Z ** 2 - 1 / 3])
    coef = rng.normal(size=(3, len(terms)))
    bg = np.einsum("ct,tkji->kjic", coef, terms)
    bg *= fct["background_m_s"] / (np.abs(bg).max() + 1e-9)
    venc = float(fct["venc_m_s"])
    sigma = np.sqrt(2) * venc / (np.pi * fct["snr"])                      # PC-MRI velocity noise
    vel = np.empty((nt,) + shape + (3,), np.float32)
    for k in range(nt):
        v = wave(ph_[k]) * (F + S) + rev_t[k] * Rv + bg + rng.normal(0, sigma, shape + (3,))
        vel[k] = (v + venc) % (2 * venc) - venc                           # phase wrapping at the VENC
    mag = np.where(lumen, 1.6, 1.0)[None].repeat(nt, 0) + rng.normal(0, 0.05, (nt,) + shape)
    A = np.eye(4); A[:3, :3] *= vox; A[:3, 3] = lo
    P = f.preprocess(vel * 1000.0, mag.astype(np.float32), np.full(3, venc * 1000.0), A, lumen_mask=lumen,
                     background_method="war+msac", unwrap_method="laplacian_then_temporal",
                     encoding_to_world=np.eye(3), verbose=False)
    v = np.asarray(P["velocity"], np.float32) / 1000.0                     # m/s
    # planes from a VMTK-like centreline (true centreline + 0.3 mm jitter), study's own frame code
    cl = G["p"][::4] + rng.normal(0, 0.3, G["p"][::4].shape)
    pc, Tc, Nc, Bc = parallel_transport_frames(cl, reference=(0, 1, 0))
    idx, s_c = select_planes(pc, n_planes, start_mm=8.0, end_mm=None)
    idx = select_planes(pc, n_planes, start_mm=8.0, end_mm=s_c[-1] - 8.0)[0]
    planes, insides, meta = [], [], []
    tree = cKDTree(G["p"])
    for i in idx:
        pf = f.plane_flow(v, A, centre=tuple(pc[i]), normal=tuple(Tc[i]), radius=float(G["r"].max() * 1.5), lumen_mask=lumen)
        area = float(np.atleast_1d(pf["area"]).mean())
        Req = np.sqrt(area / np.pi)
        Sp = sample_plane(v, A, pc[i], Tc[i], Nc[i], Bc[i], Req, mask=lumen)
        planes.append(normalise_cycle(Sp["v_local"], t_ms, RR_MS, 20, method="fourier").astype(np.float32))
        insides.append(Sp["inside"].astype(np.float32))
        g = tree.query(pc[i])[1]
        q = np.asarray(pf["flow"])                                         # m/s x mm^2 = mL/s
        meta.append(dict(s_mm=float(G["s"][g]), region=["ascending", "arch", "descending"][G["region"][g]],
                         radius_mm=float(G["r"][g]), radius_eq_mm=Req, curvature_ratio=float(G["r"][g] * G["kappa"][g]),
                         dean=float(G["dean"][g]), net_flow_ml=float(q.sum() * RR_MS / nt / 1000.0)))
    out = (np.stack(planes), np.stack(insides), meta)
    if return_volume:
        out += (dict(velocity=v, lumen=lumen, voxel_to_world=A, t_ms=t_ms, geometry=G),)
    return out


def draw_factors(rng, fixed=None):
    fct = {k: (int(rng.integers(lo, hi + 1)) if k == "frames" else float(rng.uniform(lo, hi))) for k, (lo, hi) in RANGES.items()}
    fct["jet_angle_deg"] = float(rng.uniform(-180, 180))
    fct["reversal_angle_deg"] = float(rng.uniform(-180, 180))
    if rng.random() < 0.4:                                                # 40% uniform ring, 60% localised sector
        fct["reversal_width_deg"] = 360.0
    if fixed:
        fct.update(fixed)
    return fct
