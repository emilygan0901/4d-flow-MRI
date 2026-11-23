"""Plane placement and polar sampling of 4D Flow velocity.

Each analysis plane is centred on the vessel centreline and oriented by a
local frame (T, N, B):
  T  tangent to the centreline (through-plane, "axial")
  N, B  in-plane axes, carried along the centreline by parallel transport
        (rotation-minimising frame, double-reflection method of Wang et al.
        2008, ACM Trans Graph 27(1):2), so the in-plane axes do not flip or
        twist where the centreline is straight or changes curvature, which
        the Frenet frame does.

Velocity on each plane is sampled on a polar grid (n_r radii x n_theta
angles, radii normalised to the local lumen radius) and decomposed into
axial, radial and circumferential components.

Array conventions follow slicerheart_4dflow: velocity (nt, nk, nj, ni, 3)
with world components, voxel_to_world 4x4 mapping (i, j, k) to (x, y, z) mm.
"""
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d
import slicerheart_4dflow as f


def centreline_tangents(points, smooth=2.0):
    """Unit tangents of an ordered (n, 3) centreline, lightly smoothed."""
    p = np.asarray(points, float)
    if smooth and len(p) > 5:
        p = gaussian_filter1d(p, smooth, axis=0, mode="nearest")
    t = np.gradient(p, axis=0)
    return p, t / np.linalg.norm(t, axis=1, keepdims=True)


def parallel_transport_frames(points, reference=(0.0, 1.0, 0.0), smooth=2.0):
    """Rotation-minimising frames along an ordered centreline.

    reference : world vector used to fix the in-plane angle at the first
                point (projected onto the first plane). Use the same vector
                for every subject, e.g. anterior, so theta = 0 points the
                same anatomical way in everyone.
    Returns smoothed points (n, 3), T, N, B (each (n, 3)).
    """
    p, T = centreline_tangents(points, smooth)
    n = len(p)
    N = np.zeros_like(T)
    r = np.asarray(reference, float)
    r = r - r.dot(T[0]) * T[0]
    if np.linalg.norm(r) < 1e-6:                       # reference parallel to T
        r = np.cross(T[0], [1.0, 0.0, 0.0])
    N[0] = r / np.linalg.norm(r)
    for i in range(n - 1):                             # double reflection
        v1 = p[i + 1] - p[i]
        c1 = v1.dot(v1)
        rL = N[i] - (2 / c1) * v1.dot(N[i]) * v1
        tL = T[i] - (2 / c1) * v1.dot(T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2.dot(v2)
        N[i + 1] = rL - (2 / c2) * v2.dot(rL) * v2 if c2 > 1e-12 else rL
        N[i + 1] -= N[i + 1].dot(T[i + 1]) * T[i + 1]
        N[i + 1] /= np.linalg.norm(N[i + 1])
    B = np.cross(T, N)
    return p, T, N, B


def select_planes(points, n_planes, start_mm=0.0, end_mm=None):
    """Indices of n_planes points equally spaced by arc length."""
    p = np.asarray(points, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    end_mm = s[-1] if end_mm is None else end_mm
    targets = np.linspace(start_mm, end_mm, n_planes)
    return np.array([int(np.argmin(np.abs(s - t))) for t in targets]), s


def polar_grid(centre, N, B, radius_mm, n_r=16, n_theta=32):
    """World points of a polar grid on one plane.

    Radii at bin centres (m + 0.5) / n_r * radius, angles 2 pi n / n_theta
    measured from N towards B. Returns points (n_r, n_theta, 3), radii,
    angles, and the radial and circumferential unit vectors (n_r, n_theta, 3).
    """
    r = (np.arange(n_r) + 0.5) / n_r * radius_mm
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    e_r = np.cos(th)[:, None] * N + np.sin(th)[:, None] * B          # (n_theta, 3)
    e_t = -np.sin(th)[:, None] * N + np.cos(th)[:, None] * B
    pts = np.asarray(centre)[None, None, :] + r[:, None, None] * e_r[None]
    return pts, r, th, np.broadcast_to(e_r, (n_r, n_theta, 3)), np.broadcast_to(e_t, (n_r, n_theta, 3))


def sample_plane(velocity_t, voxel_to_world, centre, T, N, B, radius_mm,
                 n_r=16, n_theta=32, mask=None, order=1):
    """Sample velocity on a polar grid and project onto the local frame.

    Returns dict:
      v_local  (nt, n_r, n_theta, 3) axial, radial, circumferential (mm/s)
      inside   (n_r, n_theta) fraction of the lumen mask at each point (or None)
      r_mm, theta, outside_fov (fraction of grid points outside the image)
    """
    v = np.asarray(velocity_t)
    if v.ndim == 4:
        v = v[None]
    pts, r, th, e_r, e_t = polar_grid(centre, N, B, radius_mm, n_r, n_theta)
    ijk = f.world_to_voxel_points(voxel_to_world, pts.reshape(-1, 3))
    coords = ijk[:, ::-1].T                                  # (k, j, i) for map_coordinates
    shape = np.array(v.shape[1:4])
    outside = np.any((coords < -0.5) | (coords > shape[:, None] - 0.5), axis=0)
    nt = v.shape[0]
    vw = np.empty((nt, coords.shape[1], 3))
    for t in range(nt):
        for c in range(3):
            vw[t, :, c] = map_coordinates(v[t, ..., c], coords, order=order, mode="nearest")
    vw = vw.reshape(nt, n_r, n_theta, 3)
    out = np.stack([vw @ np.asarray(T), (vw * e_r).sum(-1), (vw * e_t).sum(-1)], axis=-1)
    inside = None
    if mask is not None:
        inside = map_coordinates(np.asarray(mask, float), coords, order=1, mode="constant").reshape(n_r, n_theta)
    return dict(v_local=out, inside=inside, r_mm=r, theta=th, outside_fov=float(outside.mean()))


def grid_flow(v_axial, r_mm, radius_mm):
    """Through-plane flow (mm^3/s) from axial velocity on the polar grid.

    v_axial (..., n_r, n_theta). Each cell has area r dr dtheta.
    """
    n_r, n_th = v_axial.shape[-2:]
    dr, dth = radius_mm / n_r, 2 * np.pi / n_th
    return (v_axial * r_mm[:, None]).sum(axis=(-2, -1)) * dr * dth
