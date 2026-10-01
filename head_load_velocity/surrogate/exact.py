"""Float64 vertex-ray LP. No online imports of this module are required."""

import numpy as np
from scipy.optimize import linprog

from .schema import unpack


def polygon(x, meta):
  """Clip the full projected bottom face against the platform rectangle."""
  size, com, rotation, _ = unpack(x)
  bottom = x[13:16] - rotation @ (com + [0, 0, size[2] / 2])
  corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size[:2] / 2
  points = (corners @ rotation[:, :2].T + bottom)[:, :2]
  for n, b in zip(
    ([1, 0], [-1, 0], [0, 1], [0, -1]),
    np.repeat(np.array(meta["platform_size"][:2]) / 2, 2),
    strict=True,
  ):
    n = np.asarray(n)
    result = []
    for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
      ds, de = start @ n - b, end @ n - b
      if ds <= 0:
        result.append(start)
      if (ds <= 0) != (de <= 0):
        result.append(start + ds / (ds - de) * (end - start))
    points = np.array(result).reshape(-1, 2)
    if not len(points):
      return points
  # Remove duplicate vertices introduced by exact boundary coincidences.
  return points[np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1) > 1e-12]


def area_centroid(q):
  following = np.roll(q, -1, axis=0)
  cross = q[:, 0] * following[:, 1] - q[:, 1] * following[:, 0]
  area = cross.sum() / 2
  return area, ((q + following) * cross[:, None]).sum(0) / (6 * area)


def halfplanes(q):
  edge = np.roll(q, -1, axis=0) - q
  n = np.column_stack((edge[:, 1], -edge[:, 0]))
  n /= np.linalg.norm(n, axis=1)[:, None]
  return n, (n * q).sum(1)


def wrench(x, meta):
  """All vectors in torso axes: rotation covariance equals the world calculation."""
  _, _, rotation, inertia = unpack(x)
  r, a, omega, alpha = x[13:16], x[22:25], x[25:28], x[28:31]
  ell = np.asarray(meta["top_position_T"])
  ao = a + np.cross(alpha, ell) + np.cross(omega, np.cross(omega, ell))
  ac = ao + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
  force = x[6] * (ac - meta["gravity"] * x[31:34])
  it = rotation @ inertia @ rotation.T
  torque = it @ alpha + np.cross(omega, it @ omega) + np.cross(r, force)
  # Current model verified platform axes == torso axes. R_PT is metadata-bound.
  return np.r_[force, torque] / (
    x[6] * meta["gravity"] * np.r_[np.ones(3), np.full(3, meta["p"])]
  )


def generators(q, mu, tangent, p):
  t1 = np.r_[tangent, 0.0]
  t2 = np.cross([0.0, 0.0, 1.0], t1)
  rays = np.array([mu * t1, -mu * t1, mu * t2, -mu * t2]) + [0.0, 0.0, 1.0]
  positions = np.column_stack((q, np.zeros(len(q)))) / p
  force = np.tile(rays, (len(q), 1))
  torque = np.cross(np.repeat(positions, 4, axis=0), force)
  return np.column_stack((force, torque)).T


def solve(c, a, b, bounds):
  result = linprog(
    c,
    A_eq=a,
    b_eq=b,
    bounds=bounds,
    method="highs",
    options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9},
  )
  status = result.status
  if status == 0 and np.max(np.abs(a @ result.x - b)) > 1e-7:
    status = 4
  return result, status


def label(x, meta, domain_bits=0):
  """Return labels, yaw_valid, domain bits, numerical status, feasible, LP codes."""
  y = np.full(5, np.nan)
  codes = np.full(3, -1, dtype=np.int8)
  if domain_bits:
    return y, False, domain_bits, 1, False, codes
  x = np.asarray(x, dtype=np.float64)
  size, com, rotation, inertia = unpack(x)
  valid = (
    np.isfinite(x).all()
    and np.all(x[:6] > 0)
    and x[6] > 0
    and x[34] > 0
    and np.linalg.eigvalsh(inertia).min() > 0
    and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    and abs(np.linalg.norm(x[31:34]) - 1) < 1e-5
    and abs(np.linalg.norm(x[35:37]) - 1) < 1e-5
  )
  if not valid:
    return y, False, 32, 1, False, codes
  bottom = x[13:16] - rotation @ (com + [0, 0, size[2] / 2])
  gap_extent = np.abs(rotation[2, :2]) @ (size[:2] / 2)
  if (
    rotation[2, 2] < np.cos(meta["tilt_tolerance_rad"])
    or abs(bottom[2]) + gap_extent > meta["bottom_gap_tolerance_m"]
  ):
    return y, False, 2, 1, False, codes
  q = polygon(x, meta)
  if len(q) < 3:
    return y, False, 32, 1, False, codes
  area, q0 = area_centroid(q)
  if area <= meta["area_tolerance_m2"]:
    return y, False, 32, 1, False, codes
  w = wrench(x, meta)
  t1 = x[35:37]
  t2 = np.array([-t1[1], t1[0]])
  y[0] = w[2]
  y[1] = x[34] * w[2] - abs(t1 @ w[:2]) - abs(t2 @ w[:2])
  n, b = halfplanes(q)
  y[2] = np.min(b / meta["p"] * w[2] - n @ [-w[4], w[3]])
  g = generators(q, x[34], t1, meta["p"])
  low, codes[0] = solve(g[5], g[:5], w[:5], (0, None))
  high, codes[1] = solve(-g[5], g[:5], w[:5], (0, None))
  yaw_valid = codes[0] == 0 and codes[1] == 0
  if yaw_valid:
    y[3] = min(w[5] - low.fun, -high.fun - w[5])
  reference = np.array([0, 0, 1, q0[1] / meta["p"], -q0[0] / meta["p"], 0])
  a = np.column_stack((g, reference))
  objective = np.r_[np.zeros(g.shape[1]), -1.0]
  reserve, codes[2] = solve(objective, a, w, [(0, None)] * g.shape[1] + [(None, None)])
  if codes[2] == 0:
    y[4] = reserve.x[-1]
  numerical = codes[2] != 0 or not (yaw_valid or (codes[0] == codes[1] == 2))
  return y, bool(yaw_valid), 0, 2 if numerical else 0, bool(y[4] >= -1e-8), codes
