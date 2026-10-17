"""Shared contact-wrench margins for curriculum sampling and dynamic rewards."""

from itertools import product

import numpy as np
import torch

from .contact_geometry import polygon_constraints


def square_constraints(half_size: float) -> np.ndarray:
  """The 26 facets for a centered square and unit diamond friction."""
  rows = []
  for s, t in product((-1, 1), repeat=2):
    rows.append([s, t, -1, 0, 0, 0])
    rows.append([s, 0, -1, t / 2, 0, s * t / 2])
    rows.append([0, s, -1, 0, t / 2, s * t / 2])
    for z in (-1, 1):
      rows.append([s / 2, t / 2, -1, s * z / 2, t * z / 2, z / 2])
  for axis in range(3):
    for sign in (-1, 1):
      row = [0, 0, -1, 0, 0, 0]
      row[3 + axis] = sign
      rows.append(row)
  H = np.array(rows, dtype=np.float64)
  H[:, 3:] /= half_size
  return H


def margin_denominator(H, c, J, a_scale, alpha_scale):
  hf, ht = H[..., :3], H[..., 3:]
  v = hf - torch.linalg.cross(c[:, None], ht)
  r = (J[:, None] @ ht[..., None]).squeeze(-1) + torch.linalg.cross(c[:, None], v)
  return torch.sqrt(
    a_scale**2 * v.square().sum(-1) + alpha_scale**2 * r.square().sum(-1)
  )


def evaluate_margin(H, valid, c, J, w, a_scale, alpha_scale):
  kappa = margin_denominator(H, c, J, a_scale, alpha_scale)
  scores = -(H * w[:, None]).sum(-1) / torch.where(valid, kappa, 1)
  return scores.masked_fill(~valid, float("inf")).amin(-1)


def static_margin_accepts(size, mass, com, inertia, offset, mu, square, half,
                          minimum, a_scale, alpha_scale):
  """Screen a batch in a horizontal, aligned reference pose at actual friction."""
  size, mass, com, inertia, offset, mu = (
    value.double() for value in (size, mass, com, inertia, offset, mu)
  )
  c = torch.cat((offset, size[:, 2:] / 2 + com[:, 2:]), -1)
  J = inertia / mass[:, None, None]
  force = torch.zeros_like(c)
  force[:, 2] = 9.81
  w = torch.cat((force, torch.linalg.cross(c, force)), -1)
  friction = size.new_tensor([
    [1, 1, -1, 0, 0, 0], [1, -1, -1, 0, 0, 0],
    [-1, 1, -1, 0, 0, 0], [-1, -1, -1, 0, 0, 0],
  ]).expand(len(size), -1, -1).clone()
  friction[:, :, :2] /= mu[:, None, None]
  rows = torch.ones(friction.shape[:2], device=size.device, dtype=torch.bool)
  upper = evaluate_margin(friction, rows, c, J, w, a_scale, alpha_scale)
  accepted = torch.zeros(len(size), device=size.device, dtype=torch.bool)
  ids = (upper >= minimum).nonzero().flatten()
  if not len(ids):
    return accepted

  center = offset[ids] - com[ids, :2]
  low = torch.maximum(center - size[ids, :2] / 2, -half)
  high = torch.minimum(center + size[ids, :2] / 2, half)
  full = ((low == -half) & (high == half)).all(-1)
  full_ids = ids[full]
  if len(full_ids):
    H = square.expand(len(full_ids), -1, -1).clone()
    H[:, :, [0, 1, 5]] /= mu[full_ids, None, None]
    rows = torch.ones(H.shape[:2], device=size.device, dtype=torch.bool)
    accepted[full_ids] = evaluate_margin(
      H, rows, c[full_ids], J[full_ids], w[full_ids], a_scale, alpha_scale
    ) >= minimum
  partial_ids = ids[~full]
  if len(partial_ids):
    lo, hi = low[~full], high[~full]
    points = torch.stack((lo, torch.stack((hi[:, 0], lo[:, 1]), -1), hi,
                          torch.stack((lo[:, 0], hi[:, 1]), -1)), 1)
    vertices = torch.ones(points.shape[:2], device=size.device, dtype=torch.bool)
    H, rows = polygon_constraints(
      points, vertices, mu[partial_ids], parallelogram=True
    )
    accepted[partial_ids] = evaluate_margin(
      H, rows, c[partial_ids], J[partial_ids], w[partial_ids], a_scale, alpha_scale
    ) >= minimum
  return accepted


