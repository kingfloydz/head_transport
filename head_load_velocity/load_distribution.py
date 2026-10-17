"""Single-box mass properties from 10 x 10 x 10 uniform-density blocks."""

from functools import lru_cache

import torch

from .stability_math import square_constraints, static_margin_accepts

STATIC_MARGIN_MIN = (3.0, 2.5, 1.8, 1.0)
MASS_RATIO_MAX = (0.5, 1.0, 1.5, 2.0)


@lru_cache
def block_centers(device: str) -> torch.Tensor:
  axis = (torch.arange(10, device=device) + 0.5) / 5.0 - 1.0
  return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)


def mass_properties(size, mass, weights, centers):
  """Return body-frame COM and full COM inertia, including each block's inertia."""
  mean = weights @ centers
  second = weights @ (centers[:, :, None] * centers[:, None, :]).reshape(1000, 9)
  covariance = second.reshape(-1, 3, 3) - mean[:, :, None] * mean[:, None, :]
  half = size / 2
  covariance *= half[:, :, None] * half[:, None, :]
  covariance += torch.diag_embed((size / 10).square() / 12)
  eye = torch.eye(3, device=size.device)
  inertia = mass[:, None, None] * (
    covariance.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * eye - covariance
  )
  return mean * half, inertia


def sample_payload(robot_mass: torch.Tensor, stage: int, mu, tray_half,
                   a_scale, alpha_scale, batch_size):
  """Joint rejection sampling; no clipping, retry cap, or fallback distribution."""
  n, device = len(robot_mass), robot_mass.device
  centers = block_centers(str(device))
  minimum, eta_max = STATIC_MARGIN_MIN[stage], MASS_RATIO_MAX[stage]
  half = torch.tensor(tray_half, device=device, dtype=torch.float64)
  square = torch.as_tensor(square_constraints(tray_half[0]), device=device)
  size = torch.empty(n, 3, device=device)
  mass = torch.empty(n, device=device)
  com = torch.empty_like(size)
  inertia = torch.empty(n, 3, 3, device=device)
  offset = torch.empty(n, 2, device=device)
  pending = torch.arange(n, device=device)
  while pending.numel():
    count = len(pending)
    xy = 0.05 + 0.45 * torch.rand(count, 2, device=device)
    height = 0.01 + (xy.min(-1, keepdim=True).values - 0.01) * torch.rand(
      count, 1, device=device
    )
    candidate_size = torch.cat((xy, height), -1)
    minimum_eta = 1 / robot_mass[pending]
    eta = minimum_eta + (eta_max - minimum_eta) * torch.rand(count, device=device)
    candidate_mass = eta * robot_mass[pending]
    # Reject density before generating the 1000-block mass distribution.
    density = candidate_mass / candidate_size.prod(-1)
    eligible = ((density >= 40) & (density <= 4000)).nonzero().flatten()
    if not len(eligible):
      continue
    candidate_size, candidate_mass = candidate_size[eligible], candidate_mass[eligible]
    count = len(eligible)
    bias = -2 + 4 * torch.rand(count, 3, device=device)
    weights = torch.rand(count, 1000, device=device) * torch.exp(bias @ centers.T)
    weights /= weights.sum(-1, keepdim=True)
    candidate_com, candidate_inertia = mass_properties(
      candidate_size, candidate_mass, weights, centers
    )
    candidate_offset = torch.empty(count, 2, device=device).uniform_(-0.03, 0.03)
    accepted = torch.zeros(count, device=device, dtype=torch.bool)
    for group in torch.arange(count, device=device).split(batch_size):
      accepted[group] = static_margin_accepts(
        candidate_size[group], candidate_mass[group], candidate_com[group],
        candidate_inertia[group], candidate_offset[group], mu[pending[eligible[group]]],
        square, half, minimum, a_scale, alpha_scale,
      )
    ids = pending[eligible[accepted]]
    size[ids], mass[ids] = candidate_size[accepted], candidate_mass[accepted]
    com[ids], inertia[ids] = candidate_com[accepted], candidate_inertia[accepted]
    offset[ids] = candidate_offset[accepted]
    remaining = torch.ones(len(pending), device=device, dtype=torch.bool)
    remaining[eligible[accepted]] = False
    pending = pending[remaining]
  return size, mass, com, inertia, offset
