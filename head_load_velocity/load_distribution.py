"""Single-box mass properties from 10 x 10 x 10 uniform-density blocks."""

from functools import lru_cache

import torch

STAGE_LIMITS = (
  (0.5, 0.5, 0.5, 0.5, 0.3),
  (1.0, 1.0, 1.0, 1.0, 0.6),
  (1.5, 1.5, 2.0, 2.0, 1.1),
  (2.5, 2.5, 5.0, 5.0, 1.7),
)


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


def difficulty(size, mass, com, inertia, robot_mass):
  support = torch.minimum(size[:, :2] / 2 - com[:, :2].abs(), size.new_tensor(0.10))
  height = size[:, 2] / 2 + com[:, 2]
  inertia_q = inertia / mass[:, None, None] + torch.diag_embed(
    height[:, None].square() * size.new_tensor([1, 1, 0])
  )
  kappa = height[:, None] / support
  chi = inertia_q[:, :, [1, 0]].norm(dim=1) / (0.20 * support)
  return torch.cat((kappa, chi, (mass / robot_mass)[:, None]), dim=-1)


def sample_payload(robot_mass: torch.Tensor, stage: int):
  """Joint rejection sampling; no clipping, retry cap, or fallback distribution."""
  n, device = len(robot_mass), robot_mass.device
  centers = block_centers(str(device))
  limits = robot_mass.new_tensor(STAGE_LIMITS[stage])
  size = torch.empty(n, 3, device=device)
  mass = torch.empty(n, device=device)
  com = torch.empty_like(size)
  inertia = torch.empty(n, 3, 3, device=device)
  metrics = torch.empty(n, 5, device=device)
  pending = torch.arange(n, device=device)
  while pending.numel():
    count = len(pending)
    xy = 0.05 + 0.45 * torch.rand(count, 2, device=device)
    height = 0.01 + (xy.min(-1, keepdim=True).values - 0.01) * torch.rand(
      count, 1, device=device
    )
    candidate_size = torch.cat((xy, height), -1)
    minimum_eta = 1 / robot_mass[pending]
    eta = minimum_eta + (limits[4] - minimum_eta) * torch.rand(count, device=device)
    candidate_mass = eta * robot_mass[pending]
    bias = -2 + 4 * torch.rand(count, 3, device=device)
    weights = torch.rand(count, 1000, device=device) * torch.exp(bias @ centers.T)
    weights /= weights.sum(-1, keepdim=True)
    candidate_com, candidate_inertia = mass_properties(
      candidate_size, candidate_mass, weights, centers
    )
    candidate_metrics = difficulty(
      candidate_size,
      candidate_mass,
      candidate_com,
      candidate_inertia,
      robot_mass[pending],
    )
    # Engineering prior on average density; individual blocks are unrestricted.
    density = candidate_mass / candidate_size.prod(-1)
    accepted = (
      (density >= 40) & (density <= 4000) & (candidate_metrics <= limits).all(-1)
    )
    ids = pending[accepted]
    size[ids], mass[ids] = candidate_size[accepted], candidate_mass[accepted]
    com[ids], inertia[ids] = candidate_com[accepted], candidate_inertia[accepted]
    metrics[ids] = candidate_metrics[accepted]
    pending = pending[~accepted]
  return size, mass, com, inertia, metrics
