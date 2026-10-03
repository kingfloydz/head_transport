"""Fixed-rectangle square-friction reward approximation; physics unchanged."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from mjlab.entity.data import compute_velocity_from_cvel
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import Sensor, SensorCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from .asset import PLATFORM_HALF_SIZE


def stability_margins(
  rotation, acceleration, omega, alpha, mass, r, inertia_w, center, half_size, mu
):
  """Square-friction reward approximation, not the MuJoCo pyramidal cone."""
  acceleration = (
    acceleration
    + torch.cross(alpha, r, dim=-1)
    + torch.cross(omega, torch.cross(omega, r, dim=-1), dim=-1)
  )
  force_w = mass[:, None] * (acceleration - acceleration.new_tensor([0, 0, -9.81]))
  momentum = (inertia_w @ omega.unsqueeze(-1)).squeeze(-1)
  torque_w = (
    (inertia_w @ alpha.unsqueeze(-1)).squeeze(-1)
    + torch.cross(omega, momentum, dim=-1)
    + torch.cross(r, force_w, dim=-1)
  )
  inverse = rotation.transpose(-1, -2)
  force = (inverse @ force_w.unsqueeze(-1)).squeeze(-1)
  torque = (inverse @ torque_w.unsqueeze(-1)).squeeze(-1)
  q = torch.cat((center, torch.zeros_like(center[:, :1])), dim=-1)
  torque -= torch.cross(q, force, dim=-1)
  fx, fy, fn = force.unbind(-1)
  tx, ty, tz = torque.unbind(-1)
  x, y = half_size.unbind(-1)
  mg = mass * 9.81
  lower = -mu * (x + y) * fn + (y * fx - mu * tx).abs() + (x * fy - mu * ty).abs()
  upper = mu * (x + y) * fn - (y * fx + mu * tx).abs() - (x * fy + mu * ty).abs()
  return torch.stack(
    (
      fn / mg,
      (mu * fn - torch.maximum(fx.abs(), fy.abs())) / (mu * mg),
      torch.minimum((y * fn - tx.abs()) / (mg * y), (x * fn - ty.abs()) / (mg * x)),
      torch.minimum(tz - lower, upper - tz) / (mg * (2 * PLATFORM_HALF_SIZE[0])),
    ),
    dim=-1,
  )


@dataclass
class StabilitySensorCfg(SensorCfg):
  def build(self) -> StabilitySensor:
    return StabilitySensor()


class StabilitySensor(Sensor[torch.Tensor]):
  """Use Scene.update's native physics-substep hook, without changing mjlab."""

  def edit_spec(self, scene_spec, entities) -> None:
    self.robot = entities["robot"]
    self.payload = entities["payload"]

  def initialize(self, mj_model, model, data, device) -> None:
    site = self.robot.find_sites("head_platform")[0][0]
    self.site = int(self.robot.indexing.site_ids[site])
    self.body = int(mj_model.site_bodyid[self.site])
    self.root = int(mj_model.body_rootid[self.body])
    self.platform_geom = mj_model.geom("robot/head_platform_collision").id
    self.payload_geom = mj_model.geom("payload/payload_collision").id
    self.priorities = (
      mj_model.geom_priority[self.platform_geom],
      mj_model.geom_priority[self.payload_geom],
    )
    n = self.robot.data.root_link_pos_w.shape[0]
    self.center = torch.zeros(n, 2, device=device)
    self.half_size = torch.ones(n, 2, device=device)
    self.mu = torch.zeros(n, device=device)
    self.previous_velocity = torch.zeros(n, 3, device=device)
    self.previous_omega = torch.zeros(n, 3, device=device)
    self.valid = torch.zeros(n, dtype=torch.bool, device=device)
    self.total = torch.zeros(n, device=device)
    self.margin_sum = torch.zeros(n, 4, device=device)
    self.mean_margins = torch.zeros_like(self.margin_sum)
    self.substeps = 0

  def set_payload(self, env_ids, size, box_xy):
    # Include the randomized COM displacement in the reset box position.
    # Freeze this axis-aligned overlap even if the free box later slips/tilts.
    low = (box_xy - size[:, :2] / 2).clamp_min(-PLATFORM_HALF_SIZE[0])
    high = (box_xy + size[:, :2] / 2).clamp_max(PLATFORM_HALF_SIZE[0])
    self.center[env_ids] = (low + high) / 2
    self.half_size[env_ids] = (high - low) / 2
    # These geoms use dynamic contacts (no explicit pair). Match MuJoCo's
    # priority selection, or elementwise maximum when priorities are equal.
    friction = self.robot.data.model.geom_friction
    platform = friction[:, self.platform_geom, 0].expand_as(self.mu)
    payload = friction[:, self.payload_geom, 0].expand_as(self.mu)
    if self.priorities[0] == self.priorities[1]:
      mu = torch.maximum(platform, payload)
    elif self.priorities[0] > self.priorities[1]:
      mu = platform
    else:
      mu = payload
    self.mu[env_ids] = mu[env_ids]

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.valid[ids] = False
    self.previous_velocity[ids] = 0
    self.previous_omega[ids] = 0
    self.total[ids] = 0
    self.margin_sum[ids] = 0
    self.mean_margins[ids] = 0

  def update(self, dt: float) -> None:
    super().update(dt)
    # mj_step leaves these derived quantities at the pre-integration instant.
    # Read pose and cvel from that same instant on every substep (one-tick lag),
    # never mix them with already-integrated qvel or control-rate snapshots.
    data = self.robot.data.data
    rotation = data.site_xmat[:, self.site]
    site_velocity = compute_velocity_from_cvel(
      data.site_xpos[:, self.site],
      data.subtree_com[:, self.root],
      data.cvel[:, self.body],
    )
    omega = site_velocity[:, 3:]
    offset = rotation[:, :, 2] * PLATFORM_HALF_SIZE[2]
    velocity = site_velocity[:, :3] + torch.cross(omega, offset, dim=-1)
    acceleration = (velocity - self.previous_velocity) / dt
    alpha = (omega - self.previous_omega) / dt
    model = self.robot.data.model
    body = self.payload.indexing.root_body_id
    box_rotation = data.xmat[:, body]
    axes = box_rotation @ matrix_from_quat(model.body_iquat[:, body])
    inertia_w = (
      axes @ torch.diag_embed(model.body_inertia[:, body]) @ axes.transpose(-1, -2)
    )
    top = data.site_xpos[:, self.site] + offset
    r = data.xipos[:, body] - top
    margins = stability_margins(
      rotation,
      acceleration,
      omega,
      alpha,
      model.body_mass[:, body],
      r,
      inertia_w,
      self.center,
      self.half_size,
      self.mu,
    )
    # Penalize normal, friction and tipping; retain yaw only for diagnostics.
    penalty = F.softplus((0.1 - margins[:, :3]) / 0.1).sum(-1)
    self.total += torch.where(self.valid, penalty, 0.0)
    self.margin_sum += torch.where(self.valid[:, None], margins, 0.0)
    self.substeps += 1
    self.previous_velocity.copy_(velocity)
    self.previous_omega.copy_(omega)
    self.valid.fill_(True)

  def _compute_data(self) -> torch.Tensor:
    # The reward is read once after the decimation loop. Cached reads do not
    # consume twice. The first post-reset sample contributes zero to the mean.
    mean = self.total / self.substeps
    self.mean_margins.copy_(self.margin_sum / self.substeps)
    self.margin_sum.zero_()
    self.total.zero_()
    self.substeps = 0
    return mean


def stability_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor = cast(StabilitySensor, env.scene["payload_stability"])
  cost = sensor.data
  for i, name in enumerate(("normal", "friction", "tipping", "yaw")):
    env.extras["log"][f"Stability/{name}"] = sensor.mean_margins[:, i].mean()
  return cost
