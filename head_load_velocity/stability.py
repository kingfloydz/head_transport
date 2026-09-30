"""Nominal platform-only stability, sampled once per physics substep."""

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
  rotation, velocity_acc, omega, alpha, mass, height, inertia, edges, mu
):
  """Return friction, normal and (+x, -x, +y, -y) moment margins.

  Kinematics are world-frame; inertia and edges are reset-time platform-frame
  constants. The hypothetical COM stays at (0, 0, height) in the platform frame.
  """
  r = rotation[:, :, 2] * height[:, None]
  acceleration = (
    velocity_acc
    + torch.cross(alpha, r, dim=-1)
    + torch.cross(omega, torch.cross(omega, r, dim=-1), dim=-1)
  )
  gravity = acceleration.new_tensor([0.0, 0.0, -9.81])
  force_w = mass[:, None] * (acceleration - gravity)
  inertia_w = rotation @ inertia @ rotation.transpose(-1, -2)
  momentum = (inertia_w @ omega.unsqueeze(-1)).squeeze(-1)
  torque_w = (
    (inertia_w @ alpha.unsqueeze(-1)).squeeze(-1)
    + torch.cross(omega, momentum, dim=-1)
    + torch.cross(r, force_w, dim=-1)
  )
  inverse = rotation.transpose(-1, -2)
  force = (inverse @ force_w.unsqueeze(-1)).squeeze(-1)
  torque = (inverse @ torque_w.unsqueeze(-1)).squeeze(-1)
  normal, mg = force[:, 2], mass * 9.81
  friction = (mu * normal - force[:, :2].norm(dim=-1)) / mg
  moments = torch.stack(
    (torque[:, 1], -torque[:, 1], -torque[:, 0], torque[:, 0]), dim=-1
  )
  tipping = (edges * normal[:, None] + moments) / (mg[:, None] * 0.20)
  return torch.cat((friction[:, None], (normal / mg)[:, None], tipping), dim=-1)


def stability_barrier(margins):
  """Positive soft-barrier cost; the reward manager applies weight -0.2."""
  return (
    F.softplus((0.1 - margins[:, 0]) / 0.1)
    + F.softplus((0.2 - margins[:, 1]) / 0.1)
    + F.softplus((0.1 - margins[:, 2:]) / 0.1).mean(-1)
  )


@dataclass
class StabilitySensorCfg(SensorCfg):
  def build(self) -> StabilitySensor:
    return StabilitySensor()


class StabilitySensor(Sensor[torch.Tensor]):
  """Use Scene.update's native physics-substep hook, without changing mjlab."""

  def edit_spec(self, scene_spec, entities) -> None:
    self.robot = entities["robot"]

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
    self.mass = torch.ones(n, device=device)
    self.height = torch.zeros(n, device=device)
    self.inertia = torch.zeros(n, 3, 3, device=device)
    self.edges = torch.zeros(n, 4, device=device)
    self.mu = torch.zeros(n, device=device)
    self.previous_velocity = torch.zeros(n, 3, device=device)
    self.previous_omega = torch.zeros(n, 3, device=device)
    self.valid = torch.zeros(n, dtype=torch.bool, device=device)
    self.total = torch.zeros(n, device=device)
    self.substeps = 0

  def set_payload(self, env_ids, mass, size, com, inertia, platform_quat, box_quat):
    self.mass[env_ids] = mass
    self.height[env_ids] = size[:, 2] / 2 + com[:, 2]
    relative = matrix_from_quat(platform_quat).transpose(-1, -2) @ matrix_from_quat(
      box_quat
    )
    self.inertia[env_ids] = relative @ inertia @ relative.transpose(-1, -2)
    half = size[:, :2] / 2
    self.edges[env_ids] = torch.stack(
      (
        half[:, 0] - com[:, 0],
        half[:, 0] + com[:, 0],
        half[:, 1] - com[:, 1],
        half[:, 1] + com[:, 1],
      ),
      dim=-1,
    ).clamp_max(PLATFORM_HALF_SIZE[0])
    # These geoms use dynamic contacts (no explicit pair). Match MuJoCo's
    # priority selection, or elementwise maximum when priorities are equal.
    friction = self.robot.data.model.geom_friction
    platform = friction[:, self.platform_geom, 0].expand_as(self.mass)
    payload = friction[:, self.payload_geom, 0].expand_as(self.mass)
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
    margins = stability_margins(
      rotation,
      acceleration,
      omega,
      alpha,
      self.mass,
      self.height,
      self.inertia,
      self.edges,
      self.mu,
    )
    self.total += torch.where(self.valid, stability_barrier(margins), 0.0)
    self.substeps += 1
    self.previous_velocity.copy_(velocity)
    self.previous_omega.copy_(omega)
    self.valid.fill_(True)

  def _compute_data(self) -> torch.Tensor:
    # The reward is read once after the decimation loop. Cached reads do not
    # consume twice. The first post-reset sample contributes zero to the mean.
    mean = self.total / self.substeps
    self.total.zero_()
    self.substeps = 0
    return mean


def stability_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  return cast(StabilitySensor, env.scene["payload_stability"]).data
