"""Control-rate raw features and frozen inference. No wrench or LP computation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from typing import cast

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

from mjlab.entity.data import compute_velocity_from_cvel
from mjlab.sensor import Sensor, SensorCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from .schema import metadata


def inspect_model(model, step_dt):
  """Reject incompatible physics instead of quietly relabeling it."""
  torso = model.body("robot/torso_link").id
  pg = model.geom("robot/head_platform_collision").id
  bg = model.geom("payload/payload_collision").id
  if model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
    raise ValueError("This labeler requires the actual pyramidal contact model")
  if tuple(model.geom_condim[[pg, bg]]) != (3, 3):
    raise ValueError("This labeler supports current condim=3 only")
  if np.any(
    ((model.pair_geom1 == pg) & (model.pair_geom2 == bg))
    | ((model.pair_geom1 == bg) & (model.pair_geom2 == pg))
  ):
    raise ValueError("Explicit material pair requires an updated feature contract")
  if model.geom_bodyid[pg] != torso or not np.allclose(
    model.geom_quat[pg], [1, 0, 0, 0]
  ):
    raise ValueError("Platform must be aligned with torso axes as in current model")
  if not np.allclose(model.opt.gravity, [0, 0, -9.81]):
    raise ValueError("Gravity differs from current feature contract")
  if not (
    np.allclose(model.geom_pos[bg], 0)
    and np.allclose(model.geom_quat[bg], [1, 0, 0, 0])
  ):
    raise ValueError("Box geom must be centered/aligned in its free body")
  size = model.geom_size[pg] * 2
  if not np.isclose(size[0], size[1]):
    raise ValueError("Current normalization uses a square platform")
  top = model.geom_pos[pg].copy()
  top[2] += size[2] / 2
  result = metadata(step_dt, size, top)
  result["versions"] = {
    name: version(name) for name in ("mujoco", "mujoco-warp", "mjlab")
  }
  result["verified_material"] = dict(
    priorities=model.geom_priority[[pg, bg]].tolist(),
    geom_friction=model.geom_friction[[pg, bg]].tolist(),
    cone=int(model.opt.cone),
    condim=model.geom_condim[[pg, bg]].tolist(),
    explicit_pair=False,
  )
  return result


@dataclass
class RawContactFeaturesCfg(SensorCfg):
  decimation: int = 4

  def build(self):
    return RawContactFeatures(self.decimation)


class RawContactFeatures(Sensor[torch.Tensor]):
  def __init__(self, decimation):
    super().__init__()
    self.decimation = decimation
    self.tick = 0

  def edit_spec(self, scene_spec, entities):
    self.robot = entities["robot"]
    self.payload = entities["payload"]

  def initialize(self, mj_model, model, data, device):
    self.meta = inspect_model(mj_model, mj_model.opt.timestep * self.decimation)
    self.torso = mj_model.body("robot/torso_link").id
    self.root = int(mj_model.body_rootid[self.torso])
    self.box = mj_model.body("payload/head_payload").id
    self.pg = mj_model.geom("robot/head_platform_collision").id
    self.bg = mj_model.geom("payload/payload_collision").id
    self.priority = mj_model.geom_priority[[self.pg, self.bg]]
    self.ell = torch.tensor(
      self.meta["top_position_T"], device=device, dtype=torch.float32
    )
    n = self.robot.data.root_link_pos_w.shape[0]
    self.x = torch.zeros(n, 37, device=device)
    self.domain = torch.ones(n, dtype=torch.int64, device=device)
    self.valid_history = torch.zeros(n, dtype=torch.bool, device=device)
    self.prev_v = torch.zeros(n, 3, device=device)
    self.prev_w = torch.zeros_like(self.prev_v)
    self.episode = torch.full((n,), -1, device=device, dtype=torch.int64)
    self.time_index = torch.zeros_like(self.episode)
    self.sample_episode = self.episode.clone()
    self.sample_time = self.time_index.clone()

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.valid_history[ids] = False
    self.episode[ids] += 1
    self.time_index[ids] = 0
    # Keep the last pre-reset sample intact for the collector after env.step().

  def update(self, dt):
    super().update(dt)
    self.tick += 1
    if self.tick % self.decimation:
      return
    model, data = self.robot.data.model, self.robot.data.data
    n = len(self.x)
    rt = data.xmat[:, self.torso]
    inverse = rt.transpose(-1, -2)
    rb = data.xmat[:, self.box]
    relative = inverse @ rb
    vel = compute_velocity_from_cvel(
      data.xpos[:, self.torso], data.subtree_com[:, self.root], data.cvel[:, self.torso]
    )
    v, omega = vel[:, :3], vel[:, 3:]
    acc = (v - self.prev_v) / self.meta["sample_dt"]
    alpha = (omega - self.prev_w) / self.meta["sample_dt"]
    self.prev_v.copy_(v)
    self.prev_w.copy_(omega)

    def rotate(z):
      return (inverse @ z.unsqueeze(-1)).squeeze(-1)

    com = model.body_ipos[:, self.box]
    half = model.geom_size[:, self.bg]
    faces = torch.stack((half + com, half - com), -1).flatten(1)
    axes = matrix_from_quat(model.body_iquat[:, self.box])
    inertia = (
      axes @ torch.diag_embed(model.body_inertia[:, self.box]) @ axes.transpose(-1, -2)
    )
    inertia6 = inertia[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    center = rotate(data.xipos[:, self.box] - data.xpos[:, self.torso]) - self.ell
    gravity = rotate(self.x.new_tensor([0, 0, -1]).expand(n, -1))
    friction = model.geom_friction
    if self.priority[0] == self.priority[1]:
      mu = torch.maximum(friction[:, self.pg, 0], friction[:, self.bg, 0])
    else:
      geom = self.pg if self.priority[0] > self.priority[1] else self.bg
      mu = friction[:, geom, 0]
    mu = mu.expand(n)
    domain = (~self.valid_history).long()
    self.valid_history.fill_(True)
    bottom = center - (
      relative
      @ (com + torch.stack((half[:, 0] * 0, half[:, 1] * 0, half[:, 2]), -1)).unsqueeze(
        -1
      )
    ).squeeze(-1)
    gap_extent = (relative[:, 2, :2].abs() * half[:, :2]).sum(-1)
    domain |= (
      (relative[:, 2, 2] < np.cos(self.meta["tilt_tolerance_rad"]))
      | (bottom[:, 2].abs() + gap_extent > self.meta["bottom_gap_tolerance_m"])
    ).long() * 2
    # Positive-area overlap via separating axes, no inscribed rectangle.
    corners = self.x.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    points = (
      relative[:, None, :2, :2] @ (corners[None] * half[:, None, :2]).unsqueeze(-1)
    ).squeeze(-1) + bottom[:, None, :2]
    sides = torch.roll(points, -1, 1) - points
    normals = F.normalize(torch.stack((sides[..., 1], -sides[..., 0]), -1), dim=-1)
    axes2 = torch.cat(
      (torch.eye(2, device=self.x.device)[None].expand(n, -1, -1), normals), 1
    )
    projections = torch.einsum("nki,nvi->nkv", axes2, points)
    extent = axes2.abs() @ self.x.new_tensor(self.meta["platform_size"][:2]) / 2
    overlap = (
      (projections.amin(-1) < extent - 1e-8) & (projections.amax(-1) > -extent + 1e-8)
    ).all(-1)
    domain |= (~overlap).long() * 32
    domain |= (mu <= 0).long() * 64
    # Read only active actual contacts; no added simulation sensors or forward.
    contact = data.contact
    slot = torch.arange(contact.geom.shape[0], device=self.x.device)
    active = (slot < data.nacon[0]) & (contact.dist[:] < contact.includemargin[:])
    geoms = contact.geom[:]
    cargo = (geoms == self.bg).any(-1)
    platform = (geoms == self.pg).any(-1)
    ids = contact.worldid[active & cargo & platform].long()
    selected = active & cargo & platform
    other = contact.worldid[active & cargo & ~platform].long()
    count = torch.bincount(ids, minlength=n)
    domain |= (count == 0).long() * 4
    domain |= (torch.bincount(other, minlength=n) > 0).long() * 8
    tangent = (inverse[ids] @ contact.frame[selected, 1, :].unsqueeze(-1)).squeeze(-1)
    normal = (inverse[ids] @ contact.frame[selected, 0, :].unsqueeze(-1)).squeeze(-1)
    first = torch.full((n,), len(slot), device=self.x.device, dtype=torch.int64)
    first.scatter_reduce_(0, ids, slot[selected], reduce="amin")
    # Empty worlds receive zero features and an explicit no-contact domain flag.
    basis = torch.zeros(n, 2, device=self.x.device)
    present = count > 0
    basis[present] = F.normalize(
      (inverse[present] @ contact.frame[first[present], 1, :].unsqueeze(-1)).squeeze(
        -1
      )[:, :2],
      dim=-1,
    )
    tangent_xy = F.normalize(tangent[:, :2], dim=-1)
    cosine = (tangent_xy * basis[ids]).sum(-1).abs()
    equivalent = (
      torch.minimum(cosine, (1 - cosine).abs()) < self.meta["basis_tolerance"]
    )
    cf = contact.friction[selected]
    bad = (
      ~equivalent
      | (normal[:, 2].abs() < 1 - self.meta["basis_tolerance"])
      | (tangent[:, 2].abs() > self.meta["basis_tolerance"])
      | (contact.dim[selected] != 3)
      | ((cf[:, 0] - cf[:, 1]).abs() > 1e-6)
      | ((cf[:, 0] - mu[ids]).abs() > 1e-6)
    )
    domain |= (torch.bincount(ids[bad], minlength=n) > 0).long() * 16
    self.x = torch.cat(
      (
        faces,
        model.body_mass[:, self.box, None],
        inertia6,
        center,
        relative[:, :, :2].transpose(1, 2).flatten(1),
        rotate(acc),
        rotate(omega),
        rotate(alpha),
        gravity,
        mu[:, None],
        basis,
      ),
      -1,
    )
    self.domain = domain
    self.sample_episode = self.episode.clone()
    self.sample_time = self.time_index.clone()
    self.time_index += 1

  def _compute_data(self):
    return self.x


class FrozenMarginReward:
  def __init__(self, cfg, env):
    self.sensor = cast(RawContactFeatures, env.scene["surrogate_features"])
    extra = {"metadata.json": b""}
    self.model = torch.jit.load(
      cfg.params["model_path"], map_location=env.device, _extra_files=extra
    ).eval()
    meta = json.loads(extra["metadata.json"])
    if meta["physics"] != self.sensor.meta:
      raise ValueError("Frozen model and environment feature/physics metadata differ")
    if not meta["release_ready"]:
      raise ValueError("Synthetic smoke-test models cannot be used as RL rewards")
    for parameter in self.model.parameters():
      parameter.requires_grad_(False)
    self.minimum = torch.tensor(meta["feature_min"], device=env.device)
    self.maximum = torch.tensor(meta["feature_max"], device=env.device)
    padding = 0.01 * (self.maximum - self.minimum) + 1e-6
    self.minimum -= padding
    self.maximum += padding

  @torch.inference_mode()
  def __call__(self, env, model_path, delta, temperature, outside_cost):
    in_range = ((self.sensor.x >= self.minimum) & (self.sensor.x <= self.maximum)).all(
      -1
    )
    self.last_domain = self.sensor.domain | (~in_range).long() * 128
    valid = (self.sensor.domain == 0) & in_range
    cost = torch.full_like(self.sensor.x[:, 0], outside_cost)
    prediction = self.model(self.sensor.x[valid])[:, 4]
    cost[valid] = F.softplus((delta - prediction) / temperature)
    # First differenced sample unavailable: no penalty, explicitly logged.
    warmup = (self.sensor.domain & 1) != 0
    cost[warmup] = 0
    env.extras["log"]["Surrogate/outside_fraction"] = (~valid & ~warmup).float().mean()
    env.extras["log"]["Surrogate/range_fraction"] = (~in_range & ~warmup).float().mean()
    env.extras["log"]["Surrogate/warmup_fraction"] = warmup.float().mean()
    return cost
