"""Diamond-friction acceleration robustness for planar face contact.

Small contact-point hulls receive a fixed cost. This support-area criterion
is not a guarantee of planar face contact or of MuJoCo contact feasibility.
"""

from dataclasses import dataclass
from itertools import product
from typing import cast

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from mjlab.entity.data import compute_velocity_from_cvel
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import Sensor, SensorCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from .asset import PLATFORM_HALF_SIZE
from .contact_geometry import (
  contact_hull_area,
  ordered_intersection,
  polygon_constraints,
  rectangle_constraints,
)


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


def support_vertices(bottom: torch.Tensor, tray: torch.Tensor):
  """Batched convex quadrilateral intersection, including full-tray detection."""

  def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

  edges = bottom.roll(-1, 1) - bottom
  inside = cross(edges[:, :, None], tray[None, None] - bottom[:, :, None]) >= 0
  tray_inside = inside.all(1)
  half = tray.abs().amax(0)
  bottom_inside = (bottom.abs() <= half).all(-1)
  a = bottom[:, :, None]
  e = edges[:, :, None]
  b = tray[None, None]
  f = (tray.roll(-1, 0) - tray)[None, None]
  det = cross(e, f)
  nonparallel = det != 0
  divisor = torch.where(nonparallel, det, 1)
  t = cross(b - a, f) / divisor
  u = cross(b - a, e) / divisor
  valid = nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
  intersections = (a + t[..., None] * e).flatten(1, 2)
  vertices = torch.cat((tray.expand(len(bottom), -1, -1), bottom, intersections), 1)
  mask = torch.cat((tray_inside, bottom_inside, valid.flatten(1)), 1)
  return vertices, mask, tray_inside.all(-1)


def margin_denominator(H, c, J, a_scale, alpha_scale):
  hf, ht = H[..., :3], H[..., 3:]
  v = hf - torch.linalg.cross(c[:, None], ht)
  r = (J[:, None] @ ht[..., None]).squeeze(-1) + torch.linalg.cross(c[:, None], v)
  return torch.sqrt(
    a_scale**2 * v.square().sum(-1) + alpha_scale**2 * r.square().sum(-1)
  )


@dataclass
class StabilitySensorCfg(SensorCfg):
  acceleration_scale: float = 1.0
  angular_acceleration_scale: float = 10.0
  target_margin: float = 1.0
  temperature: float = 0.5
  geometry_tolerance: float = 0.0
  """Metres relative to the last rebuild; zero gives exact geometry reuse."""
  invalid_penalty: float = 10.0
  minimum_contact_area: float = 1e-6
  """Square metres; smaller contact hulls receive the fixed invalid penalty."""
  geometry_batch_size: int = 64

  def build(self):
    sensor = StabilitySensor()
    sensor.cfg = self
    return sensor


class StabilitySensor(Sensor[torch.Tensor]):
  def edit_spec(self, scene_spec, entities):
    self.robot, self.payload = entities["robot"], entities["payload"]
    scene_site = scene_spec.site("robot/head_platform")
    scene_site.parent.add_site(
      name="robot/stability_top",
      pos=scene_site.pos + [0, 0, PLATFORM_HALF_SIZE[2]],
      quat=scene_site.quat,
    )
    for name, kind in [
      ("stability_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
      ("stability_alpha", mujoco.mjtSensor.mjSENS_FRAMEANGACC),
    ]:
      scene_spec.add_sensor(
        name=name,
        type=kind,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="robot/stability_top",
      )

  def initialize(self, mj_model, model, data, device):
    self.site = mj_model.site("robot/stability_top").id
    self.body = int(mj_model.site_bodyid[self.site])
    self.root = int(mj_model.body_rootid[self.body])
    self.payload_body = int(self.payload.indexing.root_body_id)
    self.payload_geom = int(self.payload.indexing.geom_ids[0])
    self.platform_geom = mj_model.geom("robot/head_platform_collision").id
    self.priorities = (
      mj_model.geom_priority[self.platform_geom],
      mj_model.geom_priority[self.payload_geom],
    )
    self.acc_address = int(mj_model.sensor_adr[mj_model.sensor("stability_acc").id])
    self.alpha_address = int(mj_model.sensor_adr[mj_model.sensor("stability_alpha").id])
    n = self.robot.data.root_link_pos_w.shape[0]
    self.mu = torch.ones(n, device=device, dtype=torch.float64)
    self.corners = torch.tensor(
      [[-1, -1], [1, -1], [1, 1], [-1, 1]], device=device, dtype=torch.float64
    )
    self.tray = self.corners * PLATFORM_HALF_SIZE[0]
    self.square = torch.as_tensor(
      square_constraints(PLATFORM_HALF_SIZE[0]), device=device
    )
    self.square_H = self.square.expand(n, -1, -1).clone()
    self.H = torch.zeros(n, 32, 6, device=device, dtype=torch.float64)
    self.rows = torch.zeros(n, 32, device=device, dtype=torch.bool)
    self.kappa = torch.ones(n, 32, device=device, dtype=torch.float64)
    self.cached_c = torch.full((n, 3), float("inf"), device=device, dtype=torch.float64)
    self.cached_J = torch.full(
      (n, 3, 3), float("inf"), device=device, dtype=torch.float64
    )
    self.cached_bottom = torch.full(
      (n, 4, 2), float("inf"), device=device, dtype=torch.float64
    )
    self.full_tray = torch.zeros(n, device=device, dtype=torch.bool)
    self.body_J = torch.empty(n, 3, 3, device=device, dtype=torch.float64)
    self.face_contact = torch.zeros_like(self.full_tray)
    self.contact_area = torch.zeros(n, device=device, dtype=torch.float64)
    self.rectangle = torch.zeros_like(self.full_tray)
    self.total = torch.zeros(n, device=device)
    self.margin_sum = torch.zeros_like(self.total)
    self.mean_margin = torch.zeros_like(self.total)
    self.mean_penalty = torch.zeros_like(self.total)
    self.count = torch.zeros_like(self.total)

  def set_payload(self, env_ids):
    friction = self.robot.data.model.geom_friction
    platform = friction[env_ids, self.platform_geom, 0]
    payload = friction[env_ids, self.payload_geom, 0]
    p, q = self.priorities
    self.mu[env_ids] = (
      torch.maximum(platform, payload) if p == q else (platform if p > q else payload)
    ).double()
    self.square_H[env_ids] = self.square
    self.square_H[env_ids, :, 0] /= self.mu[env_ids, None]
    self.square_H[env_ids, :, 1] /= self.mu[env_ids, None]
    self.square_H[env_ids, :, 5] /= self.mu[env_ids, None]
    model = self.robot.data.model
    axes = matrix_from_quat(model.body_iquat[env_ids, self.payload_body]).double()
    self.body_J[env_ids] = (
      axes
      @ torch.diag_embed(
        (
          model.body_inertia[env_ids, self.payload_body]
          / model.body_mass[env_ids, self.payload_body, None]
        ).double()
      )
      @ axes.transpose(-1, -2)
    )
    self.cached_bottom[env_ids] = float("inf")
    self.full_tray[env_ids] = False
    self.cached_c[env_ids] = float("inf")

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.total[ids] = 0
    self.margin_sum[ids] = 0
    self.count[ids] = 0

  def _store_constraints(self, ids, H, valid):
    capacity = H.shape[1]
    if capacity > self.H.shape[1]:
      self.H = F.pad(self.H, (0, 0, 0, capacity - self.H.shape[1]))
      self.rows = F.pad(self.rows, (0, capacity - self.rows.shape[1]))
      self.kappa = F.pad(self.kappa, (0, capacity - self.kappa.shape[1]), value=1)
    self.H[ids] = 0
    self.rows[ids] = False
    self.H[ids, :capacity] = H
    self.rows[ids, :capacity] = valid

  def update(self, dt):
    super().update(dt)
    cfg = cast(StabilitySensorCfg, self.cfg)
    data, model = self.robot.data.data, self.robot.data.model
    R = data.site_xmat[:, self.site].double()
    Rt = R.transpose(-1, -2)
    top = data.site_xpos[:, self.site].double()
    body = self.payload_body
    relative = Rt @ data.xmat[:, body].double()
    position = (Rt @ (data.xpos[:, body].double() - top)[..., None]).squeeze(-1)
    size = model.geom_size[:, self.payload_geom].double()
    local = torch.cat(
      (self.corners[None] * size[:, None, :2], -size[:, None, 2:].expand(-1, 4, -1)), -1
    )
    bottom_3d = local @ relative.transpose(-1, -2) + position[:, None]
    bottom = bottom_3d[..., :2]
    contacts = self.contact_points._extract_sensor_data()
    positions = torch.einsum("nij,nkj->nki", Rt, contacts.pos.double() - top[:, None])[
      ..., :2
    ]
    self.contact_area.copy_(contact_hull_area(positions, contacts.found > 0))
    face_contact = self.contact_area >= cfg.minimum_contact_area
    edges = bottom.roll(-1, 1) - bottom
    delta = self.tray[None, None] - bottom[:, :, None]
    full = (
      edges[:, :, None, 0] * delta[..., 1] - edges[:, :, None, 1] * delta[..., 0] >= 0
    ).all((1, 2)) & face_contact
    changed = (bottom - self.cached_bottom).abs().amax((1, 2)) > cfg.geometry_tolerance
    changed = (
      (changed & ~full) | (full != self.full_tray) | (face_contact != self.face_contact)
    ) & face_contact
    full_changed = changed & full
    self.H[full_changed] = 0
    self.rows[full_changed] = False
    self.H[full_changed, : len(self.square)] = self.square_H[full_changed]
    self.rows[full_changed, : len(self.square)] = True
    ids = (changed & ~full).nonzero().flatten()
    for group in ids.split(cfg.geometry_batch_size):
      vertices, mask, _ = support_vertices(bottom[group], self.tray)
      p, valid = ordered_intersection(vertices, mask)
      low = p.masked_fill(~valid[..., None], float("inf")).amin(1)
      high = p.masked_fill(~valid[..., None], -float("inf")).amax(1)
      half = (high - low) / 2
      corners = ((p - low[:, None]).abs() <= 1e-12) | (
        (p - high[:, None]).abs() <= 1e-12
      )
      rect = (valid.sum(1) == 4) & (corners.all(-1) | ~valid).all(1)
      e0, e1 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 1]
      rotated_rect = (
        (valid.sum(1) == 4)
        & ((e0 * e1).sum(-1).abs() <= 1e-12)
        & ((p[:, 0] + p[:, 2] - p[:, 1] - p[:, 3]).abs().amax(-1) <= 1e-12)
        & ~rect
      )
      self.rectangle[group] = rect | rotated_rect
      rect_ids = group[rect]
      H = rectangle_constraints(
        half[rect], (high[rect] + low[rect]) / 2, self.mu[rect_ids]
      )
      self._store_constraints(
        rect_ids, H, torch.ones(H.shape[:2], device=R.device, dtype=torch.bool)
      )
      rotated_ids = group[rotated_rect]
      if len(rotated_ids):
        H, hv = polygon_constraints(
          p[rotated_rect], valid[rotated_rect], self.mu[rotated_ids], rectangle=True
        )
        self._store_constraints(rotated_ids, H, hv)
      nonrect = ~rect & ~rotated_rect & (valid.sum(1) >= 3) & (half.amin(1) > 1e-10)
      general_ids = group[nonrect]
      if len(general_ids):
        H, hv = polygon_constraints(p[nonrect], valid[nonrect], self.mu[general_ids])
        self._store_constraints(general_ids, H, hv)
      empty = group[~rect & ~rotated_rect & ~nonrect]
      self.rows[empty] = False
    self.cached_bottom[changed] = bottom[changed]
    self.full_tray.copy_(full)
    self.face_contact.copy_(face_contact)
    c = (Rt @ (data.xipos[:, body].double() - top)[..., None]).squeeze(-1)
    J = relative @ self.body_J @ relative.transpose(-1, -2)
    velocity = compute_velocity_from_cvel(
      data.site_xpos[:, self.site],
      data.subtree_com[:, self.root],
      data.cvel[:, self.body],
    )
    omega = (Rt @ velocity[:, 3:].double()[..., None]).squeeze(-1)
    alpha = (
      Rt
      @ data.sensordata[:, self.alpha_address : self.alpha_address + 3].double()[
        ..., None
      ]
    ).squeeze(-1)
    specific = data.sensordata[:, self.acc_address : self.acc_address + 3].double()
    force = (
      specific
      + torch.linalg.cross(alpha, c)
      + torch.linalg.cross(omega, torch.linalg.cross(omega, c))
    )
    torque = (
      (J @ alpha[..., None]).squeeze(-1)
      + torch.linalg.cross(omega, (J @ omega[..., None]).squeeze(-1))
      + torch.linalg.cross(c, force)
    )
    w = torch.cat((force, torque), -1)
    refresh = (
      changed | (c != self.cached_c).any(-1) | (J != self.cached_J).any((1, 2))
    ) & face_contact
    margin = torch.zeros_like(self.mu)
    # Full-tray evaluation stays at 26 rows regardless of general-polygon capacity.
    for active, width in [
      (full, len(self.square)),
      (face_contact & ~full, self.H.shape[1]),
    ]:
      update = refresh & active
      kappa = margin_denominator(
        self.H[update, :width],
        c[update],
        J[update],
        cfg.acceleration_scale,
        cfg.angular_acceleration_scale,
      )
      self.kappa[update, :width] = torch.where(self.rows[update, :width], kappa, 1)
      scores = (
        -(self.H[active, :width] @ w[active, :, None]).squeeze(-1)
        / self.kappa[active, :width]
      )
      margin[active] = scores.masked_fill(
        ~self.rows[active, :width], float("inf")
      ).amin(-1)
    self.cached_c.copy_(c)
    self.cached_J.copy_(J)
    feasible_geometry = self.rows.any(-1) & face_contact
    penalty = torch.where(
      feasible_geometry,
      F.softplus((cfg.target_margin - margin) / cfg.temperature),
      cfg.invalid_penalty,
    )
    self.total += penalty.float()
    self.margin_sum += torch.where(feasible_geometry, margin, 0).float()
    self.count += 1

  def _compute_data(self):
    mean = self.total / self.count
    self.mean_penalty.copy_(mean)
    self.mean_margin.copy_(self.margin_sum / self.count)
    self.total.zero_()
    self.margin_sum.zero_()
    self.count.zero_()
    return mean


def stability_penalty(env: ManagerBasedRlEnv):
  sensor = cast(StabilitySensor, env.scene["payload_stability"])
  cost = sensor.data
  env.extras["log"]["Stability/margin"] = sensor.mean_margin.mean()
  env.extras["log"]["Stability/full_tray"] = sensor.full_tray.float().mean()
  env.extras["log"]["Stability/face_contact"] = sensor.face_contact.float().mean()
  env.extras["log"]["Stability/contact_area"] = sensor.contact_area.mean()
  return cost
