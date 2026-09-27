"""Payload privilege observations and platform-contact termination."""

import math
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse


def payload_lost_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = cast(ContactSensor, env.scene[sensor_name])
  air_time = cast(torch.Tensor, sensor.data.current_air_time).squeeze(-1)
  # Compare physics ticks so float32 accumulation cannot add an extra policy step.
  return torch.round(air_time / env.physics_dt) >= math.ceil(1.0 / env.physics_dt)


def payload_state(
  env: ManagerBasedRlEnv, platform_cfg: SceneEntityCfg, sensor_name: str
) -> torch.Tensor:
  robot = env.scene[platform_cfg.name]
  payload = env.scene["payload"]
  platform_pos = robot.data.site_pos_w[:, platform_cfg.site_ids].squeeze(1)
  platform_quat = robot.data.site_quat_w[:, platform_cfg.site_ids].squeeze(1)
  platform_vel = robot.data.site_lin_vel_w[:, platform_cfg.site_ids].squeeze(1)
  platform_omega = robot.data.site_ang_vel_w[:, platform_cfg.site_ids].squeeze(1)
  offset = payload.data.root_com_pos_w - platform_pos
  position = quat_apply_inverse(platform_quat, offset)
  rotation = torch.stack(euler_xyz_from_quat(payload.data.root_link_quat_w), dim=-1)
  velocity = quat_apply_inverse(
    platform_quat,
    payload.data.root_link_lin_vel_w
    - platform_vel
    - torch.cross(platform_omega, offset, dim=-1),
  )
  angular_velocity = quat_apply_inverse(
    platform_quat, payload.data.root_link_ang_vel_w - platform_omega
  )
  size = 2.0 * env.sim.model.geom_size[:, payload.indexing.geom_ids[0]]
  mass = env.sim.model.body_mass[:, payload.indexing.root_body_id, None]
  sensor = cast(ContactSensor, env.scene[sensor_name]).data
  contact = (cast(torch.Tensor, sensor.found) > 0).float()
  return torch.cat(
    (
      position,
      rotation,
      velocity,
      angular_velocity,
      size,
      mass,
      contact,
      cast(torch.Tensor, sensor.current_air_time),
    ),
    dim=-1,
  )
