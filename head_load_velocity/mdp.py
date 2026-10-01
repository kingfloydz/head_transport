"""Payload privilege observations and platform-contact termination."""

import math
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp import rewards as velocity_rewards
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse


def moving_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(torch.Tensor, env.command_manager.get_command(command_name))
  return (command[:, :2].norm(dim=-1) >= 0.1) | (command[:, 2].abs() >= 0.05)


def moving_reward(env, reward_fn, command_name: str, **kwargs) -> torch.Tensor:
  return reward_fn(env, command_name=command_name, **kwargs) * moving_command(
    env, command_name
  )


class DeadzoneSwingHeight(velocity_rewards.feet_swing_height):
  def __call__(
    self,
    env,
    sensor_name,
    height_sensor_name,
    target_height,
    command_name,
    command_threshold,
  ) -> torch.Tensor:
    # Always update the official peak-height history, including while standing.
    return super().__call__(
      env,
      sensor_name,
      height_sensor_name,
      target_height,
      command_name,
      command_threshold,
    ) * moving_command(env, command_name)

  def reset(self, env_ids=None):
    self.peak_heights[env_ids] = 0


class DeadzonePosture(velocity_rewards.variable_posture):
  def __call__(
    self,
    env,
    std_standing,
    std_walking,
    std_running,
    asset_cfg,
    command_name,
    walking_threshold=0.5,
    running_threshold=1.5,
  ):
    command = cast(torch.Tensor, env.command_manager.get_command(command_name))
    running = command[:, :2].norm(dim=-1) + command[:, 2].abs() >= running_threshold
    std = torch.where(running[:, None], self.std_running, self.std_walking)
    std = torch.where(
      moving_command(env, command_name)[:, None], std, self.std_standing
    )
    error = (
      env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
      - self.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.exp(-(error / std).square().mean(-1))


def track_yaw_velocity(
  env: ManagerBasedRlEnv, std: float, command_name: str
) -> torch.Tensor:
  """Track body-frame yaw angular velocity without roll/pitch terms."""
  command = cast(torch.Tensor, env.command_manager.get_command(command_name))
  actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
  return torch.exp(-torch.square(command[:, 2] - actual) / std**2)


def foot_distance_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, minimum_distance: float = 0.1
) -> torch.Tensor:
  """Rickshaw's signed lateral foot separation penalty, including crossed feet."""
  robot = env.scene[asset_cfg.name]
  feet = robot.data.site_pos_w[:, asset_cfg.site_ids]
  delta = quat_apply_inverse(robot.data.root_link_quat_w, feet[:, 0] - feet[:, 1])
  return ((minimum_distance - delta[:, 1]) / minimum_distance).clamp_min(0).square()


def payload_lost_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = cast(ContactSensor, env.scene[sensor_name])
  air_time = cast(torch.Tensor, sensor.data.current_air_time).squeeze(-1)
  # Compare physics ticks so float32 accumulation cannot add an extra policy step.
  return torch.round(air_time / env.physics_dt) >= math.ceil(0.3 / env.physics_dt)


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
    payload.data.root_com_lin_vel_w
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
