"""One-stage visual unloading: reset, observations, success, and rewards."""

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)

from .asset import (
  BOX_HALF,
  BOX_MASS,
  TABLE_DEPTH,
  TABLE_HEIGHT,
  TABLE_WIDTH,
  TARGET_INSET,
)
from .robot import PLATFORM_HALF_SIZE


class ResetScene:
  def __init__(self, cfg, env):
    env.unload = self
    robot = env.scene["robot"]
    self.platform = robot.find_sites("head_platform")[0][0]
    self.hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.previous_distance = torch.zeros(env.num_envs, device=env.device)
    self.effort_limits = torch.tensor(
      [a.effort_limit for a in robot.cfg.articulation.actuators], device=env.device,
    )

  def __call__(self, env, env_ids):
    robot, box, table = (env.scene[n] for n in ("robot", "payload", "table"))
    origins = env.scene.env_origins[env_ids]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += origins
    robot.write_root_state_to_sim(root, env_ids)
    robot.write_joint_state_to_sim(
      robot.data.default_joint_pos[env_ids], robot.data.default_joint_vel[env_ids], env_ids=env_ids,
    )
    table_pose = table.data.default_root_state[env_ids, :7].clone()
    table_pose[:, :3] = origins
    table_pose[:, 0] += 0.3 + TABLE_DEPTH / 2
    table.write_mocap_pose_to_sim(table_pose, env_ids)
    with wp.ScopedDevice(env.sim.wp_device):
      mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
    platform_pos = robot.data.site_pos_w[env_ids, self.platform]
    platform_quat = robot.data.site_quat_w[env_ids, self.platform]
    offset = torch.zeros_like(platform_pos)
    offset[:, 2] = PLATFORM_HALF_SIZE[2] + BOX_HALF
    position = platform_pos + quat_apply(platform_quat, offset)
    box.write_root_link_pose_to_sim(torch.cat((position, platform_quat), -1), env_ids)
    box.write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids)
    goal = table_pose[:, :3].clone()
    goal[:, 0] += TARGET_INSET - TABLE_DEPTH / 2
    goal[:, 2] += TABLE_HEIGHT + BOX_HALF
    self.previous_distance[env_ids] = (position - goal).norm(dim=-1)
    self.hold_steps[env_ids] = 0


def proprioception(env):
  r = env.scene["robot"].data
  return torch.cat((
    r.root_link_ang_vel_b, r.projected_gravity_b,
    r.joint_pos - r.default_joint_pos, r.joint_vel - r.default_joint_vel,
    env.action_manager.action,
  ), -1)


def depth_image(env):
  depth = env.scene["depth"].data.depth.permute(0, 3, 1, 2)
  valid = (depth >= 0.1) & (depth <= 4.0)
  return torch.cat((torch.where(valid, depth / 4.0, 0.0), valid.float()), dim=1)


def goal_position(env):
  goal = env.scene["table"].data.root_link_pos_w.clone()
  goal[:, 0] += TARGET_INSET - TABLE_DEPTH / 2
  goal[:, 2] += TABLE_HEIGHT + BOX_HALF
  return goal


def contact(env, name):
  return env.scene[name].data.found > 0


def privileged_state(env):
  r, b, table = (env.scene[n].data for n in ("robot", "payload", "table"))
  platform = env.unload.platform
  pq = r.site_quat_w[:, platform]
  relative_rotation = quat_mul(quat_conjugate(pq), b.root_link_quat_w)
  edge = table.root_link_pos_w.clone()
  edge[:, 0] -= TABLE_DEPTH / 2
  edge[:, 2] += TABLE_HEIGHT
  table_direction = torch.zeros_like(edge)
  table_direction[:, 1] = 1
  contacts = torch.cat((*(contact(env, n).any(-1, keepdim=True).float() for n in (
    "box_platform", "box_table", "box_ground", "box_robot", "body_ground", "robot_table", "self_collision",
  )), contact(env, "feet_ground").float()), -1)
  supports = torch.cat(tuple(
    env.scene[n].data.force[:, :, 2].abs() / (BOX_MASS * 9.81)
    for n in ("box_platform", "box_table")
  ), -1)
  return torch.cat((
    proprioception(env), r.root_link_lin_vel_b,
    quat_apply_inverse(pq, b.root_link_pos_w - r.site_pos_w[:, platform]),
    matrix_from_quat(relative_rotation)[:, :, :2].flatten(1),
    quat_apply_inverse(r.root_link_quat_w, b.root_link_lin_vel_w),
    quat_apply_inverse(r.root_link_quat_w, b.root_link_ang_vel_w),
    quat_apply_inverse(r.root_link_quat_w, edge - r.root_link_pos_w),
    quat_apply_inverse(r.root_link_quat_w, table_direction),
    quat_apply_inverse(r.root_link_quat_w, goal_position(env) - b.root_link_pos_w),
    contacts, supports, (env.unload.hold_steps * env.step_dt)[:, None],
  ), -1)


def failure(env):
  return contact(env, "body_ground").any(-1) | contact(env, "box_ground").any(-1)


def success(env):
  b, table = (env.scene[n].data for n in ("payload", "table"))
  # Oriented cube's projected half-extents; accept any final face.
  half_extent = BOX_HALF * matrix_from_quat(b.root_link_quat_w).abs().sum(-1)
  offset = (b.root_link_pos_w - table.root_link_pos_w).abs()
  inside = ((offset[:, 0] + half_extent[:, 0] <= TABLE_DEPTH / 2)
            & (offset[:, 1] + half_extent[:, 1] <= TABLE_WIDTH / 2))
  stable = (
    inside & contact(env, "box_table").any(-1)
    & ~contact(env, "box_robot").any(-1)
    & (b.root_link_lin_vel_w.norm(dim=-1) < 0.03)
    & (b.root_link_ang_vel_w.norm(dim=-1) < 0.1)
    & ~env.termination_manager.get_term("failure")
  )
  env.unload.hold_steps[:] = torch.where(stable, env.unload.hold_steps + 1, 0)
  return env.unload.hold_steps > 5.0 / env.step_dt


def success_reward(env):
  return env.step_dt * (env.unload.hold_steps > 0).float()


def pose(env):
  r = env.scene["robot"].data
  error = (r.joint_pos - r.default_joint_pos) / 0.15
  return env.step_dt * torch.exp(-error.square().mean(-1))


def progress(env):
  distance = (env.scene["payload"].data.root_link_pos_w - goal_position(env)).norm(dim=-1)
  delta = env.unload.previous_distance - distance
  env.unload.previous_distance.copy_(distance)
  return delta


def terminal_reward(env, name):
  return env.termination_manager.get_term(name).float()


def control_cost(env):
  action_rate = (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)
  torque = (env.scene["robot"].data.actuator_force / env.unload.effort_limits).square().mean(-1)
  return env.step_dt * (0.1 * action_rate + 0.05 * torque)


def joint_limit_cost(env):
  r = env.scene["robot"].data
  limits = r.soft_joint_pos_limits
  violation = (limits[..., 0] - r.joint_pos).clamp(min=0)
  violation += (r.joint_pos - limits[..., 1]).clamp(min=0)
  return env.step_dt * violation.sum(-1)


def collision_cost(env, sensor_name):
  return env.step_dt * contact(env, sensor_name).any(-1).float()


def feet_ground_reward(env):
  return env.step_dt * contact(env, "feet_ground").all(-1).float()


def box_table_force_cost(env):
  force = env.scene["box_table"].data.force.sum(dim=1).norm(dim=-1)
  return env.step_dt * (force / (BOX_MASS * 9.81) - 1.0).square()
