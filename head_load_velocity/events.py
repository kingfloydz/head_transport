"""Reset a randomized free payload on the robot's platform."""

from typing import cast

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from .asset import PLATFORM_HALF_SIZE


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def reset_payload_size(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> None:
  payload = env.scene[asset_cfg.name]
  geom_id = payload.indexing.geom_ids[asset_cfg.geom_ids][0]
  # MuJoCo box sizes are half-extents.
  xy = 0.025 + 0.225 * torch.rand((len(env_ids), 2), device=env.device)
  z = 0.005 + (xy.min(dim=-1, keepdim=True).values - 0.005) * torch.rand(
    (len(env_ids), 1), device=env.device
  )
  env.sim.model.geom_size[env_ids, geom_id] = torch.cat((xy, z), dim=-1)
  _recompute_geom_bounds(env, env_ids.to(dtype=torch.int), asset_cfg)


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def reset_payload(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  platform_cfg: SceneEntityCfg,
  mass_kg: float | None = None,
) -> None:
  if mass_kg is None:
    manager = cast(CurriculumManager, env.curriculum_manager)
    upper = manager.get_term_cfg("payload_mass_upper").func.upper
    mass = 1.0 + (upper - 1.0) * torch.rand(len(env_ids), device=env.device)
  else:
    mass = torch.full((len(env_ids),), mass_kg, device=env.device)
  payload = env.scene[asset_cfg.name]
  body_id = payload.indexing.root_body_id
  geom_id = payload.indexing.geom_ids[asset_cfg.geom_ids][0]
  half_size = env.sim.model.geom_size[env_ids, geom_id]
  env.sim.model.body_mass[env_ids, body_id] = mass
  squared = half_size.square()
  env.sim.model.body_inertia[env_ids, body_id] = (
    mass[:, None] * (squared.sum(dim=-1, keepdim=True) - squared) / 3.0
  )

  # Robot reset events have written qpos; refresh only forward kinematics.
  with wp.ScopedDevice(env.sim.wp_device):
    mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
  robot = env.scene[platform_cfg.name]
  position = robot.data.site_pos_w[env_ids][:, platform_cfg.site_ids].squeeze(1)
  rotation = robot.data.site_quat_w[env_ids][:, platform_cfg.site_ids].squeeze(1)
  offset = torch.zeros_like(position)
  offset[:, 2] = PLATFORM_HALF_SIZE[2] + half_size[:, 2] + 0.001
  payload.write_root_link_pose_to_sim(
    torch.cat((position + quat_apply(rotation, offset), rotation), dim=-1), env_ids
  )
  payload.write_root_link_velocity_to_sim(
    torch.zeros((len(env_ids), 6), device=env.device), env_ids
  )
