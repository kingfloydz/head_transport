"""Joint payload reset after robot domain randomization."""

from typing import cast

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr.body import _eigh_3x3_jacobi
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_matrix, yaw_quat

from .asset import PLATFORM_HALF_SIZE
from .load_distribution import sample_payload
from .stability import StabilitySensor


@requires_model_fields(
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def reset_payload(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  platform_cfg: SceneEntityCfg,
) -> None:
  manager = cast(CurriculumManager, env.curriculum_manager)
  stage = manager.get_term_cfg("payload_curriculum").func.stage
  robot, payload = env.scene[platform_cfg.name], env.scene[asset_cfg.name]
  model = env.sim.model
  robot_mass = model.body_mass[env_ids][:, robot.indexing.body_ids].sum(-1)
  size, mass, com, inertia, _ = sample_payload(robot_mass, stage)
  body = payload.indexing.root_body_id
  geom = payload.indexing.geom_ids[asset_cfg.geom_ids][0]
  principal, axes = _eigh_3x3_jacobi(inertia)
  axes[:, :, 2] *= torch.linalg.det(axes).sign()[:, None]
  model.geom_size[env_ids, geom] = size / 2
  model.body_mass[env_ids, body] = mass
  model.body_ipos[env_ids, body] = com
  model.body_inertia[env_ids, body] = principal
  model.body_iquat[env_ids, body] = quat_from_matrix(axes)
  _recompute_geom_bounds(env, env_ids.to(dtype=torch.int), asset_cfg)

  with wp.ScopedDevice(env.sim.wp_device):
    mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
  position = robot.data.site_pos_w[env_ids][:, platform_cfg.site_ids].squeeze(1)
  rotation = robot.data.site_quat_w[env_ids][:, platform_cfg.site_ids].squeeze(1)
  top_offset = torch.zeros_like(position)
  top_offset[:, 2] = PLATFORM_HALF_SIZE[2]
  position = position + quat_apply(rotation, top_offset)
  platform_rotation = rotation
  rotation = yaw_quat(platform_rotation)
  cast(StabilitySensor, env.scene["payload_stability"]).set_payload(
    env_ids, mass, size, com, inertia, platform_rotation, rotation
  )
  offset = -com
  offset[:, 2] = size[:, 2] / 2
  position += quat_apply(rotation, offset)
  payload.write_root_link_pose_to_sim(torch.cat((position, rotation), -1), env_ids)
  payload.write_root_link_velocity_to_sim(
    torch.zeros_like(position).repeat(1, 2), env_ids
  )
