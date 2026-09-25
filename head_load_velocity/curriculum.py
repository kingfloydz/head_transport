"""Per-environment payload mass ceiling and reset-time inertial randomization."""

from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumManager, CurriculumTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


class PayloadMassUpper:
  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self.upper = torch.full((env.num_envs,), 3.0, device=env.device)

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
  ) -> torch.Tensor:
    survived = (env.episode_length_buf[env_ids] >= env.max_episode_length) & (
      ~env.termination_manager.terminated[env_ids]
    )
    self.upper[env_ids] = (self.upper[env_ids] + 3.0 * survived).clamp(max=60.0)
    return self.upper.mean()


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def sample_payload_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
) -> None:
  manager = cast(CurriculumManager, env.curriculum_manager)
  curriculum = manager.get_term_cfg("payload_mass_upper").func
  upper = curriculum.upper[env_ids]
  mass = 1.0 + torch.rand_like(upper) * (upper - 1.0)
  body_id = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids][0]
  env.sim.model.body_mass[env_ids, body_id] = mass
  env.sim.model.body_inertia[env_ids, body_id] = 0.015 * mass[:, None]
