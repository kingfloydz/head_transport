"""Shared payload mass curriculum with fixed success-rate windows."""

from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumManager, CurriculumTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


class PayloadMassUpper:
  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self.upper = torch.tensor(3.0, device=env.device)
    self.episode_upper = torch.full((env.num_envs,), 3.0, device=env.device)
    self.counts = torch.zeros(2, dtype=torch.long, device=env.device)

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
  ) -> torch.Tensor:
    return self.upper.clone()


def update_payload_curriculum(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, window_steps: int
) -> None:
  manager = cast(CurriculumManager, env.curriculum_manager)
  curriculum = manager.get_term_cfg("payload_mass_upper").func
  finished = env.reset_buf & (curriculum.episode_upper == curriculum.upper)
  succeeded = (
    finished
    & (env.episode_length_buf >= env.max_episode_length)
    & (~env.termination_manager.terminated)
  )
  curriculum.counts += torch.stack((succeeded.sum(), finished.sum()))
  if env.common_step_counter % window_steps == 0:
    promote = (curriculum.counts[0] * 100 >= curriculum.counts[1] * 95) & (
      curriculum.counts[1] > 0
    )
    curriculum.upper.add_(3.0 * promote).clamp_(max=60.0)
    curriculum.counts.zero_()


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def sample_payload_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
) -> None:
  manager = cast(CurriculumManager, env.curriculum_manager)
  curriculum = manager.get_term_cfg("payload_mass_upper").func
  curriculum.episode_upper[env_ids] = curriculum.upper
  upper = curriculum.episode_upper[env_ids]
  mass = 1.0 + torch.rand_like(upper) * (upper - 1.0)
  body_id = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids][0]
  env.sim.model.body_mass[env_ids, body_id] = mass
  env.sim.model.body_inertia[env_ids, body_id] = 0.015 * mass[:, None]
