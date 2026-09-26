"""Reset-time uniform payload mass sampling and synchronized inertia."""

from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def sample_payload_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  mass_kg: float | None = None,
) -> None:
  if mass_kg is None:
    manager = cast(CurriculumManager, env.curriculum_manager)
    upper = manager.get_term_cfg("payload_mass_upper").func.upper
    mass = 1.0 + (upper - 1.0) * torch.rand(len(env_ids), device=env.device)
  else:
    mass = torch.full((len(env_ids),), mass_kg, device=env.device)
  body_id = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids][0]
  env.sim.model.body_mass[env_ids, body_id] = mass
  env.sim.model.body_inertia[env_ids, body_id] = 0.015 * mass[:, None]
