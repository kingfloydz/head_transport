"""Shared payload ceiling driven by the official mean episode length."""

from collections.abc import Callable
from functools import partial
from statistics import mean
from typing import Any, cast

import torch
from rsl_rl.utils.logger import Logger

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumManager, CurriculumTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class PayloadMassUpper:
  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self.upper = torch.tensor(3.0, device=env.device)
    self.iterations = 0
    self.lengths: list[float] = []

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
  ) -> torch.Tensor:
    return self.upper.clone()

  def log_iteration(
    self, log: Callable, logger: Logger, step_dt: float, *args, **kwargs
  ) -> None:
    log(*args, **kwargs)
    self.iterations += 1
    if logger.lenbuffer:
      self.lengths.append(mean(logger.lenbuffer) * step_dt)
    if self.iterations % 100 == 0:
      if self.lengths and mean(self.lengths) > 19.0:
        self.upper.add_(3.0).clamp_(max=60.0)
      self.lengths.clear()


def bind_payload_curriculum(runner: VelocityOnPolicyRunner) -> None:
  env = runner.env.unwrapped
  manager = cast(CurriculumManager, env.curriculum_manager)
  curriculum = manager.get_term_cfg("payload_mass_upper").func
  cast(Any, runner.logger).log = partial(
    curriculum.log_iteration, runner.logger.log, runner.logger, env.step_dt
  )


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def sample_payload_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  mass_kg: float | None = None,
) -> None:
  if mass_kg is None:
    manager = cast(CurriculumManager, env.curriculum_manager)
    curriculum = manager.get_term_cfg("payload_mass_upper").func
    mass = 1.0 + torch.rand(len(env_ids), device=env.device) * (curriculum.upper - 1.0)
  else:
    mass = torch.full((len(env_ids),), mass_kg, device=env.device)
  body_id = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids][0]
  env.sim.model.body_mass[env_ids, body_id] = mass
  env.sim.model.body_inertia[env_ids, body_id] = 0.015 * mass[:, None]
