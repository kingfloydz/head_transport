"""Four-stage curriculum using completed episodes from the current stage."""

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch
from rsl_rl.utils.logger import Logger
from torch.distributed import all_reduce

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumManager, CurriculumTermCfg
from mjlab.rl.runner import MjlabOnPolicyRunner

from .load_distribution import STAGE_LIMITS

REWARD_WEIGHTS = {
  "action_rate_l2": (-0.05, -0.08, -0.12, -0.16),
  "joint_torques_l2": (-5e-6, -1e-5, -2e-5, -6e-5),
}


class PayloadCurriculum:
  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self.stage = cfg.params["initial_stage"] - 1
    self.episode_stage = torch.full((env.num_envs,), -1, device=env.device)
    self.window = torch.zeros(2, dtype=torch.float64, device=env.device)
    self.iterations = 0
    self.reward_terms = {
      name: env.reward_manager.get_term_cfg(name) for name in REWARD_WEIGHTS
    }
    self.update_reward_weights()

  def update_reward_weights(self):
    for name, term in self.reward_terms.items():
      term.weight = REWARD_WEIGHTS[name][self.stage]

  def __call__(self, env: ManagerBasedRlEnv, env_ids, initial_stage: int):
    lengths = env.episode_length_buf[env_ids]
    eligible = (
      (self.episode_stage[env_ids] == self.stage)
      & env.termination_manager.dones[env_ids]
      & (lengths > 0)
    )
    self.window[0] += lengths[eligible].sum() * env.step_dt
    self.window[1] += eligible.sum()
    self.episode_stage[env_ids] = self.stage
    return {"stage": self.stage + 1, "eta_max": STAGE_LIMITS[self.stage][4]}

  def log_iteration(self, log: Callable, logger: Logger, *args, **kwargs):
    self.iterations += 1
    if self.iterations % 100 == 0:
      totals = self.window.clone()
      if logger.gpu_world_size > 1:
        all_reduce(totals)
      if totals[1] > 0 and totals[0] / totals[1] > 18.8:
        self.stage = min(self.stage + 1, len(STAGE_LIMITS) - 1)
        self.update_reward_weights()
      self.window.zero_()
    log(*args, **kwargs)


def bind_payload_curriculum(runner: MjlabOnPolicyRunner) -> None:
  manager = cast(CurriculumManager, runner.env.unwrapped.curriculum_manager)
  curriculum = manager.get_term_cfg("payload_curriculum").func
  cast(Any, runner.logger).log = partial(
    curriculum.log_iteration, runner.logger.log, runner.logger
  )
