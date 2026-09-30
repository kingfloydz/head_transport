"""Official velocity commands with dedicated in-place turning samples."""

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class HeadLoadVelocityCommand(UniformVelocityCommand):
  cfg: "HeadLoadVelocityCommandCfg"

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    probability = self.cfg.rel_turning_envs / (
      (1 - self.cfg.rel_standing_envs) * (1 - self.cfg.rel_forward_envs)
    )
    turning = (
      ~self.is_standing_env[env_ids]
      & ~self.is_forward_env[env_ids]
      & (torch.rand(len(env_ids), device=self.device) < probability)
    )
    ids = env_ids[turning]
    magnitude = self.cfg.ranges.ang_vel_z[1] - (
      self.cfg.ranges.ang_vel_z[1] - 0.2
    ) * torch.rand(len(ids), device=self.device)
    sign = 2 * torch.randint(2, (len(ids),), device=self.device) - 1
    self.vel_command_b[ids, :2] = 0.0
    self.vel_command_b[ids, 2] = sign * magnitude
    self.vel_command_w[ids] = self.vel_command_b[ids]
    self.is_heading_env[ids] = False
    self.is_world_env[ids] = False


@dataclass(kw_only=True)
class HeadLoadVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_turning_envs: float = 0.1

  def build(self, env: ManagerBasedRlEnv) -> HeadLoadVelocityCommand:
    return HeadLoadVelocityCommand(self, env)
