"""Command dead-zone gating for the official locomotion rewards."""

from collections.abc import Callable
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity import mdp


def moving_mask(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(torch.Tensor, env.command_manager.get_command(command_name))
  return (torch.linalg.vector_norm(command[:, :2], dim=1) >= 0.1) | (
    command[:, 2].abs() >= 0.05
  )


def command_gated_reward(
  env: ManagerBasedRlEnv, reward_fn: Callable, command_name: str, **kwargs
) -> torch.Tensor:
  reward = reward_fn(env, command_name=command_name, **kwargs)
  return reward * moving_mask(env, command_name)


class CommandGatedSwingHeight(mdp.feet_swing_height):
  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    reward = super().__call__(
      env,
      sensor_name,
      height_sensor_name,
      target_height,
      command_name,
      command_threshold,
    )
    return reward * moving_mask(env, command_name)
