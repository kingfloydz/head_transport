"""Rickshaw-style teacher: dynamic history and static load properties, latent 16."""

from copy import deepcopy
from typing import cast

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import Distribution
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

HISTORY_LENGTH = 61
LATENT_DIM = 16


class CausalBlock(nn.Module):
  def __init__(self, dilation: int):
    super().__init__()
    self.left_pad = 4 * dilation
    self.conv = nn.Conv1d(64, 64, kernel_size=5, dilation=dilation)
    self.mix = nn.Conv1d(64, 64, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.elu(x + self.mix(F.elu(self.conv(F.pad(x, (self.left_pad, 0))))))


class TeacherEncoder(nn.Module):
  def __init__(self, obs_dim: int):
    super().__init__()
    self.input = nn.Conv1d(obs_dim + 3, 64, kernel_size=1)
    self.blocks = nn.Sequential(*(CausalBlock(d) for d in (1, 2, 4, 8)))
    self.static = nn.Sequential(nn.Linear(4, 16), nn.ELU())
    self.context = nn.Sequential(
      nn.Linear(144, 64), nn.ELU(), nn.Linear(64, LATENT_DIM)
    )

  def forward(
    self, history: torch.Tensor, dynamic: torch.Tensor, properties: torch.Tensor
  ) -> torch.Tensor:
    history = torch.cat((history, dynamic), dim=-1)
    encoded = self.blocks(self.input(history.transpose(1, 2)))
    return self.context(
      torch.cat(
        (encoded[:, :, -1], encoded.mean(dim=-1), self.static(properties)), dim=-1
      )
    )


class TemporalActor(MLPModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.encoder = TeacherEncoder(self.obs_dim)
    self.dynamic_normalizer = (
      EmpiricalNormalization(3) if self.obs_normalization else nn.Identity()
    )
    self.static_normalizer = (
      EmpiricalNormalization(4) if self.obs_normalization else nn.Identity()
    )

  def _get_obs_dim(self, obs: TensorDict, obs_groups, obs_set):
    groups = list(obs_groups[obs_set])
    return groups, obs[groups[0]].shape[-1]

  def _get_latent_dim(self) -> int:
    return self.obs_dim + LATENT_DIM

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    sequence = self.obs_normalizer(obs[self.obs_groups[0]])
    dynamic = self.dynamic_normalizer(obs["teacher_dynamic"])
    properties = self.static_normalizer(obs["teacher_static"])
    context = self.encoder(sequence[:, :-1], dynamic[:, :-1], properties)
    return torch.cat((sequence[:, -1], context), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      cast(EmpiricalNormalization, self.obs_normalizer).update(
        cast(torch.Tensor, obs[self.obs_groups[0]])[:, -1]
      )
      cast(EmpiricalNormalization, self.dynamic_normalizer).update(
        cast(torch.Tensor, obs["teacher_dynamic"])[:, -1]
      )
      cast(EmpiricalNormalization, self.static_normalizer).update(
        cast(torch.Tensor, obs["teacher_static"])
      )

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return TemporalPolicyExport(self)

  def as_jit(self) -> nn.Module:
    return TemporalPolicyExport(self)


class TemporalPolicyExport(nn.Module):
  is_recurrent = False

  def __init__(self, model: TemporalActor):
    super().__init__()
    self.obs_dim = model.obs_dim
    self.history_length = HISTORY_LENGTH + 1
    self.obs_normalizer = deepcopy(model.obs_normalizer)
    self.dynamic_normalizer = deepcopy(model.dynamic_normalizer)
    self.static_normalizer = deepcopy(model.static_normalizer)
    self.encoder = deepcopy(model.encoder)
    self.mlp = deepcopy(model.mlp)
    self.deterministic_output = cast(
      Distribution, model.distribution
    ).as_deterministic_output_module()

  def forward(
    self,
    observation_sequence: torch.Tensor,
    velocity_sequence: torch.Tensor,
    load_properties: torch.Tensor,
  ) -> torch.Tensor:
    sequence = self.obs_normalizer(observation_sequence)
    dynamic = self.dynamic_normalizer(velocity_sequence)
    properties = self.static_normalizer(load_properties)
    context = self.encoder(sequence[:, :-1], dynamic[:, :-1], properties)
    latent = torch.cat((sequence[:, -1], context), dim=-1)
    return self.deterministic_output(self.mlp(latent))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.history_length, self.obs_dim),
      torch.zeros(1, self.history_length, 3),
      torch.zeros(1, 4),
    )

  @property
  def input_names(self) -> list[str]:
    return ["observation_sequence", "velocity_sequence", "load_properties"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
