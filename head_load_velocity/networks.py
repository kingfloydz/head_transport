"""61-frame causal TCN, based on rickshaw's encoder, with a 16-D latent."""

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


class HistoryEncoder(nn.Module):
  def __init__(self, obs_dim: int):
    super().__init__()
    self.input = nn.Conv1d(obs_dim, 64, kernel_size=1)
    self.blocks = nn.Sequential(*(CausalBlock(d) for d in (1, 2, 4, 8)))
    self.context = nn.Sequential(
      nn.Linear(128, 64), nn.ELU(), nn.Linear(64, LATENT_DIM)
    )

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    encoded = self.blocks(self.input(history.transpose(1, 2)))
    return self.context(torch.cat((encoded[:, :, -1], encoded.mean(dim=-1)), dim=-1))


class TemporalActor(MLPModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.encoder = HistoryEncoder(self.obs_dim)

  def _get_obs_dim(self, obs: TensorDict, obs_groups, obs_set):
    groups = list(obs_groups[obs_set])
    return groups, obs[groups[0]].shape[-1]

  def _get_latent_dim(self) -> int:
    return self.obs_dim + LATENT_DIM

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    sequence = self.obs_normalizer(obs[self.obs_groups[0]])
    return torch.cat((sequence[:, -1], self.encoder(sequence[:, :-1])), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      cast(EmpiricalNormalization, self.obs_normalizer).update(
        cast(torch.Tensor, obs[self.obs_groups[0]])[:, -1]
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
    self.encoder = deepcopy(model.encoder)
    self.mlp = deepcopy(model.mlp)
    self.deterministic_output = cast(
      Distribution, model.distribution
    ).as_deterministic_output_module()

  def forward(self, observation_sequence: torch.Tensor) -> torch.Tensor:
    sequence = self.obs_normalizer(observation_sequence)
    latent = torch.cat((sequence[:, -1], self.encoder(sequence[:, :-1])), dim=-1)
    return self.deterministic_output(self.mlp(latent))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.history_length, self.obs_dim),)

  @property
  def input_names(self) -> list[str]:
    return ["observation_sequence"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
