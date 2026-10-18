"""Depth CNN plus proprioceptive MLP, with no privileged actor inputs."""

from copy import deepcopy

import torch
from rsl_rl.models import MLPModel
from torch import nn


class DepthActor(MLPModel):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.encoder = nn.Sequential(
      nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.ELU(),
      nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
      nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
      nn.AdaptiveAvgPool2d((2, 3)), nn.Flatten(), nn.Linear(192, 64), nn.ELU(),
    )

  def _get_obs_dim(self, obs, obs_groups, obs_set):
    return ["actor"], obs["actor"].shape[-1]

  def _get_latent_dim(self):
    return self.obs_dim + 64

  def get_latent(self, obs, masks=None, hidden_state=None):
    return torch.cat((self.obs_normalizer(obs["actor"]), self.encoder(obs["depth"])), -1)

  def as_onnx(self, verbose=False):
    return DepthPolicyExport(self)

  def as_jit(self):
    return DepthPolicyExport(self)


class DepthPolicyExport(nn.Module):
  is_recurrent = False
  input_names = ["proprioception", "depth_and_mask"]
  output_names = ["actions"]

  def __init__(self, model):
    super().__init__()
    self.obs_dim = model.obs_dim
    self.normalizer = deepcopy(model.obs_normalizer)
    self.encoder = deepcopy(model.encoder)
    self.mlp = deepcopy(model.mlp)
    self.output = model.distribution.as_deterministic_output_module()

  def forward(self, proprioception, depth_and_mask):
    latent = torch.cat((self.normalizer(proprioception), self.encoder(depth_and_mask)), -1)
    return self.output(self.mlp(latent))

  def get_dummy_inputs(self):
    return torch.zeros(1, self.obs_dim), torch.zeros(1, 2, 64, 96)
