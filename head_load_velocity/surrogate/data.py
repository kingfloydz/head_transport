"""Bounded rollout collection, exact labels and physically consistent extension."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.os import dump_yaml

from ..env_cfg import head_load_velocity_env_cfg
from ..networks import TemporalActor
from .exact import label
from .online import RawContactFeaturesCfg
from .schema import unpack


def save(path, data, meta):
  np.savez_compressed(path, **data, metadata=np.array(json.dumps(meta)))


def load(paths):
  arrays = []
  metadata = None
  for path in paths:
    with np.load(path, allow_pickle=False) as file:
      current = json.loads(str(file["metadata"]))
      if metadata is not None and current != metadata:
        raise ValueError("Dataset physics metadata differ")
      metadata = current
      arrays.append({k: file[k] for k in file.files if k != "metadata"})
  return {k: np.concatenate([d[k] for d in arrays]) for k in arrays[0]}, metadata


def label_data(data, meta):
  results = [
    label(x, meta, int(bits)) for x, bits in zip(data["x"], data["domain"], strict=True)
  ]
  y, yaw, bits, status, feasible, codes = zip(*results, strict=True)
  return {
    **data,
    "y": np.array(y),
    "yaw_valid": np.array(yaw),
    "domain": np.array(bits),
    "solver_status": np.array(status, dtype=np.int8),
    "feasible": np.array(feasible),
    "lp_status": np.array(codes),
  }


def collect(manifest, output, steps, num_envs, device, controller_ids=None, seed=42):
  """Sample current task at unchanged control rate; never bind a training runner."""
  from .. import rl_cfg

  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  selected = [
    c
    for c in manifest["controllers"]
    if controller_ids is None or c["id"] in controller_ids
  ]
  # Do not execute some controllers before discovering another path is missing.
  for controller in selected:
    if not controller["checkpoint"]:
      raise ValueError(f"Fill checkpoint path for {controller['id']} in manifest")
    if not Path(controller["checkpoint"]).is_file():
      raise FileNotFoundError(controller["checkpoint"])
  for controller in selected:
    cfg = head_load_velocity_env_cfg(payload_stage=controller["stage"])
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.scene.sensors += (
      RawContactFeaturesCfg(name="surrogate_features", decimation=cfg.decimation),
    )
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
      observations, _ = env.reset(seed=seed)
      actor_cfg = rl_cfg.actor
      actor = (
        TemporalActor(
          TensorDict.from_dict(cast(Any, observations), batch_size=[num_envs]),
          {"actor": ["actor"]},
          "actor",
          env.action_manager.total_action_dim,
          hidden_dims=actor_cfg.hidden_dims,
          activation=actor_cfg.activation,
          obs_normalization=actor_cfg.obs_normalization,
          distribution_cfg=deepcopy(actor_cfg.distribution_cfg),
        )
        .to(device)
        .eval()
      )
      actor.load_state_dict(
        torch.load(controller["checkpoint"], map_location=device, weights_only=True)[
          "actor_state_dict"
        ],
        strict=True,
      )
      sensor = env.scene["surrogate_features"]
      run_id = uuid4().hex
      folder = output / controller["id"] / run_id
      folder.mkdir(parents=True)
      dump_yaml(folder / "env.yaml", cast(Any, cfg))
      dump_yaml(folder / "agent.yaml", cast(Any, rl_cfg))
      provenance = {
        **controller,
        "seed": seed,
        "steps": steps,
        "num_envs": num_envs,
        "run_id": run_id,
        "checkpoint_sha256": hashlib.sha256(
          Path(controller["checkpoint"]).read_bytes()
        ).hexdigest(),
      }
      (folder / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
      )
      rows = []
      with torch.inference_mode():
        for step in range(steps):
          actions = actor(
            TensorDict.from_dict(cast(Any, observations), batch_size=[num_envs]),
            stochastic_output=False,
          )
          observations, _, terminated, truncated, _ = env.step(actions)
          rows.append(
            dict(
              x=sensor.x.cpu().numpy().copy(),
              domain=sensor.domain.cpu().numpy().copy(),
              controller=np.full(num_envs, controller["id"]),
              episode=np.char.add(
                run_id + ":",
                (
                  sensor.sample_episode * num_envs
                  + torch.arange(num_envs, device=device)
                )
                .cpu()
                .numpy()
                .astype(str),
              ),
              time_index=sensor.sample_time.cpu().numpy(),
              source=np.full(num_envs, "rollout"),
              terminated=terminated.cpu().numpy(),
              truncated=truncated.cpu().numpy(),
              parent=np.full(num_envs, -1, dtype=np.int64),
            )
          )
          if (step + 1) % 100 == 0 or step + 1 == steps:
            batch = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
            save(folder / f"raw_{step + 1:06d}.npz", batch, sensor.meta)
            rows.clear()
    finally:
      env.close()


def perturb(x, rng, meta):
  """Affine deformation of an existing box distribution, not independent I/COM noise."""
  x = x.copy().astype(np.float64)
  size, com, rotation, inertia = unpack(x)
  bottom = x[13:16] - rotation @ (com + [0, 0, size[2] / 2])
  factors = rng.uniform(0.8, 1.2, 3)
  density_scale = rng.uniform(0.75, 1.25)
  mass = x[6] * density_scale * np.prod(factors)
  # Central second moment per mass Q = tr(I)/(2m) Id - I/m.
  covariance = (np.trace(inertia) * np.eye(3) / 2 - inertia) / x[6]
  covariance = np.diag(factors) @ covariance @ np.diag(factors)
  inertia = mass * (np.trace(covariance) * np.eye(3) - covariance)
  size *= factors
  com *= factors
  yaw = rng.uniform(-0.3, 0.3)
  rz = np.array(
    [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
  )
  rotation = rz @ rotation
  bottom[:2] += rng.uniform(-0.025, 0.025, 2)
  x[:6] = np.column_stack((size / 2 + com, size / 2 - com)).ravel()
  x[6] = mass
  x[7:13] = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
  x[13:16] = bottom + rotation @ (com + [0, 0, size[2] / 2])
  x[16:22] = rotation[:, :2].T.ravel()
  x[22:25] += rng.normal(0, 6, 3)
  x[25:28] += rng.normal(0, 1, 3)
  x[28:31] += rng.normal(0, 10, 3)
  x[34] = rng.uniform(0.3, 1.2)
  return x


def extend(data, meta, count, seed=42):
  """Keep boundary and infeasible candidates; retain parent episode/controller."""
  rng = np.random.default_rng(seed)
  eligible = np.flatnonzero((data["domain"] == 0) & (data["solver_status"] == 0))
  if not len(eligible):
    raise ValueError("No valid parents for physical extension")
  parents = rng.choice(eligible, size=count * 4)
  x = np.array([perturb(data["x"][i], rng, meta) for i in parents])
  candidates = {
    k: v[parents].copy()
    for k, v in data.items()
    if k not in ("y", "yaw_valid", "solver_status", "feasible", "lp_status")
  }
  candidates.update(
    x=x,
    domain=np.zeros(len(x), dtype=np.int64),
    source=np.full(len(x), "extension"),
    parent=parents,
  )
  candidates = label_data(candidates, meta)
  ok = (candidates["solver_status"] == 0) & (candidates["domain"] == 0)
  weight = np.where(
    ok,
    1
    + 8 * np.exp(-np.nan_to_num(abs(candidates["y"][:, 4]), nan=1e6) / 0.1)
    + 3 * (candidates["y"][:, 4] < 0),
    0.0,
  )
  available = np.flatnonzero(ok)
  take = min(count, len(available))
  chosen = rng.choice(
    available, size=take, replace=False, p=weight[available] / weight[available].sum()
  )
  # No valid extension is invented if perturbed geometry leaves applicability.
  return {k: np.concatenate((data[k], candidates[k][chosen])) for k in data}
