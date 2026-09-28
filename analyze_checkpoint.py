"""Evaluate complete episodes and correlate outcomes with randomized properties."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, cast

os.environ["MUJOCO_GL"] = "disable"

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.head_load_velocity import rl_cfg
from mjlab.tasks.head_load_velocity.env_cfg import head_load_velocity_env_cfg
from mjlab.tasks.head_load_velocity.load_distribution import STAGE_LIMITS, difficulty
from mjlab.tasks.head_load_velocity.networks import TemporalActor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, matrix_from_quat
from mjlab.utils.torch import configure_torch_backends


def initial_properties(env):
  robot, payload = env.scene["robot"], env.scene["payload"]
  model = env.sim.model
  body = payload.indexing.root_body_id
  torso = robot.indexing.body_ids[robot.find_bodies("torso_link")[0][0]]
  ground = env.scene["terrain"].indexing.geom_ids[0]
  size = 2 * model.geom_size[:, payload.indexing.geom_ids[0]]
  inertia = model.body_inertia[:, body]
  mass = model.body_mass[:, body]
  com = model.body_ipos[:, body]
  axes = matrix_from_quat(model.body_iquat[:, body])
  full_inertia = axes @ torch.diag_embed(inertia) @ axes.transpose(-1, -2)
  robot_mass = model.body_mass[:, robot.indexing.body_ids].sum(-1)
  metrics = difficulty(size, mass, com, full_inertia, robot_mass)
  columns = {
    "mass_kg": mass,
    "robot_mass_kg": robot_mass,
    **{
      name: metrics[:, i]
      for i, name in enumerate(("kappa_x", "kappa_y", "chi_x", "chi_y", "eta"))
    },
    **{f"payload_com_{axis}_m": com[:, i] for i, axis in enumerate("xyz")},
    **{f"size_{axis}_m": size[:, i] for i, axis in enumerate("xyz")},
    **{f"principal_inertia_{i}": inertia[:, i] for i in range(3)},
    "inertia_ratio": inertia.max(-1).values / inertia.min(-1).values,
    "density_kg_m3": model.body_mass[:, body] / size.prod(-1),
    "torso_mass_kg": model.body_mass[:, torso],
    "ground_friction": model.geom_friction[:, ground, 0],
  }
  for i, axis in enumerate("xyz"):
    columns[f"torso_com_{axis}_m"] = model.body_ipos[:, torso, i]
    columns[f"initial_base_velocity_{axis}"] = robot.data.root_link_lin_vel_b[:, i]
    columns[f"initial_base_omega_{axis}"] = robot.data.root_link_ang_vel_b[:, i]
  columns["initial_base_height_m"] = robot.data.root_link_pos_w[:, 2]
  for name, value in zip(
    ("roll", "pitch", "yaw"),
    euler_xyz_from_quat(robot.data.root_link_quat_w),
    strict=True,
  ):
    columns[f"initial_base_{name}"] = value
  for i, name in enumerate(robot.joint_names):
    columns[f"initial_pos_{name}"] = robot.data.joint_pos[:, i]
    columns[f"initial_vel_{name}"] = robot.data.joint_vel[:, i]
  command = env.command_manager.get_command("twist")
  for i, name in enumerate(("vx", "vy", "yaw_rate")):
    columns[f"initial_command_{name}"] = command[:, i]
  return list(columns), torch.stack(list(columns.values()), dim=-1).clone(), torso


@torch.inference_mode()
def evaluate(env, policy, seed, stochastic):
  observations, _ = env.reset(seed=seed)
  names, properties, torso = initial_properties(env)
  n, device = env.num_envs, env.device
  active = torch.ones(n, dtype=torch.bool, device=device)
  lengths = torch.zeros(n, device=device)
  returns = torch.zeros_like(lengths)
  command_sum = torch.zeros(n, 3, device=device)
  tracking_error = torch.zeros_like(lengths)
  push_impulse = torch.zeros_like(lengths)
  push_peak = torch.zeros_like(lengths)
  success = torch.zeros_like(lengths)
  terms = env.termination_manager.active_terms
  causes = torch.zeros(n, len(terms), device=device)
  for step in range(env.max_episode_length):
    command = env.command_manager.get_command("twist")
    command_sum += command.abs() * active[:, None]
    velocity = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    tracking_error += (velocity - command[:, :2]).norm(dim=-1) * active
    force = env.sim.data.xfrc_applied[:, torso, :3].norm(dim=-1) * active
    push_impulse += force * env.step_dt
    push_peak = torch.maximum(push_peak, force)
    actions = policy(
      TensorDict.from_dict(observations, batch_size=[n]), stochastic_output=stochastic
    )
    observations, reward, terminated, truncated, _ = env.step(actions)
    lengths += active
    returns += reward * active
    finished = active & (terminated | truncated)
    success[finished] = (truncated & ~terminated)[finished].float()
    for i, term in enumerate(terms):
      causes[finished, i] = env.termination_manager.get_term(term)[finished].float()
    active &= ~finished
    if (step + 1) % 100 == 0:
      print(f"step {step + 1}: completed {int((~active).sum())}/{n}", flush=True)
    if not active.any():
      break
  outcomes = torch.cat(
    (
      lengths[:, None],
      (lengths * env.step_dt)[:, None],
      success[:, None],
      returns[:, None],
      (~active).float()[:, None],
      causes,
      command_sum / lengths[:, None],
      (tracking_error / lengths)[:, None],
      push_impulse[:, None],
      push_peak[:, None],
    ),
    dim=-1,
  )
  outcome_names = [
    "episode_steps",
    "episode_seconds",
    "success",
    "episode_return",
    "completed",
    *[f"termination_{term}" for term in terms],
    "mean_abs_command_vx",
    "mean_abs_command_vy",
    "mean_abs_command_yaw_rate",
    "mean_xy_tracking_error",
    "push_impulse_Ns",
    "push_peak_N",
  ]
  return names, outcome_names, torch.cat((properties, outcomes), -1).cpu().numpy()


def write_csv(path, names, rows):
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(names)
    writer.writerows(rows)


def summarize(folder, names, outcome_names, data):
  all_names = names + outcome_names
  length = data[:, all_names.index("episode_seconds")]
  success = data[:, all_names.index("success")]
  correlations, bins = [], []
  for i, name in enumerate(names):
    values = data[:, i]
    correlations.append(
      [
        name,
        pearsonr(values, length)[0],
        spearmanr(values, length)[0],
        pearsonr(values, success)[0],
      ]
    )
    groups = np.digitize(values, np.unique(np.quantile(values, np.arange(1, 10) / 10)))
    for group in np.unique(groups):
      selected = groups == group
      bins.append(
        [
          name,
          int(group),
          int(selected.sum()),
          float(values[selected].min()),
          float(values[selected].max()),
          float(length[selected].mean()),
          float(success[selected].mean()),
          *[
            float(data[selected, all_names.index(term)].mean())
            for term in outcome_names
            if term.startswith("termination_")
          ],
        ]
      )
  write_csv(folder / "episodes.csv", all_names, data)
  write_csv(
    folder / "correlations.csv",
    [
      "property",
      "pearson_episode_seconds",
      "spearman_episode_seconds",
      "pearson_success",
    ],
    correlations,
  )
  write_csv(
    folder / "property_bins.csv",
    [
      "property",
      "bin",
      "count",
      "min",
      "max",
      "mean_episode_seconds",
      "success_rate",
      *[term for term in outcome_names if term.startswith("termination_")],
    ],
    bins,
  )
  summary = {
    "episodes": len(data),
    "mean_episode_seconds": float(length.mean()),
    "mean_episode_steps": float(data[:, all_names.index("episode_steps")].mean()),
    "success_rate": float(success.mean()),
    "completed_rate": float(data[:, all_names.index("completed")].mean()),
    "termination_rates": {
      term: float(data[:, all_names.index(term)].mean())
      for term in outcome_names
      if term.startswith("termination_")
    },
  }
  (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoints", type=Path, nargs="+")
  parser.add_argument("--num-envs", type=int, default=16394)
  parser.add_argument("--episodes-per-env", type=int, default=1)
  parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), default=1)
  parser.add_argument("--platform-height", type=float, default=0.44)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--stochastic", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--output", type=Path, default=Path("checkpoint_analysis"))
  args = parser.parse_args()
  configure_torch_backends()
  cfg = head_load_velocity_env_cfg(
    platform_height=args.platform_height, payload_stage=args.stage
  )
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg, device=args.device)
  observations, _ = env.reset(seed=args.seed)
  actor_cfg = rl_cfg.actor
  actor = (
    TemporalActor(
      TensorDict.from_dict(cast(Any, observations), batch_size=[args.num_envs]),
      {"actor": ["actor"]},
      "actor",
      env.action_manager.total_action_dim,
      hidden_dims=actor_cfg.hidden_dims,
      activation=actor_cfg.activation,
      obs_normalization=actor_cfg.obs_normalization,
      distribution_cfg=actor_cfg.distribution_cfg,
    )
    .to(args.device)
    .eval()
  )
  for index, checkpoint in enumerate(args.checkpoints):
    actor.load_state_dict(
      torch.load(checkpoint, map_location=args.device, weights_only=False)[
        "actor_state_dict"
      ]
    )
    batches = []
    for cohort in range(args.episodes_per_env):
      print(f"{checkpoint}: cohort {cohort + 1}", flush=True)
      batches.append(evaluate(env, actor, args.seed + cohort, args.stochastic))
    names, outcome_names, _ = batches[0]
    folder = args.output / f"{index:02d}_{checkpoint.stem}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(
      json.dumps(
        {
          "checkpoint": str(checkpoint.resolve()),
          "num_envs": args.num_envs,
          "episodes_per_env": args.episodes_per_env,
          "stage": args.stage,
          "stage_limits_kappa_xy_chi_xy_eta": STAGE_LIMITS[args.stage - 1],
          "density_range_kg_m3": [40, 4000],
          "platform_height": args.platform_height,
          "seed": args.seed,
          "stochastic": args.stochastic,
          "physics_dt": env.physics_dt,
          "step_dt": env.step_dt,
          "episode_limit_s": cfg.episode_length_s,
          "note": "Current main environment, training observation noise and pushes enabled. "
          "One complete episode per environment per cohort; no short-episode oversampling. "
          "Correlations are marginal associations, not causal effects; size axes are coupled. "
          "Constant properties produce undefined (nan) correlations. "
          "Command/push exposure depends on episode duration. "
          "Initial cohort seeds are shared across checkpoints, subsequent trajectories are not paired.",
        },
        indent=2,
      ),
      encoding="utf-8",
    )
    summarize(
      folder, names, outcome_names, np.concatenate([batch[2] for batch in batches])
    )
  env.close()


if __name__ == "__main__":
  main()
