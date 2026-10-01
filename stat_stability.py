"""Run a checkpoint in 4096 environments and summarize stability margins."""

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


NAMES = ("normal", "friction", "tipping", "yaw")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint")
  parser.add_argument("--steps", type=int, default=250)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--delta", type=float, default=0.1)
  parser.add_argument("--temperature", type=float, default=0.1)
  args = parser.parse_args()
  import mjlab.tasks  # noqa: F401

  task = "Mjlab-Velocity-HeadLoad-Unitree-G1"
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=args.device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)
  obs = vec_env.reset()
  samples = []
  with torch.inference_mode():
    for _ in range(args.steps):
      obs, _, _, _ = vec_env.step(policy(obs))
      samples.append(env.scene["payload_stability"].mean_margins.cpu().numpy())
  env.close()
  margins = np.concatenate(samples)
  penalties = np.logaddexp(0.0, (args.delta - margins) / args.temperature)
  mean_penalty = penalties.mean(0)
  result = {
    "samples": int(len(margins)),
    "environments": args.num_envs,
    "steps": args.steps,
    "delta": args.delta,
    "temperature": args.temperature,
    "margins": {
      name: {
        "mean": float(margins[:, i].mean()),
        "p05": float(np.quantile(margins[:, i], 0.05)),
        "median": float(np.median(margins[:, i])),
        "violation_rate": float((margins[:, i] < 0).mean()),
      }
      for i, name in enumerate(NAMES)
    },
    "penalty_mean": {name: float(mean_penalty[i]) for i, name in enumerate(NAMES)},
    "penalty_ratio_to_others": {
      name: float(mean_penalty[i] / np.delete(mean_penalty, i).mean())
      for i, name in enumerate(NAMES)
    },
  }
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
