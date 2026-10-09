"""Run a checkpoint in 4096 environments and summarize stability margins."""

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


NAMES = ("margin",)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint")
  parser.add_argument("--steps", type=int, default=250)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--delta", type=float, default=1.0)
  parser.add_argument("--temperature", type=float, default=0.5)
  args = parser.parse_args()
  import mjlab.tasks  # noqa: F401

  task = "Mjlab-Velocity-HeadLoad-Unitree-G1"
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  sensor_cfg = next(s for s in env_cfg.scene.sensors if s.name == "payload_stability")
  sensor_cfg.target_margin = args.delta
  sensor_cfg.temperature = args.temperature
  agent_cfg = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=args.device)
  runner.load(
    args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device
  )
  policy = runner.get_inference_policy(device=args.device)
  reset = vec_env.reset()
  obs = reset[0] if isinstance(reset, tuple) else reset
  samples = []
  costs = []
  with torch.inference_mode():
    for _ in range(args.steps):
      obs, _, _, _ = vec_env.step(policy(obs))
      sensor = env.scene["payload_stability"]
      samples.append(sensor.mean_margin.cpu().numpy().copy()[:, None])
      costs.append(sensor.mean_penalty.cpu().numpy().copy())
  env.close()
  margins = np.concatenate(samples)
  mean_penalty = np.array([np.concatenate(costs).mean()])
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
  }
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
