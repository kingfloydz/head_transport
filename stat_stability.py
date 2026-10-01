"""Summarize the four stability margins and their softplus penalties."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


NAMES = ("normal", "friction", "tipping", "yaw")


def read(path: Path) -> np.ndarray:
  if path.suffix == ".npz":
    data = np.load(path)
    for key in ("margins", "stability_margins"):
      if key in data:
        return np.asarray(data[key], dtype=np.float64)
    raise ValueError("NPZ must contain margins or stability_margins")
  with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
  columns = []
  for name in NAMES:
    column = next(
      (key for key in rows[0] if key == name or key.endswith(f"/{name}")), None
    )
    if column is None:
      raise ValueError(f"Missing stability column: {name}")
    columns.append([float(row[column]) for row in rows])
  return np.asarray(columns, dtype=np.float64).T


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input", type=Path)
  parser.add_argument("--delta", type=float, default=0.1)
  parser.add_argument("--temperature", type=float, default=0.1)
  args = parser.parse_args()
  margins = read(args.input)
  penalties = np.logaddexp(0.0, (args.delta - margins) / args.temperature)
  mean_penalty = penalties.mean(0)
  result = {
    "samples": int(len(margins)),
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
