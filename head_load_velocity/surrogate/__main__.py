"""Small explicit commands; no long-running job is started implicitly."""

import argparse
import json
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  collect = commands.add_parser("collect")
  collect.add_argument("--manifest", type=Path, required=True)
  collect.add_argument("--output", type=Path, required=True)
  collect.add_argument("--steps", type=int, required=True)
  collect.add_argument("--num-envs", type=int, default=64)
  collect.add_argument("--device", default="cuda:0")
  collect.add_argument("--controllers", nargs="+")
  collect.add_argument("--seed", type=int, default=42)
  for name in ("label", "extend", "train"):
    sub = commands.add_parser(name)
    sub.add_argument("--input", type=Path, nargs="+", required=True)
    sub.add_argument("--output", type=Path, required=True)
    if name == "extend":
      sub.add_argument("--count", type=int, required=True)
      sub.add_argument("--seed", type=int, default=42)
    if name == "train":
      sub.add_argument("--manifest", type=Path, required=True)
      sub.add_argument("--epochs", type=int, default=50)
      sub.add_argument("--batch", type=int, default=1024)
      sub.add_argument("--device", default="cuda:0")
      sub.add_argument("--geometry-x", type=float, default=0.4)
      sub.add_argument("--geometry-mu", type=float, default=0.9)
  validate = commands.add_parser("validate")
  validate.add_argument("--output", type=Path, required=True)
  bench = commands.add_parser("benchmark")
  bench.add_argument("--model", type=Path, required=True)
  bench.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  # Local imports keep the online reward path free of scipy/LP and CLI utilities.
  if args.command == "validate":
    from .validation import validate

    print(json.dumps(validate(args.output), indent=2))
  elif args.command == "benchmark":
    import torch

    from .model import benchmark

    model = torch.jit.load(str(args.model), map_location=args.device).eval()
    print(json.dumps(benchmark(model, args.device), indent=2))
  else:
    from .data import collect, extend, label_data, load, save

    if args.command == "collect":
      manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
      collect(
        manifest,
        args.output,
        args.steps,
        args.num_envs,
        args.device,
        args.controllers,
        args.seed,
      )
    else:
      data, meta = load(args.input)
      if args.command == "train":
        from .model import train

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        print(
          json.dumps(
            train(
              data,
              manifest,
              meta,
              args.output,
              args.epochs,
              args.batch,
              args.device,
              geometry_x=args.geometry_x,
              geometry_mu=args.geometry_mu,
            ),
            indent=2,
          )
        )
      else:
        result = (
          label_data(data, meta)
          if args.command == "label"
          else extend(data, meta, args.count, args.seed)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, result, meta)
        print(f"Saved {len(result['x'])} rows to {args.output}")


if __name__ == "__main__":
  main()
