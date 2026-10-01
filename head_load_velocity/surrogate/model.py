"""Small ELU regressor, train-only statistics and auditable masked losses."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MarginMLP(nn.Module):
  x_mean: torch.Tensor
  x_scale: torch.Tensor
  y_scale: torch.Tensor

  def __init__(self):
    super().__init__()
    self.register_buffer("x_mean", torch.zeros(37))
    self.register_buffer("x_scale", torch.ones(37))
    self.register_buffer("y_scale", torch.ones(5))
    self.network = nn.Sequential(
      nn.Linear(37, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 5)
    )

  def forward(self, x):
    return self.network((x - self.x_mean) / self.x_scale) * self.y_scale


def regression_loss(
  pred, target, yaw_valid, scale, boundary_width=0.1, over_weight=2.0
):
  mask = torch.isfinite(target)
  mask[:, 3] &= yaw_valid
  clean = torch.where(mask, target, 0.0)
  weight = 1 + 4 * torch.exp(-clean.abs() / boundary_width)
  residual = (pred - clean) / scale
  losses = F.huber_loss(residual, torch.zeros_like(residual), reduction="none")
  base = ((losses * weight * mask).sum(0) / mask.sum(0).clamp_min(1)).mean()
  over = F.relu(residual[:, 4]).square() * weight[:, 4] * mask[:, 4]
  return base + over_weight * over.sum() / mask[:, 4].sum().clamp_min(1)


def split_rows(data, manifest, geometry_x=0.40, geometry_mu=0.9):
  mapping = {c["id"]: c["split"] for c in manifest["controllers"]}
  split = np.array([mapping[str(c)] for c in data["controller"]], dtype="U16")
  # Move entire episodes, including all descendants, to geometric holdout.
  held = (data["x"][:, 0] + data["x"][:, 1] > geometry_x) & (
    data["x"][:, 34] > geometry_mu
  )
  keys = np.char.add(
    np.char.add(data["controller"].astype(str), ":"), data["episode"].astype(str)
  )
  held_keys = np.unique(keys[held & (split == "train")])
  split[(split == "train") & np.isin(keys, held_keys)] = "geometry_test"
  return split


def metrics(pred, y, yaw_valid):
  if not len(y):
    return {"n": 0, "status": "no samples; unverified"}
  mask = np.isfinite(y)
  mask[:, 3] &= yaw_valid
  mae = [
    float(np.abs(pred[mask[:, i], i] - y[mask[:, i], i]).mean())
    if mask[:, i].any()
    else None
    for i in range(5)
  ]
  error = pred[:, 4] - y[:, 4]
  boundary = abs(y[:, 4]) <= 0.1
  unsafe = y[:, 4] < -1e-8
  return dict(
    n=len(y),
    mae=mae,
    boundary_n=int(boundary.sum()),
    boundary_mae=float(abs(error[boundary]).mean()) if boundary.any() else None,
    overestimate_mean=float(np.maximum(error, 0).mean()),
    overestimate_p95=float(np.quantile(np.maximum(error, 0), 0.95)),
    unsafe_n=int(unsafe.sum()),
    safe_n=int((~unsafe).sum()),
    false_unsafe_given_safe=float((pred[~unsafe, 4] < 0).mean())
    if (~unsafe).any()
    else None,
    label_counts=mask.sum(0).tolist(),
    false_safe_given_unsafe=float((pred[unsafe, 4] >= 0).mean())
    if unsafe.any()
    else None,
  )


def benchmark(model, device, batch=4096, repeats=100):
  x = torch.zeros(batch, 37, device=device)
  with torch.inference_mode():
    for _ in range(10):
      model(x)
    if str(device).startswith("cuda"):
      torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
      model(x)
    if str(device).startswith("cuda"):
      torch.cuda.synchronize()
  return dict(
    device=str(device),
    batch=batch,
    repeats=repeats,
    milliseconds=(time.perf_counter() - start) * 1000 / repeats,
    scope="normalization + MLP only; excludes feature/contact gathering",
    cpu_threads=torch.get_num_threads(),
  )


def train(
  data,
  manifest,
  physics,
  output,
  epochs=50,
  batch=1024,
  device="cpu",
  seed=42,
  smoke=False,
  geometry_x=0.40,
  geometry_mu=0.9,
):
  torch.manual_seed(seed)
  np.random.seed(seed)
  groups = [c["split"] for c in manifest["controllers"]]
  if [groups.count(k) for k in ("train", "val", "test")] != [6, 2, 2]:
    raise ValueError("Manifest must have 6/2/2 controller groups")
  split = split_rows(data, manifest, geometry_x, geometry_mu)
  good = (
    (data["domain"] == 0) & (data["solver_status"] == 0) & np.isfinite(data["y"][:, 4])
  )
  masks = {k: (split == k) & good for k in ("train", "val", "test", "geometry_test")}
  if any(not masks[k].any() for k in ("train", "val", "test")):
    raise ValueError("Missing usable train/validation/test controller samples")
  present = set(data["controller"][good])
  if not smoke and any(c["id"] not in present for c in manifest["controllers"]):
    raise ValueError("All ten controllers require valid labels before release training")
  if not smoke and any(
    not (good & (data["source"] == "rollout") & (data["controller"] == c["id"])).any()
    for c in manifest["controllers"]
  ):
    raise ValueError("Release training requires real rollouts from every controller")
  model = MarginMLP().to(device)
  train_x = data["x"][masks["train"]]
  train_y = data["y"][masks["train"]].copy()
  train_y[~data["yaw_valid"][masks["train"]], 3] = np.nan
  model.x_mean.copy_(torch.tensor(train_x.mean(0), device=device))
  model.x_scale.copy_(torch.tensor(np.maximum(train_x.std(0), 1e-4), device=device))
  scale = np.nanstd(train_y, axis=0)
  scale[~np.isfinite(scale)] = 1.0
  model.y_scale.copy_(torch.tensor(np.maximum(scale, 0.05), device=device))
  x = torch.tensor(data["x"], dtype=torch.float32, device=device)
  y = torch.tensor(data["y"], dtype=torch.float32, device=device)
  valid = torch.tensor(data["yaw_valid"], device=device)
  indices = torch.tensor(np.flatnonzero(masks["train"]), device=device)
  val_indices = torch.tensor(np.flatnonzero(masks["val"]), device=device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  best = float("inf")
  best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
  for _ in range(epochs):
    model.train()
    for ids in indices[torch.randperm(len(indices), device=device)].split(batch):
      loss = regression_loss(model(x[ids]), y[ids], valid[ids], model.y_scale)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    model.eval()
    with torch.inference_mode():
      score = regression_loss(
        model(x[val_indices]), y[val_indices], valid[val_indices], model.y_scale
      ).item()
    if score < best:
      best = score
      best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
  model.load_state_dict(best_state)
  model.eval()
  report = {}
  with torch.inference_mode():
    for key, mask in masks.items():
      pred = model(x[mask]).cpu().numpy()
      report[key] = metrics(pred, data["y"][mask], data["yaw_valid"][mask])
    for controller in manifest["controllers"]:
      mask = good & (data["controller"] == controller["id"])
      report[controller["id"]] = metrics(
        model(x[mask]).cpu().numpy(), data["y"][mask], data["yaw_valid"][mask]
      )
  frozen = torch.jit.freeze(torch.jit.script(model))
  report["benchmark"] = benchmark(frozen, device)
  report["synthetic_smoke_only"] = smoke
  report["dataset_status_counts"] = {
    str(k): int((data["solver_status"] == k).sum()) for k in (0, 1, 2)
  }
  package = dict(
    physics=physics,
    release_ready=not smoke,
    manifest=manifest,
    geometry_holdout=dict(size_x_above=geometry_x, mu_above=geometry_mu),
    normalization="train episodes only; per-feature mean/std, per-label std",
    x_mean=model.x_mean.cpu().tolist(),
    x_scale=model.x_scale.cpu().tolist(),
    y_scale=model.y_scale.cpu().tolist(),
    feature_min=train_x.min(0).tolist(),
    feature_max=train_x.max(0).tolist(),
    inference_range="train min/max plus 1% span + 1e-6; not a support guarantee",
    loss="masked scaled Huber; boundary weighting; extra positive s_all error squared",
    caveat="asymmetric loss is not a conservative guarantee",
    report=report,
  )
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  torch.jit.save(
    frozen,
    str(output / "frozen.pt"),
    _extra_files={"metadata.json": json.dumps(package)},
  )
  (output / "metadata.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
  (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
  np.savez_compressed(
    output / "split.npz",
    controller=data["controller"],
    episode=data["episode"],
    split=split,
  )
  return report
