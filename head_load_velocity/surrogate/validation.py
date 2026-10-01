"""Requested small analytical checks and synthetic pipeline smoke run only."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.tasks.head_load_velocity.env_cfg import head_load_velocity_env_cfg

from .data import label_data, perturb, save
from .exact import label, polygon, wrench
from .model import regression_loss, train
from .online import FrozenMarginReward, RawContactFeaturesCfg, inspect_model
from .schema import unpack


def centered():
  x = np.zeros(37)
  x[:6] = 0.1
  x[6] = 5
  x[7:10] = 5 * 0.2**2 / 6
  x[15] = 0.1
  x[16:22] = [1, 0, 0, 0, 1, 0]
  x[31:34] = [0, 0, -1]
  x[34] = 0.8
  x[35:37] = [1, 0]
  return x


def verified_metadata():
  # MuJoCo Windows from_file does not handle this workspace's Unicode path.
  # Only the validation scene XML location is redirected; content/config intact.
  import mjlab.scene.scene as scene_module

  original = scene_module._SCENE_XML
  with tempfile.TemporaryDirectory(prefix="surrogate_verify_") as directory:
    path = Path(directory) / "scene.xml"
    path.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    scene_module._SCENE_XML = path
    try:
      cfg = head_load_velocity_env_cfg()
      cfg.scene.num_envs = 2
      scene = Scene(cfg.scene, device="cpu")
      model = scene.spec.compile()
      cfg.sim.mujoco.apply(model)
      return inspect_model(model, cfg.sim.mujoco.timestep * cfg.decimation)
    finally:
      scene_module._SCENE_XML = original


def runtime_smoke(output):
  """Two environments, eight zero-action steps; not a controller evaluation."""
  import mjlab.scene.scene as scene_module

  original = scene_module._SCENE_XML
  with tempfile.TemporaryDirectory(prefix="surrogate_runtime_") as directory:
    xml = Path(directory) / "scene.xml"
    xml.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    scene_module._SCENE_XML = xml
    env = None
    try:
      cfg = head_load_velocity_env_cfg()
      cfg.seed = 42
      cfg.scene.num_envs = 2
      cfg.scene.sensors += (
        RawContactFeaturesCfg(name="surrogate_features", decimation=cfg.decimation),
      )
      env = ManagerBasedRlEnv(cfg, device="cpu")
      env.reset()
      sensor = env.scene["surrogate_features"]
      rows = []
      for i in range(8):
        env.step(torch.zeros(2, env.action_manager.total_action_dim))
        rows.append((sensor.x.numpy().copy(), sensor.domain.numpy().copy()))
        if i == 0:
          assert ((sensor.domain & 1) != 0).all()
      env.reset(env_ids=torch.tensor([0]))
      env.step(torch.zeros(2, env.action_manager.total_action_dim))
      assert sensor.domain[0] & 1
      assert not sensor.domain[1] & 1
      raw = dict(
        x=np.concatenate([r[0] for r in rows]),
        domain=np.concatenate([r[1] for r in rows]),
        controller=np.full(16, "zero_action_smoke"),
        episode=np.tile([0, 1], 8),
        time_index=np.repeat(np.arange(8), 2),
        source=np.full(16, "synthetic_runtime"),
        parent=np.full(16, -1),
        terminated=np.zeros(16, bool),
        truncated=np.zeros(16, bool),
      )
      labeled = label_data(raw, sensor.meta)
      save(output / "runtime_smoke.npz", labeled, sensor.meta)
      good = labeled["solver_status"] == 0
      assert good.any() and np.isfinite(labeled["y"][good][:, [0, 1, 2, 4]]).all()
      return dict(
        samples=16,
        valid=int(good.sum()),
        numerical_failures=int((labeled["solver_status"] == 2).sum()),
        partial_reset_history_isolated=True,
        physics_unchanged=True,
      )
    finally:
      if env is not None:
        env.close()
      scene_module._SCENE_XML = original


def validate(output):
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(4)
  meta = verified_metadata()
  x = centered()
  static = label(x, meta)
  np.testing.assert_allclose(static[0][[0, 1, 2, 4]], [1, 0.8, 0.5, 1], atol=1e-8)
  assert static[1] and static[4]
  friction = x.copy()
  friction[22] = 9.81
  fy = label(friction, meta)
  assert fy[0][1] < 0 and fy[0][4] < 0 and not fy[1] and fy[3] == 0
  tip = x.copy()
  tip[34] = 3
  tip[22] = 15
  ty = label(tip, meta)
  assert ty[0][1] > 0 and ty[0][2] < 0 and ty[0][4] < 0 and not ty[1]
  yaw = x.copy()
  yaw[30] = 200
  yy = label(yaw, meta)
  assert yy[1] and yy[0][3] < 0 and yy[0][4] < 0
  shifted = x.copy()
  shifted[13:15] = [0.02, -0.01]
  shifted[16:22] = Rotation.from_euler("z", 0.7).as_matrix()[:, :2].T.ravel()
  q = polygon(shifted, meta)
  assert 3 <= len(q) <= 8
  octagon = x.copy()
  octagon[16:22] = Rotation.from_euler("z", np.pi / 4).as_matrix()[:, :2].T.ravel()
  assert len(polygon(octagon, meta)) == 8
  # Identical base 35 inputs but different actual tangent axes change the labels.
  tangent = x.copy()
  tangent[22] = 0.6 * 9.81
  t1 = label(tangent, meta)[0]
  tangent[35:37] = [2**-0.5, 2**-0.5]
  t2 = label(tangent, meta)[0]
  assert t1[1] > 0 and t2[1] < 0
  # Independent world-frame dynamics must equal the torso-frame calculation.
  rng = np.random.default_rng(42)
  transform_errors = []
  for _ in range(20):
    item = perturb(x, rng, meta)
    item[10:13] = [0.001, -0.0015, 0.002]  # Nonzero products of inertia.
    _, _, rb, ib = unpack(item)
    rt = Rotation.random(random_state=rng).as_matrix()
    a, omega, alpha = rt @ item[22:25], rt @ item[25:28], rt @ item[28:31]
    ell = rt @ np.array(meta["top_position_T"])
    r = rt @ item[13:16]
    ac = a + np.cross(alpha, ell + r) + np.cross(omega, np.cross(omega, ell + r))
    force = item[6] * (ac - rt @ item[31:34] * meta["gravity"])
    iw = rt @ rb @ ib @ rb.T @ rt.T
    torque = iw @ alpha + np.cross(omega, iw @ omega) + np.cross(r, force)
    world = np.r_[rt.T @ force, rt.T @ torque] / (
      item[6] * meta["gravity"] * np.r_[np.ones(3), np.full(3, meta["p"])]
    )
    np.testing.assert_allclose(world, wrench(item, meta), atol=1e-12)
    transform_errors.append(float(abs(world - wrench(item, meta)).max()))
  # Masking invalid yaw must eliminate both the yaw error and its gradient.
  pred = torch.zeros(2, 5, requires_grad=True)
  target = torch.tensor(np.tile(fy[0], (2, 1)), dtype=torch.float32)
  loss = regression_loss(pred, target, torch.zeros(2, dtype=torch.bool), torch.ones(5))
  loss.backward()
  assert pred.grad is not None
  assert torch.isfinite(loss) and not pred.grad[:, 3].any()
  # Affine deformation preserves a realizable central second moment and box COM.
  for _ in range(20):
    item = perturb(x, rng, meta)
    size, com, _, inertia = unpack(item)
    covariance = (np.trace(inertia) / 2 * np.eye(3) - inertia) / item[6]
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert (abs(com) < size / 2).all()
  # A small 10-identity synthetic dataset tests plumbing, NOT held-out policies.
  manifest = {
    "controllers": [
      dict(
        id=f"synthetic_{i:02}",
        checkpoint=None,
        split="train" if i < 6 else "val" if i < 8 else "test",
      )
      for i in range(10)
    ]
  }
  inputs = []
  for _ in range(10):
    for episode in range(8):
      base = x.copy()
      if episode == 7:
        base[:2] = 0.23
        base[34] = 1.1
      for _ in range(8):
        item = perturb(base, rng, meta)
        if episode == 7:
          item[34] = 1.1
        inputs.append(item)
  count = len(inputs)
  raw = dict(
    x=np.array(inputs),
    domain=np.zeros(count, dtype=np.int64),
    controller=np.repeat([c["id"] for c in manifest["controllers"]], 64),
    episode=np.tile(np.repeat(np.arange(8), 8), 10),
    time_index=np.tile(np.arange(8), 80),
    source=np.full(count, "synthetic"),
    parent=np.full(count, -1),
    terminated=np.zeros(count, bool),
    truncated=np.zeros(count, bool),
  )
  labeled = label_data(raw, meta)
  save(output / "synthetic_labels.npz", labeled, meta)
  report = train(
    labeled, manifest, meta, output / "smoke_model", epochs=3, batch=128, smoke=True
  )
  from .data import extend

  extended = extend(labeled, meta, 16, seed=13)
  assert len(extended["x"]) == len(labeled["x"]) + 16
  assert np.all(extended["solver_status"][-16:] == 0)
  for i in range(len(labeled["x"]), len(extended["x"])):
    parent = extended["parent"][i]
    assert extended["controller"][i] == labeled["controller"][parent]
    assert extended["episode"][i] == labeled["episode"][parent]
  # Export normalization and outputs round-trip through the frozen artifact.
  frozen = torch.jit.load(str(output / "smoke_model/frozen.pt"))
  assert frozen(torch.tensor(np.array(inputs[:3]), dtype=torch.float32)).shape == (3, 5)
  # The release loader must explicitly reject the synthetic demonstration model.
  from types import SimpleNamespace

  fake_env = SimpleNamespace(
    device="cpu", scene={"surrogate_features": SimpleNamespace(meta=meta)}
  )
  rejected = False
  try:
    FrozenMarginReward(
      SimpleNamespace(params={"model_path": str(output / "smoke_model/frozen.pt")}),
      fake_env,
    )
  except ValueError as error:
    rejected = "Synthetic smoke-test" in str(error)
  assert rejected
  with np.load(output / "smoke_model/split.npz") as assignment:
    keys = np.char.add(
      np.char.add(assignment["controller"], ":"), assignment["episode"].astype(str)
    )
    assert all(
      len(np.unique(assignment["split"][keys == key])) == 1 for key in np.unique(keys)
    )

  def finite_labels(values):
    return [float(v) if np.isfinite(v) else None for v in values]

  results = dict(
    physics=meta,
    static_labels=finite_labels(static[0]),
    friction_exceeded=finite_labels(fy[0]),
    tipping_exceeded=finite_labels(ty[0]),
    yaw_exceeded=finite_labels(yy[0]),
    intersection_vertices=len(q),
    octagon_vertices=8,
    transform_max_error=max(transform_errors),
    tangent_dependency=[float(t1[1]), float(t2[1])],
    yaw_mask_passed=True,
    synthetic_model_rejected_by_reward=rejected,
    runtime=runtime_smoke(output),
    model_report=report,
    limitations="Synthetic identities are NOT the ten G1 controllers; no RL reward model is released",
  )
  (output / "validation.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8"
  )
  return results
