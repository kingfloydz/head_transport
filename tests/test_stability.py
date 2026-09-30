"""Analytical physics and substep/reset regressions for the stability reward."""

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab.tasks.head_load_velocity.stability import (
  StabilitySensor,
  stability_barrier,
  stability_margins,
)
from mjlab.utils.lab_api.math import matrix_from_quat


def nominal():
  return dict(
    rotation=torch.eye(3)[None],
    velocity_acc=torch.zeros(1, 3),
    omega=torch.zeros(1, 3),
    alpha=torch.zeros(1, 3),
    mass=torch.tensor([10.0]),
    height=torch.tensor([0.15]),
    inertia=torch.diag(torch.tensor([0.2, 0.3, 0.4]))[None],
    edges=torch.tensor([[0.08, 0.1, 0.06, 0.09]]),
    mu=torch.tensor([0.8]),
  )


def test_static_and_free_fall():
  args = nominal()
  margins = stability_margins(**args)
  torch.testing.assert_close(margins, torch.tensor([[0.8, 1, 0.4, 0.5, 0.3, 0.45]]))
  args["velocity_acc"][:, 2] = -9.81
  free_fall = stability_margins(**args)
  torch.testing.assert_close(free_fall, torch.zeros(1, 6))
  assert stability_barrier(free_fall) > stability_barrier(margins)
  args["velocity_acc"][:, 2] = -20
  assert stability_margins(**args)[0, 1] < 0


def test_translation_signs_and_mass_normalization():
  args = nominal()
  args["velocity_acc"][0, 0] = 9.81
  expected = torch.tensor([[-0.2, 1, 1.15, -0.25, 0.3, 0.45]])
  torch.testing.assert_close(stability_margins(**args), expected)
  args["mass"] *= 3
  args["inertia"] *= 3
  torch.testing.assert_close(stability_margins(**args), expected)
  args["velocity_acc"][0] = torch.tensor([0.0, 9.81, 0.0])
  torch.testing.assert_close(
    stability_margins(**args), torch.tensor([[-0.2, 1, 0.4, 0.5, 1.05, -0.3]])
  )


def test_full_inertia_rotating_platform_against_numpy():
  args = nominal()
  rotation = np.array([[0.0, -0.8, 0.6], [1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
  inertia = np.array([[0.2, 0.03, -0.02], [0.03, 0.3, 0.04], [-0.02, 0.04, 0.4]])
  acc, omega, alpha = (
    np.array([1.0, -2.0, 3.0]),
    np.array([2.0, 3.0, 4.0]),
    np.array([4.0, -1.0, 2.0]),
  )
  for key, value in [
    ("rotation", rotation),
    ("inertia", inertia),
    ("velocity_acc", acc),
    ("omega", omega),
    ("alpha", alpha),
  ]:
    args[key] = torch.tensor(value, dtype=torch.float32)[None]
  r = rotation @ np.array([0.0, 0.0, 0.15])
  force_w = 10 * (
    acc + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r)) - [0, 0, -9.81]
  )
  iw = rotation @ inertia @ rotation.T
  torque = rotation.T @ (
    iw @ alpha + np.cross(omega, iw @ omega) + np.cross(r, force_w)
  )
  force = rotation.T @ force_w
  fn = force[2]
  expected = np.r_[
    (0.8 * fn - np.linalg.norm(force[:2])) / 98.1,
    fn / 98.1,
    (
      np.array([0.08, 0.1, 0.06, 0.09]) * fn
      + [torque[1], -torque[1], -torque[0], torque[0]]
    )
    / 19.62,
  ]
  np.testing.assert_allclose(stability_margins(**args)[0].numpy(), expected, atol=2e-6)


def make_sensor():
  model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="robot" pos="0 0 1"><freejoint/>
      <geom name="robot/head_platform_collision" type="box" size=".1 .1 .01"
            friction=".8 .005 .0001" priority="1"/>
      <site name="head_platform"/>
    </body>
    <body pos="0 0 2"><freejoint/>
      <geom name="payload/payload_collision" type="box" size=".1 .1 .1"
            friction=".6 .005 .0001" priority="1"/>
    </body>
  </worldbody></mujoco>""")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  def batch(x):
    return (
      torch.tensor(np.array(x), dtype=torch.float32)
      .unsqueeze(0)
      .repeat(2, *([1] * x.ndim))
    )

  raw = SimpleNamespace(
    site_xpos=batch(data.site_xpos),
    site_xmat=batch(data.site_xmat.reshape(-1, 3, 3)),
    subtree_com=batch(data.subtree_com),
    cvel=batch(data.cvel),
  )
  robot = SimpleNamespace(
    find_sites=lambda name: ([0], [name]),
    indexing=SimpleNamespace(site_ids=[0]),
    data=SimpleNamespace(
      data=raw,
      root_link_pos_w=torch.zeros(2, 3),
      model=SimpleNamespace(geom_friction=batch(model.geom_friction)),
    ),
  )
  sensor = StabilitySensor()
  sensor.edit_spec(None, {"robot": robot})
  sensor.initialize(model, None, None, "cpu")
  sensor.set_payload(
    torch.arange(2),
    torch.ones(2) * 10,
    torch.tensor([[0.16, 0.12, 0.3]]).repeat(2, 1),
    torch.zeros(2, 3),
    torch.eye(3)[None].repeat(2, 1, 1),
    torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
    torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
  )
  return sensor, raw


def test_reset_constants_rotation_edges_and_pair_friction():
  sensor, _ = make_sensor()
  q = torch.tensor([[2**-0.5, 0, 0, 2**-0.5]])
  inertia = torch.diag(torch.tensor([1.0, 2.0, 3.0]))[None]
  sensor.set_payload(
    torch.tensor([0]),
    torch.tensor([5.0]),
    torch.tensor([[0.16, 0.12, 0.1]]),
    torch.tensor([[0.03, -0.02, 0.01]]),
    inertia,
    q,
    torch.tensor([[1.0, 0, 0, 0]]),
  )
  torch.testing.assert_close(sensor.edges[0], torch.tensor([0.05, 0.1, 0.08, 0.04]))
  torch.testing.assert_close(sensor.height[0], torch.tensor(0.06))
  torch.testing.assert_close(
    sensor.inertia[0], torch.diag(torch.tensor([2.0, 1.0, 3.0]))
  )
  torch.testing.assert_close(sensor.mu, torch.tensor([0.8, 0.8]))


def test_substep_mean_reset_and_top_center_velocity():
  sensor, data = make_sensor()
  baseline = stability_barrier(
    stability_margins(
      torch.eye(3)[None].repeat(2, 1, 1),
      torch.zeros(2, 3),
      torch.zeros(2, 3),
      torch.zeros(2, 3),
      sensor.mass,
      sensor.height,
      sensor.inertia,
      sensor.edges,
      sensor.mu,
    )
  )
  for _ in range(4):
    sensor.update(0.005)
  torch.testing.assert_close(sensor.data, baseline * 0.75)
  torch.testing.assert_close(
    sensor.data, baseline * 0.75
  )  # Cached, not consumed twice.
  sensor.reset(torch.tensor([0]))
  data.cvel[0, sensor.body, 3] = 100  # Reset teleport must not produce acceleration.
  for _ in range(4):
    sensor.update(0.005)
  torch.testing.assert_close(sensor.data, baseline * torch.tensor([0.75, 1.0]))
  data.cvel[:, sensor.body, :] = 0
  data.cvel[:, sensor.body, 1] = 2
  sensor.reset()
  sensor.update(0.005)
  torch.testing.assert_close(sensor.previous_velocity[:, 0], torch.full((2,), 0.02))


def test_alternating_acceleration_is_penalized_before_averaging():
  sensor, data = make_sensor()
  sensor.update(0.005)
  _ = sensor.data
  values = []
  for vx in [0.1, 0.0, 0.1, 0.0]:
    previous = sensor.previous_velocity.clone()
    data.cvel[:, sensor.body, 3] = vx
    sensor.update(0.005)
    values.append(
      stability_barrier(
        stability_margins(
          torch.eye(3)[None].repeat(2, 1, 1),
          (sensor.previous_velocity - previous) / 0.005,
          torch.zeros(2, 3),
          torch.zeros(2, 3),
          sensor.mass,
          sensor.height,
          sensor.inertia,
          sensor.edges,
          sensor.mu,
        )
      )
    )
  expected = torch.stack(values).mean(0)
  torch.testing.assert_close(sensor.data, expected)
  assert (expected > 1).all()


def test_world_frame_difference_does_not_create_body_rotation_artifact():
  sensor, data = make_sensor()
  # Site translation fixed in world while orientation changes (synthetic samples).
  data.cvel[:, sensor.body, 3] = 1
  sensor.update(0.005)
  _ = sensor.data
  data.site_xmat[:] = matrix_from_quat(torch.tensor([[2**-0.5, 0, 0, 2**-0.5]]))[
    :, None
  ]
  sensor.update(0.005)
  torch.testing.assert_close(
    sensor.previous_velocity, torch.tensor([[1.0, 0, 0]]).repeat(2, 1)
  )
  assert torch.isfinite(sensor.data).all()
