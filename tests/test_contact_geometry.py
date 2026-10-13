"""Independent hull reference for the signed-margin constraint construction."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial import ConvexHull
import torch

spec = importlib.util.spec_from_file_location(
  "contact_geometry",
  Path(__file__).parents[1] / "head_load_velocity" / "contact_geometry.py",
)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class ContactGeometryTests(unittest.TestCase):
  def test_signed_margins_against_independent_five_dimensional_hull(self):
    torch.set_num_threads(4)
    rng = np.random.default_rng(142)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    for index in range(20):
      angle = rng.uniform(-1, 1)
      rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
      )
      bottom = corners * rng.uniform(0.04, 0.16, 2) @ rotation
      bottom += rng.uniform(-0.04, 0.04, 2)
      p, valid = geometry.intersect_tray(
        torch.tensor(bottom)[None], torch.tensor([0.1, 0.1], dtype=torch.float64)
      )
      para = bool(geometry.parallelogram_mask(p, valid)[0])
      sources = torch.cat(
        (
          torch.tensor(bottom[1:3] - bottom[:2])[None],
          torch.eye(2, dtype=torch.float64)[None],
        ),
        1,
      )
      mu = float(rng.uniform(0.2, 1.2))
      H, mask = geometry.polygon_constraints(
        p,
        valid,
        torch.tensor([mu], dtype=torch.float64),
        parallelogram=para,
        edge_directions=sources,
      )
      self.assertEqual(H.shape[1], 156 if para else 812)
      H = H[0, mask[0]].numpy()
      r = np.column_stack((p[0, valid[0]].numpy(), np.zeros(int(valid.sum()))))
      d = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]])
      generators = np.concatenate(
        (np.broadcast_to(d, (len(r), 4, 3)), np.cross(r[:, None], d)), axis=-1
      ).reshape(-1, 6)
      equations = ConvexHull(generators[:, [0, 1, 3, 4, 5]]).equations
      reference = equations[:, [0, 1, 5, 2, 3, 4]]
      B = rng.normal(size=(6, 6))
      B = B @ B.T + np.eye(6)
      w = rng.normal(size=(100, 6))
      w = np.concatenate((w, rng.uniform(size=(100, len(r) * 4)) @ generators))
      actual = (-w @ H.T / np.linalg.norm(H @ B, axis=1)).min(axis=1)
      expected = (-w @ reference.T / np.linalg.norm(reference @ B, axis=1)).min(axis=1)
      np.testing.assert_allclose(
        actual, expected, atol=1e-8, rtol=1e-8, err_msg=f"overlap {index}"
      )


if __name__ == "__main__":
  unittest.main()
