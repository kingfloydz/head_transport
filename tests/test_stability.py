import unittest

import numpy as np
import torch
from scipy.optimize import linprog

from mjlab.tasks.head_load_velocity.stability import (
  margin_denominator,
  polygon_constraints,
  square_constraints,
  support_vertices,
)


class StabilityTests(unittest.TestCase):
  def test_square_matches_general_facets(self):
    p = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
    A, B = square_constraints(0.1), polygon_constraints(p)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    self.assertLess(
      np.min(np.linalg.norm(A[:, None] - B, axis=-1), axis=1).max(), 1e-12
    )
    self.assertLess(
      np.min(np.linalg.norm(B[:, None] - A, axis=-1), axis=1).max(), 1e-12
    )

  def test_general_cone_against_nonnegative_ray_lp(self):
    p = np.array([[-0.1, -0.1], [0.08, -0.1], [0.1, 0.02], [0.03, 0.1], [-0.1, 0.08]])
    H = polygon_constraints(p)
    rays = np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]])
    G = np.array([np.r_[f, np.cross(np.r_[v, 0], f)] for v in p for f in rays]).T
    rng = np.random.default_rng(42)
    for _ in range(30):
      w = rng.normal(size=6)
      w[2] = abs(w[2]) + 1
      w[3:] *= 0.1
      lp = linprog(
        np.zeros(G.shape[1]), A_eq=G, b_eq=w, bounds=(0, None), method="highs"
      )
      self.assertEqual(bool((H @ w <= 1e-10).all()), lp.success)

  def test_friction_scaling_and_reference_translation(self):
    p = np.array([[-0.06, -0.08], [0.1, -0.08], [0.1, 0.05], [-0.06, 0.05]])
    mu = 0.7
    H = polygon_constraints(p)
    H[:, [0, 1, 5]] /= mu
    rays = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]])
    G = np.array([np.r_[f, np.cross(np.r_[v, 0], f)] for v in p for f in rays])
    self.assertLess((H @ G.T).max(), 1e-12)

  def test_full_tray_and_partial_intersection(self):
    tray = torch.tensor(
      [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]], dtype=torch.float64
    )
    bottom = torch.stack((tray * 2, tray + tray.new_tensor([0.1, 0.0])))
    vertices, mask, full = support_vertices(bottom, tray)
    self.assertEqual(full.tolist(), [True, False])
    partial = vertices[1, mask[1]]
    self.assertTrue(((partial[:, 0] >= 0) & (partial[:, 0] <= 0.1)).all())
    self.assertTrue((partial[:, 1].abs() <= 0.1).all())

  def test_denominator_is_pullback_norm(self):
    H = torch.tensor(square_constraints(0.1))[None]
    c = torch.tensor([[0.02, -0.01, 0.15]], dtype=torch.float64)
    J = torch.diag(torch.tensor([0.004, 0.006, 0.003], dtype=torch.float64))[None]
    C = torch.tensor(
      [[0, -0.15, -0.01], [0.15, 0, -0.02], [0.01, 0.02, 0]], dtype=torch.float64
    )
    B = torch.cat(
      (torch.cat((torch.eye(3), -C), 1), torch.cat((C, J[0] - C @ C), 1)), 0
    )
    S = torch.diag(torch.tensor([1.0, 1.0, 1.0, 10.0, 10.0, 10.0], dtype=torch.float64))
    expected = torch.linalg.vector_norm(H[0] @ B @ S, dim=-1)
    actual = margin_denominator(H, c, J, 1.0, 10.0)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
  unittest.main()
