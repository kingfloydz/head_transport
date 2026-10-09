import unittest

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from mjlab.tasks.head_load_velocity.stability import (
  margin_denominator,
  polygon_constraints,
  square_constraints,
  support_vertices,
)
from mjlab.tasks.head_load_velocity.contact_geometry import (
  rectangle_constraints,
  contact_hull_area,
)


def torch_polygon(p):
  points = torch.zeros(1, 8, 2, dtype=torch.float64)
  points[0, : len(p)] = torch.as_tensor(p)
  valid = torch.arange(8)[None] < len(p)
  H, mask = polygon_constraints(points, valid, torch.ones(1, dtype=torch.float64))
  return H[0, mask[0]].numpy()


class StabilityTests(unittest.TestCase):
  def test_mixed_polygon_batch_matches_individual_results(self):
    polygons = [
      np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]),
      np.array([[-0.1, -0.1], [0.08, -0.1], [0.1, 0.02], [0.03, 0.1], [-0.1, 0.08]]),
      np.array([[-0.08, -0.05], [0.09, -0.03], [0.02, 0.08]]),
    ]
    p = torch.zeros(3, 8, 2, dtype=torch.float64)
    valid = torch.zeros(3, 8, dtype=torch.bool)
    for i, poly in enumerate(polygons):
      p[i, : len(poly)] = torch.tensor(poly)
      valid[i, : len(poly)] = True
    H, hv = polygon_constraints(p, valid, torch.ones(3, dtype=torch.float64))
    for i, poly in enumerate(polygons):
      self.assert_same_facets(H[i, hv[i]].numpy(), torch_polygon(poly))

  def assert_same_facets(self, A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    self.assertLess(np.min(np.linalg.norm(A[:, None] - B, axis=-1), axis=1).max(), 1e-8)
    self.assertLess(np.min(np.linalg.norm(B[:, None] - A, axis=-1), axis=1).max(), 1e-8)

  def test_rectangles_both_axes_and_square(self):
    for half in [[0.12, 0.06], [0.04, 0.11], [0.1, 0.1]]:
      center = np.array([0.015, -0.02])
      mu = 0.63
      p = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half + center
      H = rectangle_constraints(
        torch.tensor([half], dtype=torch.float64),
        torch.tensor(center[None]),
        torch.tensor([mu], dtype=torch.float64),
      )[0].numpy()
      reference = torch_polygon(p)
      reference[:, [0, 1, 5]] /= mu
      # The square limit has redundant copies of otherwise distinct facets.
      rays = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]])
      G = np.array([np.r_[f, np.cross(np.r_[v, 0], f)] for v in p for f in rays])
      self.assertLess((H @ G.T).max(), 1e-10)
      rng = np.random.default_rng(5)
      w = rng.normal(size=(1000, 6))
      w[:, 2] = abs(w[:, 2]) + 1
      w[:, 3:] *= 0.1
      np.testing.assert_array_equal(
        (H @ w.T <= 1e-10).all(0), (reference @ w.T <= 1e-10).all(0)
      )

  def test_rotated_rectangle_simplified_directions(self):
    theta = 0.37
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p = np.array([[-0.12, -0.06], [0.12, -0.06], [0.12, 0.06], [-0.12, 0.06]]) @ R.T
    points = torch.zeros(1, 8, 2, dtype=torch.float64)
    points[0, :4] = torch.tensor(p)
    mask = torch.arange(8)[None] < 4
    H, hv = polygon_constraints(
      points, mask, torch.ones(1, dtype=torch.float64), rectangle=True
    )
    self.assert_same_facets(H[0, hv[0]].numpy(), torch_polygon(p))

  def test_random_polygons_against_independent_hull(self):
    rng = np.random.default_rng(7)
    for _ in range(12):
      cloud = rng.uniform(-0.1, 0.1, size=(8, 2))
      p = cloud[ConvexHull(cloud).vertices]
      d = np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]])
      G = np.array([np.r_[f, np.cross(np.r_[v, 0], f)] for v in p for f in d])
      eq = ConvexHull(G[:, [0, 1, 3, 4, 5]]).equations[:, [0, 1, 5, 2, 3, 4]]
      self.assert_same_facets(torch_polygon(p), eq)

  def test_contact_hull_area_degenerate_and_interior_points(self):
    p = (
      torch.tensor(
        [[[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.5, 0], [0, 0], [0, 0]]],
        dtype=torch.float64,
      )
      .expand(5, -1, -1)
      .clone()
    )
    valid = torch.ones(5, 8, dtype=torch.bool)
    valid[1] = False
    valid[2, 2:] = False
    p[3, :, 1] = 0
    p[4] *= 1e-4
    torch.testing.assert_close(
      contact_hull_area(p, valid), torch.tensor([1, 0, 0, 0, 1e-8], dtype=torch.float64)
    )

  def test_contact_hull_area_random_points(self):
    rng = np.random.default_rng(21)
    p = rng.uniform(-0.1, 0.1, (20, 8, 2))
    actual = contact_hull_area(
      torch.tensor(p), torch.ones(20, 8, dtype=torch.bool)
    ).numpy()
    np.testing.assert_allclose(
      actual, [ConvexHull(x).volume for x in p], rtol=1e-12, atol=1e-14
    )

  def test_square_matches_general_facets(self):
    p = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
    A, B = square_constraints(0.1), torch_polygon(p)
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
    H = torch_polygon(p)
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
    H = torch_polygon(p)
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
