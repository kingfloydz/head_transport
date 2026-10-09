"""Device-resident diamond-friction wrench constraints."""

import torch


def contact_hull_area(points, valid):
  """Exact 2D convex-hull area via oriented supporting edges, in batched Torch."""
  same = (points[:, :, None] - points[:, None]).abs().amax(-1) <= 1e-12
  earlier = torch.ones(same.shape[-2:], device=points.device, dtype=torch.bool).tril(-1)
  valid = valid & ~(same & earlier & valid[:, None]).any(-1)
  count = valid.sum(1)
  center = (points * valid[..., None]).sum(1) / count.clamp_min(1)[:, None]
  p = points - center[:, None]
  edge = p[:, None] - p[:, :, None]
  delta = p[:, None, None] - p[:, :, None, None]
  cross = edge[..., None, 0] * delta[..., 1] - edge[..., None, 1] * delta[..., 0]
  dot = (edge[..., None, :] * delta).sum(-1)
  length2 = edge.square().sum(-1)
  # Reject non-supporting edges and skip collinear intermediate vertices.
  left = ((cross >= -1e-14) | ~valid[:, None, None]).all(-1)
  between = (
    (cross.abs() <= 1e-14)
    & (dot > 1e-14)
    & (dot < length2[..., None] - 1e-14)
    & valid[:, None, None]
  )
  hull_edge = (
    left & ~between.any(-1) & valid[:, :, None] & valid[:, None] & (length2 > 1e-24)
  )
  terms = p[:, :, None, 0] * p[:, None, :, 1] - p[:, :, None, 1] * p[:, None, :, 0]
  area = (terms * hull_edge).sum((1, 2)).abs() / 2
  return torch.where(count >= 3, area, 0)


def pack(values, valid):
  """Compact variable row counts within a batch, preserving row order."""
  counts = valid.sum(1)
  width = int(counts.max())
  order = torch.argsort(~valid, dim=1, stable=True)[:, :width]
  result = values.gather(1, order[..., None].expand(-1, -1, values.shape[-1]))
  return result, torch.arange(width, device=values.device)[None] < counts[:, None]


def unique_directions(u, valid):
  norms = torch.linalg.vector_norm(u, dim=-1)
  valid = valid & (norms > 1e-12)
  u = u / torch.where(valid, norms, 1)[..., None]
  same = (u[:, :, None] - u[:, None]).abs().amax(-1) <= 1e-12
  earlier = torch.ones(same.shape[-2:], device=u.device, dtype=torch.bool).tril(-1)
  duplicate = (same & earlier & valid[:, None]).any(-1)
  return pack(u, valid & ~duplicate)


def ordered_intersection(vertices, valid):
  same = (vertices[:, :, None] - vertices[:, None]).abs().amax(-1) <= 1e-12
  earlier = torch.ones(same.shape[-2:], device=vertices.device, dtype=torch.bool).tril(
    -1
  )
  valid = valid & ~(same & earlier & valid[:, None]).any(-1)
  center = (vertices * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
  delta = vertices - center[:, None]
  angle = torch.atan2(delta[..., 1], delta[..., 0]).masked_fill(~valid, float("inf"))
  order = angle.argsort(1)
  vertices = vertices.gather(1, order[..., None].expand(-1, -1, 2))
  valid = valid.gather(1, order)
  return vertices[:, :8], valid[:, :8]


def rectangle_constraints(half, center, mu):
  """32 analytic facets; quarter-turn symmetry handles either long axis."""
  swap = half[:, 1] > half[:, 0]
  a, b = half.amax(1), half.amin(1)
  z = torch.zeros_like(a)
  one = torch.ones_like(a)
  L, d = a + b, (a - b) / a
  rows = []
  for s in (-1, 1):
    for t in (-1, 1):
      rows.append(torch.stack((s * one, t * one, -one, z, z, z), -1))
      rows.append(torch.stack((s * one, z, -one, t / L, z, s * t / L), -1))
      rows.append(torch.stack((z, s * one, -one, z, t / L, s * t / L), -1))
      rows.append(torch.stack((s * d, z, -one, z, z, t / a), -1))
      rows.append(torch.stack((z, s * d, -one, z, t * d / a, s * t / a), -1))
      for t_z in (-1, 1):
        rows.append(
          torch.stack(
            (s * b / L, t * a / L, -one, s * t_z / L, t * t_z / L, t_z / L), -1
          )
        )
  for sign in (-1, 1):
    rows.append(torch.stack((z, z, -one, sign / b, z, z), -1))
    rows.append(torch.stack((z, z, -one, z, sign / a, z), -1))
  H = torch.stack(rows, 1)
  rotated = H[..., [1, 0, 2, 4, 3, 5]].clone()
  rotated[..., [0, 3]] *= -1
  H = torch.where(swap[:, None, None], rotated, H)
  H[..., [0, 1, 5]] /= mu[:, None, None]
  offset = torch.cat((center, torch.zeros_like(center[:, :1])), -1)
  H[..., :3] += torch.linalg.cross(offset[:, None], H[..., 3:])
  return H


def polygon_constraints(p, valid, mu, rectangle=False):
  """Finite support directions and Fourier--Motzkin elimination, on device.

  Facet filtering uses the rank of incident generators. No convex-hull library
  or host transfer of geometry is used. Inputs are CCW, padded to eight vertices.
  """
  n, width, _ = p.shape
  count = valid.sum(1)
  following = (torch.arange(width, device=p.device)[None] + 1) % count[:, None]
  edges = p.gather(1, following[..., None].expand(-1, -1, 2)) - p
  edges = torch.cat((edges, torch.zeros_like(edges[..., :1])), -1)
  if rectangle:
    # Opposite rectangle edges differ only by sign; +/- directions cover them.
    edges = edges[:, :2]
    edge_valid = valid[:, :2]
  else:
    edge_valid = valid
    # For two intersecting rectangles there are at most four unoriented edge
    # directions. Remove parallel/opposite edges before the quadratic pairing.
    flip = (edges[..., 0] < 0) | ((edges[..., 0] == 0) & (edges[..., 1] < 0))
    edges = torch.where(flip[..., None], -edges, edges)
    edges, edge_valid = unique_directions(edges, edge_valid)
  d = p.new_tensor([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1]])
  normals = torch.linalg.cross(edges[:, :, None], d[None, None])
  candidates = []
  for j in range(4):
    for k in range(j + 1, 4):
      candidates.append(
        torch.linalg.cross(normals[:, :, None, j], normals[:, None, :, k]).flatten(1, 2)
      )
  u = torch.cat(candidates, 1)
  uv = (edge_valid[:, :, None] & edge_valid[:, None]).flatten(1).repeat(1, 6)
  u = torch.cat((d.expand(n, -1, -1), u), 1)
  uv = torch.cat((torch.ones(n, 4, device=p.device, dtype=torch.bool), uv), 1)
  u, uv = unique_directions(torch.cat((u, -u), 1), uv.repeat(1, 2))
  r = torch.cat((p, torch.zeros_like(p[..., :1])), -1)
  moments = torch.linalg.cross(r[:, :, None], d[None, None])
  if rectangle:
    center = p[:, :4].mean(1)
    q1, q2 = (p[:, 1] - p[:, 0]) / 2, (p[:, 2] - p[:, 1]) / 2

    def project(vector):
      vector = torch.cat((vector, torch.zeros_like(vector[:, :1])), -1)
      return torch.einsum(
        "nuc,njc->nuj", u, torch.linalg.cross(vector[:, None], d[None])
      )

    # Rectangle support: center term + absolute projections of its half-edges.
    psi = project(center) + project(q1).abs() + project(q2).abs()
  else:
    psi = (
      torch.einsum("nuc,nvjc->nuvj", u, moments)
      .masked_fill(~valid[:, None, :, None], -float("inf"))
      .amax(2)
    )
  gamma = (psi[..., 0] + psi[..., 1] - psi[..., 2] - psi[..., 3]) / 2
  b = torch.stack(
    (
      (psi[..., 0] - psi[..., 1]) / 2,
      (psi[..., 2] - psi[..., 3]) / 2,
      (psi[..., 2] + psi[..., 3]) / 2,
    ),
    -1,
  )
  g = torch.cat((-b, u), -1)
  positive, negative = uv & (gamma > 1e-12), uv & (gamma < -1e-12)
  zero = uv & (gamma.abs() <= 1e-12)
  bounds = g / torch.where(positive | negative, gamma, 1)[..., None]
  lower, lv = pack(bounds, positive)
  upper, vv = pack(bounds, negative)
  base_l = p.new_tensor([[1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]]).expand(n, -1, -1)
  base_v = p.new_tensor([[0, 1, 1, 0, 0, 0], [0, -1, 1, 0, 0, 0]]).expand(n, -1, -1)
  lower, upper = torch.cat((base_l, lower), 1), torch.cat((base_v, upper), 1)
  ones = torch.ones(n, 2, device=p.device, dtype=torch.bool)
  lv, vv = torch.cat((ones, lv), 1), torch.cat((ones, vv), 1)
  generators = torch.cat((d.expand(n, width, -1, -1), moments), -1).flatten(1, 2)
  gv = valid.repeat_interleave(4, 1)
  kept, masks = [], []
  # Bound memory independently of batch size and number of eliminated pairs.
  for start in range(0, lower.shape[1], 8):
    H = (lower[:, start : start + 8, None] - upper[:, None]).flatten(1, 2)
    hv = (lv[:, start : start + 8, None] & vv[:, None]).flatten(1)
    norm = torch.linalg.vector_norm(H, dim=-1)
    hv &= norm > 1e-12
    H = H / torch.where(hv, norm, 1)[..., None]
    incidence = (torch.einsum("nhc,ngc->nhg", H, generators).abs() <= 1e-10) & gv[
      :, None
    ]
    possible = hv & (incidence.sum(-1) >= 5)
    kept.append(H)
    masks.append(possible)
  H0, zvalid = pack(g, zero)
  H = torch.cat((*kept, H0), 1)
  hv = torch.cat((*masks, zvalid), 1)
  H, hv = pack(H, hv)
  H, hv = unique_directions(H, hv)
  # Remove lower-dimensional supporting faces retained by the incidence count.
  incidence = (torch.einsum("nhc,ngc->nhg", H, generators).abs() <= 1e-10) & gv[:, None]
  face = generators[:, None] * incidence[..., None]
  rank = torch.linalg.matrix_rank(face, atol=1e-10, rtol=0)
  H, hv = pack(H, hv & (rank >= 5))
  H[..., [0, 1, 5]] /= mu[:, None, None]
  return H, hv
