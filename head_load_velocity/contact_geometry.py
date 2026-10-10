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


def pack(values, valid, capacity=None):
  """Stable prefix-sum compaction; fixed capacity avoids host synchronization.

  Callers using a capacity must supply a proven upper bound on valid rows.
  Variable capacities are used only at expensive algebra stage boundaries.
  """
  counts = valid.sum(1)
  width = capacity if capacity is not None else max(1, int(counts.max()))
  index = (valid.cumsum(1) - 1).masked_fill(~valid, 0)
  result = values.new_zeros(values.shape[0], width, values.shape[-1])
  result.scatter_add_(
    1, index[..., None].expand_as(values), values.masked_fill(~valid[..., None], 0)
  )
  return result, torch.arange(width, device=values.device)[None] < counts[:, None]


def unique_directions(u, valid, capacity=None):
  norms = torch.linalg.vector_norm(u, dim=-1)
  valid = valid & (norms > 1e-12)
  u = u / torch.where(valid, norms, 1)[..., None]
  same = (u[:, :, None] - u[:, None]).abs().amax(-1) <= 1e-12
  earlier = torch.ones(same.shape[-2:], device=u.device, dtype=torch.bool).tril(-1)
  duplicate = (same & earlier & valid[:, None]).any(-1)
  return pack(u, valid & ~duplicate, capacity)


def intersect_tray(bottom, half):
  """Clip a projected rectangular face against four tray half-planes.

  Four vertices grow to at most eight. Boundary order is preserved without
  candidate-point sorting, all-pairs deduplication, or variable-size buffers.
  """
  area = (
    bottom[..., 0] * bottom.roll(-1, 1)[..., 1]
    - bottom[..., 1] * bottom.roll(-1, 1)[..., 0]
  ).sum(1)
  p = torch.where((area < 0)[:, None, None], bottom.flip(1), bottom)
  valid = torch.ones(p.shape[:2], device=p.device, dtype=torch.bool)
  for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
    count = valid.sum(1)
    next_index = (
      torch.arange(p.shape[1], device=p.device)[None] + 1
    ) % count.clamp_min(1)[:, None]
    q = p.gather(1, next_index[..., None].expand(-1, -1, 2))
    a, b = half[axis] - sign * p[..., axis], half[axis] - sign * q[..., axis]
    inside = valid & (a >= 0)
    crossing = valid & (((a > 0) & (b < 0)) | ((a < 0) & (b > 0)))
    fraction = a / torch.where(crossing, a - b, 1)
    hit = p + fraction[..., None] * (q - p)
    points = torch.stack((p, hit), 2).flatten(1, 2)
    keep = torch.stack((inside, crossing), 2).flatten(1)
    p, valid = pack(points, keep, p.shape[1] + 1)
  return p, valid


def parallelogram_mask(p, valid):
  """Opposite vertices have the same midpoint; right angles are not required."""
  residual = (p[:, 0] + p[:, 2] - p[:, 1] - p[:, 3]).abs().amax(-1)
  return (valid.sum(1) == 4) & (residual <= 1e-12)


def polygon_constraints(p, valid, mu, parallelogram=False, edge_directions=None):
  """Finite support directions and Fourier--Motzkin elimination, on device.

  Facet filtering uses the rank of incident generators. No convex-hull library
  or host transfer of geometry is used. Inputs are CCW, padded to eight vertices.
  """
  n, width, _ = p.shape
  if parallelogram:
    edges = p[:, 1:3] - p[:, :2]
    edge_valid = valid[:, :2]
  elif edge_directions is not None:
    # Every edge of an intersection belongs to one of its input polygons.
    edges = edge_directions
    edge_valid = torch.ones(edges.shape[:2], device=p.device, dtype=torch.bool)
  else:
    count = valid.sum(1)
    following = (torch.arange(width, device=p.device)[None] + 1) % count[:, None]
    edges = p.gather(1, following[..., None].expand(-1, -1, 2)) - p
    edge_valid = valid
  edges = torch.cat((edges, torch.zeros_like(edges[..., :1])), -1)
  if not parallelogram:
    # For two intersecting rectangles there are at most four unoriented edge
    # directions. Remove parallel/opposite edges before the quadratic pairing.
    flip = (edges[..., 0] < 0) | ((edges[..., 0] == 0) & (edges[..., 1] < 0))
    edges = torch.where(flip[..., None], -edges, edges)
    edges, edge_valid = unique_directions(
      edges, edge_valid, edges.shape[1] if edge_directions is not None else None
    )
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
  if parallelogram:
    center = p[:, :4].mean(1)
    q1, q2 = (p[:, 1] - p[:, 0]) / 2, (p[:, 2] - p[:, 1]) / 2

    def project(vector):
      vector = torch.cat((vector, torch.zeros_like(vector[:, :1])), -1)
      return torch.einsum(
        "nuc,njc->nuj", u, torch.linalg.cross(vector[:, None], d[None])
      )

    # P = center +/- q1 +/- q2; q1 and q2 need not be perpendicular.
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
  # Both bound families share one compaction/host count instead of two.
  packed, packed_valid = pack(
    torch.cat((bounds, bounds), 0), torch.cat((positive, negative), 0)
  )
  lower, upper = packed.chunk(2, 0)
  lv, vv = packed_valid.chunk(2, 0)
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
  H = torch.cat((*kept, g), 1)
  hv = torch.cat((*masks, zero), 1)
  H, hv = pack(H, hv)
  H, hv = unique_directions(H, hv)
  # Remove lower-dimensional supporting faces retained by the incidence count.
  incidence = (torch.einsum("nhc,ngc->nhg", H, generators).abs() <= 1e-10) & gv[:, None]
  face = generators[:, None] * incidence[..., None]
  rank = torch.linalg.matrix_rank(face, atol=1e-10, rtol=0)
  H, hv = pack(H, hv & (rank >= 5))
  H[..., [0, 1, 5]] /= mu[:, None, None]
  return H, hv
