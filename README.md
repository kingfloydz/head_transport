```bash
git clone https://github.com/kingfloydz/head_transport.git
cd head_transport
python3 install.py
```

```bash
git fetch origin
git switch main
git pull --ff-only
python3 install.py
```

```bash
python3 analyze_checkpoint.py /path/to/model_1000.pt \
  --num-envs 16394 --stage 1 --episodes-per-env 1
```

```bash
MUJOCO_GL=disable python3 -m mjlab.scripts.train \
  Mjlab-Velocity-HeadLoad-Unitree-G1 \
  --env.payload-stage 1 \
  --agent.logger tensorboard
```

```bash
MUJOCO_GL=disable python3 -m mjlab.scripts.play \
  Mjlab-Velocity-HeadLoad-Unitree-G1 \
  --checkpoint-file /path/to/model_1000.pt \
  --num-envs 1 --viewer viser
```

```bash
HEAD_LOAD_STAGE=4 MUJOCO_GL=disable python3 -m mjlab.scripts.play \
  Mjlab-Velocity-HeadLoad-Unitree-G1 \
  --checkpoint-file /path/to/model_1000.pt \
  --num-envs 1 --viewer viser
```

The stability reward uses the current bottom-face/tray intersection and diamond
friction, including yaw constraints. Full coverage uses 26 analytic square
facets precomputed per reset. Parallelograms, including rotated rectangles and
affine projections of tilted rectangular faces, use their analytic support
function (center term plus absolute projections of two half-edges). Trapezoids
remain on the general path. General polygons use
finite support directions and Fourier--Motzkin elimination in batched Torch,
without CPU geometry transfers or a convex-hull library. Geometry rebuilds use batches of 512.
Intersection construction clips against four tray half-planes in fixed
5/6/7/8-vertex buffers. Four source edge directions generate candidates;
simultaneously exposed actual edges filter the reference facet normals.
Candidate and bound compaction uses fixed capacities. One scalar count is read
before final facet filtering; no geometry is transferred to the CPU.
Final facet filtering is retained because redundant rows can change negative
margin penalties. Square, parallelogram and general-polygon constraints use
separate buffers, with fixed polygon capacities of 156 and 812 facets.

The physical simulation runs at 200 Hz and control at 50 Hz. Stability is
sampled at 100 Hz (substeps 2 and 4), including contact validity, and the two
penalties are averaged for each control step. This changes temporal sampling
and can miss short events between samples. Sampling phase is global and is
not restarted by individual environment resets. CUDA execution compiles the
contact-area, clipping, margin-evaluation and temporal-encoder kernels.

`StabilitySensorCfg` defaults: acceleration scale `1 m/s^2`, angular acceleration
scale `10 rad/s^2`, target margin `1`, softplus temperature `0.5`. Reward weight
is `-0.1`. Geometry reuse defaults to `geometry_tolerance=1e-4 m`; a positive
tolerance compares projected vertices against the last rebuild and introduces
a geometric approximation. Denominators use the current COM and expressed inertia at each sampled step.
Body-frame inertia is cached at reset.
Empty support regions and insufficient contact support receive fixed cost `10`.
Contact support is rejected when the convex hull of unmerged contact points,
projected into the tray plane, has area below `minimum_contact_area=1e-6 m^2`.
Zero, one, two or collinear points have zero area. This support-area criterion
replaces corner-height detection; it does not identify every possible tilt.
The hull area is computed in batched Torch. Evaluation uses float64.

Payload reset applies a `-0.1 mm` translation along the tray normal to reduce
contact-establishment transients. This does not alter the soft-contact parameters.

After installation, run the mathematical regression checks with:

```bash
python3 -m unittest discover -s tests
```
