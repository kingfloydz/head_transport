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

The stability reward uses the current projected bottom-face/tray intersection
and diamond friction, including yaw constraints. Full coverage uses 26 analytic
square facets precomputed per reset; changing partial overlaps use SciPy's
convex-hull conversion. Wrench and margin evaluation are batched in float64.
The projected support region is an approximation when the payload tilts.

`StabilitySensorCfg` defaults: acceleration scale `1 m/s^2`, angular acceleration
scale `10 rad/s^2`, target margin `1`, softplus temperature `0.5`. Reward weight
is `-0.1`. Geometry reuse is exact by default (`geometry_tolerance=0`); a positive
tolerance compares projected vertices against the last rebuild and introduces
a geometric approximation. Denominators are reused only when geometry, COM
and expressed inertia are unchanged. Empty support regions receive cost `10`.

After installation, run the mathematical regression checks with:

```bash
python3 -m unittest discover -s tests
```
