```bash
git clone --branch load_property https://github.com/kingfloydz/head_transport.git
cd head_transport
python3 install.py
```

```bash
git fetch origin
git switch load_property
git pull --ff-only
python3 install.py
```

```bash
MUJOCO_GL=disable python3 -m mjlab.scripts.train \
  Mjlab-Velocity-HeadLoad-Unitree-G1 \
  --agent.logger tensorboard
```

```bash
MUJOCO_GL=disable python3 -m mjlab.scripts.play \
  Mjlab-Velocity-HeadLoad-Unitree-G1 \
  --checkpoint-file /path/to/model_1000.pt \
  --num-envs 1 --viewer viser
```

```bash
HEAD_LOAD_MASS_KG=20 MUJOCO_GL=disable python3 -m mjlab.scripts.play \
  Mjlab-Velocity-HeadLoad-Unitree-G1 \
  --checkpoint-file /path/to/model_1000.pt \
  --num-envs 1 --viewer viser
```
