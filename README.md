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
  --num-envs 16394 --stage 2 --episodes-per-env 1
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
HEAD_LOAD_STAGE=5 MUJOCO_GL=disable python3 -m mjlab.scripts.play \
  Mjlab-Velocity-HeadLoad-Unitree-G1 \
  --checkpoint-file /path/to/model_1000.pt \
  --num-envs 1 --viewer viser
```
