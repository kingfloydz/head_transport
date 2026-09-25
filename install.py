"""Install the head-load task into the active mjlab 1.6.0 environment."""

import shutil
from pathlib import Path

import mjlab

task_dir = Path(__file__).parent / "head_load_velocity"
mjlab_dir = Path(mjlab.__file__).parent
shutil.copytree(
  task_dir,
  mjlab_dir / "tasks" / "head_load_velocity",
  dirs_exist_ok=True,
  ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)

train_script = mjlab_dir / "scripts" / "train.py"
train_script.write_text(
  train_script.read_text(encoding="utf-8").replace(
    "init_at_random_ep_len=True", "init_at_random_ep_len=False"
  ),
  encoding="utf-8",
)
