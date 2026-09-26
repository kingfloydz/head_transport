"""Install the head-load task into the active mjlab 1.6.0 environment."""

import re
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
source = re.sub(
  r'  if task_id == "Mjlab-Velocity-HeadLoad-Unitree-G1":\n'
  r"    from mjlab.tasks.head_load_velocity.curriculum import bind_payload_curriculum\n"
  r"\s*bind_payload_curriculum\(runner\)\n(?:[ \t]*\n)*",
  "",
  train_script.read_text(encoding="utf-8"),
)
train_script.write_text(
  source.replace("init_at_random_ep_len=False", "init_at_random_ep_len=True"),
  encoding="utf-8",
)

for name in ("curriculum.py", "rewards.py"):
  (mjlab_dir / "tasks" / "head_load_velocity" / name).unlink(missing_ok=True)
