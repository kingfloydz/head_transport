"""Copy the unloading task and its reference assets into mjlab 1.6.0."""

import shutil
from pathlib import Path

import mjlab

root = Path(__file__).parent
tasks = Path(mjlab.__file__).parent / "tasks"
for name in ("head_load_velocity", "head_unload"):
  shutil.copytree(
    root / name, tasks / name, dirs_exist_ok=True,
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
  )
