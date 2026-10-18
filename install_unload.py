"""Install the standalone unloading task into mjlab 1.6.0."""

import shutil
from pathlib import Path

import mjlab

root = Path(__file__).parent
tasks = Path(mjlab.__file__).parent / "tasks"
shutil.copytree(
  root / "head_unload", tasks / "head_unload", dirs_exist_ok=True,
  ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)
