"""G1 velocity tracking with a rigid head payload."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import head_load_velocity_env_cfg

rl_cfg = unitree_g1_ppo_runner_cfg()
rl_cfg.experiment_name = "g1_head_load_velocity"

register_mjlab_task(
  task_id="Mjlab-Velocity-HeadLoad-Unitree-G1",
  env_cfg=head_load_velocity_env_cfg(),
  play_env_cfg=head_load_velocity_env_cfg(play=True),
  rl_cfg=rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)
