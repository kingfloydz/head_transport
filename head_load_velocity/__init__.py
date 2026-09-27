"""G1 velocity tracking with a free payload and a temporal actor."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import head_load_velocity_env_cfg

rl_cfg = unitree_g1_ppo_runner_cfg()
rl_cfg.experiment_name = "g1_head_load_property"
rl_cfg.obs_groups["actor"] = ("actor", "teacher_dynamic", "teacher_static")
rl_cfg.actor.class_name = "mjlab.tasks.head_load_velocity.networks:TemporalActor"

register_mjlab_task(
  task_id="Mjlab-Velocity-HeadLoad-Unitree-G1",
  env_cfg=head_load_velocity_env_cfg(),
  play_env_cfg=head_load_velocity_env_cfg(play=True),
  rl_cfg=rl_cfg,
  runner_cls=VelocityOnPolicyRunner,
)
