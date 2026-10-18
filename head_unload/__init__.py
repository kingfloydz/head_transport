"""Depth-guided G1 unloading with an asymmetric PPO critic."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import unload_env_cfg

register_mjlab_task(
  task_id="Mjlab-Unload-Unitree-G1",
  env_cfg=unload_env_cfg(),
  play_env_cfg=unload_env_cfg(play=True),
  rl_cfg=RslRlOnPolicyRunnerCfg(
    experiment_name="g1_unload",
    obs_groups={"actor": ("actor", "depth"), "critic": ("critic",)},
    actor=RslRlModelCfg(
      class_name="mjlab.tasks.head_unload.networks:DepthActor",
      hidden_dims=(256, 256, 128), activation="elu", obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(256, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),
    num_steps_per_env=24, max_iterations=30_000, save_interval=100,
    logger="tensorboard", upload_model=False,
  ),
)
