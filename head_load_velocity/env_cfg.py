"""Official G1 flat velocity task with a fixed head payload and mass curriculum."""

import os
from typing import cast

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .asset import get_head_load_robot_cfg
from .curriculum import PayloadMassUpper, sample_payload_mass
from .rewards import CommandGatedSwingHeight, command_gated_reward


def head_load_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_head_load_robot_cfg()
  articulation = cast(
    EntityArticulationInfoCfg, cfg.scene.entities["robot"].articulation
  )
  action = cast(JointPositionActionCfg, cfg.actions["joint_pos"])
  action.scale = {}
  for actuator_cfg in articulation.actuators:
    actuator = cast(BuiltinPositionActuatorCfg, actuator_cfg)
    action.scale[actuator.target_names_expr[0]] = (
      0.25 * cast(float, actuator.effort_limit) / actuator.stiffness
    )
  cfg.scene.num_envs = 4096
  cfg.episode_length_s = 20.0
  del cfg.observations["actor"].terms["base_lin_vel"]

  cfg.commands = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.5),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
      ),
    ),
  }
  cfg.curriculum = {
    "payload_mass_upper": CurriculumTermCfg(func=PayloadMassUpper),
  }

  cfg.rewards["track_linear_velocity"].weight = 4.0

  for name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "air_time",
    "foot_clearance",
    "foot_slip",
    "soft_landing",
  ):
    reward = cfg.rewards[name]
    reward.params["reward_fn"] = reward.func
    reward.func = command_gated_reward
  cfg.rewards["foot_swing_height"].func = CommandGatedSwingHeight
  for name in (
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
  ):
    cfg.rewards[name].params["command_threshold"] = 0.0

  cfg.events = {
    "reset_base": cfg.events["reset_base"],
    "reset_robot_joints": cfg.events["reset_robot_joints"],
    "payload_mass": EventTermCfg(
      func=sample_payload_mass,
      mode="reset",
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("head_payload",))},
    ),
    "torso_mass": EventTermCfg(
      func=dr.body_mass,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": (-2.0, 2.0),
      },
    ),
    "torso_com": EventTermCfg(
      func=dr.body_com_offset,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
    "terrain_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("terrain", geom_names=("terrain",)),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "axes": [0],
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.apply_body_impulse,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "force_range": (-10.0, 10.0),
        "torque_range": (0.0, 0.0),
        "duration_s": (0.2, 0.2),
        "cooldown_s": (5.0, 8.0),
      },
    ),
  }
  if play and (mass_kg := os.getenv("HEAD_LOAD_MASS_KG")) is not None:
    cfg.events["payload_mass"].params["mass_kg"] = float(mass_kg)
  return cfg
