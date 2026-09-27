"""G1 velocity tracking while balancing a free, randomized payload."""

import os
from copy import deepcopy
from typing import cast

from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .asset import get_head_load_robot_cfg, get_payload_cfg
from .curriculum import PayloadMassUpper
from .events import reset_payload, reset_payload_size
from .mdp import payload_lost_contact, payload_properties, payload_state
from .networks import HISTORY_LENGTH
from .torque_speed import TorqueSpeedActuatorCfg


def head_load_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_head_load_robot_cfg()
  cfg.scene.entities["payload"] = get_payload_cfg()
  articulation = cast(
    EntityArticulationInfoCfg, cfg.scene.entities["robot"].articulation
  )
  action = cast(JointPositionActionCfg, cfg.actions["joint_pos"])
  action.scale = {}
  for actuator_cfg in articulation.actuators:
    actuator = cast(TorqueSpeedActuatorCfg, actuator_cfg)
    action.scale[actuator.target_names_expr[0]] = (
      0.25 * actuator.effort_limit / actuator.stiffness
    )
  cfg.scene.num_envs = 4096
  cfg.episode_length_s = 20.0
  del cfg.observations["actor"].terms["base_lin_vel"]
  cfg.observations["actor"] = deepcopy(cfg.observations["actor"])
  cfg.observations["actor"].history_length = HISTORY_LENGTH + 1
  cfg.observations["actor"].flatten_history_dim = False
  cfg.observations["teacher_dynamic"] = ObservationGroupCfg(
    terms={"base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel)},
    history_length=HISTORY_LENGTH + 1,
    flatten_history_dim=False,
  )
  cfg.observations["teacher_static"] = ObservationGroupCfg(
    terms={"payload_properties": ObservationTermCfg(func=payload_properties)},
  )
  cfg.scene.sensors += (
    ContactSensorCfg(
      name="payload_platform_contact",
      primary=ContactMatch(mode="geom", pattern="payload_collision", entity="payload"),
      secondary=ContactMatch(
        mode="geom", pattern="head_platform_collision", entity="robot"
      ),
      fields=("found", "force"),
      reduce="netforce",
      track_air_time=True,
    ),
  )
  cfg.observations["critic"].terms["payload_state"] = ObservationTermCfg(
    func=payload_state,
    params={
      "platform_cfg": SceneEntityCfg("robot", site_names=("head_platform",)),
      "sensor_name": "payload_platform_contact",
    },
  )
  cfg.terminations["payload_lost_contact"] = TerminationTermCfg(
    func=payload_lost_contact,
    params={"sensor_name": "payload_platform_contact"},
  )

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
  cfg.curriculum = {"payload_mass_upper": CurriculumTermCfg(func=PayloadMassUpper)}
  cfg.rewards["track_linear_velocity"].weight = 5.0
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=-1e-5,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=".*")},
  )

  cfg.events = {
    "reset_base": cfg.events["reset_base"],
    "reset_robot_joints": cfg.events["reset_robot_joints"],
    "payload_size": EventTermCfg(
      func=reset_payload_size,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("payload", geom_names=("payload_collision",)),
      },
    ),
    "payload_mass": EventTermCfg(
      func=reset_payload,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("payload", geom_names=("payload_collision",)),
        "platform_cfg": SceneEntityCfg("robot", site_names=("head_platform",)),
      },
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
