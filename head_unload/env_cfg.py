"""Minimal single-stage unloading environment for mjlab 1.6.0."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .asset import FOOT_SOLREF, payload_cfg, robot_cfg, table_cfg


def contact_sensor(name, primary, secondary):
  return ContactSensorCfg(
    name=name, primary=primary, secondary=secondary,
    fields=("found", "force"), reduce="netforce",
  )


def unload_env_cfg(play=False):
  robot = robot_cfg()
  box = ContactMatch(mode="geom", pattern="box_collision", entity="payload")
  whole_robot = ContactMatch(mode="subtree", pattern="pelvis", entity="robot")
  ground = ContactMatch(mode="body", pattern="terrain")
  tabletop = ContactMatch(mode="geom", pattern="top", entity="table")
  return ManagerBasedRlEnvCfg(
    decimation=4,
    episode_length_s=20.0,
    scale_rewards_by_dt=False,
    scene=SceneCfg(
      num_envs=1 if play else 256,
      env_spacing=5.0,
      terrain=TerrainEntityCfg(
        terrain_type="plane", env_spacing=5.0,
        collisions=(CollisionCfg(
          geom_names_expr=("terrain",), contype=1, conaffinity=1,
          condim=3, priority=0, friction=(0.3, 0.005, 0.0001), solref=FOOT_SOLREF,
        ),),
      ),
      entities={"robot": robot, "payload": payload_cfg(), "table": table_cfg()},
      sensors=(
        CameraSensorCfg(
          name="depth", camera_name="robot/front_depth", width=96, height=64,
          data_types=("depth",), use_textures=False, enabled_geom_groups=(0, 2),
        ),
        contact_sensor("box_platform", box, ContactMatch(
          mode="geom", pattern="head_platform_collision", entity="robot")),
        contact_sensor("box_table", box, tabletop),
        contact_sensor("box_ground", box, ground),
        contact_sensor("box_robot", box, whole_robot),
        contact_sensor("feet_ground", ContactMatch(
          mode="subtree", pattern="(left|right)_ankle_roll_link", entity="robot"), ground),
        contact_sensor("body_ground", ContactMatch(
          mode="geom", pattern=r"(?!(left|right)_foot\d+_collision$).*_collision", entity="robot"), ground),
        contact_sensor("robot_table", ContactMatch(
          mode="geom", pattern=".*_collision", entity="robot",
          exclude=("head_platform_collision",)), tabletop),
        contact_sensor("self_collision", whole_robot, whole_robot),
      ),
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
      nconmax=128, njmax=600, contact_sensor_maxmatch=512,
    ),
    viewer=ViewerConfig(entity_name="robot", body_name="torso_link", distance=3.5, elevation=-15, azimuth=135),
    actions={"joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), use_default_offset=True,
      scale={a.target_names_expr[0]: 0.25 * a.effort_limit / a.stiffness
             for a in robot.articulation.actuators},
    )},
    observations={
      "actor": ObservationGroupCfg(terms={"proprio": ObservationTermCfg(func=mdp.proprioception)}),
      "depth": ObservationGroupCfg(terms={"image": ObservationTermCfg(func=mdp.depth_image)}),
      "critic": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=mdp.privileged_state)}),
    },
    events={"reset": EventTermCfg(func=mdp.ResetScene, mode="reset")},
    rewards={
      "progress": RewardTermCfg(func=mdp.progress, weight=5.0),
      "pose": RewardTermCfg(func=mdp.pose, weight=0.3),
      "success": RewardTermCfg(func=mdp.terminal_reward, params={"name": "success"}, weight=20.0),
      "failure": RewardTermCfg(func=mdp.terminal_reward, params={"name": "failure"}, weight=-10.0),
      "joint_limits": RewardTermCfg(func=mdp.joint_limit_cost, weight=-1.0),
      "self_collision": RewardTermCfg(
        func=mdp.collision_cost, params={"sensor_name": "self_collision"}, weight=-1.0),
      "robot_table": RewardTermCfg(
        func=mdp.collision_cost, params={"sensor_name": "robot_table"}, weight=-1.0),
      "control": RewardTermCfg(func=mdp.control_cost, weight=-1.0),
    },
    terminations={
      "failure": TerminationTermCfg(func=mdp.failure),
      "success": TerminationTermCfg(func=mdp.success),
      "time_out": TerminationTermCfg(func=common.time_out, time_out=True),
    },
  )
