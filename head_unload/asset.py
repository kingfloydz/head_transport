"""Fixed payload, table, and front-mounted depth camera for the reference G1."""

import math
from dataclasses import replace

import mujoco
from mjlab.entity import EntityCfg

from .robot import (
  FOOT_PATTERN,
  FOOT_SOLREF,
  PAYLOAD_SOLREF,
  get_head_load_robot_cfg,
  get_spec,
)

BOX_HALF = 0.15
BOX_MASS = 6.0
TABLE_HEIGHT = 1.0
TABLE_DEPTH = 1.0
TABLE_WIDTH = 1.2
TARGET_INSET = 0.35
FRICTION = (0.6, 0.005, 0.0001)
CAMERA_POS = (0.09, 0.0, 0.27)
# MuJoCo camera looks down local -Z, with local +Y pointing up.
_angle = math.radians(22)
CAMERA_XYAXES = (0, -1, 0, math.sin(_angle), 0, math.cos(_angle))


def robot_spec():
  spec = get_spec()
  torso = spec.body("torso_link")
  torso.add_geom(
    name="depth_camera_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(CAMERA_POS[0] - 0.02, CAMERA_POS[1], CAMERA_POS[2]),
    size=(0.015, 0.035, 0.015),
    mass=0.1, group=2, rgba=(0.12, 0.12, 0.12, 1),
  )
  camera = torso.add_camera(
    name="front_depth", pos=CAMERA_POS, fovy=60, resolution=(96, 64),
  )
  camera.alt.xyaxes = CAMERA_XYAXES
  camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
  return spec


def robot_cfg():
  cfg = get_head_load_robot_cfg()
  cfg.spec_fn = robot_spec
  cfg.collisions = (
    replace(
      cfg.collisions[0],
      contype=1, conaffinity=1, condim=3,
      priority={"head_platform_collision": 1, ".*_collision": 0},
      friction={FOOT_PATTERN: (0.3,), ".*_collision": FRICTION},
      solref=FOOT_SOLREF,
    ),
  )
  return cfg


def payload_spec():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(BOX_HALF,) * 3, mass=BOX_MASS, friction=FRICTION,
    priority=1, condim=3, solref=PAYLOAD_SOLREF,
    group=2, rgba=(0.8, 0.45, 0.15, 1),
  )
  return spec


def table_spec():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name="top", type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0, 0, TABLE_HEIGHT - 0.025),
    size=(TABLE_DEPTH / 2, TABLE_WIDTH / 2, 0.025),
    friction=FRICTION, priority=1, condim=3, group=2,
    rgba=(0.55, 0.38, 0.23, 1),
  )
  for x in (-0.43, 0.43):
    for y in (-0.53, 0.53):
      body.add_geom(
        name=f"leg_{x}_{y}", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y, 0.475), size=(0.035, 0.035, 0.475),
        friction=FRICTION, condim=3, group=2,
        rgba=(0.35, 0.25, 0.15, 1),
      )
  return spec


def payload_cfg():
  return EntityCfg(spec_fn=payload_spec,
                   init_state=EntityCfg.InitialStateCfg(pos=(0, 0, 1.5)))


def table_cfg():
  return EntityCfg(spec_fn=table_spec,
                   init_state=EntityCfg.InitialStateCfg(pos=(1.5, 0, 0)))
