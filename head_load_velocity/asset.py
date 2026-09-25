"""G1 mode-15 motors, spherical hand contacts, and a fixed head box."""

import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import cast

import mujoco
import trimesh

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  DAMPING_RATIO,
  G1_ARTICULATION,
  G1_XML,
  NATURAL_FREQ,
  get_g1_robot_cfg,
)
from mjlab.entity import EntityCfg

URDF_PATH = Path(__file__).parent / "assets" / "g1_29dof_mode_15.urdf"
MESH_DIR = URDF_PATH.parent / "meshes"
FOOT_PATTERN = r"^(left|right)_foot[1-4]_collision$"
HAND_PATTERN = r"^(left|right)_hand_collision$"
ARMATURE_5010 = 0.0021812
WRIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_.*",),
  stiffness=ARMATURE_5010 * NATURAL_FREQ**2,
  damping=2.0 * DAMPING_RATIO * ARMATURE_5010 * NATURAL_FREQ,
  armature=ARMATURE_5010,
)


def get_spec() -> mujoco.MjSpec:
  root = ET.parse(URDF_PATH).getroot()
  compiler = cast(ET.Element, root.find("mujoco/compiler"))
  compiler.set("meshdir", MESH_DIR.as_posix())
  compiler.set("strippath", "true")
  compiler.set("discardvisual", "false")
  compiler.set("fusestatic", "false")

  for link in root.findall("link"):
    name = link.attrib["name"]
    for i, visual in enumerate(link.findall("visual"), 1):
      visual.set("name", f"{name}_visual{i}")
    for i, collision in enumerate(link.findall("collision"), 1):
      prefix = name.replace("_ankle_roll_link", "_foot")
      collision.set("name", f"{prefix}{i}_collision")

  mesh_names = {Path(mesh.attrib["filename"]).name for mesh in root.iter("mesh")}
  spec = mujoco.MjSpec.from_string(
    ET.tostring(root, encoding="unicode"),
    assets={name: (MESH_DIR / name).read_bytes() for name in mesh_names},
  )
  spec.body("pelvis").add_freejoint(name="floating_base_joint")
  for visual in root.iter("visual"):
    spec.geom(visual.attrib["name"]).group = 2
  for collision in root.iter("collision"):
    spec.geom(collision.attrib["name"]).group = 3

  for side in ("left", "right"):
    hand_mesh = trimesh.load_mesh(MESH_DIR / f"{side}_rubber_hand.STL")
    spec.body(f"{side}_rubber_hand").add_geom(
      name=f"{side}_hand_collision",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(0.05, 0.0, 0.0),
      pos=hand_mesh.bounds.mean(axis=0).tolist(),
      mass=0.0,
      group=3,
    )

  # Reuse the official sites and built-in sensors required by flat G1 rewards.
  reference = mujoco.MjSpec.from_string(G1_XML.read_text(encoding="utf-8"))
  for body_name, site_name in (
    ("pelvis", "imu_in_pelvis"),
    ("torso_link", "imu_in_torso"),
    ("left_ankle_roll_link", "left_foot"),
    ("right_ankle_roll_link", "right_foot"),
  ):
    site = reference.site(site_name)
    spec.body(body_name).add_site(
      name=site_name,
      pos=site.pos.tolist(),
      quat=site.quat.tolist(),
      size=site.size.tolist(),
      type=site.type,
      group=5,
      rgba=site.rgba.tolist(),
    )
  for sensor in reference.sensors:
    spec.add_sensor(
      name=sensor.name,
      type=sensor.type,
      objtype=sensor.objtype,
      objname=sensor.objname,
      reftype=sensor.reftype,
      refname=sensor.refname,
    )

  # The supplied head mesh has identity visual origin in head_link.
  head_mesh = trimesh.load_mesh(MESH_DIR / "head_link.STL")
  payload = spec.body("head_link").add_body(
    name="head_payload", pos=(0.0, 0.0, float(head_mesh.bounds[1, 2]) + 0.15)
  )
  payload.add_geom(
    name="head_payload_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.15, 0.15, 0.15),
    mass=1.0,
    group=2,
    rgba=(0.8, 0.45, 0.15, 1.0),
  )
  return spec


def get_head_load_robot_cfg() -> EntityCfg:
  cfg = get_g1_robot_cfg()
  cfg.spec_fn = get_spec
  joints = ET.parse(URDF_PATH).getroot().findall("joint[@type='revolute']")
  cfg.articulation = replace(
    G1_ARTICULATION,
    actuators=tuple(
      replace(
        WRIST_ACTUATOR
        if "_wrist_" in joint.attrib["name"]
        else cast(BuiltinPositionActuatorCfg, actuator),
        target_names_expr=(joint.attrib["name"],),
        effort_limit=float(cast(ET.Element, joint.find("limit")).attrib["effort"]),
      )
      for actuator in G1_ARTICULATION.actuators
      for joint in joints
      if any(
        re.fullmatch(pattern, joint.attrib["name"])
        for pattern in actuator.target_names_expr
      )
    ),
  )
  cfg.collisions = (
    replace(
      cfg.collisions[0],
      condim={FOOT_PATTERN: 3, HAND_PATTERN: 3, ".*_collision": 1},
      priority=0,
      friction={FOOT_PATTERN: (0.3,)},
    ),
  )
  return cfg
