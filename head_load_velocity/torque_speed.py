"""Unitree curves from rickshaw/vel 0e474eac, adapted on mjlab's fused PD.

Curve constants and clipping follow unitree_rl_lab 4960b847 (Apache-2.0).
See licenses/unitree_rl_lab.txt. Parallel ankle/waist curves are the
reference's nominal two-N5020 approximation, not measured PR envelopes.
"""

from dataclasses import dataclass
from typing import cast

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque
from mjlab.entity import Entity, EntityArticulationInfoCfg

N7520_14 = (22.63, 35.52, 71.0, 83.3)
N7520_22 = (14.5, 22.7, 111.0, 131.0)
N5020 = (30.86, 40.13, 24.8, 31.9)
N5010 = (27.0, 41.5, 9.5, 17.0)
PARALLEL_N5020 = (30.86, 40.13, 49.6, 63.8)
CURVES = {
  "hip_pitch": N7520_22,
  "hip_roll": N7520_22,
  "knee": N7520_22,
  "hip_yaw": N7520_14,
  "waist_yaw": N7520_14,
  "ankle_pitch": PARALLEL_N5020,
  "ankle_roll": PARALLEL_N5020,
  "waist_pitch": PARALLEL_N5020,
  "waist_roll": PARALLEL_N5020,
  "shoulder_pitch": N5020,
  "shoulder_roll": N5020,
  "shoulder_yaw": N5020,
  "elbow": N5020,
  "wrist_roll": N5020,
  "wrist_pitch": N5010,
  "wrist_yaw": N5010,
}


def mode_15_curve(joint_name: str) -> tuple[float, float, float, float]:
  name = joint_name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
  return CURVES[name]


def torque_speed_clip(effort, velocity, x1, x2, y1, y2, force_limit):
  peak = torch.where(effort * velocity > 0, y1, y2)
  fraction = ((x2 - velocity.abs()) / (x2 - x1)).clamp(0.0, 1.0)
  limit = torch.minimum(peak * fraction, force_limit)
  return torch.clamp(effort, min=-limit, max=limit)


@dataclass(kw_only=True)
class TorqueSpeedActuatorCfg(IdealPdActuatorCfg):
  curve: tuple[float, float, float, float]

  def build(self, entity: Entity, target_ids: list[int], target_names: list[str]):
    return TorqueSpeedActuator(self, entity, target_ids, target_names)


class TorqueSpeedActuator(IdealPdActuator[TorqueSpeedActuatorCfg]):
  param_names = (*IdealPdActuator.param_names, "x1", "x2", "y1", "y2")

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    super().initialize(mj_model, model, data, device)
    for name, value in zip(("x1", "x2", "y1", "y2"), self.cfg.curve, strict=True):
      setattr(self, name, torch.full_like(cast(torch.Tensor, self.force_limit), value))

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    effort = pd_torque(params["stiffness"], params["damping"], cmd)
    return torque_speed_clip(
      effort,
      cmd.vel,
      params["x1"],
      params["x2"],
      params["y1"],
      params["y2"],
      params["force_limit"],
    )


def motor_metadata(robot: Entity) -> dict[str, list | str]:
  articulation = cast(EntityArticulationInfoCfg, robot.cfg.articulation)
  by_name = {
    name: cast(TorqueSpeedActuatorCfg, actuator)
    for actuator in articulation.actuators
    for name in actuator.target_names_expr
  }
  motors = [by_name[name] for name in robot.joint_names]
  return {
    "actuator_type": "unitree_torque_speed",
    "joint_stiffness": [motor.stiffness for motor in motors],
    "joint_damping": [motor.damping for motor in motors],
    "joint_effort_limit": [motor.effort_limit for motor in motors],
    **{
      f"torque_speed_{name}": [motor.curve[i] for motor in motors]
      for i, name in enumerate(("x1", "x2", "y1", "y2"))
    },
  }
