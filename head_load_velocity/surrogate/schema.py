"""Versioned feature convention, also embedded in frozen exports."""

from typing import Any

import numpy as np

FIELDS = [
  "d_x_minus",
  "d_x_plus",
  "d_y_minus",
  "d_y_plus",
  "d_z_minus",
  "d_z_plus",
  "mass",
  "Ixx",
  "Iyy",
  "Izz",
  "Ixy",
  "Ixz",
  "Iyz",
  "com_O_x_T",
  "com_O_y_T",
  "com_O_z_T",
  "R_BT_col0_x",
  "R_BT_col0_y",
  "R_BT_col0_z",
  "R_BT_col1_x",
  "R_BT_col1_y",
  "R_BT_col1_z",
  "a_T_x",
  "a_T_y",
  "a_T_z",
  "omega_T_x",
  "omega_T_y",
  "omega_T_z",
  "alpha_T_x",
  "alpha_T_y",
  "alpha_T_z",
  "gravity_unit_T_x",
  "gravity_unit_T_y",
  "gravity_unit_T_z",
  "mu",
  "tangent1_P_x",
  "tangent1_P_y",
]
LABELS = ["s_N", "s_mu", "s_tip", "s_yaw", "s_all"]
# Bitwise applicability reasons. No relative-velocity rejection: this model
# deliberately does not predict the effort needed to arrest existing sliding.
DOMAIN = {
  "history": 1,
  "not_coplanar": 2,
  "no_platform_contact": 4,
  "other_contact": 8,
  "contact_basis": 16,
  "geometry": 32,
  "friction": 64,
  "training_range": 128,
}
SOLVER = {"ok": 0, "outside_domain": 1, "numerical_failure": 2}


def metadata(step_dt, platform_size, top_position, gravity=9.81) -> dict[str, Any]:
  return dict(
    version=1,
    fields=FIELDS,
    labels=LABELS,
    input_dim=len(FIELDS),
    platform_size=list(platform_size),
    p=float(platform_size[0]),
    torso_reference="torso_link body origin (not COM)",
    top_position_T=list(top_position),
    R_PT=np.eye(3).tolist(),
    gravity=gravity,
    rotation_6d="first two columns of box-to-torso rotation, column-major",
    motion_frame="torso; world velocity differences rotated at current sample",
    sample_dt=step_dt,
    filter="none",
    time="last substep pre-integration derived state",
    cone="pyramidal",
    condim=3,
    tangents="actual contact tangent projected into platform xy; second = z cross first",
    extension="35+2: world-dependent pyramidal tangent cannot be inferred from gravity",
    tilt_tolerance_rad=0.035,
    bottom_gap_tolerance_m=0.005,
    basis_tolerance=0.005,
    area_tolerance_m2=1e-10,
    domain_bits=DOMAIN,
    solver_status=SOLVER,
    precision="float64 HiGHS LP; equality residual <= 1e-7",
    interpretation="rigid polyhedral distributed contact, not MuJoCo soft contact",
  )


def unpack(x):
  size = x[:6].reshape(3, 2).sum(1)
  com = (x[:6:2] - x[1:6:2]) / 2
  a, b = x[16:19], x[19:22]
  rotation = np.column_stack((a, b, np.cross(a, b)))
  ixx, iyy, izz, ixy, ixz, iyz = x[7:13]
  inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
  return size, com, rotation, inertia
