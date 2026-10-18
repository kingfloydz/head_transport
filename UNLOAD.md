# G1 depth-vision unloading

This task uses **mjlab 1.6.0** and the reference repository's 29-DoF G1,
head platform, fitted foot capsules, and Unitree torque-speed actuators.
It trains one PPO policy without stages, a teacher, motion templates,
fixed-foot constraints, or a payload stability reward.

## Install and run

Use Python 3.10–3.13 and a CUDA-enabled PyTorch installation for GPU training.
The pinned mjlab package supplies its matching MuJoCo/Warp and RSL-RL dependencies.

```bash
python -m pip install -r requirements-unload.txt
python install_unload.py

MUJOCO_GL=disable python -m mjlab.scripts.train Mjlab-Unload-Unitree-G1 \
  --env.scene.num-envs 256 --agent.logger tensorboard

MUJOCO_GL=disable python -m mjlab.scripts.play Mjlab-Unload-Unitree-G1 \
  --checkpoint-file /path/to/model_1000.pt --num-envs 1 --viewer viser
```

`MUJOCO_GL=disable` disables OpenGL, not the Warp depth renderer.
The Viser viewer displays the depth camera. The default runner saves PPO
checkpoints; the actor also implements a two-input ONNX export interface.
Existing temporal transport checkpoints have different observations and network
dimensions: they are not directly resumable as unloading checkpoints.

## Scene

- Uniform 6 kg cube, side 0.3 m, centered on the platform at reset. Its principal
  inertias are 0.09 kg m². Initial joint/root/payload velocities are zero.
- Fixed 1 m-high tabletop, 1.2 m wide and 1 m deep, 5 cm thick, with four legs.
- Robot faces +X; tabletop near edge is in front of the robot. No yaw or lateral
  randomization. Each reset samples **root-to-near-edge distance U(0.3, 1.7) m**.
  The implementation moves the fixed-base table's mocap pose only at reset.
- Box/platform and box/table sliding friction: **0.6**. Foot/ground sliding
  friction remains fixed at **0.3**. No other domain randomization.
- Camera optical center: torso-relative `(0.135, 0, 0.35)` m, ahead of the platform,
  looking forward and 25° down. Vertical FOV 60°, depth 96×64, valid range 0.1–4 m.
  The camera housing has collision geometry and 0.1 kg mass.
- Physics 200 Hz, actions and depth 50 Hz; episode limit 15 s.
- Box, platform, robot, table, and floor can collide. Hands may assist; there is
  no requirement to slide rather than tip, and no prescribed final cube face.

The camera mount/FOV are explicit simulation design values, not a calibration
of an existing physical camera. The box on top can be outside its field of view;
the actor does not receive a hidden box pose to compensate for this.

## Networks and observations

Actor: depth plus validity mask `[2,64,96]` → CNN channels 16/32/32 → pooled
64-D feature. Concatenate with 93-D proprioception → MLP 256/256/128 → 29
joint-position actions. Proprioception is angular velocity, projected gravity,
relative joint positions/velocities, and previous action. Only proprioception
uses running empirical normalization. Depth is divided by 4 m; invalid pixels
are zero with a separate mask. There is no recurrence or observation noise.

Critic: MLP 256/256/128. Its 131-D state contains proprioception plus root linear
velocity; platform-relative cube position and 6D orientation; cube linear and
angular velocities expressed in the root frame; root-relative table edge and
edge direction; box-to-goal vector; contact flags (platform, tabletop, floor,
robot, non-foot/floor, robot/table, and each foot); the two vertical support
force magnitudes divided by box weight; and the continuous-success timer.
Fixed mass, size, and friction are not repeated as observations. The critic
does not encode depth images. All privileged state is excluded from the actor.

## Reward and termination

The goal is 0.35 m inside the table's near edge, at its lateral center, with
cube center height 1.15 m. Let `D` be the cube-center distance to this goal:

```
r = 5 * (D_previous - D_current)
    + 20 * success - 10 * failure
    - dt * (0.1 * mean((a - a_previous)^2)
            + 0.05 * mean((torque / effort_limit)^2) + 0.05)
```

Reward-manager timestep scaling is disabled: progress and terminal rewards are
per-step/event values, and only running control/time costs are multiplied by dt.

Success must hold for 1 s: the entire projected cube fits within the tabletop
with 2 cm margin, the tabletop carries at least 80% of its weight, the cube has
no robot contact, its linear/angular speeds are below 0.03 m/s and 0.1 rad/s,
at least one foot contacts the floor, the robot does not contact the table, and
no failure condition holds. Any final cube face is accepted.

Failure: non-foot robot/floor contact, root height below 0.25 m, cube/floor
contact, or robot/cube horizontal displacement more than 4 m from the origin.
Timeout is a truncation without the failure penalty. Losing platform contact,
leaning, stepping, and temporary robot/table contact do not themselves terminate
the episode. Success and failure terminate immediately after their conditions
are met, so terminal bonuses are awarded once.

Implementation: `head_unload/asset.py`, `env_cfg.py`, `mdp.py`, `networks.py`.

## Validation

CPU smoke run with mjlab 1.6.0, MuJoCo / MuJoCo-Warp 3.11.0, Warp 1.15.0,
RSL-RL 5.4.2, and PyTorch 2.13.0: two environments, depth rendering, partial
reset, one PPO update with CNN gradients, actor/critic input isolation,
TorchScript export, and checkpoint save/load passed. Camera rays see the table
without platform/housing occlusion. A physical table-support fixture triggers
success after one stable second; cube/floor contact triggers failure.
This verifies the implementation, not a learned unloading policy. GPU training,
convergence, and hardware deployment have not been evaluated.
