# G1 头顶固定负载速度跟踪

适用于已经安装 **mjlab 1.6.0** 及其依赖的 Linux / NVIDIA GPU 服务器。
仓库仅包含任务代码、机器人资源和安装脚本，不包含 mjlab 源码或 Python 环境。

## 安装与训练

先激活服务器已有的 mjlab 环境，再执行：

```bash
git clone https://github.com/kingfloydz/head_transport.git
cd head_transport
python install.py
python -m mjlab.scripts.train Mjlab-Velocity-HeadLoad-Unitree-G1
```

`install.py` 将 `head_load_velocity/` 复制到当前 Python 环境的
`mjlab/tasks/head_load_velocity/`，由 mjlab 官方任务发现机制注册。
机器人资源随任务安装，路径相对于任务目录，无需修改服务器路径。
安装不下载依赖；更新仓库后再次执行 `python install.py` 即可。

安装还将官方 `mjlab/scripts/train.py` 中的 `init_at_random_ep_len` 设为
`False`，避免首回合未实际存活 20 秒就触发质量课程升级。
此设置影响该 mjlab 环境中使用此训练入口的所有任务。
对本任务还会在官方 iteration 日志输出后更新质量课程；继续使用官方 runner 和训练循环。

## 播放

```bash
python -m mjlab.scripts.play Mjlab-Velocity-HeadLoad-Unitree-G1 \
  --checkpoint-file /path/to/model_1000.pt --num-envs 1
```

日志和 checkpoint 由官方入口保存到
`logs/rsl_rl/g1_head_load_velocity/`。

## 任务

- 4096 个环境，回合 20 秒，策略频率 50 Hz；29 维动作、96 维 actor 观测。
- 完整复用官方 G1 flat 奖励、PPO、MLP、观测函数和有限外力脉冲。
- 使用提供的 29 DoF URDF，保留四球足部碰撞；头部固定边长 0.3 m 的立方体。
- 电机配置参考 [rickshaw 的 G1 mode-15 参数](https://github.com/well-robotics/rickshaw/blob/fbb16ec3c59b2dbf8a2eed6070ad6004be4ba501/source/g1_rickshaw_lab/g1_rickshaw_lab/g1_actuator_config.py)：
  双腕 roll / pitch / yaw 使用 N5010，armature 为 `0.0021812`，
  `Kp = armature × (2π×10)²`，`Kd = 2×2×armature×(2π×10)`。
  其余关节复用官方 PD / armature，各关节力矩上限取 URDF；
  动作缩放统一按 `0.25 × effort_limit / Kp` 计算。
- 每只手的接触为半径 `0.05 m` 的球体，球心位于该手网格包围盒中心，
  接触维度为 3（含切向摩擦）。保留手部外观、质量和惯量。
- 所有环境共享质量上限，初始 3 kg。每个 iteration 读取官方
  `mean_episode_length`，乘以策略步长 `0.02` 换算为秒。
  每 100 个 iteration 对窗口内这些值取算术平均，严格大于 19 秒时，
  上限增加 3 kg，最高 60 kg；等于 19 秒不升级。
  每个窗口结束后清空窗口数据。尚无已结束回合、未产生该指标的 iteration
  不计入平均；整个窗口没有指标时不升级。
  新上限在各环境下次 reset 时生效，从 `[1, 上限]` kg 重新采样，
  同步每轴惯量 `0.015 × 质量`，回合内质量不变。
- 官方指标本身是最近 100 个已结束回合的平均策略步数。
  课程直接复用该指标，不再统计成功比例，也不排除失败回合或旧阶段回合。
  统计周期直接跟随真实 iteration，修改每轮采样步数不需要调整课程窗口。
- 保留方案指定的速度命令、躯干质量 / COM 随机化、地面摩擦随机化和外力参数。

`head_load_velocity/` 下只有四个任务模块：注册、资源适配、环境配置和质量课程。
标准 G1 网格沿用 Unitree 的许可，见 `head_load_velocity/assets/LICENSE.unitree`。
