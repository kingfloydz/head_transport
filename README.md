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
- 所有环境共享质量上限，初始 3 kg。每 100 个 iteration 汇总已结束回合，
  满 20 秒且没有失败的比例达到 95% 时，上限增加 3 kg，最高 60 kg。
  每个统计窗口结束后清零计数；升级前开始的回合不计入新阶段统计。
  新上限在各环境下次 reset 时生效，从 `[1, 上限]` kg 重新采样，
  同步每轴惯量 `0.015 × 质量`，回合内质量不变。
- 统计窗口按官方每 iteration 24 个采样步设置为 2400 步。
  若修改 `--agent.num-steps-per-env`，同步将 `env_cfg.py` 中
  `payload_curriculum` 事件的 `window_steps` 设为其 100 倍。
- 保留方案指定的速度命令、躯干质量 / COM 随机化、地面摩擦随机化和外力参数。

`head_load_velocity/` 下只有四个任务模块：注册、资源适配、环境配置和质量课程。
标准 G1 网格沿用 Unitree 的许可，见 `head_load_velocity/assets/LICENSE.unitree`。
