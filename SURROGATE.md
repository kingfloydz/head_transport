# 接触稳定裕度代理模型

实现位于 `head_load_velocity/surrogate/`。现有物理配置、课程、奖励、终止条件不变；新奖励默认关闭。旧 `stability_margin` 仍是原有名义矩形奖励；本模型是另一个可选项，不能把二者混称为同一模型。

## 接触与输入约定

当前编译模型核实：pyramidal，货物/平台均 condim=3、priority=1，滑动摩擦系数均为0.8，无显式 pair；平台全尺寸 `(0.20,0.20,0.02)m`，torso局部顶面中心 `(0,0,0.39)m`。condim=3没有独立扭转/滚动力矩。

**输入为35+2=37维，而不是强行使用不足的35维。** MuJoCo-Warp `collision_convex.py` 使用 `make_frame(witness1-witness2)`，`math.py` 的 `orthogonals` 从世界y/z轴构造切向基。金字塔约束即使两个切向μ相等，也不具任意平面旋转不变性。绝对yaw不在35维内，重力方向不能补足它。追加实际接触第一切向在平台xy中的两个单位分量，第二切向取 `z×t1`；符号翻转/两切向交换不影响该对称金字塔。若同一贴合区域出现不等价切向基、非平面法向或各向异性，标为域外，不给错误标签。

`schema.py` 的 `FIELDS` 是唯一顺序定义（0-based切片）：

| 索引 | 含义 |
|---|---|
| 0:6 | 真实COM到六个箱面的距离：x−、x+、y−、y+、z−、z+ |
| 6 | 质量kg |
| 7:13 | 箱体系质心惯量 Ixx Iyy Izz Ixy Ixz Iyz，kg·m² |
| 13:16 | COM相对平台顶面中心，torso表达，m |
| 16:22 | box→torso旋转矩阵前两**列**，逐列展开；第三列为叉积 |
| 22:25 | torso_link body原点的运动学线加速度，torso表达 |
| 25:28 | torso角速度，torso表达 |
| 28:31 | torso角加速度，torso表达 |
| 31:34 | 单位重力方向，torso表达 |
| 34 | 有效滑动μ |
| 35:37 | 实际接触第一切向在平台xy的分量 |

惯量包含 `body_iquat @ diag(body_inertia) @ body_iquat.T`，不是仅取主惯量。有效μ按接触priority选取，同priority取两geom的最大值，再与实际contact friction核对。实际非零μ与两切向相等是适用条件。

## 时序和适用域

通过官方 `Scene.update` 的子步钩子，在每个控制步最后一个子步读取一次。使用同一时刻的 `xpos/xmat/xipos/subtree_com/cvel/contact`；这些都是 `mj_step` 积分前的派生量，比返回的qpos落后一个物理步。本实现不混用积分后qvel，不增加forward。相邻记录相差 `env.step_dt`（当前0.02s）。不滤波；线/角速度在世界系差分后再旋转到当前torso系。torso原点速度由官方 `compute_velocity_from_cvel` 转换。

reset仅清除对应环境的历史有效标记，第一条记录保留但无有效差分；采集在env.step后读取缓存，记录的是终止/reset之前的最后子步，与在线奖励消费的样本一致。完整episode的全部帧始终归属同一集合，包括派生扩展样本。

初版适用域：箱底与平台夹角≤0.035rad，底面所有角点离顶面≤0.005m，投影正面积相交，有平台接触，无货物与其他geom接触；实际接触切向和法向满足0.005容差。容差是近似贴合域的明确工程定义，不声称轻微倾斜时仍完全再现实际接触斑。域外记录也保存，标签不作为有效贴合监督。

`domain` 是位标记：1历史不足、2非贴合、4无平台接触、8其他接触（含手臂）、16接触基/摩擦模型不匹配、32无有效几何、64非正摩擦。在线另用128表示超出模型训练特征包络。多个原因可叠加。不用相对滑动速度拒绝样本，也不加入恢复已有滑动的力旋量。

## 离线标签

`exact.py` 在float64中：

1. 从原始物理参数重建实际箱底四角，裁剪与平台的**完整凸交集**，保留3–8边形。
2. 计算torso原点→顶面中心刚性偏移的平移、切向、向心加速度；再计算实际COM的假想随动加速度和完整惯量力矩。
3. 在顶面中心统一力矩，归一化为 `[F/(mg),tau/(mgp)]`，p=0.20m。
4. 每个多边形顶点使用四条射线 `(±μt1+n),(±μt2+n)`。不使用内接矩形、μ折减或独立偏航上限。
5. 输出 `[s_N,s_mu,s_tip,s_yaw,s_all]`。s_mu使用实际切向坐标的L1范数；s_tip取多边形最危险单位外法向边界；s_yaw固定其余五维、两次LP求区间；s_all用面积质心构造w0，求自由变量t的最大值。

使用SciPy现有HiGHS，原/对偶容差1e−9，等式残差检查1e−7。s_all是**沿参考支持力旋量的储备**，不是欧氏距离；精确仅相对于此刚性多面体模型，不是MuJoCo软接触复刻。不使用压缩LP，不存在mu_required或二分求解。

`solver_status`：0完成、1域外、2数值失败；另保存三次LP原始状态码。物理不可行通过 `s_all<0` / `feasible=false` 表示，不能把数值失败当作物理不可行。偏航前五维不可行时 `yaw_valid=false`，s_yaw保存NaN，仅这项损失屏蔽；数值失败样本不参加训练。

## 数据与训练

`surrogate_controllers.json` 留有10个真实控制器的空checkpoint路径。用户已确认它们均使用当前环境配置；未假定任何实际文件路径。默认6/2/2控制器拆分。每项stage为采集起始阶段，可明确设置1–4以覆盖现有课程支持范围，不改变课程规则；采集不运行PPO或晋级runner。

采样保留官方站立/行走/heading切换与现有躯干脉冲，使用确定性策略。每100步写一个文件，steps必须显式传入。每次采集独立run_id，episode ID含run_id、环境与回合编号，防止分批文件和重复采集产生ID碰撞。保存实际env.yaml、agent.yaml、checkpoint SHA256及provenance。末尾可能有未结束回合，这些片段仍按完整episode ID绑定，绝不随机按帧拆分。小规模采样不保证已覆盖全部启停/转向/脉冲状态，正式采集需检查覆盖和域外比例。

当前仿真货物μ固定0.8，采集不会擅自随机化它；不同μ通过**离线**扩展覆盖。`extend` 对现有箱内质量分布施加仿射尺度变化：尺寸、COM同时变换；由完整惯量重建中心二阶矩后统一变换，并一致更新质量与惯量。随后扰动水平位置、相对yaw、运动量及μ，再重算动力学和LP。不存在独立随机I/COM。优先保留边界与不可行样本，保留父controller/episode和来源；这是瞬时物理一致的反事实样本，不是声称可由该控制器执行的轨迹。

`train`：37→128→128→5，ELU，线性输出；训练集均值/标准差归一化；每标签独立标准差缩放；masked Huber，`1+4 exp(-|s|/0.1)`边界权重，额外2倍s_all高估平方损失。该损失**没有严格保守保证**。验证集选模型，测试集不选模型。初版几何/摩擦留出组合为Lx>0.40且μ>0.9；触及该组合的训练controller完整episode及其全部扩展均移到geometry_test，统计仅从剩余train计算。若组合没有样本，报告空集合，不虚构泛化结果。

导出 `frozen.pt`（TorchScript，含归一化、标签还原、metadata.json），附训练报告和划分清单。报告包括每项MAE、|s_all|≤0.1边界误差、高估均值/p95、真实不可行却预测安全比例及相反比例、各控制器分组统计。正常训练必须存在10个控制器的真实rollout有效标签；合成验证模型不能被RL奖励入口加载。

## 命令

先安装当前本地实现，再填实际checkpoint路径。以下是运行入口，**本次未启动真实10模型采样或PPO**。

```bash
python3 install.py
python3 -m mjlab.tasks.head_load_velocity.surrogate validate --output surrogate_validation
MUJOCO_GL=disable python3 -m mjlab.tasks.head_load_velocity.surrogate collect \
  --manifest surrogate_controllers.json --output surrogate_raw \
  --steps 1200 --num-envs 64 --device cuda:0
```

可先指定 `--controllers controller_01 --steps 20 --num-envs 2` 验证真实模型加载。1200步仅是24秒/环境的初步覆盖示例，不等于充分训练数据。正式规模自行指定。

Linux下合并采样文件进行标注；传入全部文件以维持统一划分：

```bash
python3 -m mjlab.tasks.head_load_velocity.surrogate label \
  --input surrogate_raw/*/*/raw_*.npz --output surrogate_labeled.npz
python3 -m mjlab.tasks.head_load_velocity.surrogate extend \
  --input surrogate_labeled.npz --output surrogate_extended.npz --count 10000
python3 -m mjlab.tasks.head_load_velocity.surrogate train \
  --input surrogate_extended.npz --manifest surrogate_controllers.json \
  --output surrogate_model --epochs 50 --device cuda:0
python3 -m mjlab.tasks.head_load_velocity.surrogate benchmark \
  --model surrogate_model/frozen.pt --device cuda:0
```

## 默认关闭的RL接入

普通训练默认 `surrogate_weight=0`，不创建新传感器、不加载代理。显式启用示例（需要先获得真实控制器训练模型并审阅报告）：

```bash
MUJOCO_GL=disable python3 -m mjlab.scripts.train \
  Mjlab-Velocity-HeadLoad-Unitree-G1 \
  --env.surrogate-model surrogate_model/frozen.pt \
  --env.surrogate-weight 0.05 \
  --env.surrogate-delta 0.0 --env.surrogate-temperature 0.1 \
  --env.surrogate-outside-cost 5.0 --agent.logger tensorboard
```

只使用s_all：`-w*softplus((delta-s_all)/T)`，再由官方奖励管理器按dt缩放。delta=0取真实可行边界；T=0.1表示沿支持方向归一化储备的过渡宽度；示例w=0.05是待调试初值，未从旧矩形裕度权重迁移。其余四标签不叠加进新奖励。参数、网络全冻结，不参与PPO优化，无LP或在线力旋量路径。

缺少首次差分：新奖励为0。明确翘起、失联、手部/其他接触、不匹配切向基、超训练特征包络：**不推理**，使用 `-w*outside_cost`（默认代价5），并记录outside/range/warmup比例及域标记。包络仅为train最小/最大值加1%跨度，不是完整分布支持证明。固定域外惩罚可能造成策略偏好，需要与域内代价分布一起评估；不能解释为恢复能力或多接触稳定性。现有失联终止、旧稳定奖励均保留，不自动替换。

## 当前验证边界

validate只使用640个明确标注为synthetic的样本、3个epoch验证端到端链路。它不代表10个真实G1控制器的精度；演示frozen.pt有smoke标记，奖励入口拒绝加载。CPU实测4096批次推理计入归一化和网络，不包含接触/特征提取；不能外推H200性能。部署前应在目标GPU运行benchmark并测完整环境步时延。

2026-10-01本机结果（MuJoCo/MuJoCo-Warp 3.11.0，mjlab 1.6.0）：

- 默认配置全部已有字段与main一致；Ruff、Ty、Pyright通过。
- 居中静止已知状态标签 `[1, 0.8, 0.5, 0.4, 1]`。
- 摩擦超限、倾覆超限、偏航超限均正确得到负s_all；偏航区间不存在时NaN标签被损失屏蔽。
- 支持完整8边形；独立世界系动力学与torso系结果最大误差约8.9e−16。
- 实际G1双环境零动作8步：16条记录，6条有效贴合标签，0次数值失败；其他记录域外或历史不足，部分reset历史隔离通过。
- 4096批次冻结网络：CPU 4线程、预热10次后测100次，平均约3.85ms。未测GPU和全环境开销。
- 合成留出集合的s_all边界MAE约1.62，高估均值0.62、p95为3.44；真实不可行但预测安全约1.11%，真实可行但预测不安全约92.11%。这是仅3个epoch的**流程演示，精度不合格**，不能据此声称安全或用于RL。真实10控制器指标尚缺。

完整本次结果见仓库 `surrogate_validation.json`。上述数据不替代用户checkpoint采样。
