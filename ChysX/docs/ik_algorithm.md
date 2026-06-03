# 逆运动学 (IK) 求解器 — 技术文档

本文档介绍 Newton 中实现的逆运动学算法，以及在 `example_chysx_softbody_franka.py` 中 IK 的具体使用方式。

---

## 1. 算法架构

### 1.1 模块结构

| 文件 | 职责 |
|------|------|
| `newton/ik.py` | 公共 API 入口（re-export） |
| `ik_solver.py` | 高层 `IKSolver` — 多种子采样、目标选择、委托给优化器 |
| `ik_lm_optimizer.py` | **Levenberg–Marquardt** 优化后端 |
| `ik_lbfgs_optimizer.py` | **L-BFGS** 替代后端 |
| `ik_objectives.py` | 内置目标函数 + `IKObjective` 基类 |
| `ik_common.py` | 雅可比类型枚举、批量 FK、代价核函数 |

### 1.2 核心类

```
IKSolver (前端)
├── IKOptimizerLM (默认后端 — Levenberg-Marquardt)
├── IKOptimizerLBFGS (可选后端 — L-BFGS)
├── IKObjective (目标函数基类)
│   ├── IKObjectivePosition
│   ├── IKObjectiveRotation
│   └── IKObjectiveJointLimit
└── IKSampler (初始猜测采样策略)
```

---

## 2. 优化方法

### 2.1 Levenberg-Marquardt（默认）

IK 问题被形式化为**最小二乘优化**：

$$
\min_{\mathbf{q}} \; C(\mathbf{q}) = \|\mathbf{r}(\mathbf{q})\|^2 = \sum_i r_i^2(\mathbf{q})
$$

其中 \(\mathbf{r}(\mathbf{q})\) 是残差向量（包含所有目标函数的残差行）。

每次 LM 迭代求解**阻尼法方程**：

$$
(\mathbf{J}^\top\mathbf{J} + \lambda \mathbf{I})\,\delta\mathbf{q} = -\mathbf{J}^\top\mathbf{r}
$$

关键参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `lambda_initial` | 0.1 | 初始阻尼系数 |
| `lambda_factor` | 2.0 | 接受/拒绝时 λ 的缩放因子 |
| `lambda_min` / `lambda_max` | — | λ 的上下界 |
| `rho_min` | 1e-3 | 最小实际/预测下降比，低于此值拒绝步长 |

**LM 迭代流程**：

```
1. 计算当前位形 q 处的残差 r(q) 和代价 C(q) = Σr²
2. 计算雅可比矩阵 J
3. 求解 (JᵀJ + λI)δ = -Jᵀr  （通过 Cholesky 分解）
4. 将 δ 通过 jcalc_integrate 积分到 q_proposed
5. FK → 计算 r(q_proposed) → 计算 C_proposed
6. ρ = (C - C_proposed) / predicted_reduction
7. if ρ ≥ ρ_min:
     接受：q ← q_proposed，λ /= factor
   else:
     拒绝：保持 q，λ *= factor
```

### 2.2 L-BFGS（替代方法）

使用有限记忆 BFGS 的两循环递归计算搜索方向，结合并行 Strong-Wolfe 线搜索。梯度使用 \(\nabla C = \mathbf{J}^\top\mathbf{r}\)。

通过 `optimizer=ik.IKOptimizer.LBFGS` 选择。

---

## 3. 目标函数 (Objectives)

### 3.1 基类接口 `IKObjective`

每个目标函数向全局残差向量贡献若干行，并可选择提供解析雅可比块。在优化器初始化时，每个目标被分配残差偏移量和批次布局。

### 3.2 `IKObjectivePosition` — 位置目标

**残差**（3 行）：

$$
\mathbf{r}_{\text{pos}} = w \cdot (\mathbf{p}_{\text{target}} - \mathbf{p}_{\text{ee}})
$$

其中：
- \(\mathbf{p}_{\text{ee}} = \text{transform\_point}(\text{body\_q}[\text{link\_index}], \text{link\_offset})\)
- \(w\) 为权重（默认 1.0）

**解析雅可比**：利用运动子空间 \(\mathbf{S}\) 计算末端执行器速度对关节速度的映射。对运动链上的每个自由度 \(j\)：

$$
\frac{\partial \mathbf{p}_{\text{ee}}}{\partial \dot{q}_j} = v_{\text{origin},j} + \omega_j \times \mathbf{p}_{\text{ee}}
$$

### 3.3 `IKObjectiveRotation` — 姿态目标

**残差**（3 行）：

将四元数误差转换为轴角表示：

$$
q_{\text{err}} = q_{\text{actual}} \cdot q_{\text{target}}^{-1}
$$

$$
\mathbf{r}_{\text{rot}} = w \cdot \text{axis} \cdot \text{angle}
$$

默认启用 `canonicalize_quat_err`，选择短弧路径。

**解析雅可比**：仅使用运动子空间的角速度分量：

$$
J_{\text{rot}}[k, j] = w \cdot \omega_j[k]
$$

### 3.4 `IKObjectiveJointLimit` — 关节限位目标

**残差**（每个 DOF 一行，限位内为零）：

$$
r_i = w \cdot \max(0, q_i - q_{\text{upper}}) + w \cdot \max(0, q_{\text{lower}} - q_i)
$$

**解析雅可比**：分段常数对角矩阵（违反上限时 +w，违反下限时 -w，限位内为 0）。

### 3.5 残差向量布局示例

对于 Franka 的三个目标，每个问题的全局残差向量为：

```
[ pos_x, pos_y, pos_z,                    # 3 行, 偏移 0
  rot_x, rot_y, rot_z,                    # 3 行, 偏移 3
  limit_0, limit_1, ..., limit_{n_dofs-1} # n_dofs 行, 偏移 6 ]
```

总残差维度 = `6 + joint_dof_count`。

---

## 4. 雅可比矩阵计算

### 4.1 计算模式

| 模式 | 说明 |
|------|------|
| `ANALYTIC` | 基于运动子空间（Featherstone 风格）的解析几何雅可比 |
| `AUTODIFF` | Warp 反向模式自动微分 |
| `MIXED` | 有解析雅可比的目标用解析方法，其余用自动微分 |

**注意**：Newton 不使用有限差分。

### 4.2 解析路径

1. **两趟 FK**：
   - 第 1 趟：`jcalc_transform` 计算每个关节的局部相对变换 `X_local`
   - 第 2 趟：沿父链累积 → 世界坐标 `body_q`

2. **运动子空间**：使用 Featherstone 的 `jcalc_motion` 计算空间速度列 \(\mathbf{S}_s\)

3. **逐目标雅可比块**：在独立 CUDA 流上并行计算

### 4.3 自动微分路径

1. 创建可微分的 `dq_dof`（`requires_grad`）
2. 录制 Warp tape：`dq_dof` → `joint_q_proposed` → FK → 残差
3. 对每个残差分量，以单位种子向量反向传播
4. 将 `tape.gradients[dq_dof]` 拷贝到雅可比行

---

## 5. Tiled / 批量求解机制

### 5.1 JIT 特化

`IKOptimizerLM` 根据 `(n_dofs, n_residuals, GPU 架构)` 创建**特化子类**并缓存：

- 编译时常量：`TILE_N_DOFS`、`TILE_N_RESIDUALS`
- 使用 Warp 的 **tile API**（`wp.tile_load`、`wp.tile_matmul`、`wp.tile_cholesky`）
- **每个 CUDA 线程块**处理一个批次行的完整 LM 线性求解

### 5.2 批次维度

| 层级 | 大小 | 含义 |
|------|------|------|
| `n_problems` | 用户指定 | 基础 IK 问题数（如 1 个机器人） |
| `n_seeds` | 用户指定 | 每个问题的候选初始猜测数 |
| `n_expanded` | `n_problems × n_seeds` | 传给优化器的总行数 |

### 5.3 采样策略 (`IKSampler`)

| 策略 | 行为 |
|------|------|
| `NONE` | 直接拷贝输入（`n_seeds` 必须为 1） |
| `GAUSS` | 种子 0 = 输入；种子 1..N-1 = 输入 + 高斯噪声，裁剪到关节限位 |
| `UNIFORM` | 在关节限位范围内均匀随机采样 |
| `ROBERTS` | 确定性低差异序列 |

---

## 6. 在 `example_chysx_softbody_franka.py` 中 IK 的使用

### 6.1 场景概述

这个例子模拟一个 Franka Emika Panda 机械臂抓取一个弹性体小鸭子。**IK 负责将高层的末端执行器轨迹转换为关节位置指令**，然后通过速度追踪送入 Featherstone 刚体求解器。

### 6.2 IK 目标定义 — `set_up_ik()`

```python
# 末端执行器位置目标
self.pos_obj = ik.IKObjectivePosition(
    link_index=self.endeffector_id,    # 手臂末端链接
    link_offset=wp.vec3(0.0, 0.0, 0.22),  # 工具中心点（夹爪中心上方 22cm）
    target_positions=wp.array([target_pos], dtype=wp.vec3),
)

# 末端执行器姿态目标
self.rot_obj = ik.IKObjectiveRotation(
    link_index=self.endeffector_id,
    link_offset_rotation=wp.quat_identity(),
    target_rotations=wp.array([target_rot], dtype=wp.vec4),
)

# 关节限位目标（权重 10.0，强制执行关节范围）
self.joint_limits_obj = ik.IKObjectiveJointLimit(
    joint_limit_lower=self.model.joint_limit_lower,
    joint_limit_upper=self.model.joint_limit_upper,
    weight=10.0,
)

# 创建 IK 求解器：单问题、解析雅可比、24 次 LM 迭代
self.ik_solver = ik.IKSolver(
    model=self.model, n_problems=1,
    objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
    lambda_initial=0.1,
    jacobian_mode=ik.IKJacobianType.ANALYTIC,
)
```

关键设计：
- `endeffector_id = builder.body_count - 3`：URDF 折叠固定关节后的手臂 TCP 链接
- `link_offset = (0, 0, 0.22)`：目标点在手掌坐标系上方 22cm（夹爪中心）
- 使用 `ANALYTIC` 雅可比模式（最快路径）
- 手指关节**不参与 IK 求解**，在 IK 后直接设置

### 6.3 机器人关键位姿 — `robot_key_poses`

```python
self.robot_key_poses = np.array([
    # [持续时间, px, py, pz, qx, qy, qz, qw, 夹爪]
    [2.5, -0.005, -0.5, 0.35, 1,0,0,0, 1.0],   # 接近（夹爪张开）
    [2.0, -0.005, -0.5, 0.21, 1,0,0,0, 1.0],   # 下降
    [2.5, -0.005, -0.5, 0.21, 1,0,0,0, 0.5],   # 夹紧（夹爪闭合）
    [2.0, -0.005, -0.5, 0.35, 1,0,0,0, 0.5],   # 提升
    [2.0, -0.005, -0.5, 0.35, 1,0,0,0, 0.5],   # 保持
    [2.0, -0.005, -0.5, 0.21, 1,0,0,0, 0.5],   # 放置
    [1.0, -0.005, -0.5, 0.21, 1,0,0,0, 1.0],   # 释放
    [2.0, -0.005, -0.5, 0.35, 1,0,0,0, 1.0],   # 撤回
])
```

每一行定义一个阶段：
- 第 1 列：该阶段持续时间（秒）
- 第 2~4 列：末端执行器位置 \((p_x, p_y, p_z)\)（米）
- 第 5~8 列：末端执行器姿态四元数 \((q_x, q_y, q_z, q_w)\)
- 第 9 列：夹爪激活量（1.0 = 张开，0.5 = 闭合）

相邻关键位姿之间使用**线性插值**（位置和四元数分量均线性插值，非球面插值）。

### 6.4 完整控制流程

下图展示了 **每一帧** 中 IK 和物理模拟的完整数据流：

```
┌─ 每帧 (60 Hz) ────────────────────────────────────────────┐
│                                                            │
│  1. update_ik_targets()                                    │
│     ├─ 根据 sim_time 找到当前关键位姿区间                     │
│     ├─ 线性插值 → 位置/姿态目标                              │
│     ├─ pos_obj.set_target_position(目标位置)                 │
│     ├─ rot_obj.set_target_rotation(目标姿态)                 │
│     └─ finger_pos = 夹爪激活量 × 0.04m                      │
│                                                            │
│  2. IK 求解 (一帧一次)                                      │
│     ├─ ik_solver.step(ik_joint_q, ik_joint_q, iters=24)    │
│     │   └─ 24 次 LM 迭代：残差计算→雅可比→求解→积分→评估      │
│     ├─ set_gripper_q: 覆写手指关节 = finger_pos              │
│     └─ target_joint_q ← ik_joint_q 的 1D 拷贝               │
│                                                            │
│  3. 计算目标关节速度                                         │
│     └─ target_joint_qd[i] = (target_q[i] - current_q[i])   │
│                              / frame_dt                     │
│                                                            │
│  4. 物理仿真子步 (5 次，dt = frame_dt/5)                    │
│     ┌─ 每个子步 ──────────────────────────────────────────┐ │
│     │  a. state_0.joint_qd = target_joint_qd              │ │
│     │  b. gravity = 0   (机器人无重力)                      │ │
│     │  c. robot_solver.step()   ← ChysX Featherstone      │ │
│     │     └─ 按目标速度积分关节 → 更新 body_q/body_qd       │ │
│     │  d. gravity = -9.81  (弹性体有重力)                   │ │
│     │  e. soft_solver.step()    ← ChysX VBD + 碰撞        │ │
│     │     └─ 弹性体 VBD 求解 + 刚柔耦合接触力               │ │
│     │  f. swap(state_0, state_1)                          │ │
│     └───────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### 6.5 IK 输出到 Featherstone 的桥梁

IK 求解的是**位置级**问题，而 Featherstone 需要**速度级**输入。两者之间的桥梁是显式的**比例关节追踪**：

```python
@wp.kernel
def compute_joint_qd(target_q, current_q, out_qd, inv_frame_dt):
    i = wp.tid()
    out_qd[i] = (target_q[i] - current_q[i]) * inv_frame_dt
```

$$
\dot{q}_d = \frac{q_{\text{target}} - q_{\text{current}}}{\Delta t_{\text{frame}}}
$$

这个恒定关节速度在一帧时间内将机器人从当前位形驱动到 IK 目标位形。在每个子步中，该速度被赋给 `state_0.joint_qd`，然后 Featherstone 以 `sim_dt = frame_dt / 5` 积分。

**重要**：Featherstone 在此场景中主要用作**运动学积分器** — 重力被清零，弹性体被禁用，碰撞形状被清除。这使得机器人精确跟踪 IK 轨迹，同时产生正确的体速度（用于 VBD 摩擦力计算）。

### 6.6 夹爪关节处理

两个手指关节（最后两个坐标）**不参与 IK 求解**，而是在 IK 之后直接设置：

```python
@wp.kernel
def set_gripper_q(joint_q, finger_pos, idx0, idx1):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]
```

两个手指接收相同的位置（对称平行夹爪）。`finger_pos = 夹爪激活量 × 0.04`，将归一化的 `[0.5, 1.0]` 范围映射到 `[0.02, 0.04]` 米。

---

## 7. 算法参数总结

### 7.1 Franka 示例中的参数

| 参数 | 值 | 说明 |
|------|------|------|
| `n_problems` | 1 | 单机器人 |
| `optimizer` | LM（默认） | Levenberg-Marquardt |
| `jacobian_mode` | `ANALYTIC` | 解析几何雅可比 |
| `lambda_initial` | 0.1 | 初始阻尼 |
| `ik_iters` | 24 | 每帧 LM 迭代次数 |
| `joint_limits weight` | 10.0 | 强关节限位约束 |
| `fps` | 60 | IK 每帧调用一次 |
| `sim_substeps` | 5 | 每帧物理子步数 |
| `link_offset` | (0, 0, 0.22) m | 工具中心点偏移 |
| `endeffector_id` | body_count - 3 | TCP 链接索引 |

### 7.2 关键性能特征

- IK 的 **LM tiled 求解** 在 GPU 上高效执行，每个问题一个线程块
- **解析雅可比** 避免了自动微分的开销
- IK 每帧只调用一次（60Hz），而物理模拟每帧执行 5 个子步（300Hz）
- IK 结果通过**速度追踪**平滑地分配到多个子步中

---

## 8. IK 仍然使用 Newton/Warp

在当前的 ChysX 集成中，**IK 是唯一仍然通过 Newton/Warp 运行的组件**：

| 组件 | 运行环境 |
|------|----------|
| IK 求解器 | Newton/Warp（Python + JIT CUDA） |
| Featherstone 刚体动力学 | **ChysX**（C++/CUDA） |
| VBD 弹性体求解 | **ChysX**（C++/CUDA） |
| 碰撞检测 | **ChysX**（C++/CUDA） |
| FK（初始化 + 渲染用） | Newton/Warp |
| 可视化 | Newton Viewer |

将 IK 移植到 ChysX 是未来可选的优化方向，但由于 IK 每帧只调用一次且计算量相对较小，当前架构下不是性能瓶颈。
