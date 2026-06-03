# Featherstone 铰接体刚体求解器 — 技术文档

本文档详细介绍 Newton/ChysX 中实现的 Featherstone 铰接体前向动力学算法。该算法基于**递归 Newton-Euler 算法 (RNEA)** 计算运动学和力投影，通过显式构建广义质量矩阵 \(H = J^\top M J\)（复合刚体算法结构），使用 Cholesky 分解求解 \(H\,\ddot{q} = \tau\)。

---

## 1. 算法概览

### 1.1 求解的核心方程

每个时间步求解：

$$
\underbrace{(J^\top M J + \mathrm{diag}(R))}_{H + R}\,\ddot{q} = \tau
$$

其中：
- \(J\) — 关节空间到笛卡尔空间的雅可比矩阵
- \(M\) — 块对角空间惯性矩阵
- \(R = \mathrm{diag}(\text{joint\_armature})\) — 关节 armature 正则化（转子惯量）
- \(\tau\) — 广义力（来自反向 RNEA + PD 驱动 + 关节限位）
- \(\ddot{q}\) — 广义加速度

### 1.2 算法流水线

```
1.  正向运动学 (FK)          → body_q, body_q_com
2.  外力准备                  → body_f_ext（COM→原点坐标变换）
3.  FREE/DISTANCE 力路由     → 6-DOF 关节力 → body_f_ext
4.  运动学体力清零            → kinematic bodies 不受外力
5.  速度/力坐标转换           → public ↔ internal（FREE/DISTANCE 特殊处理）
6.  正向 RNEA                → 空间速度、科氏力、空间惯量
7.  反向 RNEA                → 广义力 τ
8.  雅可比矩阵构建            → J
9.  质量矩阵构建              → M（块对角 6×6）
10. 矩阵乘法 P = MJ          → 中间矩阵
11. 矩阵乘法 H = JᵀP         → 广义质量矩阵
12. Cholesky 分解             → LLᵀ = H + diag(R)
13. 前代/回代求解             → q̈ = (H+R)⁻¹τ
14. 辛欧拉积分               → q⁺, q̇⁺
15. 正向运动学 + 速度转换      → body_q, body_qd (COM 参考)
16. FREE/DISTANCE 后校正     → 子体位姿修正
```

---

## 2. 空间代数约定

### 2.1 空间向量 (Spatial Vector)

Newton 和 ChysX 均使用 **linear-first, angular-second** 的空间向量布局：

| 索引 | 分量 | 含义 |
|------|------|------|
| 0–2 | 线性 (v) | 平动速度 / 力 |
| 3–5 | 角度 (ω) | 转动速度 / 力矩 |

运动旋量 (twist): \(\mathbf{x} = (v, \omega)\)

空间力 (wrench): \(\mathbf{f} = (f_{\text{linear}}, \tau_{\text{angular}})\)

```cpp
// ChysX C++ 实现
struct SpatialVector {
    float data[6];  // [v_x, v_y, v_z, w_x, w_y, w_z]
    Vec3f linear()  const;  // data[0:3] — 平动
    Vec3f angular() const;  // data[3:6] — 转动
};
```

### 2.2 空间惯性矩阵 (Spatial Inertia Matrix)

6×6 行主序矩阵，在体 COM 坐标系下为块对角结构：

$$
I_m = \begin{bmatrix} m \cdot I_3 & 0 \\ 0 & I_{3\times3} \end{bmatrix}
$$

其中 \(m\) 为标量质量，\(I_{3\times3}\) 为关于 COM 的惯性张量。

### 2.3 刚体变换 (Transform7)

`Transform7` / `wp.transform` 的内存布局为 7 个 float：`[px, py, pz, qx, qy, qz, qw]`（平移 + 四元数）。

变换组合 `A * B` 的含义：先施加 B、再施加 A（子到父的链式法则）。

**旋量变换 (Twist Transform)**：对于刚体变换 \(t = (R, p)\)，从源坐标系到目标坐标系：

$$
\omega' = R\,\omega, \qquad v' = R\,v + p \times \omega'
$$

**空间惯量变换**：

$$
I_{\text{new}} = T^\top I T, \quad T = \begin{bmatrix} R & [p]_\times R \\ 0 & R \end{bmatrix}
$$

### 2.4 坐标系定义

| 坐标系 | 含义 |
|--------|------|
| **世界坐标系** | 原点在世界原点，轴与世界对齐。RNEA 中的速度、加速度、力均在此坐标系下 |
| **体原点坐标系** | `body_q[child]` — 子体坐标系在世界中的位姿 |
| **体 COM 坐标系** | `body_q_com = body_q * body_X_com` — COM 在世界中的位置，方向与体坐标系一致 |
| **父关节锚点** | `X_wpj = body_q[parent] * joint_X_p` — 父侧关节坐标系在世界中 |
| **子关节锚点** | `X_wcj = X_wpj * X_j` — 经过关节位移后的子侧关节坐标系 |

---

## 3. 铰接体 (Articulation)

### 3.1 定义

**铰接体**是由关节连接的刚体组成的运动树，作为一个独立的动力学单元进行质量矩阵求解。每个铰接体拥有自己的稠密矩阵 \(H\)、雅可比 \(J\) 和 Cholesky 求解，各铰接体之间可以并行计算。

### 3.2 关节分组

关节在索引空间中**连续排列**，按**拓扑序**（DFS 序）存储：

| 数组 | 形状 | 作用 |
|------|------|------|
| `articulation_start` | `[articulation_count + 1]` | 每个铰接体的起始关节索引；最后一个为哨兵值 |
| `joint_ancestor` | `[joint_count]` | 关节 \(i\) 的树中父关节索引（用于沿运动链向上遍历） |

### 3.3 每个铰接体的矩阵尺寸

对于一个有 \(n\) 个连杆、\(d\) 个自由度的铰接体：

| 矩阵 | 尺寸 | 说明 |
|------|------|------|
| \(J\) | \(6n \times d\) | 雅可比矩阵 |
| \(M\) | \(6n \times 6n\) | 块对角空间惯性矩阵 |
| \(H\) | \(d \times d\) | 广义质量矩阵 |
| \(L\) | \(d \times d\) | 下三角 Cholesky 因子 |

---

## 4. 关节类型与关节计算函数

### 4.1 关节类型枚举

| 枚举值 | 名称 | 位置 DOF | 速度 DOF | 坐标数 |
|--------|------|----------|----------|--------|
| 0 | PRISMATIC | 1 (lin) | 1 | 1 |
| 1 | REVOLUTE | 1 (ang) | 1 | 1 |
| 2 | BALL | 3 (ang) | 3 | 4 (四元数) |
| 3 | FIXED | 0 | 0 | 0 |
| 4 | FREE | 6 | 6 | 7 (pos + 四元数) |
| 5 | DISTANCE | 6 | 6 | 7 |
| 6 | D6 | 1~6 | 1~6 | 1~6 |

### 4.2 `jcalc_transform` — 关节变换

计算关节两侧的相对变换 \(X_j\)（父锚点 → 子锚点）：

| 关节类型 | \(X_j\) |
|---------|---------|
| PRISMATIC | 沿轴平移 `axis * q` |
| REVOLUTE | 绕轴旋转角度 `q` |
| BALL | 由四元数坐标定义的旋转 |
| FIXED | 单位变换 |
| FREE / DISTANCE | 完整刚体变换（7 个坐标） |
| D6 | 棱柱轴之和 + 复合旋转 |

### 4.3 `jcalc_motion` — 运动子空间

计算运动子空间 \(S\) 和关节速度 \(v_j = S\dot{q}\)：

- **PRISMATIC**: \(S = (a, 0)\) — 线性轴
- **REVOLUTE**: \(S = (0, a)\) — 角度轴
- **BALL**: 3 个角度单位向量
- **FREE/DISTANCE**: 6 个单位基向量
- **D6**: 按每个轴分别计算

每列在世界坐标系下表示：\(S_s = \mathrm{transform\_twist}(X_{\text{wpj}}, S_{\text{local}})\)

### 4.4 `jcalc_tau` — 广义力投影

将空间力投影到关节轴上：

$$
\tau_i = -S_i^\top f_s + f_{\text{drive},i} + f_{\text{joint},i}
$$

**PD 驱动力公式**：

$$
f_{\text{drive}} = k_e (q^* - q) + k_d (\dot{q}^* - \dot{q})
$$

**关节限位弹簧**：当 \(q < q_{\text{lower}}\) 或 \(q > q_{\text{upper}}\) 时，使用弹簧 + 阻尼力代替 PD 驱动。

### 4.5 `jcalc_integrate` — 辛欧拉积分

- **1-DOF 关节**: \(\dot{q}^+ = \dot{q} + \ddot{q}\,dt\)；\(q^+ = q + \dot{q}^+\,dt\)
- **BALL**: 角速度积分后通过四元数微分方程更新：\(\dot{q}_{\text{quat}} = \frac{1}{2}\,\omega^+ \otimes q\)
- **FREE 根关节** (`parent < 0`): 在**世界 COM** 坐标系下积分
- **FREE 子关节** (`parent ≥ 0`): 在**父原点**内部坐标下积分

---

## 5. 正向 RNEA（前向传递）

### 5.1 算法描述

逐关节从根到叶遍历（Featherstone Table 5.1 的前向传递，此时 \(\ddot{q} = 0\)）：

| 步骤 | 公式 |
|------|------|
| 关节速度 | \(v_j = \texttt{jcalc\_motion}(\ldots)\) |
| 体速度 | \(v_s = v_{\text{parent}} + v_j\) |
| 体加速度 | \(a_s = a_{\text{parent}} + v_s \times v_j\)（科氏项；无 \(\ddot{q}\) 项） |
| COM 惯量 | \(I_s = \mathrm{transform\_spatial\_inertia}(X_{\text{sm}}, I_m)\) |
| 重力力 | \(f_g = (mg,\; r_{\text{com}} \times mg)\) |
| 偏置力 | \(f_b = I_s a_s + v_s \times^* (I_s v_s)\) |
| 输出 | `body_f_s = f_b - f_g` |

### 5.2 空间叉积

**运动 × 运动** (`spatial_cross`)：

$$
\omega_{\text{out}} = \omega_a \times \omega_b, \quad v_{\text{out}} = \omega_a \times v_b + v_a \times \omega_b
$$

**运动 ×* 力** (`spatial_cross_dual`)：

$$
f_{\text{out}} = \omega_a \times f_b, \quad \tau_{\text{out}} = \omega_a \times \tau_b + v_a \times f_b
$$

---

## 6. 反向 RNEA（后向传递）

从叶到根反向遍历树：

对每个关节 \(i\)（子体 \(c\)）：

$$
f_s = f_{b,s}[c] + f_{t,s}[c] + f_{\text{ext}}[c]
$$

调用 `jcalc_tau` 将 \(f_s\) 投影到关节轴写入 \(\tau\)，然后 \(f_s\) 通过原子加法累积到 `body_ft_s[parent]`。

| 缓冲区 | 作用 |
|--------|------|
| `body_f_s` | 正向传递的惯性偏置力 |
| `body_ft_s` | 子体累积的空间力 |
| `body_f_ext` | 外力 + 接触力（已转换到原点坐标） |
| `joint_tau` | 输出的广义力 |

---

## 7. 质量矩阵构建与求解

### 7.1 雅可比矩阵构建 (`eval_rigid_jacobian`)

对铰接体中的每个连杆 \(i\)，行块 \(6i\) 对应连杆 \(i\) 的空间运动。通过沿祖先链遍历填充列：

$$
J[6i+k,\; j] = S_j[k] \quad \text{（对连杆 } i \text{ 的每个祖先关节 } j\text{）}
$$

### 7.2 质量矩阵 (`eval_rigid_mass`)

块对角空间惯性矩阵：

$$
M = \mathrm{blkdiag}(I_{s,0},\; I_{s,1},\; \ldots,\; I_{s,n-1})
$$

### 7.3 GEMM（矩阵乘法）

两次 GEMM 计算：

1. \(P = M \cdot J\)
2. \(H = J^\top \cdot P\)

### 7.4 Cholesky 分解

下三角 Cholesky 分解 \(L\) 使得 \(L L^\top = H + \mathrm{diag}(R)\)：

$$
L L^\top = H + \mathrm{diag}(R_{\text{armature}})
$$

### 7.5 前代/回代求解

1. 前代：\(L y = \tau\)
2. 回代：\(L^\top \ddot{q} = y\)

结果 \(\ddot{q}\) 即为广义加速度。

### 7.6 Armature 的作用

| 用途 | 说明 |
|------|------|
| 物理意义 | 电机/齿轮的反映惯量 |
| 数值意义 | 防止 \(H\) 在冗余构型附近奇异 |
| 运动学体 | 将 kinematic 关节的 armature 设为 `1e10`，使求解产生近零加速度 |

---

## 8. 辛欧拉积分

### 8.1 积分方案

**辛欧拉** (Symplectic Euler)：先更新速度，再用更新后的速度更新位置。

$$
\dot{q}^{n+1} = \dot{q}^n + \ddot{q}\,\Delta t
$$
$$
q^{n+1} = q^n + \dot{q}^{n+1}\,\Delta t
$$

### 8.2 四元数关节的特殊处理

BALL 和 FREE 关节的旋转部分使用四元数微分方程：

$$
q_{\text{quat}}^{n+1} = \text{normalize}\left(q_{\text{quat}}^n + \frac{1}{2}\,\omega^{n+1} \otimes q_{\text{quat}}^n\,\Delta t\right)
$$

---

## 9. FREE/DISTANCE 关节特殊处理

### 9.1 两种速度约定

| 上下文 | 线速度含义 | 参考系 |
|--------|-----------|--------|
| **公共** `State.joint_qd` | 子体 **COM** 的速度 | 父关节锚点方向 |
| **内部**（求解器流水线） | 父**关节锚点**的速度 | 父关节锚点 |

转换公式（以 public → internal 为例）：

$$
r = R_p^\top (x_{\text{com\_world}} - x_{\text{anchor\_world}}), \quad v_{\text{internal}} = v_{\text{com}} - \omega \times r
$$

### 9.2 根关节 vs 子关节积分

| 情况 | 积分坐标系 | 原因 |
|------|-----------|------|
| 根 FREE (`parent = -1`) | 世界 COM | 浮动基座使用世界坐标系的 COM 运动 |
| 子 FREE (`parent ≥ 0`) | 父原点内部坐标 | 与 Featherstone 关节坐标一致 |

### 9.3 子关节位姿后校正

内部坐标积分会偏离公共 COM 约定。积分 + FK 后：

1. `correct_free_distance_body_pose` — 从公共 COM 旋量重新积分 `body_q`
2. `reconstruct_joint_q_from_body_pose` — 从校正后的 `body_q` 重建 `joint_q`
3. 从第一个子关节开始部分刷新 FK

### 9.4 力的路由

FREE/DISTANCE 的 `Control.joint_f`（6-DOF 力）**不通过** `jcalc_tau` 施加。而是：

1. 转换为子体 COM 处的空间力
2. 累加到 `body_f_ext[child]`
3. `joint_f_internal` 中对应 DOF 清零

---

## 10. 运动学体 (Kinematic Body) 处理

`BodyFlags.KINEMATIC` 标记的刚体是**预定义运动**的，不参与动力学模拟。

| 机制 | 位置 | 效果 |
|------|------|------|
| 清零外力 | `zero_kinematic_body_forces` | 接触/重力/外力无法移动运动学体 |
| 膨胀 armature | `joint_armature = 1e10` | \(H\) 刚性化 → \(\ddot{q} \approx 0\) |
| 清零加速度 | `zero_kinematic_joint_qdd` | 强制 \(\ddot{q} = 0\) |
| 拷贝预定状态 | `copy_kinematic_joint_state` | 用输入状态覆盖积分结果 |

---

## 11. Newton (Warp) vs ChysX (C++/CUDA) 实现对比

| 方面 | Newton | ChysX |
|------|--------|-------|
| 语言 | Python + Warp JIT | C++/CUDA |
| 质量矩阵更新 | 可配置间隔 (`update_mass_matrix_interval`) | 每步重建 |
| Tile GEMM | 支持（Warp tile API，硬编码 18 DOF） | 不支持 |
| Kinematic armature | Python 端膨胀到 `1e10` | Python wrapper 端处理 |
| 原子操作 | `wp.atomic_add` | `atomicAdd` |
| 接触/粒子 | 内置在 `step()` 中 | 外部处理 |
| 可微分 | 支持（Warp 自动微分） | 不支持 |
| 并行模型 | 树遍历在铰接体内串行，跨铰接体并行 | 相同 |

### 11.1 CUDA Kernel 对应表

| ChysX CUDA Kernel | Newton Warp Kernel |
|---|---|
| `eval_rigid_fk_kernel` | `eval_rigid_fk` |
| `eval_rigid_id_kernel` | `eval_rigid_id` |
| `eval_rigid_tau_kernel` | `eval_rigid_tau` |
| `eval_rigid_jacobian_kernel` | `eval_rigid_jacobian` |
| `eval_rigid_mass_kernel` | `eval_rigid_mass` |
| `eval_dense_gemm_batched_kernel` | `eval_dense_gemm_batched` |
| `eval_dense_cholesky_batched_kernel` | `eval_dense_cholesky_batched` |
| `eval_dense_solve_batched_kernel` | `eval_dense_solve_batched` |
| `integrate_generalized_joints_kernel` | `integrate_generalized_joints` |

### 11.2 数值精度验证

在 Franka 机械臂（9 关节、9 体）上进行 10 步对比测试：

| 指标 | 最大误差 |
|------|---------|
| joint_q | ~1e-10 |
| joint_qd | ~1e-8 |
| body_q | ~1e-7 |
| body_qd | ~1e-8 |

所有误差在 float32 精度范围内，两个实现**完全一致**。

---

## 12. 关键数据缓冲区总结

| 缓冲区 | 坐标系/基 | 写入时机 |
|--------|-----------|----------|
| `joint_S_s` | 世界空间，父锚点 | 正向 RNEA |
| `body_v_s`, `body_a_s` | 世界原点 | 正向 RNEA |
| `body_I_s` | 世界，COM 处 | 正向 RNEA |
| `body_f_s` | 惯性偏置 − 重力 | 正向 RNEA |
| `body_ft_s` | 子体累积力 | 反向 RNEA |
| `body_f_ext` | 外力，原点坐标，取反 | 力准备 + 接触 |
| `joint_tau` | 广义力 | 反向 RNEA |
| `joint_qdd` | 广义加速度 | Cholesky 求解 |
| `State.body_qd` | 公共 COM 旋量 | FK + 速度转换 |
| `State.body_q` | 体原点位姿 | FK |

---

## 13. 算法复杂度

- **每个铰接体**: \(O(d^2 \cdot n)\) 用于 GEMM，\(O(d^3)\) 用于 Cholesky
- **跨铰接体**: 完全并行（每个铰接体一个 CUDA 线程）
- **适用场景**: 多个独立机器人（如多机械臂）性能优秀；单个巨型机构则受串行遍历限制
