# ChysX 布料–Mesh 碰撞检测与响应

本文档总结 ChysX 中布料粒子与刚体三角网格（mesh body）之间碰撞检测和碰撞响应的完整算法管线。

---

## 1. 总体架构

ChysX 将布料–mesh 碰撞建模为 **点–三角面** 的罚函数（penalty）接触，而**非** IPC 的 log-barrier 模型。每个布料粒子独立地与最近三角面进行碰撞检测，使用 **量化无栈 BVH** 加速最近点查询。

```
┌─────────────────────────────────────────────────────────────────┐
│                   每步 step() 管线                               │
├─────────────────────────────────────────────────────────────────┤
│ 1. set_pose()     : 刚体变换 + BVH 重建                         │
│ 2. detect()       : BVH 遍历 → 最近三角面 → 缓存 (n, depth)     │
│ 3. accumulate_gradient() : 罚函数梯度 + 摩擦力 → rhs            │
│ 4. bake_diag()    : Gauss-Newton 对角 Hessian → H               │
│ 5. PCG 求解       : (M/dt² + H_elastic + H_contact) dx = rhs   │
│ 6. 更新位置       : pos = x_n + dx                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 碰撞检测（Detection）

### 2.1 宽相位：量化无栈 LBVH

**算法**：Apetrei (2014) 的 LBVH（Linear BVH），使用 Morton 码排序 + DFS 重排 + 逃逸指针，实现**无栈遍历**。

**量化**：节点 AABB 压缩为 14-bit 量化格式（`Ull2` = 16 字节/节点），用整数比较代替浮点 AABB overlap 测试。

| 阶段 | 调用 | 工作内容 |
|------|------|---------|
| **建树** | `set_mesh()` → `bvh_.build(n_tris, 1)` | 分配叶节点（每三角面一个） |
| **每帧更新** | `set_pose()` → `bvh_.refit(...)` | 场景 AABB → Morton 排序 → 完整重建内部树 + 量化节点 |

三角面的 AABB 按 `thickness` 向外膨胀，确保碰撞壳层内的粒子也能通过宽相位。

### 2.2 窄相位：最近点查询

对每个布料粒子执行 `detect_mesh_kernel`：

1. 以 `search_radius`（默认 `10 × thickness`）为半径构造球形查询区域
2. 无栈遍历 BVH，对命中的叶节点执行 `point_triangle_dist2`（Ericson 风格的顶点/边/面区域判定）
3. 维护全局最小距离²，随着找到更近的三角面动态收缩查询 AABB
4. **有符号距离**：`sd = ±dist`，由 `dot(p − closest, face_normal)` 的符号决定正负
5. **穿透深度**：`depth = thickness − sd`，若 `depth ≤ 0` 则无接触
6. 缓存结果：`contacts_[p] = Vec4f(nx, ny, nz, depth)`

> 这是**离散碰撞检测（DCD）**，在线性化点 `x_n`（上一帧收敛位置）处执行，不包含连续碰撞检测（CCD）。

---

## 3. 碰撞响应（Response）

### 3.1 接触模型：平方罚函数

碰撞能量：

$$E_c = \frac{1}{2} k \cdot \max(0,\ h - d)^2$$

其中 `h = thickness`（接触厚度），`k = stiffness`（接触刚度），`d` = 粒子到 mesh 表面的有符号距离。

梯度（力）：

$$\nabla E_c = -k \cdot \text{depth} \cdot \mathbf{n}$$

Hessian（Gauss-Newton 近似，忽略法线曲率）：

$$\nabla^2 E_c \approx k \cdot \mathbf{n}\mathbf{n}^T$$

接触仅贡献**对角块**（每个粒子独立），不引入粒子间耦合项。

### 3.2 摩擦模型

ChysX 提供两种摩擦实现方式，通过 `ipc_friction` 参数切换：

#### 方式 A：IPC 隐式摩擦（默认 `ipc_friction=True`）

来自 Li et al. 2020（Incremental Potential Contact）的**摩擦线性化**策略（注意：仅摩擦部分使用 IPC 思想，接触本身仍为罚函数而非 log-barrier）。

相对滑移量：

$$\mathbf{r} = (\mathbf{v}_p - \mathbf{v}_{\text{body}}) \cdot dt$$

切向滑移：

$$\mathbf{u}_t = \mathbf{r} - (\mathbf{r} \cdot \mathbf{n})\mathbf{n}$$

摩擦力系数：

$$\alpha = \mu \cdot f_n \cdot f_1^{SF}(\|\mathbf{u}_t^{lag}\|)$$

其中 $f_1^{SF}$ 是光滑化函数：在零滑移处 $2/\varepsilon$，超过 $\varepsilon$ 后 $1/\|\mathbf{u}_t\|$。

梯度中额外增加：
- 摩擦力：$-\alpha \cdot \mathbf{u}_t$
- 法向阻尼（当粒子趋近时）：$-k_d \cdot k \cdot r_{dn} \cdot \mathbf{n}$

Hessian 中增加：
- 摩擦块：$\alpha \cdot (\mathbf{I} - \mathbf{n}\mathbf{n}^T)$
- 阻尼块：$k_d \cdot k \cdot \mathbf{n}\mathbf{n}^T$

#### 方式 B：Coulomb 锥后投影（`ipc_friction=False`）

在 `assemble_rhs` 之后执行 `apply_coulomb_friction_kernel`：

1. 重建不含摩擦的法向力 $F_0 = \text{rhs} - k \cdot \text{depth} \cdot \mathbf{n}$
2. 计算锥半径：$\mu \cdot k \cdot \max(\text{depth},\ 0.1 \cdot \text{thickness})$
3. **Stick**（$\|F_{0,t}\| \leq$ 锥半径）：加入 $M/dt \cdot (\mathbf{v}_{\text{body},t} - \mathbf{v}_{p,t})$ 冲量
4. **Slip**（$\|F_{0,t}\| >$ 锥半径）：将切向力缩放到锥边界

---

## 4. 在 ClothSimulator::step() 中的调用位置

```
predictor: x_tilde = x_n + dt*v + dt²*g
    │
    ▼
梯度累积（在 x_n 处线性化）:
    pins_.accumulate_gradient
    fem_stretch_.accumulate_gradient
    fem_shear_.accumulate_gradient
    bending_.accumulate_gradient
    self_collision detect + gradient      (可选)
    untangle detect + gradient            (可选)
    static_contacts detect + gradient     (地面/盒子)
    sdf_contacts detect + gradient        (SDF 体)
  ★ mesh_contacts detect + gradient       (三角网格体)
    │
    ▼
assemble_rhs: rhs = M/dt²(x_tilde - x_n) - ∇E(x_n)
    │
    ▼
后投影摩擦（仅 ipc_friction=false 时）:
    static_contacts apply_coulomb_friction
    sdf_contacts apply_coulomb_friction
  ★ mesh_contacts apply_coulomb_friction
    │
    ▼
Hessian 累积:
    pins, FEM stretch/shear, bending → H
    M/dt² 惯性对角 → H
    self_collision bake_diag
    static_contacts bake_diag
    sdf_contacts bake_diag
  ★ mesh_contacts bake_diag
    │
    ▼
PCG 求解: (M/dt² + H) dx = rhs
    │
    ▼
更新: pos = x_n + dx,  vel = dx/dt · exp(-damping·dt)
```

---

## 5. 关键参数

| 参数 | Python 默认值 | 含义 | 调参建议 |
|------|-------------|------|---------|
| `thickness` | 0.005 m | 接触激活距离 $h$ | 过大→远距离吸附；过小→穿透。建议 1-3mm |
| `stiffness` | 1e4 N/m | 罚函数刚度 $k$ | 过大→PCG 收敛慢/振荡；过小→穿透深 |
| `friction` | 0.0 | Coulomb 摩擦系数 $\mu$ | 布料-金属约 0.3-0.5 |
| `friction_epsilon` | 0.01 m/s | IPC 摩擦光滑带宽 | 过大→粘滞；过小→数值不稳定 |
| `contact_kd` | 0.01 | 法向阻尼（IPC 路径） | 抑制弹跳，0.01-0.1 |
| `ipc_friction` | True | 使用 IPC 隐式摩擦 | True 更稳定但需要更多 PCG 迭代 |
| `search_radius` | 0 → 10×thickness | BVH 查询半径 | 0 即可，深穿透时增大 |

---

## 6. 与 StaticContact（地面/盒子）的对比

| 方面 | StaticContact | MeshContact |
|------|--------------|-------------|
| 几何类型 | 平面/OBB 解析几何 | 任意三角网格 |
| 检测方式 | 解析距离公式 | BVH + 最近三角面 |
| 每粒子代价 | O(平面数+盒子数) | O(log N_tris) |
| 法向来源 | 解析梯度 | 三角面法线 |
| 摩擦 | 总是 IPC 隐式 | 可选 IPC 或后投影 |
| 刚体速度 | 不支持 | 支持（线速度） |
| 对角 Hessian | 是 | 是 |

两者使用相同的罚函数数学，区别在于几何查询方式和摩擦实现细节。

---

## 7. 已知局限

1. **每粒子单接触**——每帧只缓存一个最近三角面，不支持多接触点
2. **仅支持刚体 mesh**——顶点由单一刚体变换控制
3. **无角速度**——`set_body_velocity` 仅支持线速度
4. **面法线符号歧义**——在边/顶点附近可能出现法线跳变（非伪法线）
5. **离散碰撞检测**——大 dt 时可能穿透（无 CCD）
6. **需手动更新位姿**——`set_pose()` 不在 `step()` 内自动调用

---

## 8. 算法参考

| 技术 | 是否使用 | 参考 |
|------|---------|------|
| 平方罚函数接触 | ✅ | 标准位置罚函数 + Gauss-Newton Hessian |
| IPC log-barrier | ❌ | 未使用（无 barrier 能量，无 CCD） |
| IPC 各向同性 Coulomb 摩擦 | ✅（可选） | Li et al. 2020, *Incremental Potential Contact* |
| Coulomb 锥投影 | ✅（可选） | cuda-cloth 风格后投影 |
| LBVH / Morton BVH | ✅ | Apetrei 2014; 量化无栈变体 |
| 最近三角面点 | ✅ | Ericson, *Real-Time Collision Detection* |
| 隐式 Euler + PCG | ✅ | 在 x_n 处线性化，块 Jacobi 预条件 |
