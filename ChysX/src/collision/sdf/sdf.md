# SDF 体素接触：踩坑实录与设计取舍

`SdfVolume` + `SdfContact` 是布料 ⇄ 隐式体的 penalty 接触通道。结构上跟
`StaticContactSet` 完全平行（共用 `Vec4f(n, depth)` 接触格式 + 同一套
gradient/diag/Coulomb-cone kernel），但 SDF 是**离散采样**的，比解析平面 / 盒子
多了三类陷阱。把它们写下来，避免下次再踩。

---

## 陷阱 1：网格 padding 必须 ≥ contact thickness（**最严重**）

### 现象

布料应该静止地搭在 SDF box 上，结果**高频抖动**永远不停。表面看像欠阻尼
振铃，但有几个反常信号：

- `damping` 加 4× 几乎无效（粒子平均速度 151 → 147 mm/s）
- 接触刚度调**软**反而抖得更厉害（1e4 → 1e3，速度 151 → 411 mm/s）
- 同样参数搬到 `example_chysx_cloth_drop`（解析 ground plane）就稳得纹丝不动
  （平均速度 0.6 mm/s）

### 真正机制

`SdfVolume::bake_box(hx, hy, hz, voxel_size, padding)` 把 grid 覆盖
`local ∈ [-(h+padding), +(h+padding)]`。粒子查询落在 grid 外时，`sample()`
返回 sentinel `sd = 1e30f, grad = 0`，下游 `detect_kernel` 算出
`depth = thickness - 1e30 < 0`，于是判定**无接触**。

如果 `padding < thickness`，则**整个 contact band 的一部分位于 grid 外**：

```
                                 grid top  (z = pos.z + hz + padding)
─────────────────────────────────────────────────────  z = 0.16  (padding=10mm)
                  ╳     ←  布料应该静止在 z ≈ 0.167（band 中)
                                              ↑
                                         超出 grid！
─────────────────────────────────────────────────────  z = 0.15  box top
```

于是粒子每一步在**有力 / 无力**之间翻转：

1. 进入 grid 边缘 → sample 给出真实 sd → penalty 推上去
2. 推过 grid 上界 → sentinel `1e30` → `depth < 0` → 力**消失**
3. 重力把粒子拉回 grid → 第 1 步重来

这是**力本身的开关跳变**，不是粒子在弹簧上做物理振动，所以阻尼对它无效
（阻尼只能耗散动能，对"突然消失的力"没办法）。把刚度调软，粒子能更深地
穿透 contact band → 更频繁地撞到 grid 上界 → 抖得更厉害，正好解释了
"调软反而更差"的反常。

### 修复

在 `solver_chysx.py::bake_sdf_box` 的自动 padding 中加 thickness 余量：

```python
if padding < 0.0:
    padding = max(2.0 * voxel_size,
                  thickness + 2.0 * voxel_size)
```

这样 grid 在 surface 外**至少**多伸出 thickness 距离，contact band 完全位于
grid 内部，再额外留 2 voxel 给 trilinear 梯度模板做边界处理。

### 诊断套路

下次再遇到"SDF 接触不稳"，先在 Python 里复刻一份 trilinear 采样器（30 行），
跑完 warmup 后对每个布料粒子查一次 SDF：

```
particles in grid: 0 / 625        ←  红色警报
active SDF contacts: 0
```

一眼定位是 padding 还是别的。**别先怀疑摩擦或步长**——它们都不会让粒子凭空
逃出 grid。

---

## 陷阱 2：trilinear 梯度在体素边界 C⁰ 但不 C¹

`SdfVolume::sample()` 用解析 trilinear 微分给出梯度（`gx_local` 等）。这
等价于：

- 值场 `sd(x)` 跨体素**连续**（C⁰）
- 梯度场 `∇sd(x)` 跨体素**不连续**（C¹ 断裂）

跟解析 plane 给出的恒定 `(0, 0, 1)` 法向相比，SDF 法向**在每个 voxel
边界处发生小幅跳变**。对快速运动的粒子这没事（一帧穿过好几个 voxel，平均
掉），但对**几乎静止**的粒子很糟——它在某个 voxel 内做亚毫米级随机游走，
法向方向跟着小幅震颤。

后果：
- 通常表现为"几乎静止但永远不完全静止"的 mm/s 级残余速度
- 跟 IPC lagged 摩擦正反馈时，把 mm/s 噪声放大成可见的振铃（见陷阱 3）

缓解措施按代价从低到高：
1. **减小 voxel_size**：5 mm → 2 mm，跳变幅度变成原来的 2/5
2. 改用 tricubic 采样（C¹ 连续）—— 8× 内存访问，仅在质量极敏感时考虑
3. 改用 narrow-band SDF + 解析 primitive 混合（box 用解析，复杂部分用 SDF）

实测：voxel = 5 mm、padding 修好之后，SDF box 上静止布料平均速度 ≈ 9 mm/s；
对比 `cloth_drop`（解析 plane）≈ 0.6 mm/s。差 15× 的那部分就是这条。

---

## 陷阱 4：CUDA Graph capture 会冻结 by-value kernel 参数

### 现象

`bake_sdf_box` + `set_sdf_pose(volume_index, pos)` 看上去工作正常——
host 侧每帧把新的 pose 推下去，可视化里 box 也跟着动。但布料**完全
感受不到 box 的运动**：在 `example_chysx_sdf_gripper` 里，夹爪闭合
后向上抬升 25 cm，布料原地不动；切换到不走 graph 的 path（手动
`_simulate_substeps()`）布料立刻开始跟随。

### 真正机制

`SdfContact::detect()` 这样启动 detect kernel：

```cpp
SdfVolumeView view = volume_->make_view();   // host 侧 POD
detect_kernel<<<...>>>(..., view, body_velocity, ...);
```

其中 `view` 包含 `pos_, ex_, ey_, ez_`，`body_velocity` 是 `float3`。
两者都是**按值**传给 kernel。

CUDA Graph capture 录制 kernel launch 节点时，会把 kernel 参数
**复制一份**到 graph 节点里。Replay 同一个 graph 时这些参数永远是
capture 时刻的值——后续 host 上更新的 `pos`、`body_velocity` 全
被旁路。结果就是 graph 里的 detect kernel 永远在用"开始 lift 前
那一帧"的 pose 算 contact，布料看到的 SDF body 像被钉死了。

### 修复

把 pose / body_velocity 从 host-by-value 改成 device-pointer：

- `SdfVolume` 内部加 `CudaArray<Vec3f> pose_dev_`（4 个 Vec3f：
  `[pos, ex, ey, ez]`），`set_pose(...)` 写 host cache + async H2D
  copy 进这个稳定的 device buffer。
- `SdfVolumeView::pose` 变成 `const Vec3f*` 指针。Graph capture 时
  pointer 值固定（device buffer 不会被 resize），replay 时 kernel
  读 pointer 指向的最新数据。
- `SdfContact` 同理给 `body_velocity_` 加一个 1-元素 device buffer。

device buffer 必须**lazy-alloc 一次后永不重新分配**，否则 graph
里那份 pointer 会失效。`set_pose` 路径里加 `if (gpu_size() != N)`
guard，保证只在第一次调用时 resize。

### 经验

只要某个 host-mutable 状态会被 CUDA Graph 内的 kernel 读取，就**不能**
按值传——必须挪到 device buffer 经指针访问。这条规则适用于所有
"每帧改一点 host 标量然后跑 graph"的场景：动画 pose、shader uniform、
材质参数等等都需要这样处理。

---

## 陷阱 5：IPC lagged 摩擦的 RHS 符号在 body 移动时反了

### 现象

修完陷阱 4 后，`example_chysx_sdf_gripper` 里夹爪向上 lift，布料
**反方向掉下去**（cloth 离 jaw 越来越远，从 -20mm 掉到 -30mm）。
直觉上摩擦应该把布料往 jaw 方向拽，但实际给出的力是相反的。

### 真正机制

`scatter_gradient_kernel` 原本写：

```cpp
const Vec3f slip = (v_particle - v_body) * dt;   // 切向投影后
rhs[p] += -alpha * slip;                          // 加到 grad E
```

接着 `assemble_rhs_kernel` 把累积的 `grad E` **变号**折进 Newton
残差：`rhs_final = M·v_n/dt + M·g - rhs_grad`，所以最终 rhs 里
摩擦项贡献是 `+α · slip = +α · (v_particle - v_body) · dt`。

当布料静止、body 向上 lift 时：
- `v_particle.z = 0`
- `v_body.z = +0.05`
- `slip.z = -0.05 · dt` (负)
- `rhs_final.z += +α · slip.z = -α · dt · v_body` (负 = 把布料拉下去!)

正确的 IPC lagged-Newton 推导是：摩擦力 `f_t = -α (dx - dt · v_body)_t`
在 RHS 上的贡献是 `+α · dt · v_body_t`（独立于 `v_particle`，永远
把布料朝 body 切向位置牵）。所以 `slip` 在 RHS 用法上的符号刚好反
了，原写法只能在 `v_body = 0` 的静态体场景下"看起来对"（因为静态
体 Coulomb cone post-projection 会把 stick 分支的切向直接清零，把
错误符号掩盖掉）。

### 修复

`sdf_contact.cu` 的 `scatter_gradient_kernel` 改成 `+alpha * slip`
（即不再取反）。同时 `apply_coulomb_friction_kernel` 删掉 stick
分支的"把切向清零"逻辑，只保留 slip 分支的 cone-cap——否则即使
RHS 符号修对了，post-projection 还是会把 IPC 摩擦贡献再次清零。

`static_contact.cu` 保持不变：它的用例都是静态体 (`v_body ≡ 0`)，
新旧符号给出相同结果（`slip = v_particle · dt`，stick 分支由 cone
projection 接管）。两条 contact 通道符号约定从此**有意不同**，这是
"分别针对自己使用场景做最简实现"的取舍。

### 经验

IPC lagged-Newton 摩擦在静态 body 场景下，RHS 符号错了照样能工作
（cone post-projection 兜底），所以这条 bug 在没有移动 body 的 unit
test 里**测不出来**。要 catch 这类问题，回归测试必须覆盖"body 主动
移动、布料应当随动"的场景（比如 `example_chysx_sdf_gripper` 里夹爪
带布料抬升的 `test_final`）。

---

## 陷阱 3：IPC lagged 摩擦在准静态 SDF 接触上会自激

IPC 摩擦（Li et al. 2020）在 ChysX 里是 lagged-Newton 线性化：
当前步在对角块加 `α·(I - n nᵀ)` 切向刚度，在 RHS 减 `α·u_t^lag`
（`u_t^lag` 是**上一步**的切向位移）。这相当于把摩擦建模成一个把粒子拉回
"上一步切向位置"的弹簧。

问题：当布料**几乎静止**时，
1. PCG 残差 + SDF 法向小抖（陷阱 2）→ 粒子有 μm 级真实切向位移
2. 该位移被写进 `u_t^lag`
3. 下一步摩擦把粒子拉回"几步前的位置"
4. 但布料的真实力平衡点早就漂走了 → 弹回新位置 → 新的 slip
5. 回到第 1 步 → **自激振铃**

`cloth_drop` 没有这个问题是因为它**没开摩擦**（μ=0）。

对策：
- **静态调试期间一律设 μ=0**，先确认几何/法向/penalty 单独工作
- 真正需要摩擦时，把 `friction_epsilon` (`ε_u`) 调大（5 mm 起步），
  让"准静态"判定容差变粗；同时 `damping` ≥ 0.1 帮助耗散
- 长期方案：摩擦 RHS 加缓和因子或仅在 |v| > 阈值时启用动摩擦项

---

## 设计取舍备忘

### `SdfVolume` 数据 vs pose 解耦

`bake_box` 只写 `values_, nx_, ny_, nz_, voxel_size_, origin_local_`
（静态烘焙数据）；body 的 world pose `(pos, ex, ey, ez)` 缓存在
**device buffer** `pose_dev_`（见陷阱 4），每帧 `set_pose(...)` async
H2D copy 进去。`make_view()` 给 kernel 的是 device pointer，graph
capture 时 pointer 稳定，replay 自动读到最新 pose。

好处：移动 body 不重烘焙、CUDA Graph 跨帧重放也能拿到最新 pose、
所有 SDF kernel launch 都是 graph-friendly 的。

### 为什么 SDF 法向要重新归一化

trilinear `∇sd` 即便对解析单位 SDF 也**不是单位长度**——线性内插的梯度
模长在 voxel 内部从 1.0 漂到 ~1.1 不等。如果不归一化，后续 `α·(I - n nᵀ)`
里的投影矩阵就不是真正的投影（`nnᵀ` 算出来 trace ≠ 1），会污染对角块的
条件数。在 `detect_kernel` 里用 `rsqrtf(g2 + 1e-30)` 重新归一化是廉价
保险（一个 rsqrt + 三个 mul）。

### `body_velocity` 必须在 slip 计算里减掉

否则布料骑在匀速移动的 SDF body 上时会被"误检测出 slip"，触发摩擦反弹。
`detect_kernel` 里：

```
v_rel = v_particle - body_velocity
u_t   = (v_rel - n·(n·v_rel)) · dt
```

这点上 SDF 通道**比** `StaticContactSet` 强——后者假设静态体，没有
`body_velocity` 字段，移动 plane / box 也只能传零相对速度。

---

## TL;DR 调试清单

SDF 接触不稳时，按这个顺序排查：

1. **`padding ≥ thickness + 2·voxel` 吗？**  Python 复刻 sampler 看
   `particles in grid` 是不是 0
2. **`voxel_size << min(hx, hy, hz)` 吗？**  至少 10 voxel 穿过最薄轴
3. **摩擦先关 (μ=0) 排除自激**  确认几何对了再加摩擦
4. **`damping ≥ 0.05`、`pcg_iterations ≥ 30`**  跟 `cloth_drop` 对齐
5. **stiffness 1e3 ~ 1e4 之间**  调软通常是错方向（陷阱 1 反例）
