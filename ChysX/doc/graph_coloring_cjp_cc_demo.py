#!/usr/bin/env python3
"""
CJP / CC / CC-Incremental 图着色算法的 Python 演示.

论文来源:
  [1] "Efficient Algorithms for Graph Coloring on GPU"
      Nguyen Quang Anh Pham, Rui Fan (ICPADS 2018)

本脚本包含:
  1. CJP: Counting-based Jones-Plassmann
  2. CC: Conflict Coloring (正确模拟 GPU 并行语义)
  3. CC-Incremental: 利用 CC 框架做帧间增量着色
     - 把上一帧着色结果直接作为 CC 的初始着色 (替代随机初始化)
     - max_color 取上一帧使用的最大颜色+1
     - CC 冲突检测只会修正新增冲突, 大部分顶点跳过

用法: python graph_coloring_cjp_cc_demo.py
"""

import random
from collections import defaultdict

# ========================================================================
#  图工具
# ========================================================================

def build_adj(edges, n):
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    for v in range(n):
        if v not in adj:
            adj[v] = set()
    return adj

def validate(adj, colors, n, name=""):
    violations = []
    for v in range(n):
        for nb in adj[v]:
            if nb > v and colors.get(v, -1) >= 0 and colors.get(nb, -1) >= 0:
                if colors[v] == colors[nb]:
                    violations.append((v, nb))
    nc = len(set(c for c in colors.values() if c >= 0))
    uncolored = sum(1 for v in range(n) if colors.get(v, -1) < 0)
    if violations:
        print(f"  [FAIL] {name}: {len(violations)} conflicts! {violations[:5]}")
    elif uncolored > 0:
        print(f"  [FAIL] {name}: {uncolored} vertices uncolored!")
    else:
        print(f"  [OK] {name}: {nc} colors, 0 conflicts")
    return len(violations) == 0 and uncolored == 0

def min_avail(adj, v, colors):
    used = set()
    for nb in adj[v]:
        c = colors.get(nb, -1)
        if c >= 0:
            used.add(c)
    c = 0
    while c in used:
        c += 1
    return c

SEP = "=" * 72

# ========================================================================
#  金字塔碰撞图 (15 个 body 的金字塔: 底层 3x3, 中层 2x2, 顶层 1)
# ========================================================================

PYRAMID_N = 15
PYRAMID_EDGES = [
    (0,1),(1,2),
    (3,4),(4,5),
    (6,7),(7,8),
    (0,3),(1,4),(2,5),
    (3,6),(4,7),(5,8),
    (9,10),(11,12),(9,11),(10,12),
    (0,9),(1,9),(1,10),(2,10),
    (3,9),(3,11),(4,9),(4,10),(4,11),(4,12),
    (5,10),(5,12),
    (6,11),(7,11),(7,12),(8,12),
    (9,13),(10,13),(11,13),(12,13),
    (0,14),(1,14),(2,14),(3,14),(4,14),(5,14),(6,14),(7,14),(8,14),
]

def print_graph(adj, n, label=""):
    if label:
        print(f"  {label}:")
    for v in range(n):
        nbs = sorted(adj[v])
        print(f"    body {v:2d}: deg={len(nbs):2d}, neighbors={nbs}")

# ========================================================================
#  CJP: Counting-based Jones-Plassmann
# ========================================================================

def cjp_coloring(adj, n, seed=42, verbose=True):
    rng = random.Random(seed)
    colors = {}
    val = {v: rng.randint(0, 100000) for v in range(n)}
    count = {}
    for v in range(n):
        count[v] = sum(1 for nb in adj[v]
                       if val[nb] > val[v] or (val[nb] == val[v] and nb > v))

    queue = [v for v in range(n) if count[v] == 0]

    if verbose:
        print(f"\n  CJP init:")
        for v in range(n):
            print(f"    body {v:2d}: val={val[v]:6d}, count={count[v]}")
        print(f"  initial queue: {queue}")

    round_num = 0
    while queue:
        round_num += 1
        if verbose:
            print(f"\n  CJP round {round_num}: queue_size={len(queue)}")

        next_queue = []
        for v in queue:
            c = min_avail(adj, v, colors)
            colors[v] = c
            if verbose and len(queue) <= 20:
                print(f"    body {v:2d} -> color {c}")
            for nb in adj[v]:
                if nb not in colors and count[nb] > 0:
                    count[nb] -= 1
                    if count[nb] == 0:
                        next_queue.append(nb)
        queue = next_queue

    if verbose:
        nc = len(set(colors.values()))
        print(f"  CJP done: {round_num} rounds, {nc} colors")
    return colors, round_num

# ========================================================================
#  CC: Conflict Coloring (正确模拟 GPU 并行语义)
# ========================================================================

def cc_coloring(adj, n, seed=42, verbose=True,
                init_colors=None, init_max_color=None):
    """
    CC 着色, 支持两种模式:

    模式 1 (全量): init_colors=None
      - 所有顶点从 -1 (未着色) 开始
      - max_color 初始化为 4 (论文默认)
      - 所有顶点加入初始队列

    模式 2 (增量/warm-start): init_colors=dict
      - 从上一帧着色结果开始, 已着色顶点保留颜色
      - max_color = init_max_color (上一帧 max_color)
      - 初始队列 = 所有顶点 (CC 的冲突检测会自动跳过无冲突的)
      - 第一轮不做随机选色, 只做冲突检测 (因为已经有颜色了)

    max_color 的含义 (论文 Section III-B):
      - 全局变量, 表示当前允许使用的最大颜色数
      - 初始值 = 4 (论文: "We initialize max-color to 4")
      - 当某个顶点发现 [0, max_color) 的颜色全被邻居占了,
        它 atomicAdd(max_color, 1), 并使用新值作为自己的颜色
      - max_color 只增不减, 快速收敛到足够大的值
      - 论文实测: max_color 在 2-3 轮后就稳定
    """
    rng = random.Random(seed)
    val = {v: rng.randint(0, 100000) for v in range(n)}
    degrees = {v: len(adj[v]) for v in range(n)}

    # 初始化颜色和队列
    if init_colors is not None:
        colors = dict(init_colors)
        max_color = init_max_color if init_max_color else 4
        # 增量模式: 所有顶点都参与第一轮冲突检测
        queue = list(range(n))
        skip_coloring_first_round = True
    else:
        colors = {v: -1 for v in range(n)}
        max_color = 4
        queue = list(range(n))
        skip_coloring_first_round = False

    process_count = {v: 0 for v in range(n)}
    round_num = 0

    while queue:
        round_num += 1
        active_set = set(queue)
        if verbose:
            print(f"\n  CC round {round_num}: {len(queue)} active, max_color={max_color}")

        # ---- Phase 1: 并行着色 ----
        if skip_coloring_first_round and round_num == 1:
            # 增量模式第一轮: 不选色, 直接用上一帧的颜色, 只做冲突检测
            if verbose:
                already_colored = sum(1 for v in queue if colors[v] >= 0)
                uncolored = len(queue) - already_colored
                print(f"    [warm-start] skip coloring, reuse prev frame "
                      f"({already_colored} colored, {uncolored} uncolored)")
            # 对还没有颜色的顶点 (新增的body?), 随机选色
            snapshot = dict(colors)
            for v in queue:
                if colors[v] < 0:
                    process_count[v] += 1
                    used = set()
                    for nb in adj[v]:
                        c = snapshot.get(nb, -1)
                        if c >= 0:
                            used.add(c)
                    available = [c for c in range(max_color) if c not in used]
                    if available:
                        colors[v] = rng.choice(available)
                    else:
                        max_color += 1
                        colors[v] = max_color - 1
        else:
            # 正常 CC: 快照 -> 并行选色 -> 统一写入
            snapshot = dict(colors)
            new_colors = {}
            for v in queue:
                process_count[v] += 1
                used = set()
                for nb in adj[v]:
                    c = snapshot.get(nb, -1)
                    if c >= 0:
                        used.add(c)
                available = [c for c in range(max_color) if c not in used]
                if available:
                    new_colors[v] = rng.choice(available)
                else:
                    # 论文: atomicAdd(max_color, 1), 选新值
                    max_color += 1
                    new_colors[v] = max_color - 1

            for v, c in new_colors.items():
                colors[v] = c

            if verbose and len(queue) <= 20:
                for v in queue:
                    print(f"    body {v:2d}: color={new_colors.get(v, '?')}")

        # ---- Phase 2: 冲突检测 ----
        losers = set()
        conflict_count = 0
        for v in queue:
            for nb in adj[v]:
                if nb not in active_set or nb <= v:
                    continue
                if colors[v] >= 0 and colors[v] == colors[nb]:
                    conflict_count += 1
                    if degrees[v] < degrees[nb]:
                        losers.add(v)
                    elif degrees[v] > degrees[nb]:
                        losers.add(nb)
                    elif val[v] < val[nb]:
                        losers.add(v)
                    else:
                        losers.add(nb)

        if verbose:
            if conflict_count > 0:
                print(f"    conflicts: {conflict_count} pairs, {len(losers)} losers")
                if len(losers) <= 15:
                    for v in sorted(losers):
                        # 找出和谁冲突了
                        conflicting = [nb for nb in adj[v] if nb in active_set
                                       and colors.get(v,-1) >= 0
                                       and colors.get(v,-1) == colors.get(nb,-1)]
                        # 注意 v 是 loser, 颜色已被写入但即将被清除
                        print(f"      body {v:2d} (color={colors[v]}, deg={degrees[v]}) "
                              f"loses to {conflicting}")
            else:
                print(f"    no conflicts!")

        next_queue = []
        for v in losers:
            colors[v] = -1
            next_queue.append(v)
        queue = next_queue

    nc = len(set(c for c in colors.values() if c >= 0))
    avg_proc = sum(process_count.values()) / n if n > 0 else 0
    if verbose:
        print(f"  CC done: {round_num} rounds, {nc} colors, "
              f"avg {avg_proc:.2f} proc/vertex, final max_color={max_color}")
    return colors, round_num, max_color

# ========================================================================
#  JP (经典 Jones-Plassmann, 用于基线对比)
# ========================================================================

def jp_coloring(adj, n, seed=42, verbose=False):
    rng = random.Random(seed)
    colors = {}
    round_num = 0
    while len(colors) < n:
        round_num += 1
        weights = {v: rng.randint(0, 100000) for v in range(n) if v not in colors}
        mis = []
        for v in weights:
            is_max = True
            for nb in adj[v]:
                if nb in colors:
                    continue
                if weights[nb] > weights[v] or (weights[nb] == weights[v] and nb > v):
                    is_max = False
                    break
            if is_max:
                mis.append(v)
        for v in mis:
            colors[v] = min_avail(adj, v, colors)
    return colors, round_num

# ========================================================================
#  主实验
# ========================================================================

def experiment_1_cc_full():
    """CC 全量着色, 展示 max_color 的增长过程."""
    print(f"\n{SEP}")
    print("Experiment 1: CC full coloring (max_color growth)")
    print(SEP)

    adj = build_adj(PYRAMID_EDGES, PYRAMID_N)
    print(f"\nPyramid graph: {PYRAMID_N} bodies, {len(PYRAMID_EDGES)} edges")

    print(f"\n--- CC (init max_color=4) ---")
    print("  max_color starts at 4 (paper default)")
    print("  When a vertex finds all [0,4) used by neighbors,")
    print("  it does atomicAdd(max_color,1) and takes the new value.")
    cc_colors, cc_rounds, mc = cc_coloring(adj, PYRAMID_N, seed=42)
    validate(adj, cc_colors, PYRAMID_N, "CC")
    print(f"\n  Color assignment: {[cc_colors[v] for v in range(PYRAMID_N)]}")
    return adj, cc_colors, mc


def experiment_2_cc_incremental():
    """CC 增量着色: 用上一帧结果 warm-start CC."""
    print(f"\n\n{SEP}")
    print("Experiment 2: CC-Incremental (warm-start from prev frame)")
    print(SEP)

    adj0 = build_adj(PYRAMID_EDGES, PYRAMID_N)

    # Frame 0: CC 全量着色
    print(f"\n=== Frame 0: CC full coloring ===")
    f0_colors, f0_rounds, f0_mc = cc_coloring(adj0, PYRAMID_N, seed=42, verbose=False)
    nc0 = len(set(c for c in f0_colors.values() if c >= 0))
    validate(adj0, f0_colors, PYRAMID_N, "Frame 0 CC")
    print(f"  Frame 0: {nc0} colors, {f0_rounds} rounds, max_color={f0_mc}")
    print(f"  Colors: {[f0_colors[v] for v in range(PYRAMID_N)]}")

    # 模拟帧间变化
    removed = [
        (9,13),(10,13),(11,13),(12,13),  # body13 飞走
        (0,9),(1,9),
    ]
    added_raw = [(13,14), (9,12), (6,9)]
    existing = set(tuple(sorted(e)) for e in PYRAMID_EDGES)
    real_removed = [e for e in removed if tuple(sorted(e)) in existing]
    real_added = [e for e in added_raw if tuple(sorted(e)) not in existing]

    new_edges = [e for e in PYRAMID_EDGES
                 if tuple(sorted(e)) not in set(tuple(sorted(r)) for r in real_removed)]
    for e in real_added:
        new_edges.append(e)
    adj1 = build_adj(new_edges, PYRAMID_N)

    print(f"\n=== Frame 1: graph changed ===")
    print(f"  Removed: {real_removed}")
    print(f"  Added:   {real_added}")

    # 先看上一帧的颜色在新图上有哪些冲突
    conflicts_before = []
    for u, v in real_added:
        cu = f0_colors.get(u, -1)
        cv = f0_colors.get(v, -1)
        if cu >= 0 and cv >= 0 and cu == cv:
            conflicts_before.append((u, v, cu))
    print(f"\n  Conflicts from new edges using old colors:")
    if conflicts_before:
        for u, v, c in conflicts_before:
            print(f"    edge ({u},{v}): both have color {c} -> CONFLICT")
    else:
        print(f"    None! Old coloring is already valid on new graph")

    # 方法 A: CC 全量重着色
    print(f"\n--- Method A: CC full recoloring ---")
    full_colors, full_rounds, full_mc = cc_coloring(adj1, PYRAMID_N, seed=100, verbose=False)
    nc_full = len(set(c for c in full_colors.values() if c >= 0))
    validate(adj1, full_colors, PYRAMID_N, "CC full")
    print(f"  {nc_full} colors, {full_rounds} rounds, max_color={full_mc}")
    print(f"  Colors: {[full_colors[v] for v in range(PYRAMID_N)]}")

    # 方法 B: CC-Incremental (核心!)
    print(f"\n--- Method B: CC-Incremental (warm-start) ---")
    print(f"  Key idea:")
    print(f"    1. Init colors = prev frame result (not random!)")
    print(f"    2. Init max_color = prev frame max_color = {f0_mc}")
    print(f"    3. Round 1: SKIP coloring phase, only detect conflicts")
    print(f"       -> CC's conflict resolution naturally fixes violations")
    print(f"    4. Round 2+: only recolor the losers from round 1")
    incr_colors, incr_rounds, incr_mc = cc_coloring(
        adj1, PYRAMID_N, seed=100,
        init_colors=f0_colors, init_max_color=f0_mc, verbose=True)
    nc_incr = len(set(c for c in incr_colors.values() if c >= 0))
    validate(adj1, incr_colors, PYRAMID_N, "CC-Incr")
    print(f"  Colors: {[incr_colors[v] for v in range(PYRAMID_N)]}")

    # 对比
    print(f"\n=== Comparison ===")
    print(f"  +-------------------+--------+--------+-----------+")
    print(f"  |     Method        | Colors | Rounds | max_color |")
    print(f"  +-------------------+--------+--------+-----------+")
    print(f"  | CC full           | {nc_full:6d} | {full_rounds:6d} | {full_mc:9d} |")
    print(f"  | CC-Incremental    | {nc_incr:6d} | {incr_rounds:6d} | {incr_mc:9d} |")
    print(f"  +-------------------+--------+--------+-----------+")

    return f0_colors, f0_mc, adj1


def experiment_3_multi_frame():
    """多帧连续模拟: 全量CC vs CC-Incremental."""
    print(f"\n\n{SEP}")
    print("Experiment 3: Multi-frame CC vs CC-Incremental")
    print(SEP)

    rng = random.Random(123)
    edges_set = set(tuple(sorted(e)) for e in PYRAMID_EDGES)
    adj = build_adj(PYRAMID_EDGES, PYRAMID_N)

    NUM_FRAMES = 8
    print(f"\n  Simulating {NUM_FRAMES} frames, 2-4 edge changes per frame")
    print(f"  Comparing: CC full (from scratch) vs CC-Incremental (warm-start)")

    prev_colors = None
    prev_mc = None
    all_results = []

    for frame in range(NUM_FRAMES):
        if frame == 0:
            colors, rounds, mc = cc_coloring(adj, PYRAMID_N, seed=42, verbose=False)
            nc = len(set(c for c in colors.values() if c >= 0))
            prev_colors = colors
            prev_mc = mc
            all_results.append({
                'frame': frame, 'removed': 0, 'added': 0,
                'nc_full': nc, 'r_full': rounds,
                'nc_incr': nc, 'r_incr': rounds,
                'conflicts': 0, 'losers': 0,
            })
            print(f"  Frame {frame}: init -> {nc} colors, {rounds} rounds, max_color={mc}")
            continue

        # 随机增删边
        to_remove = []
        current_edges = list(edges_set)
        for _ in range(rng.randint(1, 3)):
            if current_edges:
                e = rng.choice(current_edges)
                to_remove.append(e)
                current_edges.remove(e)

        to_add = []
        for _ in range(rng.randint(1, 3)):
            a, b = rng.randint(0, PYRAMID_N-1), rng.randint(0, PYRAMID_N-1)
            if a != b:
                e = tuple(sorted((a, b)))
                if e not in edges_set:
                    to_add.append(e)

        for e in to_remove:
            edges_set.discard(e)
        for e in to_add:
            edges_set.add(e)
        adj = build_adj(list(edges_set), PYRAMID_N)

        # CC 全量
        full_c, full_r, full_mc = cc_coloring(adj, PYRAMID_N, seed=42+frame, verbose=False)
        nc_full = len(set(c for c in full_c.values() if c >= 0))

        # CC-Incremental
        incr_c, incr_r, incr_mc = cc_coloring(
            adj, PYRAMID_N, seed=42+frame,
            init_colors=prev_colors, init_max_color=prev_mc, verbose=False)
        nc_incr = len(set(c for c in incr_c.values() if c >= 0))
        validate(adj, incr_c, PYRAMID_N, f"Frame {frame} CC-Incr")

        # 统计上一帧颜色在新图上的冲突数
        conflict_pairs = 0
        for u, v in to_add:
            cu = prev_colors.get(u, -1)
            cv = prev_colors.get(v, -1)
            if cu >= 0 and cv >= 0 and cu == cv:
                conflict_pairs += 1

        all_results.append({
            'frame': frame, 'removed': len(to_remove), 'added': len(to_add),
            'nc_full': nc_full, 'r_full': full_r,
            'nc_incr': nc_incr, 'r_incr': incr_r,
            'conflicts': conflict_pairs,
        })

        prev_colors = incr_c
        prev_mc = incr_mc

        tag = " *" if conflict_pairs > 0 else ""
        print(f"  Frame {frame}: -{len(to_remove)}+{len(to_add)} edges, "
              f"full={nc_full}c/{full_r}r, incr={nc_incr}c/{incr_r}r, "
              f"new conflicts={conflict_pairs}{tag}")

    print(f"\n=== Summary ===")
    print(f"  {'Frame':>5} | {'Del':>3} | {'Add':>3} | "
          f"{'Full-C':>6} | {'Full-R':>6} | "
          f"{'Incr-C':>6} | {'Incr-R':>6} | {'Confl':>5}")
    print(f"  {'-'*5}-+-{'-'*3}-+-{'-'*3}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}")
    for r in all_results:
        print(f"  {r['frame']:5d} | {r['removed']:3d} | {r['added']:3d} | "
              f"{r['nc_full']:6d} | {r['r_full']:6d} | "
              f"{r['nc_incr']:6d} | {r['r_incr']:6d} | {r['conflicts']:5d}")

    # 统计
    full_total_rounds = sum(r['r_full'] for r in all_results[1:])
    incr_total_rounds = sum(r['r_incr'] for r in all_results[1:])
    frames_with_conflict = sum(1 for r in all_results[1:] if r['conflicts'] > 0)
    print(f"\n  Full CC total rounds:  {full_total_rounds}")
    print(f"  Incr CC total rounds:  {incr_total_rounds}")
    if full_total_rounds > 0:
        print(f"  Round savings: {100*(1-incr_total_rounds/full_total_rounds):.0f}%")
    print(f"  Frames with new conflicts: {frames_with_conflict}/{NUM_FRAMES-1}")


def experiment_4_max_color_explained():
    """深入解释 max_color 的来源和溢出处理."""
    print(f"\n\n{SEP}")
    print("Experiment 4: max_color deep dive")
    print(SEP)

    print("""
  Q: max_color 是怎么得到的?
  A: 论文初始值 = 4 (硬编码). 这是色数 chi(G) 的下界估计.

  Q: 为什么是 4?
  A: 大多数稀疏图 (如碰撞图) 的色数 >= 3-4.
     选太小: 前几轮很多顶点溢出, 需要 atomicAdd 扩展
     选太大: 第一轮就用了太多颜色, 浪费并行度
     4 是个合理的折中. 论文实测 2-3 轮后 max_color 就稳定了.

  Q: 如果超过 max_color 怎么办?
  A: 当顶点 v 发现 [0, max_color) 全被邻居占了:
     1. v 执行 atomicAdd(&max_color, 1)  // GPU 原子操作
     2. v 拿到返回值 old_max_color
     3. v 的颜色 = old_max_color  // 即新增的那个颜色号
     同一轮多个顶点可能同时溢出, 各自 atomicAdd, 各拿不同的新颜色号.
     max_color 只增不减, 保证收敛.

  Q: CC-Incremental 中 max_color 怎么设?
  A: max_color = prev_frame_max_color (上一帧最终值).
     因为上一帧的图结构和当前帧相似, 色数也相似.
     如果新图更密, CC 会自动 atomicAdd 扩展.
     如果新图更稀疏, 多余的颜色不会被使用 (但 max_color 不缩减).
""")

    # 用一个更密的图演示 max_color 增长过程
    print("  --- Demo: max_color growth on a dense subgraph ---")
    dense_n = 6
    # K6 完全图: 每个顶点和其他5个都相邻, chi = 6
    dense_edges = [(i, j) for i in range(dense_n) for j in range(i+1, dense_n)]
    adj = build_adj(dense_edges, dense_n)
    print(f"  K6 complete graph: {dense_n} vertices, {len(dense_edges)} edges, chi=6")
    print(f"  Starting with max_color=4, need 6 colors -> will overflow twice\n")

    colors, rounds, mc = cc_coloring(adj, dense_n, seed=42, verbose=True)
    validate(adj, colors, dense_n, "K6 CC")
    print(f"\n  Final max_color = {mc} (grew from 4 to {mc})")


# ========================================================================
#  主函数
# ========================================================================

def main():
    print("GPU Graph Coloring: CC / CC-Incremental Experiments")
    print(f"Paper: 'Efficient Algorithms for Graph Coloring on GPU' (ICPADS 2018)")
    print(SEP)

    experiment_1_cc_full()
    experiment_2_cc_incremental()
    experiment_3_multi_frame()
    experiment_4_max_color_explained()

    print(f"\n\n{SEP}")
    print("Key Takeaways")
    print(SEP)
    print("""
  1. CC warm-start: Replace random init with prev frame coloring.
     - Round 1 only does conflict detection (no coloring phase)
     - Only vertices on new conflicting edges need recoloring
     - Most frames: 1-2 rounds vs 3+ rounds for full CC

  2. max_color management:
     - Init = 4 (or prev frame value for incremental)
     - Overflow: atomicAdd(max_color, 1), take new color
     - Only grows, never shrinks
     - Converges in 2-3 rounds

  3. Why CC is ideal for incremental:
     - CC already assumes conflicts exist and handles them gracefully
     - Warm-start is just "more conflicts already resolved"
     - No need for separate diff/mark/recolor kernels
     - Same kernel code, different initial state

  4. GPU implementation:
     - Full CC: colors[] = -1, max_color = 4
     - Incremental: colors[] = prev_colors[], max_color = prev_max_color
     - Everything else (snapshot, parallel color, conflict detect) is identical
""")


if __name__ == "__main__":
    main()
