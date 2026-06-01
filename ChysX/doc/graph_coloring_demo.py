#!/usr/bin/env python3
"""
四种 GPU 并行图着色算法的 Python 演示。
在一个简单的碰撞图上逐步输出每种算法的完整执行过程。

用法: python graph_coloring_demo.py
"""

import random
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════
#  构建一个小型碰撞图（模拟金字塔底部 10 个 body 的碰撞关系）
# ═══════════════════════════════════════════════════════════════════════════

# 10 个 body 的碰撞邻接表
# 类似金字塔底层 3x3+1 的碰撞拓扑
EDGES = [
    (0, 1), (0, 3),
    (1, 2), (1, 3), (1, 4),
    (2, 4), (2, 5),
    (3, 4), (3, 6), (3, 7),
    (4, 5), (4, 7), (4, 8),
    (5, 8), (5, 9),
    (6, 7),
    (7, 8),
    (8, 9),
]
N = 10

def build_adj():
    """构建邻接表。"""
    adj = defaultdict(set)
    for a, b in EDGES:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def degree(adj, v):
    return len(adj[v])

def print_graph(adj):
    print("  碰撞图（邻接表）：")
    for v in range(N):
        nbs = sorted(adj[v])
        print(f"    body {v}: 度={len(nbs)}, 邻居={nbs}")
    max_d = max(len(adj[v]) for v in range(N))
    min_d = min(len(adj[v]) for v in range(N) if len(adj[v]) > 0)
    print(f"  最大度 Δ={max_d}, 最小度 δ={min_d}")

def validate(adj, colors, name):
    violations = 0
    for a, b in EDGES:
        if colors[a] == colors[b]:
            violations += 1
    nc = len(set(colors.values()))
    if violations > 0:
        print(f"  [FAIL] {name}: {violations} conflicts!")
    else:
        print(f"  [OK] {name}: {nc} colors, 0 conflicts")
    return violations == 0

SEPARATOR = "=" * 70

# ═══════════════════════════════════════════════════════════════════════════
#  算法 1: Brooks-Vizing 随机着色 (Vivace)
# ═══════════════════════════════════════════════════════════════════════════

def vivace_coloring(adj, seed=42):
    print(f"\n{SEPARATOR}")
    print("算法 1: Brooks-Vizing 随机着色（Vivace）")
    print(SEPARATOR)
    print()
    print("核心思想：每个顶点维护一个\"调色盘\"（可选颜色集合）。")
    print("每轮三步：(1)随机选色 (2)冲突检测 (3)补充空调色盘。")
    print("调色盘的大小初始为 floor(degree/s)+1, 其中 s = 最小度。")
    print()

    rng = random.Random(seed)

    # 初始化
    min_deg = min(len(adj[v]) for v in range(N) if len(adj[v]) > 0)
    s = max(1, min_deg)
    print(f"缩减因子 s = min_degree = {s}")
    print()

    colors = {v: -1 for v in range(N)}
    palettes = {}
    for v in range(N):
        d = len(adj[v])
        palette_size = max(1, d // s) + 1
        palettes[v] = set(range(palette_size))
        print(f"  body {v}: 度={d}, 初始调色盘={sorted(palettes[v])}")

    next_new_color = max(max(p) for p in palettes.values() if p) + 1
    print(f"\n下一个可用新颜色 = {next_new_color}")

    round_num = 0
    while any(colors[v] < 0 for v in range(N)):
        round_num += 1
        uncolored = [v for v in range(N) if colors[v] < 0]
        print(f"\n--- 第 {round_num} 轮 ---")
        print(f"未着色: {uncolored}")

        # Step 1: Tentative Coloring
        print(f"\n  步骤 1 (Tentative Coloring): 每个未着色顶点从调色盘随机选一个颜色")
        tentative = {}
        for v in uncolored:
            if not palettes[v]:
                tentative[v] = -1
                print(f"    body {v}: 调色盘为空，跳过")
            else:
                c = rng.choice(sorted(palettes[v]))
                tentative[v] = c
                print(f"    body {v}: 从 {sorted(palettes[v])} 中随机选了颜色 {c}")

        # Step 2: Conflict Resolution
        print(f"\n  步骤 2 (Conflict Resolution): 检查邻居是否选了相同颜色")
        confirmed = set()
        for v in uncolored:
            if tentative[v] < 0:
                continue
            conflict = False
            for nb in adj[v]:
                if nb in tentative and tentative[nb] == tentative[v]:
                    # 匈牙利启发式：索引大的胜出
                    if v < nb:
                        conflict = True
                        print(f"    body {v}: 颜色 {tentative[v]} 与邻居 body {nb} 冲突"
                              f"（{v}<{nb}，body {v} 放弃）")
                        break
            if not conflict:
                confirmed.add(v)
                colors[v] = tentative[v]
                print(f"    body {v}: 颜色 {tentative[v]} 无冲突 -> 确认着色!")

        # 从邻居调色盘中移除已确认的颜色
        for v in confirmed:
            c = colors[v]
            for nb in adj[v]:
                if colors[nb] < 0:
                    palettes[nb].discard(c)

        if confirmed:
            print(f"  已着色: {sorted(confirmed)}")
            print(f"  更新后的调色盘：")
            for v in range(N):
                if colors[v] < 0:
                    print(f"    body {v}: {sorted(palettes[v])}")

        # Step 3: Feed the Hungry
        hungry = [v for v in range(N) if colors[v] < 0 and not palettes[v]]
        if hungry:
            print(f"\n  步骤 3 (Feed the Hungry): 调色盘为空的顶点 = {hungry}")
            print(f"    所有饥饿顶点共享新颜色 {next_new_color}")
            for v in hungry:
                palettes[v].add(next_new_color)
            next_new_color += 1
        else:
            print(f"\n  步骤 3 (Feed the Hungry): 无饥饿顶点")

    print(f"\n完成！共 {round_num} 轮")
    print(f"着色结果: {dict(sorted(colors.items()))}")
    print(f"颜色数: {len(set(colors.values()))}")
    validate(adj, colors, "Vivace")
    return colors


# ═══════════════════════════════════════════════════════════════════════════
#  算法 2: Luby MIS 着色
# ═══════════════════════════════════════════════════════════════════════════

def luby_coloring(adj, seed=42):
    print(f"\n{SEPARATOR}")
    print("算法 2: Luby MIS (最大独立集) 着色")
    print(SEPARATOR)
    print()
    print("核心思想：每轮找一个\"最大独立集\"(MIS)——通过随机权重选局部极大值。")
    print("独立集中的所有顶点赋同一颜色（因为它们互不相邻）。")
    print("然后移除已着色顶点，用下一个颜色重复。")
    print()

    rng = random.Random(seed)
    colors = {v: -1 for v in range(N)}
    current_color = 0

    round_num = 0
    while any(colors[v] < 0 for v in range(N)):
        round_num += 1
        uncolored = [v for v in range(N) if colors[v] < 0]
        print(f"--- 第 {round_num} 轮 (颜色 {current_color}) ---")
        print(f"未着色: {uncolored}")

        # 赋随机权重
        weights = {v: rng.randint(0, 1000) for v in uncolored}
        print(f"  随机权重: {weights}")

        # 找局部极大值 -> 独立集
        mis = []
        for v in uncolored:
            is_max = True
            for nb in adj[v]:
                if colors[nb] >= 0:
                    continue
                if weights[nb] > weights[v] or (weights[nb] == weights[v] and nb > v):
                    is_max = False
                    break
            if is_max:
                mis.append(v)

        print(f"  局部极大值(独立集): {mis}")
        print(f"  验证MIS中任意两点互不相邻: ", end="")
        valid_mis = True
        for i, a in enumerate(mis):
            for b in mis[i+1:]:
                if b in adj[a]:
                    valid_mis = False
                    print(f"X({a},{b}相邻!) ", end="")
        print("OK" if valid_mis else "")

        for v in mis:
            colors[v] = current_color
            print(f"    body {v} -> 颜色 {current_color}")

        current_color += 1
        print()

    print(f"完成！共 {round_num} 轮")
    print(f"着色结果: {dict(sorted(colors.items()))}")
    print(f"颜色数: {len(set(colors.values()))}")
    validate(adj, colors, "Luby MIS")
    return colors


# ═══════════════════════════════════════════════════════════════════════════
#  算法 3: Jones-Plassmann (JP) 着色
# ═══════════════════════════════════════════════════════════════════════════

def jp_coloring(adj, seed=42):
    print(f"\n{SEPARATOR}")
    print("算法 3: Jones-Plassmann (JP) 着色")
    print(SEPARATOR)
    print()
    print("核心思想：和 Luby 类似，用随机权重找独立集，但关键区别是：")
    print("每个独立集顶点不是赋同一颜色，而是赋\"最小可用色\"——")
    print("即不与已着色邻居冲突的最小颜色号。这大幅减少了总颜色数。")
    print()

    rng = random.Random(seed)
    colors = {v: -1 for v in range(N)}

    round_num = 0
    while any(colors[v] < 0 for v in range(N)):
        round_num += 1
        uncolored = [v for v in range(N) if colors[v] < 0]
        print(f"--- 第 {round_num} 轮 ---")
        print(f"未着色: {uncolored}")

        # 赋随机权重
        weights = {v: rng.randint(0, 1000) for v in uncolored}
        print(f"  随机权重: {weights}")

        # 找局部极大值 -> 独立集
        mis = []
        for v in uncolored:
            is_max = True
            for nb in adj[v]:
                if colors[nb] >= 0:
                    continue
                if weights[nb] > weights[v] or (weights[nb] == weights[v] and nb > v):
                    is_max = False
                    break
            if is_max:
                mis.append(v)
        print(f"  局部极大值(独立集): {mis}")

        # 每个独立集顶点赋最小可用色
        for v in mis:
            used = set()
            for nb in adj[v]:
                if colors[nb] >= 0:
                    used.add(colors[nb])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            print(f"    body {v}: 邻居已用色={used if used else '{}'} -> 最小可用色={c}")

        print()

    print(f"完成！共 {round_num} 轮")
    print(f"着色结果: {dict(sorted(colors.items()))}")
    print(f"颜色数: {len(set(colors.values()))}")
    validate(adj, colors, "JP")
    return colors


# ═══════════════════════════════════════════════════════════════════════════
#  算法 4: LDF (Largest-Degree-First) 着色
# ═══════════════════════════════════════════════════════════════════════════

def ldf_coloring(adj):
    print(f"\n{SEPARATOR}")
    print("算法 4: LDF (Largest-Degree-First) 着色")
    print(SEPARATOR)
    print()
    print("核心思想：和 JP 类似，但优先级不是随机权重，而是按度数大小。")
    print("度最高的顶点优先着色，度相同时索引大的优先。")
    print("每轮后更新残余图的度数（移除已着色邻居）。")
    print("颜色数最少，但轮次最多。")
    print()

    colors = {v: -1 for v in range(N)}
    # 初始残余度 = 实际度
    residual_deg = {v: len(adj[v]) for v in range(N)}

    print("初始度数:")
    for v in range(N):
        print(f"  body {v}: 度={residual_deg[v]}")

    round_num = 0
    while any(colors[v] < 0 for v in range(N)):
        round_num += 1
        uncolored = [v for v in range(N) if colors[v] < 0]
        print(f"\n--- 第 {round_num} 轮 ---")
        print(f"未着色: {uncolored}")
        print(f"  残余度: {{{', '.join(f'{v}:{residual_deg[v]}' for v in uncolored)}}}")

        # 找度最大的局部极大值
        mis = []
        for v in uncolored:
            is_max = True
            for nb in adj[v]:
                if colors[nb] >= 0:
                    continue
                if (residual_deg[nb] > residual_deg[v] or
                        (residual_deg[nb] == residual_deg[v] and nb > v)):
                    is_max = False
                    break
            if is_max:
                mis.append(v)
        print(f"  度最大局部极大值: {mis}")

        for v in mis:
            used = set()
            for nb in adj[v]:
                if colors[nb] >= 0:
                    used.add(colors[nb])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            print(f"    body {v} (度={residual_deg[v]}): 邻居已用色={used if used else '{}'}"
                  f" -> 最小可用色={c}")

        # 更新残余度
        for v in uncolored:
            if colors[v] >= 0:
                continue
            d = sum(1 for nb in adj[v] if colors[nb] < 0)
            residual_deg[v] = d

    print(f"\n完成！共 {round_num} 轮")
    print(f"着色结果: {dict(sorted(colors.items()))}")
    print(f"颜色数: {len(set(colors.values()))}")
    validate(adj, colors, "LDF")
    return colors


# ═══════════════════════════════════════════════════════════════════════════
#  比较汇总
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("GPU 并行图着色算法演示")
    print("=" * 70)
    print()
    adj = build_adj()
    print_graph(adj)

    results = {}
    results['Vivace'] = vivace_coloring(adj)
    results['Luby'] = luby_coloring(adj)
    results['JP'] = jp_coloring(adj)
    results['LDF'] = ldf_coloring(adj)

    print(f"\n{'=' * 70}")
    print("最终对比")
    print(f"{'=' * 70}")
    print(f"{'算法':<12} {'颜色数':>6} {'着色结果'}")
    print("-" * 60)
    for name, col in results.items():
        nc = len(set(col.values()))
        seq = [col[v] for v in range(N)]
        print(f"{name:<12} {nc:>6}   {seq}")

    print()
    print("关键观察：")
    print("  - LDF 颜色最少：优先处理高度顶点，贪心效果最好")
    print("  - JP 接近最优：随机独立集 + 最小可用色")
    print("  - Luby 颜色较多：独立集内所有顶点赋同一颜色，无法复用")
    print("  - Vivace 速度最快：调色盘机制使得每轮着色更多顶点")


if __name__ == "__main__":
    main()
