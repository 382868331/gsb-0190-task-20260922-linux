"""demo.py — 体素等值面提取的固定输入演示（离线、确定性、无第三方依赖）。

运行：python demo.py
内容：
  1. 正常结果：16^3 球谐场（解析球）整块提取，输出面积与非流形边计数等统计；
  2. 分块拼接：同一张 16^3 网格切成 8 个相邻体块分别提取再拼接，结果与
     整块提取逐项相同（顶点、面、面积、接缝边计数），并展示接缝共享顶点
     唯一（不靠浮点距离）；
  3. 拒绝边界：实际触发两类 InvalidInputError——标量含 NaN、接缝两侧
     共享节点值不一致。
"""

import itertools
import math
import time

from iso3d import (
    InvalidInputError,
    ScalarBlock3D,
    ScalarField3D,
    extract_isosurface,
    extract_isosurface_blocks,
)


def sphere_values(origin, ni, nj, nk, n):
    """确定性生成：f = 到中心距离的平方，等值面即半径 4 的球面。"""
    center = (n - 1) / 2.0
    oi, oj, ok = origin
    return tuple(
        (oi + li - center) ** 2 + (oj + lj - center) ** 2 + (ok + lk - center) ** 2
        for lk in range(nk)
        for lj in range(nj)
        for li in range(ni)
    )


def split_grid(n, counts):
    """每轴按节点数 counts 切段（相邻段共享一层节点），生成全部体块。"""
    def starts(cs):
        s = 0
        out = []
        for c in cs:
            out.append((s, c))
            s += c - 1
        return out

    blocks = []
    for (si, ni_), (sj, nj_), (sk, nk_) in itertools.product(
        starts(counts), starts(counts), starts(counts)
    ):
        blocks.append(
            ScalarBlock3D(
                origin=(si, sj, sk),
                ni=ni_,
                nj=nj_,
                nk=nk_,
                values=sphere_values((si, sj, sk), ni_, nj_, nk_, n),
            )
        )
    return blocks


def demo_sphere():
    print("=== 1. 正常结果：16^3 网格上的球面等值面（整块提取，iso = r^2 = 16）===")
    n = 16
    field = ScalarField3D(
        ni=n, nj=n, nk=n, values=sphere_values((0, 0, 0), n, n, n, n)
    )
    t0 = time.perf_counter()
    mesh = extract_isosurface(field, 16.0)
    elapsed = time.perf_counter() - t0
    s = mesh.stats
    analytic = 4.0 * math.pi * 16.0
    print(f"节点数           : {s.node_count} (16^3 满上限)")
    print(f"立方体/四面体    : {s.cell_count} / {s.tetrahedron_count}")
    print(f"顶点数 / 三角形数: {s.vertex_count} / {s.face_count}")
    print(f"总表面积         : {s.total_area:.6f}  (解析球面积 {analytic:.6f})")
    print(f"边界边 / 非流形边: {s.boundary_edge_count} / {s.nonmanifold_edge_count}"
          f"  (闭合球面应为 0 / 0)")
    print(f"丢弃零面积面数   : {s.degenerate_face_count}")
    print(f"提取耗时         : {elapsed * 1000.0:.1f} ms")
    ok = (
        s.boundary_edge_count == 0
        and s.nonmanifold_edge_count == 0
        and 0.75 * analytic < s.total_area <= analytic * (1.0 + 1e-9)
    )
    print(f"闭合性与面积合理性: {'通过' if ok else '未通过'}")
    print()
    return mesh


def demo_blocks(whole):
    print("=== 2. 分块拼接：16^3 网格沿中点切成 2x2x2 = 8 个相邻体块 ===")
    n = 16
    blocks = split_grid(n, (8, 9))  # 节点数 8 和 9，共享 i/j/k = 7 这一层
    print(f"体块数           : {len(blocks)}（每块 7 或 8 个立方体/轴，共享一层节点）")
    t0 = time.perf_counter()
    merged = extract_isosurface_blocks(blocks, 16.0, shape=(n, n, n))
    elapsed = time.perf_counter() - t0
    sm, wm = merged.stats, whole.stats
    print(f"拼接后顶点/三角形: {sm.vertex_count} / {sm.face_count}"
          f"  （整块 {wm.vertex_count} / {wm.face_count}）")
    same = (
        merged.vertices == whole.vertices
        and merged.faces == whole.faces
        and merged.degenerate_faces == whole.degenerate_faces
        and merged.boundary_edges == whole.boundary_edges
        and merged.nonmanifold_edges == whole.nonmanifold_edges
        and merged.edge_use_count == whole.edge_use_count
    )
    area_delta = abs(sm.total_area - wm.total_area)
    print(f"总面积差         : {area_delta:.3e}（同一浮点运算序列，应为 0.0）")
    print(f"接缝非流形边数   : {sm.nonmanifold_edge_count}（应为 0）")
    print(f"拼接耗时         : {elapsed * 1000.0:.1f} ms")
    print(f"与整块提取逐项一致: {'通过' if same and area_delta == 0.0 else '未通过'}")

    # 手算展示接缝唯一所有者：取共享面 i=7 与 k=7 交线上一条穿越球面
    # 的网格棱 (7,3,7)-(7,4,7)：
    #   f(7,3,7)=20.75 > 16（球外），f(7,4,7)=12.75 < 16（球内），
    # 故该棱被等值面穿越；它同时落在 i、k 两个共享面上，被 4 个相邻
    # 体块引用，但只应产生一个顶点。
    def nid(i, j, k):
        return (k * n + j) * n + i

    key = tuple(sorted((nid(7, 3, 7), nid(7, 4, 7))))
    seam_verts = [v for v in merged.vertices if v.edge_key == key]
    print(f"共享棱 {key} 上的交点数: {len(seam_verts)}（4 块引用，唯一所有者，应为 1）")
    print()


def demo_rejection():
    print("=== 3a. 拒绝边界：整块标量场含 NaN ===")
    values = [0.0] * (4 * 4 * 4)
    values[37] = math.nan
    try:
        field = ScalarField3D(ni=4, nj=4, nk=4, values=values)
        extract_isosurface(field, 0.5)
    except InvalidInputError as exc:
        print(f"已拒绝并定位: InvalidInputError: {exc}")
        print(f"错误参数名   : {exc.parameter}")
    else:
        print("未触发拒绝（不符合预期）")
    print()

    print("=== 3b. 拒绝边界：分块拼接时接缝两侧共享节点值不一致 ===")
    n = 4

    def plane_values(origin, ni, nj, nk):
        oi, oj, ok = origin
        return tuple(
            float((oi + li) + (oj + lj) + (ok + lk))
            for lk in range(nk)
            for lj in range(nj)
            for li in range(ni)
        )

    b0 = ScalarBlock3D((0, 0, 0), 2, n, n, plane_values((0, 0, 0), 2, n, n))
    right_vals = list(plane_values((1, 0, 0), 3, n, n))
    right_vals[0] = 99.0  # 篡改共享节点 (1,0,0)：与左块不一致
    b1 = ScalarBlock3D((1, 0, 0), 3, n, n, right_vals)
    before = (b0.values, b1.values)
    try:
        extract_isosurface_blocks([b0, b1], 2.5, shape=(n, n, n))
    except InvalidInputError as exc:
        print(f"已拒绝并定位: InvalidInputError: {exc}")
        print(f"错误参数名   : {exc.parameter}")
    else:
        print("未触发拒绝（不符合预期）")
    untouched = (b0.values, b1.values) == before
    print(f"失败后输入体块未被修改: {'通过' if untouched else '未通过'}")
    print()


def main():
    t0 = time.perf_counter()
    whole = demo_sphere()
    demo_blocks(whole)
    demo_rejection()
    print(f"demo 总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
