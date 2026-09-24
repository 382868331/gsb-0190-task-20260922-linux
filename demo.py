"""demo.py — 体素等值面提取的固定输入演示（离线、确定性、无第三方依赖）。

运行：python demo.py
内容：
  1. 正常结果：16^3 球谐场（解析球）等值面提取，输出面积与非流形边计数等统计；
  2. 相邻体块拼接：两个同分辨率体块沿面接缝合并，演示共享采样点唯一所有者、
     接缝顶点不重复，拼接结果与整块提取逐面一致（面/边/角接缝，2x2x2 八块）；
  3. 拒绝边界：标量场含 NaN；接缝节点值冲突——均触发可定位的 InvalidInputError。
"""

import math
import time

from iso3d import (
    InvalidInputError,
    ScalarField3D,
    ScalarFieldBlock,
    extract_isosurface,
    extract_isosurface_blocks,
)


def sphere_field(n=16, radius=4.0):
    """确定性生成：f(i,j,k) = 到中心距离的平方，等值面即球面。"""
    center = (n - 1) / 2.0
    values = tuple(
        (i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2
        for k in range(n)
        for j in range(n)
        for i in range(n)
    )
    return ScalarField3D(ni=n, nj=n, nk=n, values=values), radius ** 2, center


def demo_sphere():
    print("=== 正常结果：16^3 网格上的球面等值面（半径 4，iso = r^2 = 16）===")
    field, iso, center = sphere_field()
    t0 = time.perf_counter()
    mesh = extract_isosurface(field, iso)
    elapsed = time.perf_counter() - t0
    s = mesh.stats
    analytic = 4.0 * math.pi * 16.0
    print(f"节点数           : {s.node_count} (16^3 满上限)")
    print(f"立方体/四面体    : {s.cell_count} / {s.tetrahedron_count}")
    print(f"顶点数           : {s.vertex_count}")
    print(f"三角形数         : {s.face_count}")
    print(f"总表面积         : {s.total_area:.6f}  (解析球面积 {analytic:.6f}，"
          f"多边形逼近应小于它)")
    print(f"边界边数         : {s.boundary_edge_count}  (闭合曲面应为 0)")
    print(f"非流形边数       : {s.nonmanifold_edge_count}")
    print(f"丢弃零面积面数   : {s.degenerate_face_count}")
    print(f"提取耗时         : {elapsed * 1000.0:.1f} ms")
    ok = (
        s.boundary_edge_count == 0
        and s.nonmanifold_edge_count == 0
        and 0.75 * analytic < s.total_area <= analytic * (1.0 + 1e-9)
    )
    print(f"闭合性与面积合理性: {'通过' if ok else '未通过'}")
    print()


def _block(n, func, oi, oj, ok):
    ni, nj, nk = n
    values = tuple(
        float(func(oi + i, oj + j, ok + k))
        for k in range(nk)
        for j in range(nj)
        for i in range(ni)
    )
    field = ScalarField3D(ni=ni, nj=nj, nk=nk, values=values)
    return ScalarFieldBlock(origin=(oi, oj, ok), field=field)


def demo_blocks():
    print("=== 相邻体块拼接：9^3 球面场，2x2x2 八块（面/边/角三类接缝并存）===")
    n = 9
    iso = 7.0
    func = lambda i, j, k: (i - 4.0) ** 2 + (j - 4.0) ** 2 + (k - 4.0) ** 2
    whole_values = tuple(
        float(func(i, j, k))
        for k in range(n)
        for j in range(n)
        for i in range(n)
    )
    whole = ScalarField3D(ni=n, nj=n, nk=n, values=whole_values)

    cuts = (0, 4, 8)  # 每轴切成 [0..4] 与 [4..8] 两块（5 节点，接缝共享 i=4）
    blocks = []
    for kx in range(2):
        for ky in range(2):
            for kz in range(2):
                blocks.append(
                    _block(
                        (5, 5, 5), func,
                        cuts[kx], cuts[ky], cuts[kz],
                    )
                )
    t0 = time.perf_counter()
    stitched = extract_isosurface_blocks(blocks, iso)
    elapsed = time.perf_counter() - t0
    mono = extract_isosurface(whole, iso)

    # 逐顶点（含 edge_key 与 t）与逐面（含 vertex_ids）比较，不依赖浮点距离猜测。
    same_vertices = (
        [v.edge_key for v in stitched.vertices] == [v.edge_key for v in mono.vertices]
        and [v.point for v in stitched.vertices] == [v.point for v in mono.vertices]
    )
    same_faces = [f.vertex_ids for f in stitched.faces] == [
        f.vertex_ids for f in mono.faces
    ]
    print(f"体块数           : {len(blocks)}（角接缝节点 (4,4,4) 被 8 块共享）")
    print(f"拼接顶点/三角形  : {stitched.stats.vertex_count} / {stitched.stats.face_count}")
    print(f"整块顶点/三角形  : {mono.stats.vertex_count} / {mono.stats.face_count}")
    print(f"边界边/非流形边  : {stitched.stats.boundary_edge_count} / "
          f"{stitched.stats.nonmanifold_edge_count}  (闭合曲面应为 0/0)")
    print(f"总面积           : {stitched.stats.total_area:.9f}  "
          f"(整块 {mono.stats.total_area:.9f})")
    print(f"拼接耗时         : {elapsed * 1000.0:.1f} ms")
    ok = same_vertices and same_faces and stitched.stats == mono.stats
    print(f"拼接结果与整块逐面一致: {'通过' if ok else '未通过'}")
    print()

    print("=== 手算边界：3x2x2，f=i，iso=0.5，两块沿 i=1 面拼接 ===")
    lin = lambda i, j, k: float(i)
    a = _block((2, 2, 2), lin, 0, 0, 0)
    b = _block((2, 2, 2), lin, 1, 0, 0)
    hand = extract_isosurface_blocks((a, b), 0.5)
    # 手算：9 条穿越边、8 个三角形，几何并集是 x=0.5 处面积 1 的单位正方形。
    hand_ok = (
        hand.stats.vertex_count == 9
        and hand.stats.face_count == 8
        and abs(hand.stats.total_area - 1.0) < 1e-12
    )
    print(f"顶点数 {hand.stats.vertex_count} (手算 9)，"
          f"三角形数 {hand.stats.face_count} (手算 8)，"
          f"面积 {hand.stats.total_area:.12f} (手算 1.0)")
    print(f"手算核对: {'通过' if hand_ok else '未通过'}")
    print()


def demo_rejection():
    print("=== 拒绝边界 1：单块标量场含 NaN ===")
    values = [0.0] * (4 * 4 * 4)
    values[37] = math.nan  # 节点索引 37
    try:
        field = ScalarField3D(ni=4, nj=4, nk=4, values=values)
        extract_isosurface(field, 0.5)
    except InvalidInputError as exc:
        print(f"已拒绝并定位: InvalidInputError: {exc}")
        print(f"错误参数名   : {exc.parameter}")
    else:
        print("未触发拒绝（不符合预期）")
    print()

    print("=== 拒绝边界 2：相邻体块接缝节点值冲突 ===")
    va = (0.0,) * 8
    vb = [0.0] * 8
    vb[2] = 9.0  # 块 B 局部节点 (0,1,0) 即全局接缝节点 (1,1,0)
    blocks = (
        ScalarFieldBlock(
            origin=(0, 0, 0),
            field=ScalarField3D(ni=2, nj=2, nk=2, values=va),
        ),
        ScalarFieldBlock(
            origin=(1, 0, 0),
            field=ScalarField3D(ni=2, nj=2, nk=2, values=tuple(vb)),
        ),
    )
    try:
        extract_isosurface_blocks(blocks, 0.5)
    except InvalidInputError as exc:
        print(f"已拒绝并定位: InvalidInputError: {exc}")
        print(f"错误参数名   : {exc.parameter}")
    else:
        print("未触发拒绝（不符合预期）")
    print()


def main():
    t0 = time.perf_counter()
    demo_sphere()
    demo_blocks()
    demo_rejection()
    print(f"demo 总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
