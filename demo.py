"""demo.py — 体素等值面提取的固定输入演示（离线、确定性、无第三方依赖）。

运行：python demo.py
内容：
  1. 正常结果：16^3 球谐场（解析球）等值面提取，输出面积与非流形边计数等统计；
  2. 增量更新：局部修改体数据后只重算受影响单元，几何与全量重提取一致；
  3. 拒绝边界：标量场含 NaN，演示可定位的 InvalidInputError。
"""

import math
import time

from iso3d import (
    InvalidInputError,
    IsosurfaceExtractor,
    ScalarField3D,
    extract_isosurface,
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


def demo_rejection():
    print("=== 拒绝边界：标量场含 NaN ===")
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


def demo_incremental():
    print("=== 增量更新：修改 2x2x2 节点，只重算受影响单元 ===")
    field, iso, _ = sphere_field()
    n = field.ni
    ext = IsosurfaceExtractor(field, iso)
    base = ext.mesh()
    print(f"更新前           : {base.stats.vertex_count} 顶点 / "
          f"{base.stats.face_count} 三角形")
    # 在中心附近挖一个 2x2x2 的凹坑（局部修改）。
    changes = {}
    for dk in (0, 1):
        for dj in (0, 1):
            for di in (0, 1):
                idx = ((8 + dk) * n + (8 + dj)) * n + (8 + di)
                changes[idx] = field.values[idx] - 5.0
    t0 = time.perf_counter()
    updated = ext.update(changes)
    elapsed = time.perf_counter() - t0
    print(f"更新后           : {updated.stats.vertex_count} 顶点 / "
          f"{updated.stats.face_count} 三角形 (增量耗时 {elapsed * 1000.0:.1f} ms)")
    # 与全量重提取逐位比较几何（顶点坐标集合 + 三角形集合）。
    new_values = list(field.values)
    for idx, value in changes.items():
        new_values[idx] = value
    full = extract_isosurface(
        ScalarField3D(ni=n, nj=n, nk=n, values=tuple(new_values)), iso
    )
    def signature(mesh):
        pts = {v.id: v.point for v in mesh.vertices}
        return (
            set(pts.values()),
            {tuple(sorted(pts[v] for v in f.vertex_ids)) for f in mesh.faces},
        )
    same = signature(updated) == signature(full)
    # 未受影响顶点身份保留：比较更新前后共同顶点 id 的数量。
    kept = len({v.id for v in base.vertices} & {v.id for v in updated.vertices})
    print(f"几何与全量重提取一致: {'是' if same else '否'}")
    print(f"保留的旧顶点 id 数  : {kept} / {base.stats.vertex_count}")
    print(f"边界/非流形边       : {updated.stats.boundary_edge_count} / "
          f"{updated.stats.nonmanifold_edge_count} (闭合球面应均为 0)")
    # 更新失败时旧结果仍可使用。
    try:
        ext.update({0: math.nan})
    except InvalidInputError as exc:
        print(f"非法更新已拒绝     : {exc}；旧网格仍可用: "
              f"{'是' if ext.mesh().stats.face_count == updated.stats.face_count else '否'}")
    print()


def main():
    t0 = time.perf_counter()
    demo_sphere()
    demo_incremental()
    demo_rejection()
    print(f"demo 总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
