"""demo.py — 体素等值面提取的固定输入演示（离线、确定性、无第三方依赖）。

运行：python demo.py
内容：
  1. 正常结果：16^3 球谐场（解析球）等值面提取，输出面积与非流形边计数等统计；
  2. 拒绝边界：标量场含 NaN，演示可定位的 InvalidInputError。
  3. 增量更新：只改一小片体数据后局部更新等值面，未受影响部分身份保留，
     几何与完整重提取一致。
"""

import math
import time

from iso3d import (
    InvalidInputError,
    ScalarField3D,
    extract_isosurface,
    update_isosurface,
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


def demo_incremental_update():
    print("=== 增量更新：只改 2x2x2 一片体数据，局部更新等值面 ===")
    field, iso, _ = sphere_field()
    old_mesh = extract_isosurface(field, iso)

    n = field.ni
    changed = []
    new_values = list(field.values)
    for k in (7, 8):
        for j in (7, 8):
            for i in (11, 12):
                idx = (k * n + j) * n + i
                changed.append(idx)
                new_values[idx] = 30.0  # 把球面边界附近一小片推到等值面外侧
    new_field = ScalarField3D(ni=n, nj=n, nk=n, values=tuple(new_values))

    t0 = time.perf_counter()
    updated = update_isosurface(field, old_mesh, new_field, changed_nodes=changed)
    update_elapsed = time.perf_counter() - t0
    t0 = time.perf_counter()
    reference = extract_isosurface(new_field, iso)
    full_elapsed = time.perf_counter() - t0

    # 与 ID 无关的几何指纹：顶点坐标 + 三角形坐标多重集。
    def fingerprint(mesh):
        by_id = {v.id: v for v in mesh.vertices}
        return (
            sorted(v.point for v in mesh.vertices),
            sorted(
                tuple(sorted(by_id[vid].point for vid in f.vertex_ids))
                for f in mesh.faces
            ),
        )

    same_geometry = fingerprint(updated) == fingerprint(reference)
    old_vertex_ids = {v.id for v in old_mesh.vertices}
    old_face_ids = {f.id for f in old_mesh.faces}
    kept_vertices = sum(1 for v in updated.vertices if v.id in old_vertex_ids)
    kept_faces = sum(1 for f in updated.faces if f.id in old_face_ids)
    print(f"修改节点数       : {len(changed)} / {field.ni ** 3}")
    print(
        f"更新后顶点/三角形: {updated.stats.vertex_count} / {updated.stats.face_count}"
    )
    print(f"保留旧顶点/旧面  : {kept_vertices} / {kept_faces}  (身份未变，可继续引用)")
    print(
        f"增量更新耗时     : {update_elapsed * 1000.0:.1f} ms  "
        f"(完整重提取 {full_elapsed * 1000.0:.1f} ms)"
    )
    print(f"与完整重提取几何一致: {'是' if same_geometry else '否'}")
    ok = same_geometry and kept_faces > 0
    print(f"增量更新正确性   : {'通过' if ok else '未通过'}")
    print()


def main():
    t0 = time.perf_counter()
    demo_sphere()
    demo_rejection()
    demo_incremental_update()
    print(f"demo 总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
