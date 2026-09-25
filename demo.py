"""demo.py — 体素等值面提取与共享交点溯源的固定输入演示（离线、纯标准库）。

运行：python demo.py
内容：
  1. 正常结果：16^3 球面等值面提取，输出面积与非流形边计数等统计；
  2. 拒绝边界：标量场含 NaN，演示可定位的 InvalidInputError；
  3. 交点溯源：线性斜坡体上手算交点，附来源边/插值权重，并用双侧有限
     差分实际核对交点对两端标量值与 iso 的三个解析导数；
  4. 共享唯一性：反向遍历 / 跨立方体共享面的同一条物理边只有一条记录；
  5. 不可微边界：标量差极小（不稳定分母）与端点恰 iso 只标记不可微，
     不输出巨大伪精确数；
  6. 节点影响：实际改变一个输入节点，展示只有以它为来源的交点移动。
"""

import math
import time

from iso3d import (
    NONDIFF_ENDPOINT_ON_ISO,
    NONDIFF_UNSTABLE_DENOMINATOR,
    NONDIFF_ZERO_DENOMINATOR,
    InvalidInputError,
    ScalarField3D,
    extract_isosurface,
)

_NONDIFF_CN = {
    NONDIFF_ZERO_DENOMINATOR: "分母为零（两端值相等且恰为 iso，取中点）",
    NONDIFF_ENDPOINT_ON_ISO: "端点恰在 iso 上（拓扑分支点）",
    NONDIFF_UNSTABLE_DENOMINATOR: "分母极小，解析导数不稳定",
}


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
    print("=== 1. 正常结果：16^3 网格上的球面等值面（半径 4，iso = r^2 = 16）===")
    field, iso, center = sphere_field()
    t0 = time.perf_counter()
    mesh = extract_isosurface(field, iso)
    elapsed = time.perf_counter() - t0
    s = mesh.stats
    analytic = 4.0 * math.pi * 16.0
    print(f"节点数           : {s.node_count} (16^3 满上限)")
    print(f"立方体/四面体    : {s.cell_count} / {s.tetrahedron_count}")
    print(f"顶点(共享交点)数 : {s.vertex_count}；溯源记录数: {len(mesh.intersections)}")
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
        and len(mesh.intersections) == s.vertex_count
    )
    print(f"闭合性、面积合理性与溯源一一对应: {'通过' if ok else '未通过'}")
    print()


def demo_rejection():
    print("=== 2. 拒绝边界：标量场含 NaN ===")
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


def _record(mesh, key):
    return next(rec for rec in recs_iter(mesh) if rec.edge_key == key)


def recs_iter(mesh):
    return iter(mesh.intersections)


def demo_sensitivity():
    print("=== 3. 交点溯源与三个解析导数（线性斜坡 f=i+j+k，iso=2.5）===")
    n = 3
    values = tuple(
        float(i + j + k)
        for k in range(n) for j in range(n) for i in range(n)
    )
    field = ScalarField3D(ni=n, nj=n, nk=n, values=values)
    iso = 2.5
    mesh = extract_isosurface(field, iso)

    # 手算边：(1,1,0) v=2 -> (2,1,0) v=3，坐标轴边 edge_vec=(1,0,0)
    id_a = field.node_id(1, 1, 0)
    id_b = field.node_id(2, 1, 0)
    key = (id_a, id_b)
    rec = _record(mesh, key)
    v = mesh.vertices[rec.vertex_id]
    print(f"来源网格边       : 节点 {id_a}(1,1,0),v=2 -> 节点 {id_b}(2,1,0),v=3")
    print(f"插值权重 (wa,wb) : ({rec.weights[0]:.16g}, {rec.weights[1]:.16g})，和=1")
    print(f"交点 (手算 t=0.5): ({v.point[0]:.16g}, {v.point[1]:.16g}, {v.point[2]:.16g})")
    print("解析公式: t=(iso-va)/(vb-va)；dP/dx = edge_vec * dt/dx")
    s = rec.sensitivity
    print(f"  dP/dva 解析    : {tuple(round(c, 12) for c in s.d_point_d_va)}")
    print(f"  dP/dvb 解析    : {tuple(round(c, 12) for c in s.d_point_d_vb)}")
    print(f"  dP/diso 解析   : {tuple(round(c, 12) for c in s.d_point_d_iso)}")

    # 双侧有限差分（实际重新提取），独立核对三个导数。
    h = 1e-6

    def point_with(node_overrides=None, iso_override=None):
        vals = list(values)
        for nid, val in (node_overrides or {}).items():
            vals[nid] = val
        f = ScalarField3D(ni=n, nj=n, nk=n, values=tuple(vals))
        m = extract_isosurface(f, iso if iso_override is None else iso_override)
        return next(x.point for x in m.vertices if x.edge_key == key)

    def fd(overrides_lo, overrides_hi, iso_lo=None, iso_hi=None):
        p_lo = point_with(overrides_lo, iso_lo)
        p_hi = point_with(overrides_hi, iso_hi)
        return tuple((p_hi[d] - p_lo[d]) / (2 * h) for d in range(3))

    fd_va = fd({id_a: values[id_a] - h}, {id_a: values[id_a] + h})
    fd_vb = fd({id_b: values[id_b] - h}, {id_b: values[id_b] + h})
    fd_iso = fd(None, None, iso - h, iso + h)
    for name, analytic, numeric in (
        ("dP/dva ", s.d_point_d_va, fd_va),
        ("dP/dvb ", s.d_point_d_vb, fd_vb),
        ("dP/diso", s.d_point_d_iso, fd_iso),
    ):
        err = max(abs(analytic[d] - numeric[d]) for d in range(3))
        print(f"  {name} 双侧差分: {tuple(round(c, 10) for c in numeric)}"
              f"   最大分量误差 {err:.2e}")
    assert max(
        max(abs(a[d] - b[d]) for d in range(3))
        for a, b in ((s.d_point_d_va, fd_va), (s.d_point_d_vb, fd_vb),
                     (s.d_point_d_iso, fd_iso))
    ) < 1e-7
    print("三个导数与有限差分一致（阈值 1e-7）。")
    print("注：导数仅对当前拓扑不变的小扰动有效；端点越过 iso 会改变交点集合。")
    print()


def demo_shared_record():
    print("=== 4. 共享唯一性：反向遍历与跨立方体共享面只生成一条记录 ===")
    n = 3
    values = tuple(
        float(i + j + k)
        for k in range(n) for j in range(n) for i in range(n)
    )
    field = ScalarField3D(ni=n, nj=n, nk=n, values=values)
    mesh = extract_isosurface(field, 2.0)
    # k=0 立方体顶面与 k=1 立方体底面的同一条面对角线：
    # 节点 (0,0,1) v=1 与 (1,1,1) v=3，iso=2。
    n_lo = field.node_id(0, 0, 1)
    n_hi = field.node_id(1, 1, 1)
    key = (min(n_lo, n_hi), max(n_lo, n_hi))
    edge_recs = [r for r in mesh.intersections if r.edge_key == key]
    edge_verts = [v for v in mesh.vertices if v.edge_key == key]
    users = [f.id for f in mesh.faces if edge_verts[0].id in f.vertex_ids]
    print(f"共享面对角线 {key}（下层角点 4-7，上层角点 0-3，同一物理边）：")
    print(f"  溯源记录数 = {len(edge_recs)}（要求 1），顶点数 = {len(edge_verts)}（要求 1）")
    print(f"  该交点被 {len(users)} 个三角形引用: {users}")
    # 全部记录的来源边键两两不同。
    all_keys = [r.edge_key for r in mesh.intersections]
    print(f"  全部 {len(all_keys)} 条记录的来源边键唯一: {len(all_keys) == len(set(all_keys))}")
    assert len(edge_recs) == 1 and len(edge_verts) == 1
    assert len(all_keys) == len(set(all_keys))
    print()


def demo_nondifferentiable():
    print("=== 5. 不可微边界：只标记，不输出巨大伪精确数 ===")
    # (a) 标量差极小：v0=1, v1=1+1e-12，严格跨过 iso。
    vals = [10.0] * 8
    vals[0], vals[1] = 1.0, 1.0 + 1e-12
    field = ScalarField3D(ni=2, nj=2, nk=2, values=tuple(vals))
    mesh = extract_isosurface(field, 1.0 + 5e-13)
    rec = _record(mesh, (0, 1))
    v = mesh.vertices[rec.vertex_id]
    print(f"(a) 标量差 1e-12 的边：t={v.t:.4f}（几何照常，在边内），"
          f"可微={rec.sensitivity.differentiable}")
    print(f"    原因: {_NONDIFF_CN[rec.sensitivity.reason]}；三个导数均为 "
          f"{rec.sensitivity.d_point_d_va is None and None}，而非 1/1e-12 式爆炸值")
    assert not rec.sensitivity.differentiable
    assert rec.sensitivity.reason == NONDIFF_UNSTABLE_DENOMINATOR

    # (b) 端点恰 iso：f=i，iso=1。
    vals = tuple(float(i) for k in range(3) for j in range(3) for i in range(3))
    field = ScalarField3D(ni=3, nj=3, nk=3, values=vals)
    mesh = extract_isosurface(field, 1.0)
    reasons = {r.sensitivity.reason for r in mesh.intersections}
    print(f"(b) f=i, iso=1（整层节点恰在等值面）：{len(mesh.intersections)} 条记录全部不可微；")
    print(f"    出现的原因代码: {sorted(reasons)}")
    print(f"      - {NONDIFF_ENDPOINT_ON_ISO}: {_NONDIFF_CN[NONDIFF_ENDPOINT_ON_ISO]}")
    print(f"      - {NONDIFF_ZERO_DENOMINATOR}: {_NONDIFF_CN[NONDIFF_ZERO_DENOMINATOR]}（中点，权重 0.5/0.5）")
    assert reasons <= {NONDIFF_ENDPOINT_ON_ISO, NONDIFF_ZERO_DENOMINATOR}
    print()


def demo_node_impact():
    print("=== 6. 一个输入节点变化影响哪些交点（实际扰动，非整体优化）===")
    n = 3
    values = tuple(
        float(i + j + k)
        for k in range(n) for j in range(n) for i in range(n)
    )
    field = ScalarField3D(ni=n, nj=n, nk=n, values=values)
    iso = 2.5
    mesh = extract_isosurface(field, iso)
    node = field.node_id(2, 1, 0)  # v=3
    impacted = mesh.intersections_for_node(node)
    print(f"扰动节点 {node}=(2,1,0), v=3 -> 3.002；当前拓扑下它作为来源端点的交点：")
    for rec in impacted:
        other = rec.edge_key[1] if rec.edge_key[0] == node else rec.edge_key[0]
        print(f"  顶点 {rec.vertex_id}: 边 {rec.edge_key}（对端节点 {other}），"
              f"权重 w_{node}={rec.weights[0 if rec.edge_key[0] == node else 1]:.4f}")

    vals2 = list(values)
    vals2[node] += 0.002
    mesh2 = extract_isosurface(ScalarField3D(ni=n, nj=n, nk=n, values=tuple(vals2)), iso)
    before = {x.edge_key: x.point for x in mesh.vertices}
    after = {x.edge_key: x.point for x in mesh2.vertices}
    moved = [k for k in before if any(
        abs(after[k][d] - before[k][d]) > 1e-12 for d in range(3))]
    print(f"实际重新提取后发生移动的交点: {sorted(moved)}")
    print(f"与溯源查询的来源边集合一致: {sorted(moved) == sorted(r.edge_key for r in impacted)}")
    print("（只逐节点查询/单点扰动，未做分块拼接或整体场优化。）")
    assert sorted(moved) == sorted(r.edge_key for r in impacted)
    print()


def main():
    t0 = time.perf_counter()
    demo_sphere()
    demo_rejection()
    demo_sensitivity()
    demo_shared_record()
    demo_nondifferentiable()
    demo_node_impact()
    print(f"demo 总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
