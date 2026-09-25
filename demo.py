"""demo.py — 体素等值面提取与共享边交点敏感度的固定输入演示（离线、确定性）。

运行：python demo.py
内容：
  1. 正常结果：16^3 球场等值面，输出面积与非流形边计数等统计；
  2. 拒绝边界：标量场含 NaN，演示可定位的 InvalidInputError；
  3. 共享边交点敏感度（本次新增）：
     - 线性斜坡 f=i+j+k 上手算一条边的交点、来源权重与三个解析导数；
     - 双侧有限差分现场核对 d/dv_a、d/dv_b、d/diso；
     - 反向遍历同一条共享边，记录完全一致；
     - 标量差极小（不稳定分母）与端点恰 iso：标不可微，不输出伪精确大数；
     - 一个输入节点变化影响哪些交点，并用有限差分验证位移预测。
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
    intersect_edge,
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


# ---------------------------------------------------------------------------
# 敏感度演示
# ---------------------------------------------------------------------------

def _node_coord(ni, nj, nk, node_id):
    i = node_id % ni
    j = (node_id // ni) % nj
    k = node_id // (ni * nj)
    return (float(i), float(j), float(k))


def demo_hand_computed_and_finite_difference():
    print("=== 敏感度 1：线性斜坡上手算交点，并用双侧有限差分核对三个导数 ===")
    ni, nj, nk, iso = 8, 6, 5, 7.5
    field = ScalarField3D(
        ni=ni,
        nj=nj,
        nk=nk,
        values=tuple(
            float(i + j + k)
            for k in range(nk)
            for j in range(nj)
            for i in range(ni)
        ),
    )
    mesh = extract_isosurface(field, iso)
    n_diff = sum(1 for r in mesh.intersections if r.differentiable)
    print(f"斜坡 f=i+j+k, iso={iso}：共 {len(mesh.intersections)} 个共享交点，"
          f"其中 {n_diff} 个可微")

    # 取第一条可微交点（真实来自网格遍历）。
    rec = next(r for r in mesh.intersections if r.differentiable)
    a, b = rec.edge_key
    va, vb = field.values[a], field.values[b]
    pa, pb = _node_coord(ni, nj, nk, a), _node_coord(ni, nj, nk, b)
    print(f"来源网格边       : 节点 {a}{pa} -> 节点 {b}{pb}")
    print(f"两端标量         : v_a={va:g}, v_b={vb:g}, iso={iso:g}")
    print(f"插值权重 t       : {rec.t:.12f}  (point = (1-t)P_a + t P_b)")
    (na, wa), (nb, wb) = rec.weights
    print(f"来源权重         : 节点 {na} 权重 {wa:.12f}；节点 {nb} 权重 {wb:.12f}；"
          f"和 = {wa + wb:.12f}")
    print(f"交点坐标         : ({rec.point[0]:.12f}, {rec.point[1]:.12f}, "
          f"{rec.point[2]:.12f})")

    s = rec.sensitivity
    print("解析导数（t）    : "
          f"dt/dv_a={s.dt_dva:.12f}, dt/dv_b={s.dt_dvb:.12f}, "
          f"dt/diso={s.dt_diso:.12f}")

    # 双侧有限差分独立核对（重新调用纯函数 intersect_edge）。
    h = 1e-6

    def recall(dva=0.0, dvb=0.0, diso=0.0):
        return intersect_edge(
            a, b, va + dva, vb + dvb, iso + diso, pa, pb
        ).t

    fd = (
        (recall(dva=h) - recall(dva=-h)) / (2 * h),
        (recall(dvb=h) - recall(dvb=-h)) / (2 * h),
        (recall(diso=h) - recall(diso=-h)) / (2 * h),
    )
    an = (s.dt_dva, s.dt_dvb, s.dt_diso)
    names = ("dt/dv_a", "dt/dv_b", "dt/diso")
    for name, num, exact in zip(names, fd, an):
        err = abs(num - exact)
        print(f"有限差分核对 {name}: 数值 {num:.10f} vs 解析 {exact:.10f}，"
              f"绝对差 {err:.2e}")
    print()


def demo_shared_and_reverse():
    print("=== 敏感度 2：反向遍历同一条共享边，记录必须完全一致 ===")
    ni = nj = nk = 6
    field = ScalarField3D(
        ni=ni, nj=nj, nk=nk,
        values=tuple(
            float(i + j + k)
            for k in range(nk) for j in range(nj) for i in range(ni)
        ),
    )
    mesh = extract_isosurface(field, 5.5)
    rec = mesh.intersections[0]
    a, b = rec.edge_key
    pa, pb = _node_coord(ni, nj, nk, a), _node_coord(ni, nj, nk, b)

    # 统计包含该顶点的三角形数量（共享的直接证据）。
    vid = next(v.id for v in mesh.vertices if v.edge_key == (a, b))
    used = sum(
        1 for face in mesh.faces if vid in face.vertex_ids
    )
    # 反向（b,a）连同标量、坐标整体交换调用。
    rev = intersect_edge(
        b, a, field.values[b], field.values[a], mesh.iso, pb, pa
    )
    print(f"边 {a}->{b}（规范键 {(a, b)}），对应顶点出现在 {used} 个三角形角点引用中")
    print(f"反向 (b,a) 调用后规范化边键 : {rev.edge_key}")
    print(f"t / point / 权重 / 导数全等 : {rev == rec}")
    print(f"跨面查询为同一记录对象      : "
          f"{mesh.intersection_for_vertex(vid) is mesh.intersection_for_edge(a, b)}")
    print()


def demo_nondifferentiable():
    print("=== 敏感度 3：标量差极小与端点恰 iso → 标不可微，不输出巨大伪精确数 ===")
    # (a) 严格跨过但标量差 1e-13（低于相对稳定阈值 1e-12）。
    va, vb, iso = 1.0, 1.0 + 1e-13, 1.0 + 5e-14
    rec = intersect_edge(0, 1, va, vb, iso, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    print(f"[极小标量差] v_a={va!r}, v_b={vb!r}, iso={iso!r}, "
          f"严格跨过={va < iso < vb}")
    print(f"  |v_b-v_a|={abs(vb - va):.3e} 低于相对阈值 1e-12*max(1,|v|)=1e-12")
    print(f"  可微={rec.differentiable}，原因={rec.nondifferentiable_reason}，"
          f"sensitivity={rec.sensitivity}")
    print(f"  几何位置仍给出且有限: t={rec.t:g}, point={rec.point}")
    assert rec.nondifferentiable_reason == NONDIFF_UNSTABLE_DENOMINATOR

    # (b) 端点恰 iso：交点在端点，拓扑由符号扰动决定。
    rec2 = intersect_edge(2, 5, 1.0, 2.0, 1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    print(f"[端点恰 iso] v_a=1.0 恰等于 iso=1.0: t={rec2.t:g}, "
          f"可微={rec2.differentiable}，原因={rec2.nondifferentiable_reason}")
    assert rec2.nondifferentiable_reason == NONDIFF_ENDPOINT_ON_ISO

    # (c) 两端都恰 iso：零分母，取中点。
    rec3 = intersect_edge(4, 9, 1.0, 1.0, 1.0, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    print(f"[两端恰 iso/零分母] t={rec3.t:g}, point={rec3.point}, "
          f"原因={rec3.nondifferentiable_reason}")
    assert rec3.nondifferentiable_reason == NONDIFF_ZERO_DENOMINATOR
    print()


def demo_node_influence():
    print("=== 敏感度 4：一个输入节点变化影响哪些交点（不做分块/整体传播）===")
    ni, nj, nk, iso = 7, 6, 5, 8.5
    field = ScalarField3D(
        ni=ni, nj=nj, nk=nk,
        values=tuple(
            float(i + j + k)
            for k in range(nk) for j in range(nj) for i in range(ni)
        ),
    )
    mesh = extract_isosurface(field, iso)
    node = field.node_id(3, 3, 2)  # 内点，f = 8 < iso
    affected = mesh.intersections_affected_by_node(node)
    print(f"扰动节点 {node} = (3,3,2)，标量 {field.values[node]:g}")
    print(f"受影响的共享交点数 : {len(affected)}（恰为来源边端点含该节点者）")
    for rec in affected[:4]:
        a, b = rec.edge_key
        other = b if a == node else a
        s = rec.sensitivity
        grad = s.dpoint_dva if a == node else s.dpoint_dvb
        print(f"  边 {rec.edge_key}（对端节点 {other}）: "
              f"dpoint/dv_node=({grad[0]:+.3f},{grad[1]:+.3f},{grad[2]:+.3f})")
    if len(affected) > 4:
        print(f"  …（其余 {len(affected) - 4} 条省略）")

    # 现场扰动该节点 +h，重新提取，核对：只有受影响边移动，且位移符合预测。
    h = 1e-6
    values = list(field.values)
    values[node] += h
    mesh2 = extract_isosurface(
        ScalarField3D(ni=ni, nj=nj, nk=nk, values=tuple(values)), iso
    )
    by_key = {v.edge_key: v.point for v in mesh2.vertices}
    affected_keys = {r.edge_key for r in affected}
    max_unrelated = 0.0
    max_pred_err = 0.0
    for rec in mesh.intersections:
        p2 = by_key[rec.edge_key]
        moved = math.dist(p2, rec.point)
        if rec.edge_key not in affected_keys:
            max_unrelated = max(max_unrelated, moved)
        elif rec.differentiable:
            a, b = rec.edge_key
            grad = rec.sensitivity.dpoint_dva if a == node else rec.sensitivity.dpoint_dvb
            for d in range(3):
                max_pred_err = max(
                    max_pred_err, abs((p2[d] - rec.point[d]) / h - grad[d])
                )
    print(f"节点 +{h:g} 后重新提取：")
    print(f"  非相关交点的最大位移     : {max_unrelated:.2e}（应为 0，逐位不变）")
    print(f"  受影响交点位移/ h 与预测差: {max_pred_err:.2e}（一阶预测误差）")
    print("  注：敏感度仅在当前拓扑（穿越判定）不变的小扰动下有效，"
          "不承诺跨拓扑变化。")
    print()


def main():
    t0 = time.perf_counter()
    demo_sphere()
    demo_rejection()
    demo_hand_computed_and_finite_difference()
    demo_shared_and_reverse()
    demo_nondifferentiable()
    demo_node_influence()
    print(f"demo 总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
