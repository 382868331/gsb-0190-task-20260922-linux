"""demo_iteration.py — extract_band 数值带提取的离线真实计算演示。

运行：python demo_iteration.py（应远低于 90 秒，无网络/账号/sleep/预录）。

内容：
  1. 线性标量场 f=x+y+z 切盒子，独立解析体积 vs 有向三角网体积积分；
  2. 每条非退化边恰有两个相反方向面邻接的流形检查；
  3. 数值带触边、全包含、全空、零厚度四类情形；
  4. 一个真实触发的输入拒绝边界。
"""

import time
from collections import Counter

from iso3d import (
    InvalidInputError,
    ScalarField3D,
    extract_band,
    signed_volume,
)


def cdf_diag(t, length):
    """x+y+z <= t 在 [0,length]^3 内的独立解析体积（分段三次）。"""
    if t <= 0:
        return 0.0
    if t <= length:
        return t ** 3 / 6.0
    if t <= 2 * length:
        return (t ** 3 - 3.0 * (t - length) ** 3) / 6.0
    if t <= 3 * length:
        return length ** 3 - (3.0 * length - t) ** 3 / 6.0
    return length ** 3


def linear_field(n):
    values = tuple(
        i + j + k
        for k in range(n)
        for j in range(n)
        for i in range(n)
    )
    return ScalarField3D(ni=n, nj=n, nk=n, values=values)


def edge_audit(mesh):
    directed = Counter()
    for face in mesh.faces:
        v = face.vertex_ids
        for a, b in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            directed[(a, b)] += 1
    undirected = {}
    for (a, b) in directed:
        undirected[(a, b) if a < b else (b, a)] = None
    bad = 0
    for x, y in undirected:
        if directed[(x, y)] != 1 or directed[(y, x)] != 1:
            bad += 1
    return bad, len(undirected)


def demo_volume_and_manifold():
    print("=== 线性场 f=x+y+z 切盒子：解析体积 vs 有向三角网积分 ===")
    n, low, high = 6, 5.0, 10.0
    field = linear_field(n)
    t0 = time.perf_counter()
    mesh = extract_band(field, low, high)
    elapsed = time.perf_counter() - t0
    length = n - 1
    analytic = cdf_diag(high, length) - cdf_diag(low, length)
    integral = signed_volume(mesh.vertices, mesh.faces)
    s = mesh.stats
    print(f"网格/带区          : {n}^3 节点, {low} <= f <= {high}（带触网格边界）")
    print(f"顶点/三角形        : {s.vertex_count} / {s.face_count} "
          f"(低墙 {s.low_wall_face_count}, 高墙 {s.high_wall_face_count}, "
          f"盖片 {s.cap_face_count})")
    print(f"独立解析体积       : {analytic:.9f}")
    print(f"有向体积积分       : {integral:.9f}  (stats: {s.signed_volume:.9f})")
    bad, edge_total = edge_audit(mesh)
    print(f"非退化边数         : {edge_total}，方向不相反/非两邻接的坏边: {bad}")
    print(f"边界边/非流形边    : {s.boundary_edge_count} / {s.nonmanifold_edge_count}")
    print(f"提取耗时           : {elapsed * 1000.0:.1f} ms")
    ok = (
        abs(integral - analytic) < 1e-9
        and bad == 0
        and s.boundary_edge_count == 0
        and s.nonmanifold_edge_count == 0
    )
    print(f"结论               : {'通过' if ok else '未通过'}")
    print()


def demo_coverage_cases():
    print("=== 触边 / 全包含 / 全空 / 零厚度 ===")
    n = 5
    field = linear_field(n)
    length = n - 1

    touch = extract_band(field, 2.0, 4.0)
    v_ref = cdf_diag(4.0, length) - cdf_diag(2.0, length)
    print(f"触边   : 盖片 {touch.stats.cap_face_count} 面, 体积 {touch.stats.signed_volume:.6f} "
          f"(解析 {v_ref:.6f}), 边界边 {touch.stats.boundary_edge_count}")

    full = extract_band(field, -10.0, 100.0)
    print(f"全包含 : 等值墙 0 面, 盖片 {full.stats.cap_face_count} 面, "
          f"体积 {full.stats.signed_volume:.6f} (整盒 {float(length ** 3):.1f}), "
          f"边界边 {full.stats.boundary_edge_count}")

    empty = extract_band(field, -100.0, -50.0)
    print(f"全空   : 面 {empty.stats.face_count}, 顶点 {empty.stats.vertex_count}, "
          f"体积 {empty.stats.signed_volume:.1f}")

    zero = extract_band(field, 7.5, 7.5)
    print(f"零厚度 : thickness_zero={zero.thickness_zero}, 仅低墙 "
          f"{zero.stats.low_wall_face_count} 面, 盖片 {zero.stats.cap_face_count}, "
          f"体积 {zero.stats.signed_volume:.1f}（开口曲面边界边 "
          f"{zero.stats.boundary_edge_count}，不伪装实体）")
    ok = (
        abs(touch.stats.signed_volume - v_ref) < 1e-9
        and abs(full.stats.signed_volume - float(length ** 3)) < 1e-9
        and empty.stats.face_count == 0
        and zero.thickness_zero and zero.stats.signed_volume == 0.0
    )
    print(f"结论   : {'通过' if ok else '未通过'}")
    print()


def demo_rejection():
    print("=== 拒绝边界：low > high ===")
    field = linear_field(3)
    try:
        extract_band(field, 5.0, 4.0)
    except InvalidInputError as exc:
        print(f"已拒绝并定位: InvalidInputError: {exc}")
        print(f"错误参数名   : {exc.parameter}")
    else:
        print("未触发拒绝（不符合预期）")
    print()


def main():
    t0 = time.perf_counter()
    demo_volume_and_manifold()
    demo_coverage_cases()
    demo_rejection()
    print(f"demo 总耗时: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
