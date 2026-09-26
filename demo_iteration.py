"""demo_iteration.py — 数值带提取（extract_band）的真实离线演示。

运行：python demo_iteration.py（全程离线、确定性，实测约 1 秒，远低于 90 秒上限）
内容：
  1. 线性场切盒子：数值带体积与独立的切片积分交叉核对；
  2. 有向三角网检查：每条非退化边恰被两个相反方向的面邻接；
  3. 数值带触边 / 全包含 / 全空 / 零厚度（low == high）四类形态；
  4. 实际触发一次拒绝（low > high）。
"""

import time

from iso3d import InvalidInputError, ScalarField3D, extract_band, extract_isosurface


def linear_field(n, a, b, c, d):
    values = tuple(
        a * i + b * j + c * k + d
        for k in range(n)
        for j in range(n)
        for i in range(n)
    )
    return ScalarField3D(ni=n, nj=n, nk=n, values=values)


def shell_field(n):
    c = (n - 1) / 2.0
    values = tuple(
        (i - c) ** 2 + (j - c) ** 2 + (k - c) ** 2
        for k in range(n)
        for j in range(n)
        for i in range(n)
    )
    return ScalarField3D(ni=n, nj=n, nk=n, values=values)


def slice_integrated_volume(a, b, c, d, low, high, size):
    """独立核对：沿 z 切片，每片做二维多边形裁剪求面积，再 Simpson 积分。

    不调用 iso3d 的任何代码，作为体积积分的独立参照。
    """
    def area2d(lo2, hi2):
        poly = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
        for level, keep_ge in ((lo2, True), (hi2, False)):
            out = []
            m = len(poly)
            for idx in range(m):
                cur, nxt = poly[idx], poly[(idx + 1) % m]
                fc, fn = a * cur[0] + b * cur[1], a * nxt[0] + b * nxt[1]
                ci = fc >= level if keep_ge else fc <= level
                ni = fn >= level if keep_ge else fn <= level
                if ci:
                    out.append(cur)
                if ci != ni:
                    t = (level - fc) / (fn - fc)
                    out.append((cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])))
            poly = out
        if len(poly) < 3:
            return 0.0
        acc = 0.0
        for idx in range(len(poly)):
            x1, y1 = poly[idx]
            x2, y2 = poly[(idx + 1) % len(poly)]
            acc += x1 * y2 - x2 * y1
        return abs(acc) / 2.0

    slices = 700
    vol = 0.0
    for s in range(slices):
        z0, z1 = size * s / slices, size * (s + 1) / slices
        zm = (z0 + z1) / 2.0
        a0 = area2d(low - c * z0 - d, high - c * z0 - d)
        am = area2d(low - c * zm - d, high - c * zm - d)
        a1 = area2d(low - c * z1 - d, high - c * z1 - d)
        vol += (z1 - z0) / 6.0 * (a0 + 4.0 * am + a1)
    return vol


def check_edge_pairing(mesh):
    """每条非退化边恰被两个面以相反方向使用。"""
    directed = {}
    for face in mesh.faces:
        v = face.vertex_ids
        for x, y in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            directed[(x, y)] = directed.get((x, y), 0) + 1
    return all(
        count == 1 and directed.get((y, x), 0) == 1
        for (x, y), count in directed.items()
    )


def demo_linear_band():
    print("=== 1. 线性场 f = 0.9i + 1.1j + 0.7k + 0.3，数值带 [4, 9]，8^3 网格 ===")
    field = linear_field(8, 0.9, 1.1, 0.7, 0.3)
    t0 = time.perf_counter()
    mesh = extract_band(field, 4.0, 9.0)
    elapsed = time.perf_counter() - t0
    s = mesh.stats
    reference = slice_integrated_volume(0.9, 1.1, 0.7, 0.3, 4.0, 9.0, 7.0)
    print(f"顶点/三角形/盖片 : {s.vertex_count} / {s.face_count} / {s.cap_face_count}")
    print(f"有向体积积分     : {s.signed_volume:.9f}")
    print(f"独立切片积分参照 : {reference:.9f}")
    print(f"体积偏差         : {abs(s.signed_volume - reference):.3e}")
    print(f"边界边/非流形边  : {s.boundary_edge_count} / {s.nonmanifold_edge_count}")
    print(f"边对向邻接检查   : {'通过' if check_edge_pairing(mesh) else '未通过'}")
    print(f"提取耗时         : {elapsed * 1000.0:.1f} ms")
    ok = abs(s.signed_volume - reference) < 1e-6 and check_edge_pairing(mesh)
    print(f"体积核对         : {'通过' if ok else '未通过'}")
    print()
    return ok


def demo_shell_band():
    print("=== 2. 球壳数值带（半径 2~3，不触边）：闭合性 ===")
    mesh = extract_band(shell_field(12), 4.0, 9.0)
    s = mesh.stats
    print(f"三角形数         : {s.face_count}  (盖片 {s.cap_face_count}，应为 0)")
    print(f"边界边/非流形边  : {s.boundary_edge_count} / {s.nonmanifold_edge_count}")
    print(f"边对向邻接检查   : {'通过' if check_edge_pairing(mesh) else '未通过'}")
    ok = (
        s.cap_face_count == 0
        and s.boundary_edge_count == 0
        and check_edge_pairing(mesh)
    )
    print(f"闭合性           : {'通过' if ok else '未通过'}")
    print()
    return ok


def demo_edge_cases():
    print("=== 3. 触边 / 全包含 / 全空 / 零厚度 ===")
    field = linear_field(6, 0.9, 1.1, 0.7, 0.3)
    touch = extract_band(field, 4.0, 9.0)
    print(f"触边（带截边界） : 盖片 {touch.stats.cap_face_count} 个，"
          f"边界边 {touch.stats.boundary_edge_count}")
    full = extract_band(field, -1.0, 100.0)
    print(f"全包含           : 体积 {full.stats.signed_volume:.6f}"
          f"（盒子体积 125），边界边 {full.stats.boundary_edge_count}")
    empty = extract_band(field, 50.0, 60.0)
    print(f"全空             : 三角形 {empty.stats.face_count} 个（合法空结果）")
    zero = extract_band(field, 5.5, 5.5)
    iso = extract_isosurface(field, 5.5)
    print(f"零厚度 low==high : zero_thickness={zero.zero_thickness}，"
          f"面数 {zero.stats.face_count}（单层等值面 {iso.stats.face_count}），"
          f"体积 {zero.stats.signed_volume}（不伪装实体）")
    ok = (
        touch.stats.cap_face_count > 0
        and abs(full.stats.signed_volume - 125.0) < 1e-9
        and empty.stats.face_count == 0
        and zero.zero_thickness
        and zero.stats.face_count == iso.stats.face_count
        and zero.stats.signed_volume == 0.0
    )
    print(f"四类形态检查     : {'通过' if ok else '未通过'}")
    print()
    return ok


def demo_rejection():
    print("=== 4. 拒绝边界：low > high ===")
    field = linear_field(4, 1.0, 1.0, 1.0, 0.0)
    try:
        extract_band(field, 5.0, 2.0)
    except InvalidInputError as exc:
        print(f"已拒绝并定位: InvalidInputError: {exc}")
        print(f"错误参数名   : {exc.parameter}")
        print()
        return True
    print("未触发拒绝（不符合预期）")
    print()
    return False


def main():
    t0 = time.perf_counter()
    results = [
        demo_linear_band(),
        demo_shell_band(),
        demo_edge_cases(),
        demo_rejection(),
    ]
    elapsed = time.perf_counter() - t0
    print(f"demo 总耗时: {elapsed:.2f} s（真实离线计算，无 sleep / 预录）")
    print(f"总体: {'全部通过' if all(results) else '存在未通过项'}")


if __name__ == "__main__":
    main()

