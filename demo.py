#!/usr/bin/env python3
"""mtiso 固定输入演示：真实计算，离线、确定性、无第三方依赖。

内容：
  1. 正常结果：解析球面 |p-c|^2 = R^2（R^2=7，勒让德三平方定理保证 iso
     不落任何网格节点）提取闭合等值面，打印顶点/面数、面积及解析参考
     4*pi*R^2、非流形边计数、开口边计数、Euler 示性数与散度定理体积。
  2. 斜平面解析对照：平面截盒截面面积（独立 shoelace 参考）与网格曲面
     面积逐位量级对照，打印面积与非流形边计数。
  3. 零面积丢弃展示：单 above 顶点恰在 iso 上，6 个退化三角形被丢弃
     并给出可定位记录。
  4. 拒绝边界：标量数据中含 NaN，构造网格时被拒绝（可定位错误）。

运行：python demo.py
"""
from __future__ import annotations

import math
import time

from mtiso import ScalarGrid, extract_isosurface


def make_uniform_grid(n, f, h=1.0):
    c = (n - 1) * h / 2.0
    vals = [0.0] * (n ** 3)
    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                x, y, z = ix * h, iy * h, iz * h
                vals[ix + n * (iy + n * iz)] = float(f(x, y, z, c))
    return ScalarGrid(n, n, n, vals, spacing=(h, h, h))


def mesh_edge_counts(result):
    counts: dict[tuple[int, int], int] = {}
    for tri in result.faces:
        w = tri.vertex_ids
        for i in range(3):
            a, b = w[i], w[(i + 1) % 3]
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return counts


def signed_volume(result):
    v = 0.0
    for tri in result.faces:
        p0 = result.vertices[tri.vertex_ids[0]]
        p1 = result.vertices[tri.vertex_ids[1]]
        p2 = result.vertices[tri.vertex_ids[2]]
        v += (
            p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
            - p0[1] * (p1[0] * p2[2] - p1[2] * p2[0])
            + p0[2] * (p1[0] * p2[1] - p1[1] * p2[0])
        ) / 6.0
    return v


def plane_box_section_area(a, b, c, d, iso, L):
    """独立参考：平面 ax+by+cz+d=iso 截立方体 [0,L]^3 的截面多边形面积。"""
    pts = []
    for axis in range(3):
        for f1 in (0, L):
            for f2 in (0, L):
                p0 = [0.0, 0.0, 0.0]
                p0[(axis + 1) % 3] = f1
                p0[(axis + 2) % 3] = f2
                p1 = list(p0)
                p1[axis] = L
                f0v = a * p0[0] + b * p0[1] + c * p0[2] + d
                f1v = a * p1[0] + b * p1[1] + c * p1[2] + d
                if (f0v - iso) * (f1v - iso) > 0.0 or f0v == f1v:
                    continue
                t = (iso - f0v) / (f1v - f0v)
                if 0.0 <= t <= 1.0:
                    pts.append(tuple(p0[k] + t * (p1[k] - p0[k])
                                     for k in range(3)))
    uniq = []
    for p in pts:
        if all(any(abs(p[k] - q[k]) > 1e-12 for k in range(3)) for q in uniq):
            uniq.append(p)
    nlen = math.sqrt(a * a + b * b + c * c)
    nhat = (a / nlen, b / nlen, c / nlen)
    ax0 = min(range(3), key=lambda i: abs(nhat[i]))
    tmp = [0.0, 0.0, 0.0]
    tmp[ax0] = 1.0
    dot = sum(tmp[k] * nhat[k] for k in range(3))
    e1 = tuple(tmp[k] - dot * nhat[k] for k in range(3))
    el = math.sqrt(sum(x * x for x in e1))
    e1 = tuple(x / el for x in e1)
    e2 = (nhat[1] * e1[2] - nhat[2] * e1[1],
          nhat[2] * e1[0] - nhat[0] * e1[2],
          nhat[0] * e1[1] - nhat[1] * e1[0])
    c0 = uniq[0]
    uv = [(sum((p[k] - c0[k]) * e1[k] for k in range(3)),
           sum((p[k] - c0[k]) * e2[k] for k in range(3))) for p in uniq]
    mcx = sum(u for u, _ in uv) / len(uv)
    mcy = sum(v for _, v in uv) / len(uv)
    uv.sort(key=lambda q: math.atan2(q[1] - mcy, q[0] - mcx))
    return abs(sum(uv[i][0] * uv[(i + 1) % len(uv)][1]
                   - uv[i][1] * uv[(i + 1) % len(uv)][0]
                   for i in range(len(uv)))) / 2.0


def main() -> None:
    t_start = time.time()

    print("=" * 68)
    print("mtiso marching tetrahedra 等值面共享边交点提取 — 演示")
    print("=" * 68)

    # --- 1. 正常结果：闭合解析球 -----------------------------------------
    print("\n[1] 正常结果：球 |p-c|^2 = 7, 13^3 节点, 步长 0.5")
    n, h, R2 = 13, 0.5, 7.0
    grid = make_uniform_grid(
        n, lambda x, y, z, c: (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2, h
    )
    t0 = time.time()
    res = extract_isosurface(grid, R2)
    dt = time.time() - t0
    ref_area = 4.0 * math.pi * R2
    ref_vol = 4.0 / 3.0 * math.pi * R2 ** 1.5
    ec = mesh_edge_counts(res)
    vol = signed_volume(res)
    print(f"    顶点数 (共享边交点) : {res.stats.vertex_count}")
    print(f"    三角形面数          : {res.stats.face_count}")
    print(f"    曲面面积            : {res.stats.surface_area:.9f}")
    print(f"    解析 4*pi*R^2 参考  : {ref_area:.9f}"
          f"   (相对差 {abs(res.stats.surface_area - ref_area) / ref_area:+.4%})")
    print(f"    散度定理体积        : {vol:.9f}   解析 {ref_vol:.9f}"
          f"   ({abs(vol - ref_vol) / ref_vol:+.4%})")
    print(f"    非流形边计数 (>=3面): {res.stats.nonmanifold_edge_count}")
    print(f"    开口边计数 (=1面)   : {res.stats.boundary_edge_count}")
    print(f"    Euler V-E+F         : "
          f"{res.stats.vertex_count - len(ec) + res.stats.face_count} (球面=2)")
    print(f"    零面积丢弃面数      : {res.stats.zero_area_faces_discarded}")
    print(f"    提取耗时            : {dt:.3f} s")

    # --- 2. 斜平面解析对照 -----------------------------------------------
    print("\n[2] 斜平面 x - 2y + 3z - 4 = 2.5, 6^3 单位网格")
    pg = ScalarGrid(
        6, 6, 6,
        [float(x - 2 * y + 3 * z - 4)
         for z in range(6) for y in range(6) for x in range(6)],
    )
    pres = extract_isosurface(pg, 2.5)
    ref_parea = plane_box_section_area(1.0, -2.0, 3.0, -4.0, 2.5, 5)
    print(f"    顶点数/面数         : {pres.stats.vertex_count}/"
          f"{pres.stats.face_count}")
    print(f"    网格曲面面积        : {pres.stats.surface_area:.12f}")
    print(f"    解析截面面积参考    : {ref_parea:.12f}"
          f"   (相对差 {abs(pres.stats.surface_area - ref_parea) / ref_parea:.2e})")
    print(f"    非流形边计数        : {pres.stats.nonmanifold_edge_count}")
    print(f"    开口边计数(域边界)  : {pres.stats.boundary_edge_count}")

    # --- 3. 几何零面积丢弃与记录 -----------------------------------------
    print("\n[3] 退化边界：2^3 网格仅偶 id 节点 0 恰在 iso 上（值 0），其余 -1")
    dvals = [-1.0] * 8
    dvals[0] = 0.0
    dres = extract_isosurface(ScalarGrid(2, 2, 2, dvals), 0.0)
    print(f"    生成三角形          : {dres.stats.triangles_generated}")
    print(f"    保留面数            : {dres.stats.face_count}")
    print(f"    零面积丢弃并记录    : {dres.stats.zero_area_faces_discarded}")
    rec = dres.discarded_faces[0]
    print(f"    首条记录可定位      : voxel={rec.voxel} "
          f"tetrahedron={rec.tetrahedron} triangle={rec.triangle} "
          f"reason={rec.reason!r}")
    print(f"    非流形边计数        : {dres.stats.nonmanifold_edge_count}")

    # --- 4. 实际触发的拒绝边界 -------------------------------------------
    print("\n[4] 拒绝边界：标量数组中含 NaN（不静默纠正，直接拒绝）")
    bad = [0.0] * 8
    bad[5] = float("nan")
    try:
        ScalarGrid(2, 2, 2, bad)
    except (TypeError, ValueError) as exc:
        print(f"    触发 {type(exc).__name__}: {exc}")
    else:  # pragma: no cover - 演示路径必须真的触发
        raise AssertionError("非法输入未被拒绝")

    # 同组拒绝：bool 充当整数维度。
    try:
        ScalarGrid(True, 2, 2, [0.0] * 8)
    except (TypeError, ValueError) as exc:
        print(f"    触发 {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("bool 维度未被拒绝")

    print(f"\n总耗时 {time.time() - t_start:.3f} s（含真实计算，无 sleep/预录）。")


if __name__ == "__main__":
    main()
