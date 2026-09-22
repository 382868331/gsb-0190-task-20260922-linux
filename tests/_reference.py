"""测试用的独立参考实现。

本模块刻意不 import iso3d：所有期望值由这里独立书写的算法给出，
避免“用被测核心验证被测核心”的循环论证。
"""

import math

# 与公开契约一致的分解表（测试内独立副本，用于独立枚举穿越边）。
REF_CUBE_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)

REF_TETRA_TABLE = (
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 2, 3, 7),
    (0, 2, 6, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
)

REF_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def ref_below(value, iso, node_id):
    """按公开扰动规则独立实现的拓扑判定。"""
    if value < iso:
        return True
    if value > iso:
        return False
    return node_id % 2 == 0


def ref_crossing_edge_keys(ni, nj, nk, values, iso):
    """独立枚举所有四面体穿越边的规范边键集合。

    values 为行优先扁平序列：values[(k*nj + j)*ni + i]。
    """

    def nid(i, j, k):
        return (k * nj + j) * ni + i

    keys = set()
    for k in range(nk - 1):
        for j in range(nj - 1):
            for i in range(ni - 1):
                corner_ids = tuple(
                    nid(i + di, j + dj, k + dk) for (di, dj, dk) in REF_CUBE_CORNERS
                )
                below = tuple(ref_below(values[c], iso, c) for c in corner_ids)
                for tet in REF_TETRA_TABLE:
                    tb = tuple(below[c] for c in tet)
                    if all(tb) or not any(tb):
                        continue
                    nodes = tuple(corner_ids[c] for c in tet)
                    for ea, eb in REF_TET_EDGES:
                        if tb[ea] != tb[eb]:
                            a, b = nodes[ea], nodes[eb]
                            keys.add((a, b) if a < b else (b, a))
    return keys


def _polygon_area_3d(poly):
    if len(poly) < 3:
        return 0.0
    nx = ny = nz = 0.0
    for idx in range(len(poly)):
        p = poly[idx]
        q = poly[(idx + 1) % len(poly)]
        nx += p[1] * q[2] - p[2] * q[1]
        ny += p[2] * q[0] - p[0] * q[2]
        nz += p[0] * q[1] - p[1] * q[0]
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def plane_box_intersection_area(a, b, c, iso, xmax, ymax, zmax):
    """平面 a*x + b*y + c*z = iso 与长方体 [0,xmax]x[0,ymax]x[0,zmax] 的截面面积。

    独立参考：先求平面法向与截距，再用半平面裁剪盒子的投影轮廓。
    实现方式：枚举盒子 12 条棱与平面的交点，去重后绕质心按角排序，
    计算多边形面积。仅用于 a,b,c 非零且截面非退化的测试场景。
    """
    corners = [
        (x, y, z)
        for x in (0.0, xmax)
        for y in (0.0, ymax)
        for z in (0.0, zmax)
    ]
    edges = []
    for i0, p in enumerate(corners):
        for q in corners[i0 + 1:]:
            diff = sum(1 for d in range(3) if p[d] != q[d])
            if diff == 1:
                edges.append((p, q))
    pts = []
    for p, q in edges:
        fp = a * p[0] + b * p[1] + c * p[2] - iso
        fq = a * q[0] + b * q[1] + c * q[2] - iso
        if fp == 0.0 and fq == 0.0:
            raise ValueError("plane contains a box edge; reference not applicable")
        if fp * fq > 0.0:
            continue
        if fp == 0.0:
            t = 0.0
        elif fq == 0.0:
            t = 1.0
        else:
            t = fp / (fp - fq)
        pt = tuple(p[d] + t * (q[d] - p[d]) for d in range(3))
        if not any(
            abs(pt[0] - r[0]) < 1e-12
            and abs(pt[1] - r[1]) < 1e-12
            and abs(pt[2] - r[2]) < 1e-12
            for r in pts
        ):
            pts.append(pt)
    if len(pts) < 3:
        return 0.0
    # 绕质心按平面内角度排序。
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    n = (a, b, c)
    norm = math.sqrt(a * a + b * b + c * c)
    n = (n[0] / norm, n[1] / norm, n[2] / norm)
    # 平面内基向量。
    ref = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = (
        n[1] * ref[2] - n[2] * ref[1],
        n[2] * ref[0] - n[0] * ref[2],
        n[0] * ref[1] - n[1] * ref[0],
    )
    un = math.sqrt(sum(v * v for v in u))
    u = (u[0] / un, u[1] / un, u[2] / un)
    w = (
        n[1] * u[2] - n[2] * u[1],
        n[2] * u[0] - n[0] * u[2],
        n[0] * u[1] - n[1] * u[0],
    )

    def angle(p):
        rel = (p[0] - cx, p[1] - cy, p[2] - cz)
        du = rel[0] * u[0] + rel[1] * u[1] + rel[2] * u[2]
        dw = rel[0] * w[0] + rel[1] * w[1] + rel[2] * w[2]
        return math.atan2(dw, du)

    pts.sort(key=angle)
    return _polygon_area_3d(pts)
