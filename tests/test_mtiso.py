"""端到端行为测试。

期望值全部由独立解析/组合参考直接算出，不调用被测函数的输出作为期望：
  * 平面场：逐边独立判断是否被截并独立计算交点坐标；整盒截面多边形用
    独立 shoelace 参考算面积；朝向、流形边数独立统计。
  * 共享面：在内部界面上独立选出"面对角线"型共享边，核对其唯一顶点被
    界面左右两侧立方体中的三角形共同使用。
  * iso 落顶点：用两组手算最小场对照（iso 节点为偶/奇 id），验证符号
    扰动只改变拓扑归属；交点几何仍在节点处（插值 t 恰为 0/1）。
  * 常量场：无面；iso==const 是全维等值集，四面体整体跳过并计数。
  * 球场：解析 4πR² / 4/3πR³ 参考 + 散度定理体积、Euler 示性数、
    独立边计数、径向偏差上界 h²/(2R) 与两分辨率收敛。
  * 零面积：确定性退化场（单 above 顶点恰在 iso 上），6 个退化三角形
    全部丢弃并有可定位记录。

枚举候选边时借用公开数据表 CUBE_TETRAHEDRA/TETRA_EDGES（它们是契约
的一部分，不是被测算法的输出）；每条边是否被截、交点在哪、面积多少，
均由本文件内的独立公式给出。
"""
from __future__ import annotations

import math
import unittest

from mtiso import (
    CUBE_TETRAHEDRA,
    ScalarGrid,
    extract_isosurface,
)

EPS = 2.220446049250313e-16  # float64 机器精度，误差口径见 README。


def make_grid(n, f, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    nx = ny = nz = n
    vals = [0.0] * (nx * ny * nz)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                vals[ix + nx * (iy + ny * iz)] = float(
                    f(origin[0] + ix * spacing[0],
                      origin[1] + iy * spacing[1],
                      origin[2] + iz * spacing[2])
                )
    return ScalarGrid(nx, ny, nz, vals, origin=origin, spacing=spacing)


def mesh_edge_counts(result):
    """独立统计每条三角形边被多少个面引用。"""
    counts: dict[tuple[int, int], int] = {}
    for tri in result.faces:
        w = tri.vertex_ids
        for i in range(3):
            a, b = w[i], w[(i + 1) % 3]
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return counts


def signed_volume(result):
    """散度定理有向体积 det(p0,p1,p2)/6 之和，外法向时为正。"""
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


def face_normal(tri, vertices):
    w = tri.vertex_ids
    p0, p1, p2 = vertices[w[0]], vertices[w[1]], vertices[w[2]]
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    return (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)


def node_world(grid: ScalarGrid, node: int):
    nx, ny = grid.nx, grid.ny
    z, rem = divmod(node, nx * ny)
    y, x = divmod(rem, nx)
    return (
        grid.origin[0] + x * grid.spacing[0],
        grid.origin[1] + y * grid.spacing[1],
        grid.origin[2] + z * grid.spacing[2],
    )


def candidate_edge_keys(n):
    """枚举所有立方体内四面体可能引用的网格边键（公开分解表的几何边）。

    包括 12 条轴对齐网格边、6 个面上各 1 条共享面对角线、统一体对角线。
    """
    nx = ny = nz = n
    keys = set()

    def nid(x, y, z):
        return x + nx * (y + ny * z)

    for iz in range(n - 1):
        for iy in range(n - 1):
            for ix in range(n - 1):
                base = nid(ix, iy, iz)
                cg = (
                    base,
                    base + 1,
                    base + 1 + nx,
                    base + nx,
                    base + nx * ny,
                    base + nx * ny + 1,
                    base + nx * ny + 1 + nx,
                    base + nx * ny + nx,
                )
                for tet in CUBE_TETRAHEDRA:
                    gids = (cg[tet[0]], cg[tet[1]], cg[tet[2]], cg[tet[3]])
                    for i in range(4):
                        for j in range(i + 1, 4):
                            a, b = gids[i], gids[j]
                            keys.add((a, b) if a < b else (b, a))
    return keys


class PlaneAnalyticTests(unittest.TestCase):
    """f(x,y,z) = x - 2y + 3z - 4，iso=2.5（半整数，不落任何整数节点）。"""

    A, B, C, D = 1.0, -2.0, 3.0, -4.0
    ISO = 2.5

    def setUp(self):
        self.n = 6

        def f(x, y, z):
            return self.A * x + self.B * y + self.C * z + self.D

        self.f = f
        self.grid = make_grid(self.n, f)
        self.result = extract_isosurface(self.grid, self.ISO)
        self.assertGreater(self.result.stats.face_count, 0)

    def test_all_vertices_on_plane_tight_numeric_bound(self):
        # 实数意义上插值点严格落平面；三项分别浮点舍入，残差界取
        # ~64 ulp*|iso|（线性场系数 1..3 的放大），约 1.5e-14，
        # 这是舍入口径而非几何容差（README 误差口径 (2)）。
        for px, py, pz in self.result.vertices:
            r = self.A * px + self.B * py + self.C * pz + self.D - self.ISO
            self.assertLessEqual(abs(r), 64 * EPS * (abs(self.ISO) + 8.0))

    def test_intersection_edges_and_coords_match_independent_reference(self):
        # 参考：枚举全部候选网格边，独立按端点标量严格异号判定被截，
        # 独立给出 t 与交点坐标。iso 不落节点 => 被截边集必须与输出
        # edge_vertices 的键集精确相等，且坐标逐位相等。
        n = self.n
        expected = {}
        for a, b in candidate_edge_keys(n):
            pa, pb = node_world(self.grid, a), node_world(self.grid, b)
            fa = self.f(*pa)
            fb = self.f(*pb)
            if (fa - self.ISO) * (fb - self.ISO) < 0.0:
                t = (self.ISO - fa) / (fb - fa)
                expected[(a, b)] = (
                    pa[0] + t * (pb[0] - pa[0]),
                    pa[1] + t * (pb[1] - pa[1]),
                    pa[2] + t * (pb[2] - pa[2]),
                )
        self.assertEqual(set(self.result.edge_vertices), set(expected))
        for key, coord in expected.items():
            self.assertEqual(
                self.result.vertices[self.result.edge_vertices[key]], coord
            )

    def test_area_matches_whole_box_cross_section_shoelace(self):
        # 参考：平面截整盒 [0,L]^3 得凸 3..6 边形；取平面与盒 12 棱的
        # 全部交点，在平面内正交基上投影，极角排序后 shoelace 求面积。
        n, L = self.n, self.n - 1
        pts = []
        for axis in range(3):
            for f1 in (0, L):
                for f2 in (0, L):
                    p0 = [0, 0, 0]
                    p0[(axis + 1) % 3] = f1
                    p0[(axis + 2) % 3] = f2
                    p1 = list(p0)
                    p1[axis] = L
                    f0v = self.f(*p0)
                    f1v = self.f(*p1)
                    if (f0v - self.ISO) * (f1v - self.ISO) > 0.0 or f0v == f1v:
                        continue
                    t = (self.ISO - f0v) / (f1v - f0v)
                    if 0.0 <= t <= 1.0:
                        pts.append(tuple(
                            p0[k] + t * (p1[k] - p0[k]) for k in range(3)
                        ))
        uniq = []
        for p in pts:
            if all(any(abs(p[k] - q[k]) > 1e-12 for k in range(3)) for q in uniq):
                uniq.append(p)
        self.assertGreaterEqual(len(uniq), 3)

        nhat = (self.A, self.B, self.C)
        nlen = math.sqrt(sum(x * x for x in nhat))
        nhat = tuple(x / nlen for x in nhat)
        ax0 = min(range(3), key=lambda i: abs(nhat[i]))
        tmp = [0.0, 0.0, 0.0]
        tmp[ax0] = 1.0
        d = sum(tmp[k] * nhat[k] for k in range(3))
        e1 = tuple(tmp[k] - d * nhat[k] for k in range(3))
        e1len = math.sqrt(sum(x * x for x in e1))
        e1 = tuple(x / e1len for x in e1)
        e2 = (
            nhat[1] * e1[2] - nhat[2] * e1[1],
            nhat[2] * e1[0] - nhat[0] * e1[2],
            nhat[0] * e1[1] - nhat[1] * e1[0],
        )
        c0 = uniq[0]
        uv = [
            (sum((p[k] - c0[k]) * e1[k] for k in range(3)),
             sum((p[k] - c0[k]) * e2[k] for k in range(3)))
            for p in uniq
        ]
        mcx = sum(u for u, _ in uv) / len(uv)
        mcy = sum(v for _, v in uv) / len(uv)
        uv.sort(key=lambda q: math.atan2(q[1] - mcy, q[0] - mcx))
        ref_area = abs(sum(
            uv[i][0] * uv[(i + 1) % len(uv)][1]
            - uv[i][1] * uv[(i + 1) % len(uv)][0]
            for i in range(len(uv))
        )) / 2.0

        # 同一多边形面积，仅求和顺序/投影构造不同：相对差 1e-12 以内。
        self.assertLess(
            abs(self.result.stats.surface_area - ref_area) / ref_area, 1e-12
        )

    def test_faces_point_toward_increasing_scalar(self):
        grad = (self.A, self.B, self.C)
        for tri in self.result.faces:
            nrm = face_normal(tri, self.result.vertices)
            self.assertGreater(
                nrm[0] * grad[0] + nrm[1] * grad[1] + nrm[2] * grad[2], 0.0
            )

    def test_manifold_with_boundary_on_domain_walls(self):
        counts = mesh_edge_counts(self.result)
        self.assertTrue(all(c in (1, 2) for c in counts.values()))
        self.assertEqual(self.result.stats.nonmanifold_edge_count, 0)
        self.assertGreater(sum(1 for c in counts.values() if c == 1), 0)
        # 独立复核域边界顶点数。
        L = self.n - 1
        on_bdy = sum(
            1 for px, py, pz in self.result.vertices
            if px in (0.0, float(L)) or py in (0.0, float(L))
            or pz in (0.0, float(L))
        )
        self.assertEqual(
            self.result.stats.surface_vertices_on_domain_boundary, on_bdy
        )
        self.assertGreater(on_bdy, 0)

    def test_no_degenerate_faces(self):
        self.assertEqual(self.result.stats.zero_area_faces_discarded, 0)
        self.assertEqual(self.result.discarded_faces, ())


class OriginSpacingPlaneTests(unittest.TestCase):
    def test_nondefault_origin_spacing(self):
        n = 4
        origin = (-1.5, 2.0, -7.0)
        spacing = (0.75, 1.25, 0.3)

        def f(x, y, z):
            return 2.0 * x + y - z + 1.0

        grid = make_grid(n, f, origin=origin, spacing=spacing)
        # 该域节点值范围约 [6.1, 15.25]，取 iso=10 穿域。
        iso = 10.0
        res = extract_isosurface(grid, iso)
        self.assertGreater(res.stats.face_count, 0)
        for px, py, pz in res.vertices:
            r = 2.0 * px + py - pz + 1.0 - iso
            self.assertLessEqual(abs(r), 64 * EPS * 16.0)
        # 独立逐边参考核对键集（严格异号边必被取、坐标逐位相等）。
        for a, b in candidate_edge_keys(n):
            pa, pb = node_world(grid, a), node_world(grid, b)
            fa, fb = f(*pa), f(*pb)
            if (fa - iso) * (fb - iso) < 0.0:
                key = (a, b) if a < b else (b, a)
                self.assertIn(key, res.edge_vertices)
                t = (iso - fa) / (fb - fa)
                expect = tuple(pa[k] + t * (pb[k] - pa[k]) for k in range(3))
                self.assertEqual(res.vertices[res.edge_vertices[key]], expect)


class SharedFaceTests(unittest.TestCase):
    def test_one_vertex_per_edge_key(self):
        n = 5
        res = extract_isosurface(make_grid(n, lambda x, y, z: x * x - y * y + z),
                                 1.25)
        self.assertGreater(res.stats.face_count, 0)
        self.assertEqual(len(res.edge_vertices), res.stats.vertex_count)
        self.assertEqual(len(set(res.edge_vertices.values())),
                         res.stats.vertex_count)

    def test_face_diagonal_intersection_shared_by_both_sides(self):
        # 线性场一般位置（iso=1.7 不落整数节点）。在内部界面 x=k 上独立
        # 找一条"面对角线"型四面体边（两端都在界面上且 y、z 都相差 1），
        # 其端点严格异号；该边只可能被界面左右两个立方体各一个四面体
        # 使用，故其唯一顶点必须同时出现在左侧和右侧立方体的三角形中。
        n = 6
        iso = 1.7

        def f(x, y, z):
            return x + 2.0 * y - 3.0 * z

        grid = make_grid(n, f)
        res = extract_isosurface(grid, iso)

        def nid(x, y, z):
            return x + n * (y + n * z)

        shared_found = 0
        for k in (2, 3):
            for y in range(n - 1):
                for z in range(n - 1):
                    for dy, dz in ((1, 1), (1, -1)):
                        y2, z2 = y + dy, z + dz
                        if not (0 <= y2 < n and 0 <= z2 < n):
                            continue
                        a, b = nid(k, y, z), nid(k, y2, z2)
                        # 只保留真正属于公开分解表的面对角线。
                        key = (a, b) if a < b else (b, a)
                        if key not in candidate_edge_keys(n):
                            continue
                        fa, fb = f(k, y, z), f(k, y2, z2)
                        if (fa - iso) * (fb - iso) >= 0.0:
                            continue
                        shared_found += 1
                        self.assertIn(key, res.edge_vertices)
                        vid = res.edge_vertices[key]
                        left = right = False
                        for tri in res.faces:
                            if vid not in tri.vertex_ids:
                                continue
                            xs = [res.vertices[w][0] for w in tri.vertex_ids]
                            if min(xs) < k - 1e-12:
                                left = True
                            if max(xs) > k + 1e-12:
                                right = True
                        self.assertTrue(left and right,
                                        f"diagonal {key} not shared at x={k}")
        self.assertGreater(shared_found, 0)

    def test_interpolation_parameter_independent_reference(self):
        n = 4
        iso = 0.0

        def f(x, y, z):
            return x * 0.3 + y * 0.7 - z * 1.1

        grid = make_grid(n, f)
        res = extract_isosurface(grid, iso)
        for (a, b), vid in res.edge_vertices.items():
            va, vb = grid.values[a], grid.values[b]
            if va == iso or vb == iso:
                continue  # 端点恰等情形由专门测试覆盖
            t = (iso - va) / (vb - va)
            self.assertTrue(0.0 < t < 1.0)
            pa, pb = node_world(grid, a), node_world(grid, b)
            expect = tuple(pa[k] + t * (pb[k] - pa[k]) for k in range(3))
            self.assertEqual(res.vertices[vid], expect)


class IsoAtNodeTests(unittest.TestCase):
    def test_perturbation_rule_pairwise_contract(self):
        # 直接核对公开约定：偶 id 值恰 iso 归 above，奇 id 归 below；
        # 严格大于/小于不受 id 影响。（规则本身是契约，手算期望。）
        from mtiso.core import _classified_above
        self.assertTrue(_classified_above(0, 1.0, 1.0))   # 偶 id 恰等 -> above
        self.assertFalse(_classified_above(1, 1.0, 1.0))  # 奇 id 恰等 -> below
        self.assertTrue(_classified_above(1, 1.5, 1.0))   # 严格大于，与 id 无关
        self.assertFalse(_classified_above(0, 0.5, 1.0))  # 严格小于，与 id 无关
        self.assertFalse(_classified_above(7, 1.0, 1.0))  # 奇 id 恰等 -> below
        self.assertTrue(_classified_above(8, 1.0, 1.0))   # 偶 id 恰等 -> above

    def test_even_id_iso_node_is_above_degenerate_geometry(self):
        # 2x2x2：仅偶 id 节点 0 取 iso=0，其余 -1。
        # 扰动：节点 0 above，其余 below。6 个含角点 0 的四面体各生成
        # 1 个三角形；3 条跨立边交点 t 全为 0 => 三顶点重合于节点 0，
        # 几何零面积 => 6 个三角形全部丢弃并记录；7 条边键各建 1 个
        # 顶点（坐标全部恰为 (0,0,0)，几何未被扰动改变）。
        vals = [-1.0] * 8
        vals[0] = 0.0
        res = extract_isosurface(ScalarGrid(2, 2, 2, vals), 0.0)
        self.assertEqual(res.stats.face_count, 0)
        self.assertEqual(res.stats.zero_area_faces_discarded, 6)
        self.assertEqual(len(res.discarded_faces), 6)
        self.assertEqual(res.stats.vertex_count, 7)
        self.assertTrue(all(p == (0.0, 0.0, 0.0) for p in res.vertices))
        tets = sorted(d.tetrahedron for d in res.discarded_faces)
        self.assertEqual(tets, list(range(6)))
        for d in res.discarded_faces:
            self.assertEqual(d.voxel, (0, 0, 0))
            self.assertEqual(d.reason, "zero_area")
            self.assertTrue(all(0 in ek for ek in d.edge_keys))

    def test_odd_id_iso_node_is_below_no_surface(self):
        # 同构对照：仅奇 id 节点 1 取 iso=0，其余 -1。
        # 扰动：节点 1 below => 全 below => 无任何交点/面/丢弃。
        vals = [-1.0] * 8
        vals[1] = 0.0
        res = extract_isosurface(ScalarGrid(2, 2, 2, vals), 0.0)
        self.assertEqual(res.stats.vertex_count, 0)
        self.assertEqual(res.stats.face_count, 0)
        self.assertEqual(res.stats.zero_area_faces_discarded, 0)

    def test_plane_through_node_layer_geometry_unchanged(self):
        # iso 平面 x=2 穿过整层节点：归属按 id 奇偶交替，但所有交点的
        # x 坐标必须恰为 2.0（扰动不动几何），面全部朝 +x，组合流形。
        n = 5
        res = extract_isosurface(make_grid(n, lambda x, y, z: float(x)), 2.0)
        self.assertGreater(res.stats.face_count, 0)
        self.assertTrue(all(p[0] == 2.0 for p in res.vertices))
        counts = mesh_edge_counts(res)
        self.assertTrue(all(c in (1, 2) for c in counts.values()))
        self.assertEqual(res.stats.nonmanifold_edge_count, 0)
        for tri in res.faces:
            self.assertGreater(face_normal(tri, res.vertices)[0], 0.0)

    def test_intersection_at_iso_endpoint_uses_exact_node_position(self):
        # f=x+y+z-3, iso=0：节点 (1,1,1) 恰在面上。凡端点值恰为 iso 的
        # 边，交点必须精确位于该节点；其余边按独立线性插值逐位相等。
        n = 5
        iso = 0.0

        def f(x, y, z):
            return x + y + z - 3

        grid = make_grid(n, f)
        res = extract_isosurface(grid, iso)
        self.assertGreater(res.stats.face_count, 0)
        for (a, b), vid in res.edge_vertices.items():
            va, vb = grid.values[a], grid.values[b]
            pos = res.vertices[vid]
            if va == iso:
                self.assertEqual(pos, node_world(grid, a))
            elif vb == iso:
                self.assertEqual(pos, node_world(grid, b))
            else:
                t = (iso - va) / (vb - va)
                pa, pb = node_world(grid, a), node_world(grid, b)
                self.assertEqual(pos, tuple(
                    pa[k] + t * (pb[k] - pa[k]) for k in range(3)
                ))


class ConstantFieldTests(unittest.TestCase):
    def test_constant_off_iso_empty(self):
        n = 4
        for const, iso in ((1.0, 2.0), (3.0, 2.0)):
            res = extract_isosurface(make_grid(n, lambda x, y, z: const), iso)
            self.assertEqual(res.stats.vertex_count, 0)
            self.assertEqual(res.stats.face_count, 0)
            self.assertEqual(res.stats.surface_area, 0.0)
            self.assertEqual(res.discarded_faces, ())
            self.assertEqual(res.stats.level_tetrahedra_skipped, 0)

    def test_constant_equal_iso_full_dimensional_skip(self):
        n = 4
        res = extract_isosurface(make_grid(n, lambda x, y, z: 5.0), 5.0)
        self.assertEqual(res.stats.face_count, 0)
        self.assertEqual(res.stats.vertex_count, 0)
        self.assertEqual(res.stats.surface_area, 0.0)
        voxels = (n - 1) ** 3
        self.assertEqual(res.stats.level_tetrahedra_skipped, 6 * voxels)
        self.assertEqual(res.stats.tetrahedron_count, 6 * voxels)


class SphereClosureTests(unittest.TestCase):
    """f=|p-c|², iso=R²=7。

    勒让德三平方定理：形如 4^a(8b+7) 的数不能表示为三个整数平方和，
    7=8*0+7，故单位（及 h=0.5）网格上 |p-c|²=7 不落任何节点（c 取整数/
    半整数时，|p-c|² 的 4 倍为整数平方和且 =28 同样不可表示），等值面
    拓扑必是一张闭合球面膜。
    """

    N, H = 13, 0.5
    R2 = 7.0

    @classmethod
    def setUpClass(cls):
        cls.R = math.sqrt(cls.R2)
        cls.c = (cls.N - 1) * cls.H / 2
        c = cls.c

        def f(x, y, z):
            return (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2

        cls.grid = make_grid(cls.N, f, spacing=(cls.H, cls.H, cls.H))
        cls.result = extract_isosurface(cls.grid, cls.R2)

    def test_closed_manifold_euler_sphere(self):
        counts = mesh_edge_counts(self.result)
        self.assertGreater(len(counts), 0)
        self.assertTrue(all(c == 2 for c in counts.values()))
        self.assertEqual(self.result.stats.boundary_edge_count, 0)
        self.assertEqual(self.result.stats.nonmanifold_edge_count, 0)
        self.assertEqual(self.result.boundary_edge_keys, ())
        self.assertEqual(self.result.nonmanifold_edge_keys, ())
        self.assertEqual(
            self.result.stats.vertex_count - len(counts)
            + self.result.stats.face_count,
            2,
        )

    def test_strictly_inside_domain(self):
        self.assertEqual(
            self.result.stats.surface_vertices_on_domain_boundary, 0
        )
        L = (self.N - 1) * self.H
        for px, py, pz in self.result.vertices:
            self.assertTrue(0.0 < px < L and 0.0 < py < L and 0.0 < pz < L)

    def test_radial_error_analytic_bound(self):
        # 节点上二次函数精确，边内线性插值截断，径向偏差上界 h²/(2R)。
        bound = self.H * self.H / (2 * self.R)
        for px, py, pz in self.result.vertices:
            r = math.sqrt((px - self.c) ** 2 + (py - self.c) ** 2
                          + (pz - self.c) ** 2)
            self.assertLessEqual(abs(r - self.R), bound + 1e-12)

    def test_area_volume_analytic_references(self):
        ref_a = 4 * math.pi * self.R2
        ref_v = 4 / 3 * math.pi * self.R2 ** 1.5
        self.assertLess(
            abs(self.result.stats.surface_area - ref_a) / ref_a, 0.02
        )
        vol = signed_volume(self.result)
        self.assertGreater(vol, 0.0)  # |p-c|² 向外增大 -> 外法向
        self.assertLess(abs(vol - ref_v) / ref_v, 0.05)

    def test_orientation_outward(self):
        for tri in self.result.faces:
            nrm = face_normal(tri, self.result.vertices)
            w = tri.vertex_ids
            mid = tuple(
                sum(self.result.vertices[w[j]][k] for j in range(3)) / 3.0
                for k in range(3)
            )
            grad = (mid[0] - self.c, mid[1] - self.c, mid[2] - self.c)
            self.assertGreater(
                nrm[0] * grad[0] + nrm[1] * grad[1] + nrm[2] * grad[2], 0.0
            )

    def test_two_resolution_convergence(self):
        # h=1 与 h=0.5 同一物理球：相对面积误差加密后至少减半（一阶以上，
        # 实测约二阶）。参考面积解析给定，不来自被测实现。
        ref_a = 4 * math.pi * self.R2
        errs = []
        for n, h in ((7, 1.0), (13, 0.5)):
            c = (n - 1) * h / 2

            def f(x, y, z, c=c):
                return (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2

            r = extract_isosurface(
                make_grid(n, f, spacing=(h, h, h)), self.R2
            )
            errs.append(abs(r.stats.surface_area - ref_a) / ref_a)
        self.assertLess(errs[1], errs[0] / 2)


class ValidationTests(unittest.TestCase):
    def test_reject_nan_inf_iso(self):
        grid = make_grid(2, lambda x, y, z: 0.0)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                extract_isosurface(grid, bad)

    def test_reject_nan_inf_values_localizable(self):
        for bad, word in ((float("nan"), "NaN"), (float("inf"), "infinity")):
            vals = [0.0] * 8
            vals[5] = bad
            with self.assertRaises(ValueError) as cm:
                ScalarGrid(2, 2, 2, vals)
            msg = str(cm.exception)
            self.assertIn("values[5]", msg)
            self.assertIn("ix=1, iy=0, iz=1", msg)
            self.assertIn(word, msg)

    def test_reject_bool_everywhere(self):
        with self.assertRaises(TypeError):
            ScalarGrid(True, 2, 2, [0.0] * 8)
        with self.assertRaises(TypeError):
            ScalarGrid(2, False, 2, [0.0] * 8)
        with self.assertRaises(TypeError):
            ScalarGrid(2, 2, 2, [True] * 8)
        with self.assertRaises(TypeError):
            ScalarGrid(2, 2, 2, [0] * 8, spacing=(1, 1, True))
        with self.assertRaises(TypeError):
            extract_isosurface(make_grid(2, lambda x, y, z: 0.0), True)

    def test_dimension_bounds_and_types(self):
        for bad in (1, 17, 0, -3):
            with self.assertRaises(ValueError):
                ScalarGrid(bad, 2, 2, [0.0] * 8)
        for bad in (2.0, "3", None):
            with self.assertRaises(TypeError):
                ScalarGrid(bad, 2, 2, [0.0] * 8)
        g = ScalarGrid(16, 2, 2, [0.0] * 64)  # 边界值恰好接受
        self.assertEqual((g.nx, g.ny, g.nz), (16, 2, 2))

    def test_values_length_mismatch_localizable(self):
        with self.assertRaises(ValueError) as cm:
            ScalarGrid(2, 2, 2, [0.0] * 7)
        msg = str(cm.exception)
        self.assertIn("length 7", msg)
        self.assertIn("nx*ny*nz = 8", msg)

    def test_spacing_and_origin_rules(self):
        with self.assertRaises(ValueError):
            ScalarGrid(2, 2, 2, [0.0] * 8, spacing=(1, 0, 1))
        with self.assertRaises(ValueError):
            ScalarGrid(2, 2, 2, [0.0] * 8, spacing=(1, -1.0, 1))
        with self.assertRaises(ValueError):
            ScalarGrid(2, 2, 2, [0.0] * 8, spacing=(1, float("nan"), 1))
        with self.assertRaises(ValueError):
            ScalarGrid(2, 2, 2, [0.0] * 8, origin=(0, 0))
        with self.assertRaises(ValueError):
            ScalarGrid(2, 2, 2, [0.0] * 8, origin=(0, 0, float("inf")))

    def test_wrong_grid_type(self):
        with self.assertRaises(TypeError):
            extract_isosurface([[0.0]], 0.0)
        with self.assertRaises(TypeError):
            extract_isosurface("grid", 0.0)

    def test_results_are_frozen(self):
        import dataclasses
        res = extract_isosurface(make_grid(3, lambda x, y, z: float(x)), 0.5)
        with self.assertRaises(Exception):
            res.vertices.append((1.0, 2.0, 3.0))
        with self.assertRaises(Exception):
            res.faces[0].vertex_ids = (1, 2, 3)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            res.stats.face_count = 3

    def test_pure_function_does_not_mutate_input(self):
        grid = make_grid(3, lambda x, y, z: float(x))
        before = grid.values
        extract_isosurface(grid, 0.5)
        self.assertEqual(grid.values, before)


if __name__ == "__main__":
    unittest.main()
