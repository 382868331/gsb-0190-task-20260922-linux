"""extract_band 的增量测试。

独立参考：数值带的期望体积由这里的解析/几何裁剪给出，不调用 iso3d；
有向三角网体积积分直接按面顶点重算，仅作为与 stats 的交叉核对。
"""

import math
import unittest
from collections import Counter

from iso3d import (
    InvalidInputError,
    ScalarField3D,
    extract_band,
    signed_volume,
)


def linear_field(n, coeffs=(1.0, 1.0, 1.0)):
    a, b, c = coeffs
    values = tuple(
        a * i + b * j + c * k
        for k in range(n)
        for j in range(n)
        for i in range(n)
    )
    return ScalarField3D(ni=n, nj=n, nk=n, values=values)


def cdf_diag(t, L):
    """单位斜率对角场 x+y+z 在 [0,L]^3 中 <= t 的体积（分段三次，独立）。"""
    if t <= 0:
        return 0.0
    if t <= L:
        return t ** 3 / 6.0
    if t <= 2 * L:
        return (t ** 3 - 3.0 * (t - L) ** 3) / 6.0
    if t <= 3 * L:
        return L ** 3 - (3.0 * L - t) ** 3 / 6.0
    return L ** 3


def diag_slab_volume(n, low, high):
    L = n - 1
    return cdf_diag(high, L) - cdf_diag(low, L)


def edge_orientation_check(mesh):
    """每条非退化边恰好被两个方向相反的面邻接；返回 (bad, use_counter)。"""
    directed = Counter()
    for face in mesh.faces:
        v = face.vertex_ids
        for a, b in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            directed[(a, b)] += 1
    undirected = Counter()
    for (a, b), count in directed.items():
        undirected[(a, b) if a < b else (b, a)] += count
    bad = []
    for key, total in undirected.items():
        x, y = key
        if total != 2 or directed[(x, y)] != 1 or directed[(y, x)] != 1:
            bad.append(key)
    return bad, undirected


class BandVolumeTest(unittest.TestCase):
    def setUp(self):
        self.n = 6
        self.field = linear_field(self.n)

    def test_slab_volume_matches_reference_and_integral(self):
        low, high = 5.0, 10.0
        mesh = extract_band(self.field, low, high)
        analytic = diag_slab_volume(self.n, low, high)
        self.assertAlmostEqual(mesh.stats.signed_volume, analytic, places=9)
        # 独立重算有向体积积分，不通过 stats。
        direct = signed_volume(mesh.vertices, mesh.faces)
        self.assertAlmostEqual(direct, analytic, places=9)
        self.assertGreater(direct, 0.0)

    def test_every_nonmanifold_edge_has_two_opposite_faces(self):
        mesh = extract_band(self.field, 5.0, 10.0)
        bad, uses = edge_orientation_check(mesh)
        self.assertEqual(bad, [])
        self.assertEqual(mesh.stats.nonmanifold_edge_count, 0)
        self.assertEqual(mesh.stats.boundary_edge_count, 0)
        self.assertTrue(all(c == 2 for c in uses.values()))

    def test_low_and_high_walls_not_merged(self):
        mesh = extract_band(self.field, 7.0, 8.0)
        self.assertGreater(mesh.stats.low_wall_face_count, 0)
        self.assertGreater(mesh.stats.high_wall_face_count, 0)
        # 同一条网格边可同时穿过两阈值；两交点必须是不同顶点身份
        # （按 edge_key + level 去重，绝不被错误合并）。
        shared = {}
        for v in mesh.vertices:
            if v.kind == "edge":
                shared.setdefault(v.edge_key, set()).add((v.id, v.level))
        merged = [k for k, items in shared.items()
                  if len({level for _, level in items}) != len(items)]
        self.assertEqual(merged, [])
        self.assertTrue(any(len(items) == 2 for items in shared.values()))


class BandCoverageTest(unittest.TestCase):
    def setUp(self):
        self.n = 5
        self.field = linear_field(self.n)

    def test_touching_boundary_sealed(self):
        mesh = extract_band(self.field, 2.0, 4.0)
        self.assertGreater(mesh.stats.cap_face_count, 0)
        self.assertEqual(mesh.stats.boundary_edge_count, 0)
        bad, _ = edge_orientation_check(mesh)
        self.assertEqual(bad, [])
        self.assertAlmostEqual(
            mesh.stats.signed_volume,
            diag_slab_volume(self.n, 2.0, 4.0),
            places=9,
        )

    def test_fully_contained_box(self):
        mesh = extract_band(self.field, -10.0, 100.0)
        self.assertEqual(mesh.stats.low_wall_face_count, 0)
        self.assertEqual(mesh.stats.high_wall_face_count, 0)
        # 盖片恰好覆盖盒子的 6 个面：体积为整盒，无边界边。
        L = self.n - 1
        self.assertAlmostEqual(mesh.stats.signed_volume, float(L ** 3), places=9)
        self.assertEqual(mesh.stats.boundary_edge_count, 0)
        bad, _ = edge_orientation_check(mesh)
        self.assertEqual(bad, [])

    def test_completely_empty(self):
        mesh = extract_band(self.field, -100.0, -50.0)
        self.assertEqual(mesh.stats.face_count, 0)
        self.assertEqual(mesh.stats.vertex_count, 0)
        self.assertEqual(mesh.stats.signed_volume, 0.0)
        self.assertEqual(mesh.stats.cap_face_count, 0)


class BandThresholdTest(unittest.TestCase):
    def test_exact_threshold_belongs_to_band_uniformly(self):
        n = 5
        field = linear_field(n)
        # 阈值恰落在整数节点值层 x+y+z = 4 上。
        mesh = extract_band(field, 4.0, 6.0)
        self.assertEqual(mesh.stats.boundary_edge_count, 0)
        bad, _ = edge_orientation_check(mesh)
        self.assertEqual(bad, [])
        self.assertAlmostEqual(
            mesh.stats.signed_volume, diag_slab_volume(n, 4.0, 6.0), places=9
        )
        # 盖片中落在阈值层上的节点必须是 node 身份（与墙共享），
        # 不允许同一节点因奇偶扰动产生重复盖面。
        cap_node_ids = {
            v.id for v in mesh.vertices
            if v.kind == "node" and any(
                f.surface == "cap" and v.id in f.vertex_ids for f in mesh.faces
            )
        }
        self.assertTrue(cap_node_ids)

    def test_zero_thickness_is_surface_not_solid(self):
        n = 6
        field = linear_field(n)
        mesh = extract_band(field, 7.5, 7.5)
        self.assertTrue(mesh.thickness_zero)
        self.assertEqual(mesh.stats.high_wall_face_count, 0)
        self.assertEqual(mesh.stats.cap_face_count, 0)
        self.assertGreater(mesh.stats.low_wall_face_count, 0)
        self.assertEqual(mesh.stats.signed_volume, 0.0)
        # 零厚度是开口曲面（在盒子内部被截断），如实报告边界边，
        # 不伪装成闭合实体。
        self.assertGreater(mesh.stats.boundary_edge_count, 0)

    def test_zero_thickness_internal_surface_closed(self):
        # 球面状场中零厚度等值面完全位于网格内部，闭合但仍非实体。
        n = 8
        c = (n - 1) / 2.0
        values = tuple(
            (i - c) ** 2 + (j - c) ** 2 + (k - c) ** 2
            for k in range(n) for j in range(n) for i in range(n)
        )
        field = ScalarField3D(n, n, n, values)
        mesh = extract_band(field, 4.0, 4.0)
        self.assertTrue(mesh.thickness_zero)
        self.assertEqual(mesh.stats.boundary_edge_count, 0)
        self.assertEqual(mesh.stats.signed_volume, 0.0)


class BandValidationTest(unittest.TestCase):
    def test_low_greater_than_high_rejected(self):
        field = linear_field(3)
        with self.assertRaises(InvalidInputError) as ctx:
            extract_band(field, 5.0, 4.0)
        self.assertEqual(ctx.exception.parameter, "low")

    def test_non_field_rejected(self):
        with self.assertRaises(InvalidInputError):
            extract_band("not a field", 0.0, 1.0)

    def test_non_finite_rejected(self):
        field = linear_field(3)
        for kwargs in ({"low": math.nan, "high": 1.0},
                       {"low": 0.0, "high": math.inf}):
            with self.assertRaises(InvalidInputError):
                extract_band(field, **kwargs)

    def test_bool_rejected(self):
        field = linear_field(3)
        with self.assertRaises(InvalidInputError):
            extract_band(field, True, 1.0)


if __name__ == "__main__":
    unittest.main()
