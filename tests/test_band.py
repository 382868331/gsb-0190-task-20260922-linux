"""数值带提取（extract_band）的几何、拓扑与错误语义测试。

期望值来源：
- 线性场数值带体积：tests/_reference.py 的独立凸多面体裁剪实现；
- 闭合性 / 对向邻接：拓扑不变量（边界边计数、有向边配对）；
- 零厚度：与 extract_isosurface 单层等值面的对照。
"""

import unittest

from iso3d import (
    InvalidInputError,
    ScalarField3D,
    extract_band,
    extract_isosurface,
)

from _reference import band_box_volume

REL_TOL = 1e-9  # 线性场下提取是精确的，仅含浮点累加舍入


def make_field(ni, nj, nk, func):
    values = tuple(
        float(func(i, j, k))
        for k in range(nk)
        for j in range(nj)
        for i in range(ni)
    )
    return ScalarField3D(ni=ni, nj=nj, nk=nk, values=values)


def linear_field(n, a, b, c, d):
    return make_field(n, n, n, lambda i, j, k: a * i + b * j + c * k + d)


def shell_field(n, center=None):
    c = (n - 1) / 2.0 if center is None else center
    return make_field(n, n, n, lambda i, j, k: (i - c) ** 2 + (j - c) ** 2 + (k - c) ** 2)


def face_normal(mesh, face):
    p = [mesh.vertices[v].point for v in face.vertex_ids]
    u = tuple(p[1][d] - p[0][d] for d in range(3))
    v = tuple(p[2][d] - p[0][d] for d in range(3))
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def face_centroid(mesh, face):
    p = [mesh.vertices[v].point for v in face.vertex_ids]
    return tuple(sum(q[d] for q in p) / 3.0 for d in range(3))


def assert_closed_and_paired(test_case, mesh):
    """每条非退化边恰被两个面以相反方向使用。"""
    test_case.assertEqual(mesh.stats.boundary_edge_count, 0)
    test_case.assertEqual(mesh.stats.nonmanifold_edge_count, 0)
    directed = {}
    for face in mesh.faces:
        v = face.vertex_ids
        for a, b in ((v[0], v[1]), (v[1], v[2]), (v[2], v[0])):
            directed[(a, b)] = directed.get((a, b), 0) + 1
    for (a, b), count in directed.items():
        test_case.assertEqual(count, 1, f"directed edge {(a, b)} used {count} times")
        test_case.assertIn((b, a), directed, f"edge {(a, b)} has no opposite partner")


class LinearBandVolumeTest(unittest.TestCase):
    """线性标量场切盒子：有向三角网体积积分对照独立凸多面体体积。"""

    CASES = (
        # (n, a, b, c, d, low, high)：截面均不经过网格节点
        (8, 0.9, 1.1, 0.7, 0.3, 4.0, 9.0),     # 斜向，数值带触边
        (8, 0.9, 1.1, 0.7, 0.3, 6.5, 7.5),     # 薄带
        (6, 1.3, -0.7, 0.9, 2.1, 3.0, 8.0),    # 负系数
        (4, 10.0, 0.0, 0.0, 0.0, 2.0, 8.0),    # 单轴：同边穿越 low 与 high
        (16, 0.31, 0.17, 0.23, 0.11, 2.0, 6.0),  # 满上限网格
    )

    def test_signed_volume_matches_independent_reference(self):
        for n, a, b, c, d, low, high in self.CASES:
            with self.subTest(case=(n, a, b, c, d, low, high)):
                field = linear_field(n, a, b, c, d)
                mesh = extract_band(field, low, high)
                # 场为 a*i+b*j+c*k+d：等价于对 f-d 求 [low-d, high-d] 区域
                expected = band_box_volume(
                    a, b, c, low - d, high - d, n - 1, n - 1, n - 1
                )
                self.assertGreater(expected, 0.0)
                self.assertLess(
                    abs(mesh.stats.signed_volume - expected), REL_TOL * expected
                )

    def test_closed_mesh_with_opposite_edge_pairing(self):
        for n, a, b, c, d, low, high in self.CASES:
            with self.subTest(case=(n, a, b, c, d, low, high)):
                field = linear_field(n, a, b, c, d)
                mesh = extract_band(field, low, high)
                assert_closed_and_paired(self, mesh)

    def test_band_touching_boundary_has_caps(self):
        field = linear_field(8, 0.9, 1.1, 0.7, 0.3)
        mesh = extract_band(field, 4.0, 9.0)
        self.assertGreater(mesh.stats.cap_face_count, 0)
        self.assertFalse(mesh.zero_thickness)

    def test_low_and_high_surfaces_not_merged(self):
        # f = 10i：同一条网格边同时穿越 low=2 与 high=8，
        # 两层交点必须注册为不同顶点（t = 0.2 与 t = 0.8）。
        field = linear_field(4, 10.0, 0.0, 0.0, 0.0)
        mesh = extract_band(field, 2.0, 8.0)
        by_edge = {}
        for v in mesh.vertices:
            if v.edge_key[0] != v.edge_key[1]:
                by_edge.setdefault(v.edge_key, set()).add(round(v.t, 12))
        shared = {k: ts for k, ts in by_edge.items() if len(ts) == 2}
        self.assertTrue(shared, "expected edges crossed by both levels")
        for ts in shared.values():
            self.assertEqual(ts, {0.2, 0.8})
        self.assertAlmostEqual(mesh.stats.signed_volume, 0.6 * 3.0 * 3.0, places=9)


class ShellBandTest(unittest.TestCase):
    """球壳数值带（不触边）：两层等值面均为闭曲面，无盖片。"""

    @classmethod
    def setUpClass(cls):
        cls.n = 12
        cls.center = (cls.n - 1) / 2.0
        cls.field = shell_field(cls.n)
        cls.mesh = extract_band(cls.field, 4.0, 9.0)  # 半径 2 与 3 之间

    def test_no_caps_when_band_is_interior(self):
        self.assertEqual(self.mesh.stats.cap_face_count, 0)

    def test_closed_and_paired(self):
        assert_closed_and_paired(self, self.mesh)

    def test_normals_point_outward_from_band(self):
        # 高面法向远离中心，低面法向指向中心（均为数值带外侧）。
        for face in self.mesh.faces:
            cen = face_centroid(self.mesh, face)
            normal = face_normal(self.mesh, face)
            radial = tuple(cen[d] - self.center for d in range(3))
            r2 = sum(x * x for x in radial)
            dot = sum(normal[d] * radial[d] for d in range(3))
            if r2 > 6.5:  # 高面一侧（半径约 3）
                self.assertGreater(dot, 0.0)
            else:  # 低面一侧（半径约 2）
                self.assertLess(dot, 0.0)

    def test_signed_volume_matches_shell_volume_loosely(self):
        # 分片线性逼近：体积应接近解析球壳体积（宽松合理性检查）。
        import math

        analytic = 4.0 / 3.0 * math.pi * (27.0 - 8.0)
        self.assertLess(abs(self.mesh.stats.signed_volume - analytic), 0.15 * analytic)


class FullContainmentTest(unittest.TestCase):
    """数值带覆盖整个网格：边界即网格外表面，体积为盒子体积。"""

    def test_full_band(self):
        n = 6
        field = linear_field(n, 0.9, 1.1, 0.7, 0.3)
        mesh = extract_band(field, -1.0, 100.0)
        self.assertAlmostEqual(mesh.stats.signed_volume, float((n - 1) ** 3), places=9)
        self.assertGreater(mesh.stats.cap_face_count, 0)
        assert_closed_and_paired(self, mesh)

    def test_full_band_exact_limits(self):
        # low/high 恰等于场最小/最大值：仍应全包含（归属一致）。
        n = 5
        field = linear_field(n, 0.9, 1.1, 0.7, 0.3)
        lo = min(field.values)
        hi = max(field.values)
        mesh = extract_band(field, lo, hi)
        self.assertAlmostEqual(mesh.stats.signed_volume, float((n - 1) ** 3), places=9)


class EmptyBandTest(unittest.TestCase):
    """数值带处处为空：合法空结果，不抛异常。"""

    def test_band_above_range(self):
        field = linear_field(6, 0.9, 1.1, 0.7, 0.3)
        mesh = extract_band(field, 50.0, 60.0)
        self.assertEqual(mesh.stats.face_count, 0)
        self.assertEqual(mesh.stats.vertex_count, 0)
        self.assertEqual(mesh.stats.signed_volume, 0.0)
        self.assertEqual(mesh.stats.cap_face_count, 0)

    def test_band_below_range(self):
        field = linear_field(6, 0.9, 1.1, 0.7, 0.3)
        mesh = extract_band(field, -10.0, -1.0)
        self.assertEqual(mesh.stats.face_count, 0)

    def test_single_layer_grid_is_empty(self):
        field = make_field(1, 4, 4, lambda i, j, k: i + j + k)
        mesh = extract_band(field, 0.0, 100.0)
        self.assertEqual(mesh.stats.face_count, 0)


class ZeroThicknessTest(unittest.TestCase):
    """low == high：零厚度特殊结果，不伪装成有体积实体。"""

    def test_zero_thickness_matches_single_isosurface(self):
        field = linear_field(8, 0.9, 1.1, 0.7, 0.3)
        mesh = extract_band(field, 5.5, 5.5)
        iso_mesh = extract_isosurface(field, 5.5)
        self.assertTrue(mesh.zero_thickness)
        self.assertEqual(mesh.stats.face_count, iso_mesh.stats.face_count)
        self.assertAlmostEqual(mesh.stats.total_area, iso_mesh.stats.total_area, places=9)
        self.assertEqual(mesh.stats.cap_face_count, 0)
        self.assertEqual(mesh.stats.signed_volume, 0.0)

    def test_zero_thickness_constant_field(self):
        field = make_field(5, 5, 5, lambda i, j, k: 5.0)
        mesh = extract_band(field, 5.0, 5.0)
        self.assertTrue(mesh.zero_thickness)
        self.assertEqual(mesh.stats.signed_volume, 0.0)
        self.assertEqual(mesh.stats.cap_face_count, 0)


class ExactThresholdTest(unittest.TestCase):
    """标量恰等于阈值：归属统一，邻单元不重复盖面。"""

    def assert_no_duplicate_faces(self, mesh):
        seen = set()
        for face in mesh.faces:
            key = tuple(
                sorted(tuple(mesh.vertices[v].point) for v in face.vertex_ids)
            )
            self.assertNotIn(key, seen, "duplicate cap/surface face")
            seen.add(key)

    def test_constant_field_low_equals_value(self):
        field = make_field(6, 6, 6, lambda i, j, k: 5.0)
        mesh = extract_band(field, 5.0, 7.0)
        self.assert_no_duplicate_faces(mesh)
        # 整个边界都在带内：盖片覆盖全部 6 个外表面。
        self.assertEqual(mesh.stats.cap_face_count, 6 * 5 * 5 * 2)

    def test_constant_field_high_equals_value(self):
        field = make_field(6, 6, 6, lambda i, j, k: 5.0)
        mesh = extract_band(field, 3.0, 5.0)
        self.assert_no_duplicate_faces(mesh)
        self.assertEqual(mesh.stats.cap_face_count, 6 * 5 * 5 * 2)

    def test_mixed_exact_values_on_boundary(self):
        # 边界与内部都散布恰等于 low/high 的节点。
        def func(i, j, k):
            if (i + j + k) % 3 == 0:
                return 2.0   # == low
            if (i * j + k) % 4 == 0:
                return 8.0   # == high
            return 0.4 * i + 0.6 * j + 0.5 * k

        field = make_field(7, 7, 7, func)
        mesh = extract_band(field, 2.0, 8.0)
        self.assert_no_duplicate_faces(mesh)
        again = extract_band(field, 2.0, 8.0)
        self.assertEqual(mesh.stats, again.stats)  # 确定性

    def test_exact_equality_closed_mesh(self):
        # 节点精确落在 high 上（0.9*4+1.1*4+0.7+0.3 = 9.0）：网格仍应闭合。
        field = linear_field(8, 0.9, 1.1, 0.7, 0.3)
        self.assertIn(9.0, field.values)
        mesh = extract_band(field, 4.0, 9.0)
        assert_closed_and_paired(self, mesh)
        expected = band_box_volume(0.9, 1.1, 0.7, 4.0 - 0.3, 9.0 - 0.3, 7.0, 7.0, 7.0)
        self.assertLess(abs(mesh.stats.signed_volume - expected), REL_TOL * expected)


class BandValidationTest(unittest.TestCase):
    """extract_band 的输入校验与错误语义。"""

    def setUp(self):
        self.field = linear_field(4, 1.0, 1.0, 1.0, 0.0)

    def test_low_greater_than_high_rejected(self):
        with self.assertRaises(InvalidInputError) as ctx:
            extract_band(self.field, 5.0, 2.0)
        self.assertEqual(ctx.exception.parameter, "high")

    def test_non_finite_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidInputError):
                    extract_band(self.field, bad, 10.0)
                with self.assertRaises(InvalidInputError):
                    extract_band(self.field, 0.0, bad)

    def test_bool_rejected(self):
        with self.assertRaises(InvalidInputError):
            extract_band(self.field, True, 10.0)
        with self.assertRaises(InvalidInputError):
            extract_band(self.field, 0.0, False)

    def test_non_field_rejected(self):
        with self.assertRaises(InvalidInputError) as ctx:
            extract_band(object(), 0.0, 1.0)
        self.assertEqual(ctx.exception.parameter, "field")

    def test_int_bounds_accepted(self):
        mesh = extract_band(self.field, 1, 3)
        self.assertEqual(mesh.low, 1.0)
        self.assertEqual(mesh.high, 3.0)


if __name__ == "__main__":
    unittest.main()
