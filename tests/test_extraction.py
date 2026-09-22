"""核心提取算法的几何与拓扑测试。

期望值来源：
- 平面截面面积：tests/_reference.py 的独立解析实现；
- 穿越边键集合：tests/_reference.py 的独立枚举；
- 球场闭合性 / 常量场：拓扑不变量（边界边、非流形边计数）。
"""

import math
import unittest

from iso3d import ScalarField3D, extract_isosurface

from _reference import plane_box_intersection_area, ref_crossing_edge_keys

REL_TOL = 1e-9  # 平面面积比较的相对容差（线性场下提取是精确的，仅含浮点舍入）


def make_field(ni, nj, nk, func):
    values = tuple(
        float(func(i, j, k))
        for k in range(nk)
        for j in range(nj)
        for i in range(ni)
    )
    return ScalarField3D(ni=ni, nj=nj, nk=nk, values=values)


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


class PlaneFieldTest(unittest.TestCase):
    """f = i + j + k，iso = 7.5（无节点恰好落在等值面上），16^3 满上限网格。"""

    @classmethod
    def setUpClass(cls):
        cls.n = 16
        cls.field = make_field(cls.n, cls.n, cls.n, lambda i, j, k: i + j + k)
        cls.iso = 7.5
        cls.mesh = extract_isosurface(cls.field, cls.iso)

    def test_area_matches_analytic_plane_section(self):
        expected = plane_box_intersection_area(1.0, 1.0, 1.0, self.iso, 15.0, 15.0, 15.0)
        self.assertGreater(expected, 0.0)
        self.assertLess(
            abs(self.mesh.stats.total_area - expected), REL_TOL * expected
        )

    def test_normals_point_toward_increasing_scalar(self):
        # f 的梯度方向是 (1,1,1)，所有面法向与其点积必须为正。
        for face in self.mesh.faces:
            n = face_normal(self.mesh, face)
            self.assertGreater(n[0] + n[1] + n[2], 0.0)

    def test_open_surface_has_boundary_but_no_nonmanifold_edges(self):
        self.assertGreater(self.mesh.stats.boundary_edge_count, 0)
        self.assertEqual(self.mesh.stats.nonmanifold_edge_count, 0)
        self.assertEqual(len(self.mesh.nonmanifold_edges), 0)
        # 边界边必须躺在定义域边界面上。
        limit = float(self.n - 1)
        for key in self.mesh.boundary_edges:
            for vid in key:
                p = self.mesh.vertices[vid].point
                self.assertTrue(
                    any(c == 0.0 or c == limit for c in p),
                    f"boundary vertex {p} not on domain boundary",
                )

    def test_no_degenerate_faces_for_generic_plane(self):
        self.assertEqual(self.mesh.stats.degenerate_face_count, 0)
        self.assertEqual(self.mesh.degenerate_faces, ())

    def test_stats_consistency(self):
        s = self.mesh.stats
        self.assertEqual(s.vertex_count, len(self.mesh.vertices))
        self.assertEqual(s.face_count, len(self.mesh.faces))
        self.assertEqual(s.node_count, self.n ** 3)
        self.assertEqual(s.cell_count, (self.n - 1) ** 3)
        self.assertEqual(s.tetrahedron_count, 6 * (self.n - 1) ** 3)
        self.assertAlmostEqual(
            s.total_area, sum(f.area for f in self.mesh.faces), places=12
        )
        self.assertEqual(s.iso, self.iso)


class SharedEdgeTest(unittest.TestCase):
    """共享面两侧立方体产生的交点必须按全局边键合并。"""

    @classmethod
    def setUpClass(cls):
        cls.field = make_field(3, 3, 3, lambda i, j, k: i + j + k)
        cls.iso = 2.5
        cls.mesh = extract_isosurface(cls.field, cls.iso)

    def test_vertex_count_matches_independent_edge_enumeration(self):
        expected_keys = ref_crossing_edge_keys(3, 3, 3, self.field.values, self.iso)
        self.assertGreater(len(expected_keys), 0)
        self.assertEqual(self.mesh.stats.vertex_count, len(expected_keys))

    def test_every_vertex_edge_key_unique_and_canonical(self):
        keys = [v.edge_key for v in self.mesh.vertices]
        self.assertEqual(len(keys), len(set(keys)))
        for a, b in keys:
            self.assertLess(a, b)

    def test_vertex_ids_match_edge_key_set(self):
        expected_keys = ref_crossing_edge_keys(3, 3, 3, self.field.values, self.iso)
        self.assertEqual({v.edge_key for v in self.mesh.vertices}, expected_keys)

    def test_mesh_edges_used_by_one_or_two_faces(self):
        for key, count in self.mesh.edge_use_count.items():
            self.assertIn(count, (1, 2), f"edge {key} used {count} times")

    def test_deterministic_repeat(self):
        again = extract_isosurface(self.field, self.iso)
        self.assertEqual(self.mesh.vertices, again.vertices)
        self.assertEqual(self.mesh.faces, again.faces)


class IsoOnVertexTest(unittest.TestCase):
    """iso 恰好等于部分节点值：符号扰动只影响拓扑，几何位置不变。"""

    @classmethod
    def setUpClass(cls):
        # f = i，iso = 1：等值面是平面 x = 1，整层节点值恰为 iso。
        cls.field = make_field(5, 5, 5, lambda i, j, k: i)
        cls.iso = 1
        cls.mesh = extract_isosurface(cls.field, cls.iso)

    def test_total_area_is_exact_unit_square_times_16(self):
        # 平面 x=1 在 [0,4]^3 内的截面是 4x4 的正方形，面积 16。
        self.assertLess(abs(self.mesh.stats.total_area - 16.0), 1e-9 * 16.0)

    def test_all_vertices_on_plane(self):
        for v in self.mesh.vertices:
            self.assertAlmostEqual(v.point[0], 1.0, places=12)

    def test_no_nonmanifold_edges(self):
        # 扰动后的组合曲面每条边至多被 2 个三角形使用；丢弃零面积面只会减少计数。
        self.assertEqual(self.mesh.stats.nonmanifold_edge_count, 0)
        for count in self.mesh.edge_use_count.values():
            self.assertIn(count, (1, 2))

    def test_degenerate_faces_recorded_consistently(self):
        # 该场中 iso 恰好等于部分节点值，必然产生并记录零面积面。
        self.assertEqual(
            self.mesh.stats.degenerate_face_count, len(self.mesh.degenerate_faces)
        )
        self.assertGreater(self.mesh.stats.degenerate_face_count, 0)
        for record in self.mesh.degenerate_faces:
            self.assertEqual(len(record.edge_keys), 3)
            i, j, k = record.cell
            self.assertTrue(0 <= i < 4 and 0 <= j < 4 and 0 <= k < 4)

    def test_deterministic_repeat(self):
        again = extract_isosurface(self.field, self.iso)
        self.assertEqual(self.mesh.vertices, again.vertices)
        self.assertEqual(self.mesh.faces, again.faces)
        self.assertEqual(self.mesh.degenerate_faces, again.degenerate_faces)


class ConstantFieldTest(unittest.TestCase):
    def test_strictly_below_iso_gives_empty_mesh(self):
        field = make_field(5, 5, 5, lambda i, j, k: 0.0)
        mesh = extract_isosurface(field, 1.0)
        self.assertEqual(mesh.vertices, ())
        self.assertEqual(mesh.faces, ())
        self.assertEqual(mesh.stats.vertex_count, 0)
        self.assertEqual(mesh.stats.face_count, 0)
        self.assertEqual(mesh.stats.total_area, 0.0)
        self.assertEqual(mesh.stats.boundary_edge_count, 0)
        self.assertEqual(mesh.stats.nonmanifold_edge_count, 0)
        self.assertEqual(mesh.stats.degenerate_face_count, 0)

    def test_strictly_above_iso_gives_empty_mesh(self):
        field = make_field(5, 5, 5, lambda i, j, k: 0.0)
        mesh = extract_isosurface(field, -1.0)
        self.assertEqual(mesh.vertices, ())
        self.assertEqual(mesh.faces, ())

    def test_iso_equal_to_constant_field_is_combinatorially_valid(self):
        # 所有节点值恰等于 iso：完全由符号扰动决定拓扑。
        # 不规定具体形状，只验证确定性、组合合法性与统计一致性。
        field = make_field(5, 5, 5, lambda i, j, k: 0.0)
        mesh = extract_isosurface(field, 0.0)
        for key, count in mesh.edge_use_count.items():
            self.assertIn(count, (1, 2), f"edge {key} used {count} times")
        self.assertEqual(mesh.stats.nonmanifold_edge_count, 0)
        self.assertEqual(mesh.stats.vertex_count, len(mesh.vertices))
        self.assertEqual(mesh.stats.face_count, len(mesh.faces))
        self.assertEqual(
            mesh.stats.degenerate_face_count, len(mesh.degenerate_faces)
        )
        again = extract_isosurface(field, 0.0)
        self.assertEqual(mesh.vertices, again.vertices)
        self.assertEqual(mesh.faces, again.faces)


class SphereFieldTest(unittest.TestCase):
    """球场闭合性：16^3 网格，半径 4，无节点恰在等值面上。"""

    @classmethod
    def setUpClass(cls):
        cls.radius = 4.0
        cls.center = 7.5
        cls.field = make_field(
            16,
            16,
            16,
            lambda i, j, k: (i - 7.5) ** 2 + (j - 7.5) ** 2 + (k - 7.5) ** 2,
        )
        cls.mesh = extract_isosurface(cls.field, cls.radius ** 2)

    def test_closed_manifold(self):
        self.assertEqual(self.mesh.stats.boundary_edge_count, 0)
        self.assertEqual(self.mesh.stats.nonmanifold_edge_count, 0)
        self.assertEqual(self.mesh.degenerate_faces, ())

    def test_normals_point_outward(self):
        c = self.center
        for face in self.mesh.faces:
            n = face_normal(self.mesh, face)
            g = face_centroid(self.mesh, face)
            outward = tuple(g[d] - c for d in range(3))
            dot = sum(n[d] * outward[d] for d in range(3))
            self.assertGreater(dot, 0.0)

    def test_area_sanity_against_analytic_sphere(self):
        # 分段线性逼近只能不超过真实球面积太多；这里只做宽松合理性检查，
        # 精确口径以平面解析测试为准（见 README 误差口径）。
        analytic = 4.0 * math.pi * self.radius ** 2
        ratio = self.mesh.stats.total_area / analytic
        self.assertGreater(ratio, 0.75)
        self.assertLess(ratio, 1.0 + 1e-9)

    def test_vertices_near_sphere_radius(self):
        for v in self.mesh.vertices:
            r = math.dist(v.point, (self.center,) * 3)
            self.assertGreater(r, self.radius - 2.0)
            self.assertLess(r, self.radius + 2.0)


if __name__ == "__main__":
    unittest.main()
