"""增量更新（IsosurfaceExtractor）测试。

核心不变量：局部更新后的几何必须与对修改后的场全量重提取一致；
未受影响顶点/面的身份（id）保留；更新失败时旧结果仍可使用。
"""

import unittest

from iso3d import (
    InvalidInputError,
    IsosurfaceExtractor,
    ScalarField3D,
    extract_isosurface,
)


def sphere_field(n=12, radius=4.0):
    center = (n - 1) / 2.0
    values = [
        (i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2
        for k in range(n)
        for j in range(n)
        for i in range(n)
    ]
    return values, radius ** 2


def make_extractor(n=12, radius=4.0):
    values, iso = sphere_field(n, radius)
    field = ScalarField3D(ni=n, nj=n, nk=n, values=tuple(values))
    return IsosurfaceExtractor(field, iso), values, iso


def geometry_signature(mesh):
    """与 id 无关的几何签名：顶点坐标集合 + 三角形（坐标三元组）集合。"""
    points = {v.id: v.point for v in mesh.vertices}
    tris = set()
    for face in mesh.faces:
        tris.add(tuple(sorted(points[v] for v in face.vertex_ids)))
    return set(points.values()), tris


def assert_same_geometry(case, mesh_a, mesh_b):
    verts_a, tris_a = geometry_signature(mesh_a)
    verts_b, tris_b = geometry_signature(mesh_b)
    case.assertEqual(verts_a, verts_b)
    case.assertEqual(tris_a, tris_b)
    case.assertEqual(mesh_a.stats.vertex_count, mesh_b.stats.vertex_count)
    case.assertEqual(mesh_a.stats.face_count, mesh_b.stats.face_count)
    case.assertEqual(
        mesh_a.stats.degenerate_face_count, mesh_b.stats.degenerate_face_count
    )
    case.assertEqual(mesh_a.stats.boundary_edge_count, mesh_b.stats.boundary_edge_count)
    case.assertEqual(
        mesh_a.stats.nonmanifold_edge_count, mesh_b.stats.nonmanifold_edge_count
    )
    case.assertAlmostEqual(mesh_a.stats.total_area, mesh_b.stats.total_area, places=12)


def full_reextract(values, n, iso):
    field = ScalarField3D(ni=n, nj=n, nk=n, values=tuple(values))
    return extract_isosurface(field, iso)


class InitialStateTest(unittest.TestCase):
    def test_initial_mesh_equals_full_extraction_exactly(self):
        n = 12
        ext, values, iso = make_extractor(n)
        full = full_reextract(values, n, iso)
        initial = ext.mesh()
        # 初始构建与全量提取完全一致（含 id 与顺序）。
        self.assertEqual(initial, full)


class UpdateGeometryTest(unittest.TestCase):
    def test_small_region_update_matches_full_reextraction(self):
        n = 12
        ext, values, iso = make_extractor(n)
        # 在 (6,6,6) 附近挖一个凹坑：降低 2x2x2 节点的值。
        changes = {}
        for dk in (0, 1):
            for dj in (0, 1):
                for di in (0, 1):
                    idx = ((6 + dk) * n + (6 + dj)) * n + (6 + di)
                    changes[idx] = values[idx] - 5.0
        mesh = ext.update(changes)
        for idx, value in changes.items():
            values[idx] = value
        assert_same_geometry(self, mesh, full_reextract(values, n, iso))

    def test_iso_crossing_adds_and_removes_triangles(self):
        n = 12
        ext, values, iso = make_extractor(n)
        before = ext.mesh().stats.face_count
        # 把一个内部节点从等值面下方抬到上方（跨越 iso）。
        idx = (6 * n + 6) * n + 6
        self.assertLess(values[idx], iso)
        mesh_up = ext.update({idx: iso + 3.0})
        self.assertNotEqual(mesh_up.stats.face_count, before)
        values[idx] = iso + 3.0
        assert_same_geometry(self, mesh_up, full_reextract(values, n, iso))
        # 再改回去：新增三角形消失，几何回到初始状态。
        mesh_back = ext.update({idx: values[idx] - 3.0 if False else 36.0})
        values[idx] = 36.0
        assert_same_geometry(self, mesh_back, full_reextract(values, n, iso))

    def test_value_landing_exactly_on_iso(self):
        n = 12
        ext, values, iso = make_extractor(n)
        idx = (6 * n + 6) * n + 6
        mesh = ext.update({idx: iso})  # 恰好落在等值面上：走符号扰动路径
        values[idx] = iso
        assert_same_geometry(self, mesh, full_reextract(values, n, iso))

    def test_sequential_updates_match_single_full_reextraction(self):
        n = 12
        ext, values, iso = make_extractor(n)
        # 确定性伪随机序列修改 30 个节点，分 5 批提交。
        picked = sorted({(37 * m * m + 11 * m + 5) % (n ** 3) for m in range(30)})
        for batch in range(5):
            changes = {}
            for idx in picked[batch * 6:(batch + 1) * 6]:
                delta = ((idx * 7919) % 17 - 8) * 1.25
                changes[idx] = values[idx] + delta
            mesh = ext.update(changes)
            for idx, value in changes.items():
                values[idx] = value
        assert_same_geometry(self, mesh, full_reextract(values, n, iso))

    def test_closed_surface_stays_closed_after_update(self):
        n = 12
        ext, values, iso = make_extractor(n)
        self.assertEqual(ext.mesh().stats.boundary_edge_count, 0)
        idx = (6 * n + 6) * n + 6
        mesh = ext.update({idx: iso + 2.0, idx + 1: iso - 2.0})
        # 修改区域边缘的共享边交点一致更新，不留裂缝。
        self.assertEqual(mesh.stats.boundary_edge_count, 0)
        self.assertEqual(mesh.stats.nonmanifold_edge_count, 0)

    def test_empty_update_is_noop(self):
        ext, values, iso = make_extractor()
        before = ext.mesh()
        after = ext.update({})
        self.assertEqual(before, after)


class IdentityPreservationTest(unittest.TestCase):
    def test_unaffected_vertices_and_faces_keep_ids(self):
        n = 12
        ext, values, iso = make_extractor(n)
        before = ext.mesh()
        # 修改角落 (0,0,0) 附近；远处 (10,10,10) 区域不受影响。
        idx = 0
        ext.update({idx: values[idx] + 1.5})
        after = ext.mesh()
        before_verts = {v.id: v for v in before.vertices}
        after_verts = {v.id: v for v in after.vertices}
        before_faces = {f.id: f for f in before.faces}
        after_faces = {f.id: f for f in after.faces}
        # 受影响单元仅涉及节点坐标 <=1 的角点；边键两端节点都远离该处的
        # 顶点必须原样保留（同 id、同坐标、同边键）。
        def far(vertex):
            return all(node >= 2 * (n * n) + 2 * n + 2 for node in vertex.edge_key)

        kept = [v for v in before_verts.values() if far(v)]
        self.assertTrue(kept)
        for v in kept:
            self.assertIn(v.id, after_verts)
            self.assertEqual(after_verts[v.id], v)
        # 由远端顶点组成的面也必须保留 id 与顶点引用。
        kept_faces = [
            f for f in before_faces.values() if all(far(before_verts[v]) for v in f.vertex_ids)
        ]
        self.assertTrue(kept_faces)
        for f in kept_faces:
            self.assertIn(f.id, after_faces)
            self.assertEqual(after_faces[f.id], f)

    def test_changed_edge_keeps_vertex_id_with_refreshed_position(self):
        n = 12
        ext, values, iso = make_extractor(n)
        before = ext.mesh()
        idx = (5 * n + 5) * n + 9  # dist^2 = 12.75 < iso，邻节点 (10,5,5) 在外侧
        target = None
        for v in before.vertices:
            if idx in v.edge_key:
                target = v
                break
        self.assertIsNotNone(target)
        after = ext.update({idx: values[idx] + 0.7})  # 仍在同一侧，边仍穿越
        after_verts = {v.id: v for v in after.vertices}
        self.assertIn(target.id, after_verts)
        moved = after_verts[target.id]
        self.assertEqual(moved.edge_key, target.edge_key)
        self.assertNotEqual(moved.point, target.point)  # 坐标原地刷新，id 不变


class FailureAtomicityTest(unittest.TestCase):
    def test_invalid_update_leaves_state_usable(self):
        n = 12
        ext, values, iso = make_extractor(n)
        before = ext.mesh()
        bad_changes = [
            {0: math_nan()},
            {0: float("inf")},
            {n ** 3: 1.0},   # 索引越界
            {-1: 1.0},
            {0: True},       # bool 被拒绝
            {True: 1.0},
            [(0, 1.0), (0, 2.0)],  # 重复索引
            "not-a-mapping",
        ]
        for bad in bad_changes:
            with self.assertRaises(InvalidInputError, msg=repr(bad)):
                ext.update(bad)
        # 旧结果完全不变，提取器仍可正常更新。
        self.assertEqual(ext.mesh(), before)
        idx = (6 * n + 6) * n + 6
        mesh = ext.update({idx: iso + 1.0})
        values[idx] = iso + 1.0
        assert_same_geometry(self, mesh, full_reextract(values, n, iso))

    def test_constructor_rejects_invalid_input(self):
        with self.assertRaises(InvalidInputError):
            IsosurfaceExtractor("not a field", 1.0)
        values, iso = sphere_field(4, 2.0)
        field = ScalarField3D(ni=4, nj=4, nk=4, values=tuple(values))
        with self.assertRaises(InvalidInputError):
            IsosurfaceExtractor(field, float("nan"))
        with self.assertRaises(InvalidInputError):
            IsosurfaceExtractor(field, True)


def math_nan():
    return float("nan")


if __name__ == "__main__":
    unittest.main()
