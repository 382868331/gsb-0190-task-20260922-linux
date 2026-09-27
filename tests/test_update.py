"""增量更新（update_isosurface）测试。

几何一致性期望来自对同一新场的完整重提取（extract_isosurface），
并辅以 tests/_reference.py 的独立穿越边枚举做顶点级交叉验证。
"""

import unittest

from iso3d import (
    InvalidInputError,
    ScalarField3D,
    extract_isosurface,
    update_isosurface,
)

from _reference import ref_crossing_edge_keys


def make_field(n, func):
    values = tuple(
        func(i, j, k)
        for k in range(n)
        for j in range(n)
        for i in range(n)
    )
    return ScalarField3D(ni=n, nj=n, nk=n, values=values)


def base_func(i, j, k):
    # 确定性伪随机有符号场（无第三方依赖、无随机状态）。
    return (
        ((i * 73856093) ^ (j * 19349663) ^ (k * 83492791)) % 2001 - 1000
    ) / 250.0 + 0.13 * i - 0.07 * k


def geometry_of(mesh):
    """与 ID 无关的几何指纹：顶点坐标、三角形坐标多重集、退化面记录。"""
    by_id = {v.id: v for v in mesh.vertices}
    pts = sorted(v.point for v in mesh.vertices)
    tris = sorted(
        tuple(sorted(by_id[vid].point for vid in f.vertex_ids)) for f in mesh.faces
    )
    degenerates = sorted(
        (d.cell, d.tet_index, tuple(sorted(d.edge_keys)))
        for d in mesh.degenerate_faces
    )
    return pts, tris, degenerates




def edge_pairs_geometry(mesh, edges):
    """把顶点 ID 边集合映射为坐标边集合（与 ID 无关）。"""
    by_id = {v.id: v for v in mesh.vertices}
    return sorted(
        tuple(sorted((by_id[a].point, by_id[b].point))) for a, b in edges
    )


def edge_use_geometry(mesh):
    by_id = {v.id: v for v in mesh.vertices}
    return {
        tuple(sorted((by_id[a].point, by_id[b].point))): count
        for (a, b), count in mesh.edge_use_count.items()
    }


class UpdateConsistencyTest(unittest.TestCase):
    """更新结果必须与完整重提取几何一致。"""

    def check_update(self, n, iso, changes, changed_nodes="auto"):
        old_field = make_field(n, base_func)
        new_values = list(old_field.values)
        for idx, value in changes.items():
            new_values[idx] = value
        new_field = ScalarField3D(ni=n, nj=n, nk=n, values=tuple(new_values))

        old_mesh = extract_isosurface(old_field, iso)
        kwargs = {} if changed_nodes == "auto" else {"changed_nodes": changed_nodes}
        updated = update_isosurface(old_field, old_mesh, new_field, **kwargs)
        reference = extract_isosurface(new_field, iso)

        self.assertEqual(geometry_of(updated), geometry_of(reference))
        self.assertEqual(updated.stats, reference.stats)
        # 边级输出按顶点 ID 键控；ID 在增量更新后与全量重提取不同，
        # 因此按几何坐标比较（几何一致性以坐标为准）。
        self.assertEqual(
            edge_pairs_geometry(updated, updated.boundary_edges),
            edge_pairs_geometry(reference, reference.boundary_edges),
        )
        self.assertEqual(
            edge_pairs_geometry(updated, updated.nonmanifold_edges),
            edge_pairs_geometry(reference, reference.nonmanifold_edges),
        )
        self.assertEqual(edge_use_geometry(updated), edge_use_geometry(reference))
        # 顶点集合与独立参考枚举的穿越边键一致（补充交叉验证）。
        self.assertEqual(
            {v.edge_key for v in updated.vertices},
            ref_crossing_edge_keys(new_field.ni, new_field.nj, new_field.nk, new_field.values, iso),
        )
        return old_mesh, updated

    def test_single_node_change(self):
        n, iso = 8, 0.5
        idx = (3 * n + 3) * n + 3
        self.check_update(n, iso, {idx: 9.0})

    def test_small_patch_change(self):
        n, iso = 10, 0.25
        changes = {}
        for k in (4, 5):
            for j in (4, 5):
                for i in (4, 5):
                    changes[(k * n + j) * n + i] = -7.0 + i + 2 * j + 3 * k
        self.check_update(n, iso, changes)

    def test_value_crossing_iso_exactly(self):
        # 修改后值恰等于 iso：拓扑由符号扰动判定，新增/消失三角形都要正确处理。
        n, iso = 8, 1.0
        changes = {}
        for k in (2, 3):
            for j in (2, 3):
                for i in (2, 3):
                    changes[(k * n + j) * n + i] = iso
        self.check_update(n, iso, changes)

    def test_surface_vanishes_locally(self):
        # 把一片区域整体推到 iso 同侧，局部三角形应消失。
        n, iso = 8, 0.5
        changes = {}
        for k in range(2, 6):
            for j in range(2, 6):
                for i in range(2, 6):
                    changes[(k * n + j) * n + i] = -100.0
        self.check_update(n, iso, changes)

    def test_explicit_changed_nodes_superset(self):
        # 显式给出的变化集合可以是实际变化集合的超集（多算但结果一致）。
        n, iso = 8, 0.5
        idx = (4 * n + 4) * n + 4
        self.check_update(n, iso, {idx: 5.0}, changed_nodes=[idx, 0, 1, 2])

    def test_no_change_returns_equal_geometry(self):
        n, iso = 6, 0.5
        old_mesh, updated = self.check_update(n, iso, {})
        self.assertEqual(geometry_of(old_mesh), geometry_of(updated))

    def test_iso_change_falls_back_to_full_extraction(self):
        n = 8
        old_field = make_field(n, base_func)
        new_field = make_field(n, lambda i, j, k: base_func(i, j, k) * 1.5)
        old_mesh = extract_isosurface(old_field, 0.5)
        updated = update_isosurface(old_field, old_mesh, new_field, iso=0.9)
        reference = extract_isosurface(new_field, 0.9)
        self.assertEqual(geometry_of(updated), geometry_of(reference))
        self.assertEqual(updated.stats, reference.stats)


class IdentityPreservationTest(unittest.TestCase):
    """未受影响部分的顶点/三角形身份（ID 与对象）应尽量保留。"""

    def setUp(self):
        self.n = 10
        self.iso = 0.5
        self.old_field = make_field(self.n, base_func)
        # 只改角落 (8,8,8) 附近一个节点。
        idx = (8 * self.n + 8) * self.n + 8
        new_values = list(self.old_field.values)
        new_values[idx] = 50.0
        self.changed_idx = idx
        self.new_field = ScalarField3D(
            ni=self.n, nj=self.n, nk=self.n, values=tuple(new_values)
        )
        self.old_mesh = extract_isosurface(self.old_field, self.iso)
        self.updated = update_isosurface(
            self.old_field, self.old_mesh, self.new_field,
            changed_nodes=[self.changed_idx],
        )

    def test_far_vertices_keep_identity(self):
        old_by_edge = {v.edge_key: v for v in self.old_mesh.vertices}
        new_by_edge = {v.edge_key: v for v in self.updated.vertices}
        # 远离修改区域（两端节点坐标均 <= 6）的交点必须保留原对象。
        def far(key):
            for node in key:
                i = node % self.n
                j = (node // self.n) % self.n
                k = node // (self.n * self.n)
                if max(i, j, k) > 6:
                    return False
            return True

        far_keys = [key for key in old_by_edge if far(key)]
        self.assertTrue(far_keys)
        kept = 0
        for key in far_keys:
            if key in new_by_edge:
                self.assertIs(old_by_edge[key], new_by_edge[key])
                kept += 1
        self.assertTrue(kept)

    def test_unaffected_faces_keep_identity(self):
        # 受影响立方体只覆盖 (7..8)^3；其余旧面必须原样保留（同一对象）。
        from iso3d.core import _run_cubes

        affected = sorted(
            (ci, cj, ck)
            for ck in (7, 8)
            for cj in (7, 8)
            for ci in (7, 8)
            if ci <= self.n - 2 and cj <= self.n - 2 and ck <= self.n - 2
        )
        _, old_tris, _ = _run_cubes(
            affected, self.old_field.values, self.n, self.n, self.iso
        )
        by_id = {v.id: v for v in self.old_mesh.vertices}
        triple_to_faces = {}
        for f in self.old_mesh.faces:
            triple = tuple(by_id[vid].edge_key for vid in f.vertex_ids)
            triple_to_faces.setdefault(triple, []).append(f)
        removed_ids = set()
        for triple in old_tris:
            removed_ids.add(triple_to_faces[triple].pop().id)

        new_face_by_id = {f.id: f for f in self.updated.faces}
        kept = [f for f in self.old_mesh.faces if f.id not in removed_ids]
        self.assertTrue(kept)
        for f in kept:
            self.assertIs(new_face_by_id[f.id], f)

    def test_shared_edge_vertices_match_full_extraction(self):
        # 修改区域边缘的共享交点：与完整重提取的边键集合与坐标一致。
        ref = extract_isosurface(self.new_field, self.iso)
        ref_by_edge = {v.edge_key: v for v in ref.vertices}
        self.assertEqual(
            {v.edge_key for v in self.updated.vertices}, set(ref_by_edge)
        )
        for v in self.updated.vertices:
            rv = ref_by_edge[v.edge_key]
            self.assertEqual(v.point, rv.point)
            self.assertEqual(v.t, rv.t)

    def test_no_cracks_on_update_boundary(self):
        # 组合一致性：边使用计数与完整重提取完全相同（边缘无裂缝）。
        ref = extract_isosurface(self.new_field, self.iso)
        self.assertEqual(edge_use_geometry(self.updated), edge_use_geometry(ref))
        self.assertEqual(
            self.updated.stats.nonmanifold_edge_count,
            ref.stats.nonmanifold_edge_count,
        )


class UpdateValidationTest(unittest.TestCase):
    """非法输入必须在计算前拒绝，且旧结果保持可用。"""

    def setUp(self):
        self.n = 6
        self.iso = 0.5
        self.old_field = make_field(self.n, base_func)
        new_values = list(self.old_field.values)
        new_values[7] = 9.0
        self.new_field = ScalarField3D(
            ni=self.n, nj=self.n, nk=self.n, values=tuple(new_values)
        )
        self.old_mesh = extract_isosurface(self.old_field, self.iso)

    def assert_rejected(self, param, **kwargs):
        with self.assertRaises(InvalidInputError) as ctx:
            update_isosurface(
                self.old_field, self.old_mesh, self.new_field, **kwargs
            )
        self.assertIn(param, str(ctx.exception))

    def test_changed_node_out_of_range(self):
        self.assert_rejected("changed_nodes[0]", changed_nodes=[self.n ** 3])

    def test_changed_node_negative(self):
        self.assert_rejected("changed_nodes[0]", changed_nodes=[-1])

    def test_changed_node_bool(self):
        self.assert_rejected("changed_nodes[0]", changed_nodes=[True])

    def test_changed_node_float(self):
        self.assert_rejected("changed_nodes[0]", changed_nodes=[1.5])

    def test_changed_nodes_not_sequence(self):
        self.assert_rejected("changed_nodes", changed_nodes=42)

    def test_dimension_mismatch(self):
        other = make_field(self.n + 1, base_func)
        with self.assertRaises(InvalidInputError) as ctx:
            update_isosurface(self.old_field, self.old_mesh, other)
        self.assertIn("new_field", str(ctx.exception))

    def test_iso_nan(self):
        self.assert_rejected("iso", iso=float("nan"))

    def test_iso_bool(self):
        self.assert_rejected("iso", iso=True)

    def test_wrong_types(self):
        with self.assertRaises(InvalidInputError):
            update_isosurface("x", self.old_mesh, self.new_field)
        with self.assertRaises(InvalidInputError):
            update_isosurface(self.old_field, "x", self.new_field)
        with self.assertRaises(InvalidInputError):
            update_isosurface(self.old_field, self.old_mesh, "x")

    def test_old_mesh_usable_after_failure(self):
        before = geometry_of(self.old_mesh)
        for bad in ([-1], [True], [10 ** 6]):
            with self.assertRaises(InvalidInputError):
                update_isosurface(
                    self.old_field, self.old_mesh, self.new_field,
                    changed_nodes=bad,
                )
        # 旧网格对象未被破坏，仍可用于后续成功更新。
        self.assertEqual(geometry_of(self.old_mesh), before)
        updated = update_isosurface(
            self.old_field, self.old_mesh, self.new_field, changed_nodes=[7]
        )
        reference = extract_isosurface(self.new_field, self.iso)
        self.assertEqual(geometry_of(updated), geometry_of(reference))

    def test_chained_updates(self):
        # 连续多次增量更新：第二次以第一次的输出为旧结果。
        idx_a = (2 * self.n + 2) * self.n + 2
        idx_b = (3 * self.n + 3) * self.n + 3
        values_a = list(self.old_field.values)
        values_a[idx_a] = 8.0
        field_a = ScalarField3D(
            ni=self.n, nj=self.n, nk=self.n, values=tuple(values_a)
        )
        mesh_a = update_isosurface(
            self.old_field, self.old_mesh, field_a, changed_nodes=[idx_a]
        )
        values_b = list(values_a)
        values_b[idx_b] = -8.0
        field_b = ScalarField3D(
            ni=self.n, nj=self.n, nk=self.n, values=tuple(values_b)
        )
        mesh_b = update_isosurface(
            field_a, mesh_a, field_b, changed_nodes=[idx_b]
        )
        reference = extract_isosurface(field_b, self.iso)
        self.assertEqual(geometry_of(mesh_b), geometry_of(reference))
        self.assertEqual(mesh_b.stats, reference.stats)


if __name__ == "__main__":
    unittest.main()
