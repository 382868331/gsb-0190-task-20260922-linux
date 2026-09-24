"""同分辨率相邻体块分块提取与拼接的测试。

核对方式（不使用被测实现生成期望值）：
- 手算小例：3x2x2 网格沿 i 分两块，共享面上的交点坐标/边键/t 可手算；
- 解析关系：线性场 f=i+j+k 的截面总面积与独立解析截面面积比较；
- 独立枚举：顶点边键集合与 tests/_reference.py 的独立穿越边枚举比较；
- 关系性质：多种切分（面缝、边缝、角缝）的拼接结果与整块提取逐字段
  相等，且与输入块顺序无关——这是本题“拼接后的面与整块提取一致”的
  直接验收。
"""

import itertools
import math
import unittest

from iso3d import (
    InvalidInputError,
    ScalarBlock3D,
    ScalarField3D,
    extract_isosurface,
    extract_isosurface_blocks,
)

from _reference import plane_box_intersection_area, ref_crossing_edge_keys


# ---------------------------------------------------------------------------
# 确定性数据构造辅助
# ---------------------------------------------------------------------------

def flat_values(ni, nj, nk, func):
    return tuple(
        float(func(i, j, k))
        for k in range(nk)
        for j in range(nj)
        for i in range(ni)
    )


def block_values(origin, ni, nj, nk, func):
    oi, oj, ok = origin
    return tuple(
        float(func(oi + li, oj + lj, ok + lk))
        for lk in range(nk)
        for lj in range(nj)
        for li in range(ni)
    )


def _starts(counts):
    """每轴节点数分段 -> (起点, 节点数) 列表，相邻段共享一层节点。"""
    s = 0
    out = []
    for c in counts:
        out.append((s, c))
        s += c - 1
    return out


def make_blocks(shape, cuts, func):
    """按 cuts（每轴的节点数分段）把规则网格切成体块。"""
    gni, gnj, gnk = shape
    spans = [_starts(cuts[a]) for a in range(3)]
    blocks = []
    for (si, ni), (sj, nj), (sk, nk) in itertools.product(*spans):
        blocks.append(
            ScalarBlock3D(
                origin=(si, sj, sk),
                ni=ni,
                nj=nj,
                nk=nk,
                values=block_values((si, sj, sk), ni, nj, nk, func),
            )
        )
    return blocks


def mesh_equal(a, b):
    return (
        a.vertices == b.vertices
        and a.faces == b.faces
        and a.stats == b.stats
        and a.degenerate_faces == b.degenerate_faces
        and a.boundary_edges == b.boundary_edges
        and a.nonmanifold_edges == b.nonmanifold_edges
        and a.edge_use_count == b.edge_use_count
    )


# ---------------------------------------------------------------------------
# 手算小例：3x2x2 网格，沿 i 分成两个单立方体块，共享面 i=1
# ---------------------------------------------------------------------------

class HandComputedFaceSeamTest(unittest.TestCase):
    """f = i + j + k，iso = 1.5；共享面 i=1 上存在双方都计算的交点。"""

    @classmethod
    def setUpClass(cls):
        cls.shape = (3, 2, 2)
        cls.func = staticmethod(lambda i, j, k: i + j + k)
        cls.iso = 1.5
        cls.blocks = make_blocks(cls.shape, ((2, 2), (2,), (2,)), cls.func)
        cls.mesh = extract_isosurface_blocks(cls.blocks, cls.iso, shape=cls.shape)

    def test_two_blocks_share_plane(self):
        self.assertEqual(len(self.blocks), 2)
        self.assertEqual(self.blocks[0].end, (1, 1, 1))
        self.assertEqual(self.blocks[1].origin, (1, 0, 0))

    def test_hand_computed_shared_vertex_on_seam_j_edge(self):
        # 全局节点 ID（ni=3）：(1,0,0)->1，(1,1,0)->4。
        # 两端值 1 与 2，iso=1.5：交点恰为 (1, 0.5, 0)，t=0.5。
        key = (1, 4)
        verts = [v for v in self.mesh.vertices if v.edge_key == key]
        self.assertEqual(len(verts), 1)  # 接缝双方只产生一个顶点
        v = verts[0]
        self.assertEqual(v.point, (1.0, 0.5, 0.0))
        self.assertEqual(v.t, 0.5)
        self.assertEqual(v.id, self.mesh.vertices.index(v))

    def test_hand_computed_shared_vertex_on_seam_k_edge(self):
        # (1,0,0)->1，(1,0,1)->7：值 1 与 2，交点 (1, 0, 0.5)。
        key = (1, 7)
        verts = [v for v in self.mesh.vertices if v.edge_key == key]
        self.assertEqual(len(verts), 1)
        self.assertEqual(verts[0].point, (1.0, 0.0, 0.5))

    def test_hand_computed_shared_vertex_on_seam_face_diagonal(self):
        # 共享面上的面对角线 (1,0,0)->1 与 (1,1,1)->10：值 1 与 3，
        # 交点 (1, 0.25, 0.25)，t=0.25；该对角线只存在于四面体网格中，
        # 两个立方体共享它，因此同样必须按边键唯一合并。
        key = (1, 10)
        verts = [v for v in self.mesh.vertices if v.edge_key == key]
        self.assertEqual(len(verts), 1)
        v = verts[0]
        self.assertEqual(v.point, (1.0, 0.25, 0.25))
        self.assertAlmostEqual(v.t, 0.25, places=15)

    def test_shared_vertex_used_by_faces_on_both_sides(self):
        # 对角线交点 (1,.25,.25) 在 i=0 单元中是体对角线交点、在 i=1
        # 单元中是共享面的面对角线交点。引用它的三角形必须横跨两侧：
        # 同一张面的另外两个顶点中，必有 x<1（来自 i=0 单元）和 x>1
        # （来自 i=1 单元）的点，否则接缝在组合上就是裂开的。
        vid = next(v.id for v in self.mesh.vertices if v.edge_key == (1, 10))
        other_x = set()
        for face in self.mesh.faces:
            if vid in face.vertex_ids:
                for other in face.vertex_ids:
                    if other != vid:
                        other_x.add(round(self.mesh.vertices[other].point[0], 10))
        self.assertTrue(any(x < 1.0 for x in other_x), f"no face extends to x<1: {other_x}")
        self.assertTrue(any(x > 1.0 for x in other_x), f"no face extends to x>1: {other_x}")

    def test_area_matches_analytic_plane_section(self):
        # 平面 x+y+z=1.5 截盒子 [0,2]x[0,1]x[0,1] 的独立解析面积。
        expected = plane_box_intersection_area(1.0, 1.0, 1.0, 1.5, 2.0, 1.0, 1.0)
        self.assertGreater(expected, 0.0)
        self.assertLess(
            abs(self.mesh.stats.total_area - expected), 1e-9 * expected
        )

    def test_vertex_edge_keys_match_independent_enumeration(self):
        full = flat_values(3, 2, 2, self.func)
        expected = ref_crossing_edge_keys(3, 2, 2, full, self.iso)
        self.assertEqual({v.edge_key for v in self.mesh.vertices}, expected)
        self.assertEqual(self.mesh.stats.vertex_count, len(expected))

    def test_normals_point_toward_increasing_scalar(self):
        for face in self.mesh.faces:
            p = [self.mesh.vertices[v].point for v in face.vertex_ids]
            u = tuple(p[1][d] - p[0][d] for d in range(3))
            w = tuple(p[2][d] - p[0][d] for d in range(3))
            n = (
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            )
            self.assertGreater(n[0] + n[1] + n[2], 0.0)

    def test_identical_to_monolithic_extraction(self):
        whole = extract_isosurface(
            ScalarField3D(3, 2, 2, flat_values(3, 2, 2, self.func)), self.iso
        )
        self.assertTrue(mesh_equal(self.mesh, whole))

    def test_result_independent_of_block_order(self):
        rev = extract_isosurface_blocks(list(reversed(self.blocks)), self.iso,
                                        shape=self.shape)
        self.assertTrue(mesh_equal(rev, self.mesh))


# ---------------------------------------------------------------------------
# 关系性质：面缝 / 边缝 / 角缝，多场多切分，全部与整块提取一致
# ---------------------------------------------------------------------------

SPHERE = lambda i, j, k: (i - 3.5) ** 2 + (j - 3.5) ** 2 + (k - 3.5) ** 2
PLANE = lambda i, j, k: i + 2.0 * j - k
PERTURBED = lambda i, j, k: float(i)  # iso 落整层节点，退化面众多


class SeamEquivalenceTest(unittest.TestCase):
    def _check(self, n, func, iso, cuts, seed=0):
        shape = (n, n, n)
        whole = extract_isosurface(ScalarField3D(n, n, n, flat_values(n, n, n, func)), iso)
        blocks = make_blocks(shape, cuts, func)
        merged = extract_isosurface_blocks(blocks, iso, shape=shape)
        self.assertTrue(mesh_equal(merged, whole),
                        f"cuts={cuts}: merged mesh differs from monolithic")
        # 输入块顺序打乱后必须仍然逐字段相同（顶点 ID 只取决于全局边键）。
        ordered = list(blocks)
        if seed:
            import random
            random.Random(seed).shuffle(ordered)
        else:
            ordered.reverse()
        merged2 = extract_isosurface_blocks(ordered, iso, shape=shape)
        self.assertTrue(mesh_equal(merged2, whole),
                        f"cuts={cuts}: result depends on block input order")

    def test_single_block_is_trivial_tiling(self):
        blocks = make_blocks((8, 8, 8), ((8,), (8,), (8,)), SPHERE)
        m = extract_isosurface_blocks(blocks, 9.0)  # 省略 shape：用包围盒推断
        whole = extract_isosurface(ScalarField3D(8, 8, 8, flat_values(8, 8, 8, SPHERE)), 9.0)
        self.assertTrue(mesh_equal(m, whole))
        self.assertEqual(m.stats.node_count, 512)

    def test_face_seam_two_blocks(self):
        self._check(8, SPHERE, 9.0, ((4, 5), (8,), (8,)))

    def test_face_seam_three_slabs_each_axis(self):
        self._check(8, SPHERE, 9.0, ((3, 3, 4), (3, 3, 4), (8,)), seed=7)

    def test_edge_seam_four_blocks(self):
        # 两条切缝交于一条全局网格线。
        self._check(8, PLANE, 3.25, ((4, 5), (4, 5), (8,)), seed=11)

    def test_corner_seam_eight_blocks(self):
        # 三条切缝交于一个全局节点：8 块共享角点。
        self._check(8, SPHERE, 9.0, ((4, 5), (4, 5), (4, 5)), seed=13)

    def test_corner_seam_with_iso_on_shared_node(self):
        # 8 块共享节点 (4,4,4)（n=9 时），iso 穿过该节点层，扰动按全局 ID。
        n = 9
        func = lambda i, j, k: i + j + k
        shape = (n, n, n)
        whole = extract_isosurface(ScalarField3D(n, n, n, flat_values(n, n, n, func)), 12)
        blocks = make_blocks(shape, ((5, 5), (5, 5), (5, 5)), func)
        merged = extract_isosurface_blocks(blocks, 12, shape=shape)
        self.assertTrue(mesh_equal(merged, whole))

    def test_iso_on_node_layer_through_seams(self):
        # 退化面众多的场，接缝处退化记录也必须与整块一致。
        self._check(7, PERTURBED, 3, ((4, 4), (4, 4), (4, 4)), seed=17)

    def test_asymmetric_shape_and_splits(self):
        # 非立方体网格：6x4x5，三轴切分点不对称。
        ni, nj, nk = 6, 4, 5
        func = lambda i, j, k: math.sin(0.7 * i) + 0.5 * j - 0.3 * k * k
        iso = 0.4
        shape = (ni, nj, nk)
        whole = extract_isosurface(
            ScalarField3D(ni, nj, nk, flat_values(ni, nj, nk, func)), iso
        )
        blocks = make_blocks(shape, ((2, 3, 3), (2, 3), (3, 3)), func)
        merged = extract_isosurface_blocks(blocks, iso, shape=shape)
        self.assertTrue(mesh_equal(merged, whole))
        self.assertEqual(merged.stats.cell_count, (ni - 1) * (nj - 1) * (nk - 1))
        self.assertEqual(merged.stats.node_count, ni * nj * nk)

    def test_empty_field_split_is_empty(self):
        n = 4
        func = lambda i, j, k: 0.0
        blocks = make_blocks((n, n, n), ((2, 3), (2, 3), (n,)), func)
        m = extract_isosurface_blocks(blocks, 1.0, shape=(n, n, n))
        self.assertEqual(m.vertices, ())
        self.assertEqual(m.faces, ())
        self.assertEqual(m.stats.total_area, 0.0)
        self.assertEqual(m.stats.boundary_edge_count, 0)

    def test_zero_cell_slab_sharing_end_layer(self):
        # ni=1 的块不含立方体、只含一层共享节点：合法，结果不变。
        n = 4
        func = SPHERE
        main = ScalarBlock3D(
            origin=(0, 0, 0), ni=n, nj=n, nk=n,
            values=flat_values(n, n, n, func),
        )
        slab_values = block_values((n - 1, 0, 0), 1, n, n, func)
        slab = ScalarBlock3D(origin=(n - 1, 0, 0), ni=1, nj=n, nk=n,
                             values=slab_values)
        whole = extract_isosurface(ScalarField3D(n, n, n, flat_values(n, n, n, func)), 9.0)
        merged = extract_isosurface_blocks([main, slab], 9.0, shape=(n, n, n))
        self.assertTrue(mesh_equal(merged, whole))


# ---------------------------------------------------------------------------
# 接缝几何核对：接缝边两侧面数与整条网格的流形性
# ---------------------------------------------------------------------------

class SeamTopologyTest(unittest.TestCase):
    def test_edges_across_face_seam_are_manifold(self):
        n = 8
        blocks = make_blocks((n, n, n), ((4, 5), (n,), (n,)), SPHERE)
        m = extract_isosurface_blocks(blocks, 9.0, shape=(n, n, n))
        self.assertEqual(m.stats.nonmanifold_edge_count, 0)
        for key, count in m.edge_use_count.items():
            self.assertIn(count, (1, 2))

    def test_corner_seam_closed_sphere_has_no_boundary(self):
        n = 8
        blocks = make_blocks((n, n, n), ((4, 5), (4, 5), (4, 5)), SPHERE)
        m = extract_isosurface_blocks(blocks, 9.0, shape=(n, n, n))
        self.assertEqual(m.stats.boundary_edge_count, 0)
        self.assertEqual(m.stats.nonmanifold_edge_count, 0)
        # 顶点 id 即其边键在全部穿越边键排序中的全局排名（切分无关身份）。
        for idx, v in enumerate(m.vertices):
            self.assertEqual(v.id, idx)
        self.assertEqual(
            [v.edge_key for v in m.vertices],
            sorted(v.edge_key for v in m.vertices),
        )


# ---------------------------------------------------------------------------
# 失败与边界：明确错误、无半更新状态
# ---------------------------------------------------------------------------

def _two_blocks(func=PLANE, tamper=None):
    n = 4
    blocks = make_blocks((n, n, n), ((2, 3), (n,), (n,)), func)
    if tamper is not None:
        blocks = tamper(blocks)
    return blocks, n


class BlockValidationTest(unittest.TestCase):
    def test_blocks_not_sequence(self):
        with self.assertRaisesRegex(InvalidInputError, "blocks"):
            extract_isosurface_blocks(ScalarBlock3D((0, 0, 0), 2, 2, 2, (0.0,) * 8), 1.0)

    def test_empty_blocks_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "blocks"):
            extract_isosurface_blocks([], 1.0)

    def test_non_block_element_rejected(self):
        good = ScalarBlock3D((0, 0, 0), 2, 2, 2, (0.0,) * 8)
        with self.assertRaisesRegex(InvalidInputError, r"blocks\[1\]"):
            extract_isosurface_blocks([good, ((0.0,) * 8,)], 1.0)

    def test_nan_iso_rejected_before_any_work(self):
        blocks, n = _two_blocks()
        with self.assertRaisesRegex(InvalidInputError, "iso"):
            extract_isosurface_blocks(blocks, math.nan, shape=(n, n, n))

    def test_bad_shape_types(self):
        blocks, n = _two_blocks()
        with self.assertRaisesRegex(InvalidInputError, "shape"):
            extract_isosurface_blocks(blocks, 1.0, shape=(4, 4))
        with self.assertRaisesRegex(InvalidInputError, "shape"):
            extract_isosurface_blocks(blocks, 1.0, shape=[4, 4, 4])
        with self.assertRaisesRegex(InvalidInputError, "shape"):
            extract_isosurface_blocks(blocks, 1.0, shape=(True, 4, 4))

    def test_shape_too_large_rejected(self):
        blocks, _ = _two_blocks()
        big = ScalarBlock3D((0, 0, 0), 16, 16, 16, values=(0.0,) * 4096)
        with self.assertRaisesRegex(InvalidInputError, "shape"):
            extract_isosurface_blocks([big], 1.0, shape=(17, 16, 16))

    def test_inferred_shape_too_large_rejected(self):
        # 两块首尾相接拼成 31 节点轴：推断形状超 MAX_DIM，必须拒绝。
        b1 = ScalarBlock3D((0, 0, 0), 16, 1, 1, values=(0.0,) * 16)
        b2 = ScalarBlock3D((15, 0, 0), 16, 1, 1, values=(0.0,) * 16)
        with self.assertRaisesRegex(InvalidInputError, "shape"):
            extract_isosurface_blocks([b1, b2], 1.0)

    def test_block_out_of_bounds_rejected(self):
        n = 4
        blocks = make_blocks((n, n, n), ((2, 3), (n,), (n,)), PLANE)
        over = ScalarBlock3D(
            origin=(3, 0, 0), ni=2, nj=n, nk=n,
            values=block_values((3, 0, 0), 2, n, n, PLANE),
        )
        blocks = [blocks[0], over]
        with self.assertRaisesRegex(InvalidInputError, "blocks"):
            extract_isosurface_blocks(blocks, 1.0, shape=(n, n, n))

    def test_missing_cell_gap_rejected(self):
        # shape 比块覆盖范围大一层：末端整层立方体无人拥有。
        n = 4
        blocks = make_blocks((n, n, n), ((2, 3), (n,), (n,)), PLANE)
        with self.assertRaisesRegex(InvalidInputError, r"cell \(3, 0, 0\)"):
            extract_isosurface_blocks(blocks, 1.0, shape=(n + 1, n, n))

    def test_internal_gap_rejected(self):
        # 第二块从 i=2 起（应从 i=1 起）：i=1 列立方体缺失。
        n = 4
        b0 = ScalarBlock3D((0, 0, 0), 2, n, n,
                           values=block_values((0, 0, 0), 2, n, n, PLANE))
        b1 = ScalarBlock3D((2, 0, 0), 2, n, n,
                           values=block_values((2, 0, 0), 2, n, n, PLANE))
        with self.assertRaisesRegex(InvalidInputError, r"cell \(1, 0, 0\)"):
            extract_isosurface_blocks([b0, b1], 1.0, shape=(n, n, n))

    def test_overlapping_cells_rejected(self):
        # 第二块从 i=0 起：与第一块立方体重叠。
        n = 4
        b0 = ScalarBlock3D((0, 0, 0), 3, n, n,
                           values=block_values((0, 0, 0), 3, n, n, PLANE))
        b1 = ScalarBlock3D((1, 0, 0), 3, n, n,
                           values=block_values((1, 0, 0), 3, n, n, PLANE))
        b2 = ScalarBlock3D((2, 0, 0), 2, n, n,
                           values=block_values((2, 0, 0), 2, n, n, PLANE))
        with self.assertRaisesRegex(InvalidInputError, "both blocks"):
            extract_isosurface_blocks([b0, b1, b2], 1.0, shape=(n, n, n))

    def test_inconsistent_shared_node_value_rejected(self):
        n = 4

        def tamper(blocks):
            vals = list(blocks[1].values)
            # 块1 局部 (0,0,0) 即共享节点 (1,0,0)，改成与块0不一致的值。
            vals[0] = 99.0
            return [
                blocks[0],
                ScalarBlock3D(origin=blocks[1].origin, ni=blocks[1].ni,
                              nj=blocks[1].nj, nk=blocks[1].nk, values=vals),
            ]

        blocks = make_blocks((n, n, n), ((2, 3), (n,), (n,)), PLANE)
        blocks = tamper(blocks)
        with self.assertRaisesRegex(InvalidInputError, r"shared node \(1, 0, 0\)"):
            extract_isosurface_blocks(blocks, 1.0, shape=(n, n, n))

    def test_failed_call_leaves_blocks_untouched(self):
        n = 4
        blocks = make_blocks((n, n, n), ((2, 3), (n,), (n,)), PLANE)
        snapshots = [b.values for b in blocks]
        bad = list(blocks)
        bad[1] = ScalarBlock3D(origin=bad[1].origin, ni=bad[1].ni, nj=bad[1].nj,
                               nk=bad[1].nk,
                               values=tuple(99.0 if i == 0 else v
                                            for i, v in enumerate(bad[1].values)))
        with self.assertRaises(InvalidInputError):
            extract_isosurface_blocks(bad, 1.0, shape=(n, n, n))
        for blk, snap in zip(blocks, snapshots):
            self.assertEqual(blk.values, snap)
        # 同一批块在修正输入后仍可正常使用（无残留状态）。
        ok = extract_isosurface_blocks(blocks, 1.0, shape=(n, n, n))
        self.assertGreaterEqual(ok.stats.face_count, 0)

    def test_block_constructor_validation(self):
        with self.assertRaisesRegex(InvalidInputError, "origin"):
            ScalarBlock3D(origin=(0, 0), ni=2, nj=2, nk=2, values=(0.0,) * 8)
        with self.assertRaisesRegex(InvalidInputError, "origin"):
            ScalarBlock3D(origin=(-1, 0, 0), ni=2, nj=2, nk=2, values=(0.0,) * 8)
        with self.assertRaisesRegex(InvalidInputError, "origin"):
            ScalarBlock3D(origin=(True, 0, 0), ni=2, nj=2, nk=2, values=(0.0,) * 8)
        with self.assertRaisesRegex(InvalidInputError, "ni"):
            ScalarBlock3D((0, 0, 0), ni=17, nj=1, nk=1, values=(0.0,) * 17)
        with self.assertRaisesRegex(InvalidInputError, r"values\[2\]"):
            ScalarBlock3D((0, 0, 0), ni=3, nj=1, nk=1,
                          values=(0.0, 0.0, math.nan))
        with self.assertRaisesRegex(InvalidInputError, "values"):
            ScalarBlock3D((0, 0, 0), ni=2, nj=2, nk=2, values=(0.0,) * 7)

    def test_end_and_global_node_id(self):
        b = ScalarBlock3D((2, 3, 4), ni=3, nj=2, nk=2,
                          values=(0.0,) * 12)
        self.assertEqual(b.end, (4, 4, 5))
        # 全局形状 (6,5,6)：局部 (0,0,0) -> 全局 (2,3,4)
        # nid = (4*5+3)*6+2 = 140
        self.assertEqual(b.global_node_id((6, 5, 6), 0, 0, 0), (4 * 5 + 3) * 6 + 2)
        self.assertEqual(b.global_node_id((6, 5, 6), 1, 1, 1), (5 * 5 + 4) * 6 + 3)


if __name__ == "__main__":
    unittest.main()
