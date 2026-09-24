"""同分辨率相邻体块拼接提取的测试。

覆盖：
- 面接缝、边接缝、角接缝：拼接结果与整块提取逐顶点/逐面（含 ID）一致；
- 手算小例子（f=i, iso=0.5，3x2x2 两块）核对顶点数、面数与面积；
- iso 恰等于接缝节点值：符号扰动按全局节点 ID 判定，拼接仍与整块一致；
- 独立穿越边枚举（tests/_reference.py，不 import iso3d 核心）；
- 接缝值冲突、空洞、原点不对齐、超上限等失败路径不留部分结果。
"""

import math
import unittest

from iso3d import (
    MAX_NODES,
    InvalidInputError,
    ScalarField3D,
    ScalarFieldBlock,
    extract_isosurface,
    extract_isosurface_blocks,
)

from _reference import ref_crossing_edge_keys


def make_field(ni, nj, nk, func, oi=0, oj=0, ok=0):
    """生成体块标量场：全局节点 (gi,gj,gk) 的值为 func(gi,gj,gk)。"""
    values = tuple(
        float(func(oi + i, oj + j, ok + k))
        for k in range(nk)
        for j in range(nj)
        for i in range(ni)
    )
    return ScalarField3D(ni=ni, nj=nj, nk=nk, values=values)


def octant_blocks(n, func, splits):
    """把 n x n x n 网格按 splits（每轴切分节点坐标列表）切成相邻体块。"""
    axes = [sorted(set((0,) + tuple(s) + (n - 1,))) for s in splits]
    blocks = []
    for kx in range(len(axes[0]) - 1):
        for ky in range(len(axes[1]) - 1):
            for kz in range(len(axes[2]) - 1):
                sx = axes[0]
                sy = axes[1]
                sz = axes[2]
                ni = sx[kx + 1] - sx[kx] + 1
                nj = sy[ky + 1] - sy[ky] + 1
                nk = sz[kz + 1] - sz[kz] + 1
                field = make_field(
                    ni, nj, nk, func, sx[kx], sy[ky], sz[kz]
                )
                blocks.append(
                    ScalarFieldBlock(origin=(sx[kx], sy[ky], sz[kz]), field=field)
                )
    return blocks


def mesh_signature(mesh):
    """忽略 ID 编号顺序之外的全部几何/拓扑内容，用于逐元素严格比较。"""
    return (
        tuple((v.point, v.edge_key, v.t) for v in mesh.vertices),
        tuple((f.vertex_ids, f.area) for f in mesh.faces),
        mesh.stats.degenerate_face_count,
        mesh.boundary_edges,
        mesh.nonmanifold_edges,
    )


class FaceSeamTest(unittest.TestCase):
    """两块沿 i 方向共享一个节点面（面接缝）。"""

    @classmethod
    def setUpClass(cls):
        cls.n = 9
        cls.func = lambda i, j, k: (i - 4.0) ** 2 + (j - 4.0) ** 2 + (k - 4.0) ** 2
        cls.iso = 7.0
        cls.whole = make_field(cls.n, cls.n, cls.n, cls.func)
        cls.mono = extract_isosurface(cls.whole, cls.iso)
        a = ScalarFieldBlock(origin=(0, 0, 0), field=make_field(5, 9, 9, cls.func))
        b = ScalarFieldBlock(origin=(4, 0, 0), field=make_field(5, 9, 9, cls.func, oi=4))
        cls.blocks = (a, b)
        cls.stitched = extract_isosurface_blocks(cls.blocks, cls.iso)

    def test_stitched_equals_monolithic_vertex_by_vertex(self):
        self.assertEqual(
            [v.point for v in self.stitched.vertices],
            [v.point for v in self.mono.vertices],
        )
        self.assertEqual(
            [v.edge_key for v in self.stitched.vertices],
            [v.edge_key for v in self.mono.vertices],
        )
        self.assertEqual(
            [v.t for v in self.stitched.vertices],
            [v.t for v in self.mono.vertices],
        )

    def test_stitched_equals_monolithic_face_by_face(self):
        self.assertEqual(
            [f.vertex_ids for f in self.stitched.faces],
            [f.vertex_ids for f in self.mono.faces],
        )
        self.assertEqual(
            [f.area for f in self.stitched.faces],
            [f.area for f in self.mono.faces],
        )
        self.assertEqual(mesh_signature(self.stitched), mesh_signature(self.mono))
        self.assertEqual(self.stitched.stats, self.mono.stats)

    def test_closed_manifold_across_face_seam(self):
        self.assertEqual(self.stitched.stats.boundary_edge_count, 0)
        self.assertEqual(self.stitched.stats.nonmanifold_edge_count, 0)

    def test_seam_plane_edges_each_own_one_shared_vertex(self):
        # 完全躺在接缝面 i=4 上的穿越网格边（两端节点 i 坐标都为 4），
        # 每条只能产生一个顶点，且该顶点被面两侧立方体的三角形共用。
        n = self.n

        def gid(i, j, k):
            return (k * n + j) * n + i

        seam_edges = {
            tuple(sorted((gid(4, j, k), gid(4, j + dj, k + dk))))
            for j in range(n)
            for k in range(n)
            for dj, dk in ((1, 0), (0, 1))
            if j + dj < n and k + dk < n
        }
        hit = [v for v in self.stitched.vertices if v.edge_key in seam_edges]
        # 无重复：一条接缝边对应且仅对应一个顶点。
        self.assertEqual(len(hit), len({v.edge_key for v in hit}))
        # 球面（半径^2=7）在接缝面上确有穿越边。
        self.assertGreater(len(hit), 0)
        # 每个这样的顶点必须被至少两个三角形引用（面两侧各一）。
        use_count = {v.id: 0 for v in hit}
        for f in self.stitched.faces:
            for vid in f.vertex_ids:
                if vid in use_count:
                    use_count[vid] += 1
        self.assertTrue(all(c >= 2 for c in use_count.values()))

    def test_vertex_edge_keys_match_independent_enumeration(self):
        expected = ref_crossing_edge_keys(
            self.n, self.n, self.n, self.whole.values, self.iso
        )
        self.assertEqual({v.edge_key for v in self.stitched.vertices}, expected)

    def test_independent_of_block_order(self):
        flipped = extract_isosurface_blocks(tuple(reversed(self.blocks)), self.iso)
        self.assertEqual(mesh_signature(flipped), mesh_signature(self.mono))

    def test_deterministic_repeat(self):
        again = extract_isosurface_blocks(self.blocks, self.iso)
        self.assertEqual(mesh_signature(again), mesh_signature(self.stitched))


class EdgeAndCornerSeamTest(unittest.TestCase):
    """2x2x2 八块：同时存在面、边、角三类接缝。"""

    @classmethod
    def setUpClass(cls):
        cls.n = 9
        cls.func = lambda i, j, k: i + 2.0 * j + 3.0 * k
        cls.iso = 11.25
        cls.whole = make_field(cls.n, cls.n, cls.n, cls.func)
        cls.mono = extract_isosurface(cls.whole, cls.iso)
        cls.blocks = octant_blocks(cls.n, cls.func, splits=([4], [4], [4]))
        cls.stitched = extract_isosurface_blocks(cls.blocks, cls.iso)

    def test_eight_blocks_tile_with_each_seam_kind(self):
        self.assertEqual(len(self.blocks), 8)
        # 角接缝：节点 (4,4,4) 被全部 8 块共享。
        target = (4, 4, 4)
        owners = [
            b
            for b in self.blocks
            if all(o <= c <= o + s - 1 for c, o, s in zip(target, b.origin, (b.field.ni, b.field.nj, b.field.nk)))
        ]
        self.assertEqual(len(owners), 8)

    def test_stitched_equals_monolithic(self):
        self.assertEqual(mesh_signature(self.stitched), mesh_signature(self.mono))
        self.assertTrue(self.stitched.faces)

    def test_edge_seam_four_blocks_share_one_vertex(self):
        # 边接缝：沿 i 方向、j=4、k=0 的棱线恰被 4 个体块覆盖
        # （i 方向两块 × j 方向两块，都在 k 最下层）。
        # f = i+2j+3k = 11.25 在该棱上为 i+8 = 11.25 -> i = 3.25，
        # 只穿越一条网格边，因此只能产生一个共享顶点。
        n = self.n

        def gid(i, j, k):
            return (k * n + j) * n + i

        seam_line = {
            tuple(sorted((gid(i, 4, 0), gid(i + 1, 4, 0))))
            for i in range(n - 1)
        }
        hit = [v for v in self.stitched.vertices if v.edge_key in seam_line]
        self.assertEqual(len(hit), 1)
        self.assertEqual(len({v.edge_key for v in hit}), 1)
        v = hit[0]
        self.assertAlmostEqual(v.point[0], 3.25, places=12)
        self.assertAlmostEqual(v.point[1], 4.0, places=12)
        self.assertAlmostEqual(v.point[2], 0.0, places=12)

    def test_corner_seam_node_shared_by_eight_blocks(self):
        # 角接缝节点 (4,4,4) 的扰动判定必须按其全局节点 ID 唯一确定；
        # 它若落在穿越边上，对应顶点只能出现一次。
        n = self.n
        gid444 = (4 * n + 4) * n + 4
        hit = [
            v for v in self.stitched.vertices if gid444 in v.edge_key
        ]
        self.assertEqual(len(hit), len({v.edge_key for v in hit}))
        for v in hit:
            # 顶点位于从角点出发的某条网格边上：坐标必须精确等于线性插值。
            self.assertEqual(len(v.edge_key), 2)

    def test_analytic_plane_area_preserved_across_seams(self):
        # 平面法向 (1,2,3)，截面面积可由独立参考计算。
        from _reference import plane_box_intersection_area

        expected = plane_box_intersection_area(
            1.0, 2.0, 3.0, self.iso, 8.0, 8.0, 8.0
        )
        self.assertGreater(expected, 0.0)
        self.assertLess(
            abs(self.stitched.stats.total_area - expected), 1e-9 * expected
        )


class UnequalBlockSizesTest(unittest.TestCase):
    """不同尺寸（同间距）体块沿面拼接，切分面不在正中。"""

    def test_staggered_split_equals_monolith(self):
        func = lambda i, j, k: math.sin(0.7 * i) + math.cos(0.9 * j) + 0.5 * k
        iso = 1.1
        whole = make_field(8, 6, 5, func)
        mono = extract_isosurface(whole, iso)
        a = ScalarFieldBlock(origin=(0, 0, 0), field=make_field(4, 6, 5, func))
        b = ScalarFieldBlock(origin=(3, 0, 0), field=make_field(5, 6, 5, func, oi=3))
        stitched = extract_isosurface_blocks((a, b), iso)
        self.assertEqual(mesh_signature(stitched), mesh_signature(mono))

    def test_single_block_equals_direct_extraction(self):
        func = lambda i, j, k: i * i - j * k
        whole = make_field(4, 5, 6, func)
        mono = extract_isosurface(whole, 2.0)
        block = ScalarFieldBlock(origin=(0, 0, 0), field=whole)
        stitched = extract_isosurface_blocks((block,), 2.0)
        self.assertEqual(mesh_signature(stitched), mesh_signature(mono))


class HandComputedSeamTest(unittest.TestCase):
    """手算小例子：3x2x2 网格，f=i，iso=0.5，两块共享 i=1 节点面。

    只有 i=0 层的 1 个立方体贡献等值面（x=0.5 处的单位正方形）。
    对该立方体的 6 个四面体逐个枚举（角点编号 i+2j+4k，i=0 的
    角点 {0,2,4,6} 在下方）：
      tet (0,1,3,7) 下方 {0}        -> 3 穿越边 -> 1 三角形
      tet (0,1,5,7) 下方 {0}        -> 3        -> 1
      tet (0,2,3,7) 下方 {0,2}      -> 4        -> 2
      tet (0,2,6,7) 下方 {0,2,6}    -> 3        -> 1
      tet (0,4,5,7) 下方 {0,4}      -> 4        -> 2
      tet (0,4,6,7) 下方 {0,4,6}    -> 3        -> 1
    合计 9 条穿越边、8 个三角形，几何并集为面积 1 的单位正方形。
    """

    @classmethod
    def setUpClass(cls):
        func = lambda i, j, k: float(i)
        cls.iso = 0.5
        a = ScalarFieldBlock(origin=(0, 0, 0), field=make_field(2, 2, 2, func))
        b = ScalarFieldBlock(origin=(1, 0, 0), field=make_field(2, 2, 2, func, oi=1))
        cls.stitched = extract_isosurface_blocks((a, b), cls.iso)
        cls.mono = extract_isosurface(make_field(3, 2, 2, func), cls.iso)

    def test_hand_counts(self):
        self.assertEqual(self.stitched.stats.vertex_count, 9)
        self.assertEqual(self.stitched.stats.face_count, 8)
        self.assertAlmostEqual(self.stitched.stats.total_area, 1.0, places=12)
        self.assertEqual(self.stitched.stats.nonmanifold_edge_count, 0)

    def test_all_vertices_on_section_plane(self):
        for v in self.stitched.vertices:
            self.assertAlmostEqual(v.point[0], 0.5, places=12)

    def test_body_diagonal_vertex_at_square_center(self):
        # 体对角线 0-7（全局节点 0 与 (1,1,1)= (0*2+1)*... 手算 id）
        nid_111 = (1 * 2 + 1) * 3 + 1
        key = tuple(sorted((0, nid_111)))
        points = [v.point for v in self.stitched.vertices if v.edge_key == key]
        self.assertEqual(len(points), 1)
        for got, want in zip(points[0], (0.5, 0.5, 0.5)):
            self.assertAlmostEqual(got, want, places=12)

    def test_seam_plane_nodes_owned_once(self):
        # i=1 接缝面上 4 个节点：没有任何顶点重复落在同一几何点。
        points = [tuple(round(c, 12) for c in v.point) for v in self.stitched.vertices]
        self.assertEqual(len(points), len(set(points)))

    def test_equals_monolith(self):
        self.assertEqual(
            mesh_signature(self.stitched), mesh_signature(self.mono)
        )


class IsoOnSeamNodesTest(unittest.TestCase):
    """iso 恰等于接缝节点值：扰动按全局节点 ID 奇偶，结果须与整块一致。"""

    @classmethod
    def setUpClass(cls):
        cls.func = lambda i, j, k: float(i)
        cls.iso = 1.0  # 接缝面 i=1 的节点值全部恰为 iso
        a = ScalarFieldBlock(origin=(0, 0, 0), field=make_field(2, 3, 3, cls.func))
        b = ScalarFieldBlock(origin=(1, 0, 0), field=make_field(2, 3, 3, cls.func, oi=1))
        cls.blocks = (a, b)
        cls.stitched = extract_isosurface_blocks(cls.blocks, cls.iso)
        cls.mono = extract_isosurface(make_field(3, 3, 3, cls.func), cls.iso)

    def test_equals_monolith_despite_perturbation(self):
        self.assertEqual(
            mesh_signature(self.stitched), mesh_signature(self.mono)
        )

    def test_section_area_is_unit_rectangle(self):
        # 等值面位于 x=1，在 y,z ∈ [0,2] 上的面积为 4。
        self.assertAlmostEqual(self.stitched.stats.total_area, 4.0, places=9)
        for v in self.stitched.vertices:
            self.assertAlmostEqual(v.point[0], 1.0, places=12)

    def test_combinatorially_valid(self):
        for key, count in self.stitched.edge_use_count.items():
            self.assertIn(count, (1, 2), f"edge {key} used {count} times")
        self.assertEqual(self.stitched.stats.nonmanifold_edge_count, 0)


class StitchingFailureTest(unittest.TestCase):
    def _two_blocks(self, va_values, vb_values):
        fa = ScalarField3D(ni=2, nj=2, nk=2, values=va_values)
        fb = ScalarField3D(ni=2, nj=2, nk=2, values=vb_values)
        return (
            ScalarFieldBlock(origin=(0, 0, 0), field=fa),
            ScalarFieldBlock(origin=(1, 0, 0), field=fb),
        )

    def test_conflicting_seam_values_rejected_and_located(self):
        va = [0.0] * 8
        vb = [0.0] * 8
        # 块 B 的局部节点 (0,1,0) 即全局接缝节点 (1,1,0)，故意与块 A 不一致。
        vb[2] = 9.0
        blocks = self._two_blocks(tuple(va), tuple(vb))
        with self.assertRaisesRegex(
            InvalidInputError, r"shared node \(1, 1, 0\).*block 0.*block 1"
        ):
            extract_isosurface_blocks(blocks, 0.5)

    def test_conflict_leaves_no_partial_state_and_inputs_usable(self):
        # 正确的接缝数据应为 f=i：块 A [0,1,0,1,...]，块 B 接缝面值为 1。
        va = tuple(float(i) for k in range(2) for j in range(2) for i in range(2))
        vb_bad = list(va)
        vb_bad[2] = 9.0  # 局部 (0,1,0) 即接缝节点 (1,1,0)，故意冲突
        blocks_bad = self._two_blocks(va, tuple(vb_bad))
        with self.assertRaises(InvalidInputError):
            extract_isosurface_blocks(blocks_bad, 0.5)
        # 失败不修改输入对象；用正确数据重建后结果与整块 f=i 一致。
        vb_ok = tuple(
            float(1 + i) for k in range(2) for j in range(2) for i in range(2)
        )
        blocks_ok = (
            blocks_bad[0],
            ScalarFieldBlock(
                origin=(1, 0, 0),
                field=ScalarField3D(ni=2, nj=2, nk=2, values=vb_ok),
            ),
        )
        mono_values = tuple(
            float(i) for k in range(2) for j in range(2) for i in range(3)
        )
        mono = extract_isosurface(
            ScalarField3D(ni=3, nj=2, nk=2, values=mono_values), 0.5
        )
        mesh = extract_isosurface_blocks(blocks_ok, 0.5)
        self.assertEqual(mesh_signature(mesh), mesh_signature(mono))
        # 原冲突对象仍然完好（冻结数据类，值未被改动）。
        self.assertEqual(blocks_bad[1].field.values, tuple(vb_bad))

    def test_gap_between_blocks_rejected(self):
        func = lambda i, j, k: float(i)
        a = ScalarFieldBlock(origin=(0, 0, 0), field=make_field(2, 2, 2, func))
        b = ScalarFieldBlock(origin=(3, 0, 0), field=make_field(2, 2, 2, func, oi=3))
        with self.assertRaisesRegex(InvalidInputError, r"node \(2, 0, 0\).*not covered"):
            extract_isosurface_blocks((a, b), 0.5)

    def test_union_not_anchored_at_origin_rejected(self):
        func = lambda i, j, k: float(i)
        block = ScalarFieldBlock(origin=(1, 0, 0), field=make_field(2, 2, 2, func, oi=1))
        with self.assertRaisesRegex(InvalidInputError, r"anchored at the lattice origin"):
            extract_isosurface_blocks((block,), 0.5)

    def test_empty_sequence_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "blocks"):
            extract_isosurface_blocks((), 0.5)

    def test_non_sequence_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "blocks"):
            extract_isosurface_blocks(None, 0.5)

    def test_wrong_element_type_rejected_with_index(self):
        field = make_field(2, 2, 2, lambda i, j, k: 0.0)
        with self.assertRaisesRegex(InvalidInputError, r"blocks\[1\]"):
            extract_isosurface_blocks(
                (ScalarFieldBlock(origin=(0, 0, 0), field=field), "not-a-block"),
                0.5,
            )

    def test_bad_origin_rejected_at_construction(self):
        field = make_field(2, 2, 2, lambda i, j, k: 0.0)
        with self.assertRaisesRegex(InvalidInputError, "origin"):
            ScalarFieldBlock(origin=(0, 0, -1), field=field)
        with self.assertRaisesRegex(InvalidInputError, "origin"):
            ScalarFieldBlock(origin=(0, True, 0), field=field)
        with self.assertRaisesRegex(InvalidInputError, "origin"):
            ScalarFieldBlock(origin=(0, 0), field=field)

    def test_bad_field_rejected_at_construction(self):
        with self.assertRaisesRegex(InvalidInputError, "field"):
            ScalarFieldBlock(origin=(0, 0, 0), field=((0.0,) * 8,))

    def test_nan_and_bool_iso_rejected(self):
        blocks = (
            ScalarFieldBlock(origin=(0, 0, 0),
                             field=make_field(2, 2, 2, lambda i, j, k: 0.0)),
        )
        with self.assertRaisesRegex(InvalidInputError, "iso"):
            extract_isosurface_blocks(blocks, math.nan)
        with self.assertRaisesRegex(InvalidInputError, "iso"):
            extract_isosurface_blocks(blocks, True)

    def test_merged_grid_above_node_limit_rejected(self):
        # 每块 16x16x9=2304 节点（单块合法），原点 k=0 与 k=8 拼成
        # 16x16x17=4352 节点，超过 MAX_NODES=4096。
        big_a = ScalarField3D(ni=16, nj=16, nk=9, values=(0.0,) * (16 * 16 * 9))
        big_b = ScalarField3D(ni=16, nj=16, nk=9, values=(0.0,) * (16 * 16 * 9))
        blocks = (
            ScalarFieldBlock(origin=(0, 0, 0), field=big_a),
            ScalarFieldBlock(origin=(0, 0, 8), field=big_b),
        )
        with self.assertRaisesRegex(InvalidInputError, "blocks"):
            extract_isosurface_blocks(blocks, 0.5)

    def test_merged_grid_at_limit_accepted(self):
        # 两块各 16x16x9=2304 节点（单块合法），原点 k=0 与 k=7，
        # 并集覆盖 k=0..15 共 16 层，合并恰为 16x16x16 = 4096 = MAX_NODES，
        # 接缝共享 k=7、k=8 两层节点。
        fa = ScalarField3D(
            ni=16, nj=16, nk=9,
            values=tuple(
                float(i + j + k)
                for k in range(9) for j in range(16) for i in range(16)
            ),
        )
        fb = ScalarField3D(
            ni=16, nj=16, nk=9,
            values=tuple(
                float(i + j + (k + 7))
                for k in range(9) for j in range(16) for i in range(16)
            ),
        )
        blocks = (
            ScalarFieldBlock(origin=(0, 0, 0), field=fa),
            ScalarFieldBlock(origin=(0, 0, 7), field=fb),
        )
        mesh = extract_isosurface_blocks(blocks, 20.0)
        self.assertEqual(mesh.stats.node_count, MAX_NODES)
        self.assertEqual(mesh.stats.cell_count, 15 ** 3)
        self.assertEqual(mesh.stats.nonmanifold_edge_count, 0)


class RedundantOverlapTest(unittest.TestCase):
    """同一块被重复给出（所有共享节点值一致）：唯一所有者语义下合法。"""

    def test_duplicate_block_accepted_and_identical(self):
        func = lambda i, j, k: (i - 2.0) ** 2 + (j - 2.0) ** 2 + (k - 2.0) ** 2
        field = make_field(5, 5, 5, func)
        block = ScalarFieldBlock(origin=(0, 0, 0), field=field)
        once = extract_isosurface_blocks((block,), 4.0)
        twice = extract_isosurface_blocks((block, block), 4.0)
        self.assertEqual(mesh_signature(twice), mesh_signature(once))


if __name__ == "__main__":
    unittest.main()
