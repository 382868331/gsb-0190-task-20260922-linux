"""共享交点溯源与标量敏感度测试。

覆盖：
- 线性斜坡体上手算交点（来源边、插值权重、解析导数）；
- 安全边上三个导数（两端标量值 + iso）的双侧有限差分数值验证，
  含坐标轴边与面对角线边（edge_vec 多个非零分量）；
- 反向遍历：无论从哪个端点方向枚举，同一网格边只有一条记录；
- 标量差极小：标记 unstable_denominator，导数为 None 且无巨大伪精确数；
- 端点恰 iso：标记 endpoint_on_iso / zero_denominator，导数为 None；
- 一条网格边被多个四面体引用时来源记录全局唯一；
- intersections_for_node：一个输入节点影响哪些交点。

解析期望值由本文件内从 t=(iso-va)/(vb-va) 独立推导的公式与手算常数给出，
不调用被测核心的内部计算；有限差分是完全独立的数值核对。
"""

import math
import unittest

from iso3d import (
    NONDIFF_ENDPOINT_ON_ISO,
    NONDIFF_UNSTABLE_DENOMINATOR,
    NONDIFF_ZERO_DENOMINATOR,
    ScalarField3D,
    canonical_edge_key,
    extract_isosurface,
)

from _reference import REF_CUBE_CORNERS, REF_TETRA_TABLE, REF_TET_EDGES, ref_below

FD_H = 1e-6
FD_TOL = 1e-7


def ramp_values(ni, nj, nk):
    return tuple(
        float(i + j + k)
        for k in range(nk)
        for j in range(nj)
        for i in range(ni)
    )


def point_on_edge(mesh, key):
    for v in mesh.vertices:
        if v.edge_key == key:
            return v.point
    raise KeyError(f"no vertex on edge {key}")


def record_on_edge(mesh, key):
    for rec in mesh.intersections:
        if rec.edge_key == key:
            return rec
    raise KeyError(f"no record on edge {key}")


def perturbed_point(ni, nj, nk, base_values, iso, overrides, key):
    """用 overrides {node_id: value} 替换节点值后重新提取，返回 key 上交点。"""
    values = list(base_values)
    for nid, val in overrides.items():
        values[nid] = val
    field = ScalarField3D(ni=ni, nj=nj, nk=nk, values=tuple(values))
    mesh = extract_isosurface(field, iso)
    return point_on_edge(mesh, key)


def central_fd(base_values, dims, iso, key, endpoint_id, h=FD_H):
    """扰动单个节点标量值的双侧有限差分（3 个坐标分量）。"""
    ni, nj, nk = dims
    lo = perturbed_point(
        ni, nj, nk, base_values, iso, {endpoint_id: base_values[endpoint_id] - h}, key
    )
    hi = perturbed_point(
        ni, nj, nk, base_values, iso, {endpoint_id: base_values[endpoint_id] + h}, key
    )
    return tuple((hi[d] - lo[d]) / (2.0 * h) for d in range(3))


def central_fd_iso(base_values, dims, iso, key, h=FD_H):
    ni, nj, nk = dims
    field = ScalarField3D(ni=ni, nj=nj, nk=nk, values=base_values)
    lo = point_on_edge(extract_isosurface(field, iso - h), key)
    hi = point_on_edge(extract_isosurface(field, iso + h), key)
    return tuple((hi[d] - lo[d]) / (2.0 * h) for d in range(3))


class HandComputedAxisEdgeTest(unittest.TestCase):
    """3x3x3 斜坡 f=i+j+k，iso=2.5：边 (1,1,0)->(2,1,0) 上的手算交点。"""

    @classmethod
    def setUpClass(cls):
        cls.ni = cls.nj = cls.nk = 3
        cls.values = ramp_values(3, 3, 3)
        cls.field = ScalarField3D(ni=3, nj=3, nk=3, values=cls.values)
        cls.iso = 2.5
        cls.mesh = extract_isosurface(cls.field, cls.iso)
        cls.id_a = cls.field.node_id(1, 1, 0)  # v=2
        cls.id_b = cls.field.node_id(2, 1, 0)  # v=3
        cls.key = (cls.id_a, cls.id_b)

    def test_hand_computed_intersection(self):
        v = record_on_edge(self.mesh, self.key)
        vertex = next(x for x in self.mesh.vertices if x.edge_key == self.key)
        # t = (2.5 - 2) / (3 - 2) = 0.5，点 = (1.5, 1, 0)
        self.assertEqual(vertex.t, 0.5)
        for got, want in zip(vertex.point, (1.5, 1.0, 0.0)):
            self.assertAlmostEqual(got, want, places=15)
        self.assertEqual(v.edge_key, self.key)
        self.assertEqual(v.vertex_id, vertex.id)

    def test_weights(self):
        rec = record_on_edge(self.mesh, self.key)
        wa, wb = rec.weights
        self.assertAlmostEqual(wa, 0.5, places=15)
        self.assertAlmostEqual(wb, 0.5, places=15)
        self.assertAlmostEqual(wa + wb, 1.0, places=15)
        # point = wa*P_a + wb*P_b，权重顺序对应升序节点 ID。
        coords = ((1.0, 1.0, 0.0), (2.0, 1.0, 0.0))
        for d in range(3):
            self.assertAlmostEqual(
                self.mesh.vertices[rec.vertex_id].point[d],
                wa * coords[0][d] + wb * coords[1][d],
                places=15,
            )

    def test_analytic_derivatives_hand_values(self):
        # edge_vec = (1,0,0)，denom=1，denom^2=1：
        # dP/dva = (iso-vb)  * e = -0.5 (1,0,0)
        # dP/dvb = (va-iso)  * e = -0.5 (1,0,0)
        # dP/diso = 1/denom  * e =  1.0 (1,0,0)
        rec = record_on_edge(self.mesh, self.key)
        s = rec.sensitivity
        self.assertTrue(s.differentiable)
        self.assertEqual(s.reason, "")
        for got, want in zip(s.d_point_d_va, (-0.5, 0.0, 0.0)):
            self.assertAlmostEqual(got, want, places=14)
        for got, want in zip(s.d_point_d_vb, (-0.5, 0.0, 0.0)):
            self.assertAlmostEqual(got, want, places=14)
        for got, want in zip(s.d_point_d_iso, (1.0, 0.0, 0.0)):
            self.assertAlmostEqual(got, want, places=14)

    def test_derivatives_sum_to_zero(self):
        # t 只依赖差值结构：dt/dva + dt/dvb + dt/diso == 0（沿边方向投影）。
        rec = record_on_edge(self.mesh, self.key)
        s = rec.sensitivity
        for d in range(3):
            self.assertAlmostEqual(
                s.d_point_d_va[d] + s.d_point_d_vb[d] + s.d_point_d_iso[d],
                0.0,
                places=14,
            )

    def test_finite_difference_all_three_inputs(self):
        rec = record_on_edge(self.mesh, self.key)
        s = rec.sensitivity
        fd_va = central_fd(
            self.values, (3, 3, 3), self.iso, self.key, self.id_a
        )
        fd_vb = central_fd(
            self.values, (3, 3, 3), self.iso, self.key, self.id_b
        )
        fd_iso = central_fd_iso(self.values, (3, 3, 3), self.iso, self.key)
        for d in range(3):
            self.assertAlmostEqual(s.d_point_d_va[d], fd_va[d], delta=FD_TOL)
            self.assertAlmostEqual(s.d_point_d_vb[d], fd_vb[d], delta=FD_TOL)
            self.assertAlmostEqual(s.d_point_d_iso[d], fd_iso[d], delta=FD_TOL)

    def test_small_perturbation_moves_point_as_predicted(self):
        # 对 id_a 施加实际小扰动，位移 ≈ dP/dva * δ（一阶）。
        rec = record_on_edge(self.mesh, self.key)
        delta = 0.001
        p1 = perturbed_point(
            3, 3, 3, self.values, self.iso, {self.id_a: 2.0 + delta}, self.key
        )
        p0 = point_on_edge(self.mesh, self.key)
        for d in range(3):
            self.assertAlmostEqual(
                p1[d] - p0[d],
                rec.sensitivity.d_point_d_va[d] * delta,
                delta=1e-5,
            )


class HandComputedDiagonalEdgeTest(unittest.TestCase):
    """4x4x4 斜坡 iso=3.5：面对角线 (1,1,1)->(2,2,1)，edge_vec=(1,1,0)。

    该边是四面体 0-3 边，同时出现在两个四面体中，且在相邻枚举中会被
    不同方向遍历——必须共享同一条记录。
    """

    @classmethod
    def setUpClass(cls):
        cls.ni = cls.nj = cls.nk = 4
        cls.values = ramp_values(4, 4, 4)
        cls.field = ScalarField3D(ni=4, nj=4, nk=4, values=cls.values)
        cls.iso = 3.5
        cls.mesh = extract_isosurface(cls.field, cls.iso)
        cls.id_a = cls.field.node_id(1, 1, 1)  # v=3
        cls.id_b = cls.field.node_id(2, 2, 1)  # v=5
        cls.key = (cls.id_a, cls.id_b)

    def test_hand_computed_diagonal_intersection(self):
        # t = (3.5-3)/(5-3) = 0.25，点 = (1.25,1.25,1)
        vertex = next(x for x in self.mesh.vertices if x.edge_key == self.key)
        self.assertAlmostEqual(vertex.t, 0.25, places=15)
        for got, want in zip(vertex.point, (1.25, 1.25, 1.0)):
            self.assertAlmostEqual(got, want, places=14)

    def test_diagonal_analytic_derivatives(self):
        # denom=2, denom^2=4：
        # dt/dva = (3.5-5)/4 = -0.375 ; dt/dvb = (3-3.5)/4 = -0.125
        # dt/diso = 0.5；edge_vec=(1,1,0)
        rec = record_on_edge(self.mesh, self.key)
        s = rec.sensitivity
        self.assertTrue(s.differentiable)
        for got, want in zip(s.d_point_d_va, (-0.375, -0.375, 0.0)):
            self.assertAlmostEqual(got, want, places=14)
        for got, want in zip(s.d_point_d_vb, (-0.125, -0.125, 0.0)):
            self.assertAlmostEqual(got, want, places=14)
        for got, want in zip(s.d_point_d_iso, (0.5, 0.5, 0.0)):
            self.assertAlmostEqual(got, want, places=14)

    def test_diagonal_finite_differences(self):
        rec = record_on_edge(self.mesh, self.key)
        s = rec.sensitivity
        fd_va = central_fd(self.values, (4, 4, 4), self.iso, self.key, self.id_a)
        fd_vb = central_fd(self.values, (4, 4, 4), self.iso, self.key, self.id_b)
        fd_iso = central_fd_iso(self.values, (4, 4, 4), self.iso, self.key)
        for d in range(3):
            self.assertAlmostEqual(s.d_point_d_va[d], fd_va[d], delta=FD_TOL)
            self.assertAlmostEqual(s.d_point_d_vb[d], fd_vb[d], delta=FD_TOL)
            self.assertAlmostEqual(s.d_point_d_iso[d], fd_iso[d], delta=FD_TOL)


class ReverseTraversalSharingTest(unittest.TestCase):
    """反向遍历 / 多四面体引用：同一条物理边只能产生一条记录。"""

    @classmethod
    def setUpClass(cls):
        cls.ni = cls.nj = cls.nk = 3
        cls.values = ramp_values(3, 3, 3)
        cls.field = ScalarField3D(ni=3, nj=3, nk=3, values=cls.values)
        cls.iso = 2.0
        cls.mesh = extract_isosurface(cls.field, cls.iso)

    def _all_crossings_both_directions(self):
        """独立枚举穿越边，分别以正向与反向端点对返回。"""
        ni, nj, nk = 3, 3, 3
        forward, reversed_ = [], []

        def nid(i, j, k):
            return (k * nj + j) * ni + i

        for k in range(nk - 1):
            for j in range(nj - 1):
                for i in range(ni - 1):
                    corners = tuple(
                        nid(i + di, j + dj, k + dk)
                        for (di, dj, dk) in REF_CUBE_CORNERS
                    )
                    below = tuple(
                        ref_below(self.values[c], self.iso, c) for c in corners
                    )
                    for tet in REF_TETRA_TABLE:
                        nodes = tuple(corners[c] for c in tet)
                        tb = tuple(below[c] for c in tet)
                        for ea, eb in REF_TET_EDGES:
                            if tb[ea] != tb[eb]:
                                forward.append((nodes[ea], nodes[eb]))
                                # 故意反向遍历同一条边。
                                reversed_.append((nodes[eb], nodes[ea]))
        return forward, reversed_

    def test_reversed_keys_resolve_to_single_records(self):
        forward, reversed_ = self._all_crossings_both_directions()
        self.assertEqual(len(forward), len(reversed_))
        record_by_key = {rec.edge_key: rec for rec in self.mesh.intersections}
        for (a, b), (b2, a2) in zip(forward, reversed_):
            self.assertEqual((a, b), (a2, b2))  # 枚举确实反向
            key = canonical_edge_key(a, b)
            key_rev = canonical_edge_key(b2, a2)
            self.assertEqual(key, key_rev)
            self.assertIn(key, record_by_key)
        # 每个规范边键恰好一条记录。
        canonical_keys = [canonical_edge_key(a, b) for a, b in forward]
        self.assertEqual(len(record_by_key), len(set(canonical_keys)))

    def test_shared_face_diagonal_one_record_across_cells(self):
        # k=0 立方体的顶面对角线 4-7 与 k=1 立方体的底面对角线 0-3
        # 是同一条物理边：节点 (0,0,1)=(v=1) 与 (1,1,1)=(v=3)，iso=2。
        lower_4 = self.field.node_id(0, 0, 1)
        lower_7 = self.field.node_id(1, 1, 1)
        upper_0 = self.field.node_id(0, 0, 1)
        upper_3 = self.field.node_id(1, 1, 1)
        self.assertEqual((lower_4, lower_7), (upper_0, upper_3))
        key = (lower_4, lower_7)
        recs = [rec for rec in self.mesh.intersections if rec.edge_key == key]
        self.assertEqual(len(recs), 1)
        verts = [v for v in self.mesh.vertices if v.edge_key == key]
        self.assertEqual(len(verts), 1)
        # 该共享顶点必须被两个立方体中的三角形同时引用。
        vid = verts[0].id
        using_faces = [
            f for f in self.mesh.faces if vid in f.vertex_ids
        ]
        self.assertGreaterEqual(len(using_faces), 2)

    def test_record_and_vertex_align_one_to_one(self):
        self.assertEqual(
            len(self.mesh.intersections), len(self.mesh.vertices)
        )
        for vid, rec in enumerate(self.mesh.intersections):
            self.assertEqual(rec.vertex_id, vid)
            self.assertEqual(
                rec.edge_key, self.mesh.vertices[vid].edge_key
            )


class TinyScalarDifferenceTest(unittest.TestCase):
    """标量差极小：分母不稳定，拒绝输出巨大伪精确导数。"""

    @classmethod
    def setUpClass(cls):
        ni = nj = nk = 2
        values = [10.0] * 8
        values[0] = 1.0
        values[1] = 1.0 + 1e-12  # 边 0-1 严格跨过 iso，但分母极小
        cls.iso = 1.0 + 5e-13
        cls.field = ScalarField3D(ni=2, nj=2, nk=2, values=tuple(values))
        cls.mesh = extract_isosurface(cls.field, cls.iso)

    def test_unstable_edge_marked_nondifferentiable(self):
        rec = record_on_edge(self.mesh, (0, 1))
        s = rec.sensitivity
        self.assertFalse(s.differentiable)
        self.assertEqual(s.reason, NONDIFF_UNSTABLE_DENOMINATOR)
        self.assertIsNone(s.d_point_d_va)
        self.assertIsNone(s.d_point_d_vb)
        self.assertIsNone(s.d_point_d_iso)

    def test_geometry_still_emitted_inside_edge(self):
        # 几何交点照常给出且落在边内，只是不宣称导数。
        v = next(x for x in self.mesh.vertices if x.edge_key == (0, 1))
        self.assertTrue(0.0 <= v.t <= 1.0)
        self.assertTrue(0.0 <= v.point[0] <= 1.0)
        wa, wb = record_on_edge(self.mesh, (0, 1)).weights
        self.assertAlmostEqual(wa + wb, 1.0, places=15)

    def test_no_huge_pseudo_precise_numbers(self):
        # 所有已输出的可微导数必须是温和数值，不允许 1/(极小分母) 式爆炸。
        for rec in self.mesh.intersections:
            s = rec.sensitivity
            if not s.differentiable:
                continue
            for deriv in (s.d_point_d_va, s.d_point_d_vb, s.d_point_d_iso):
                for comp in deriv:
                    self.assertLess(abs(comp), 1e6)
                    self.assertTrue(math.isfinite(comp))


class EndpointOnIsoTest(unittest.TestCase):
    """f=i，iso=1：整层节点恰在等值面上，拓扑扰动分支点不可微。"""

    @classmethod
    def setUpClass(cls):
        ni = nj = nk = 5
        values = tuple(
            float(i)
            for k in range(nk)
            for j in range(nj)
            for i in range(ni)
        )
        cls.field = ScalarField3D(ni=ni, nj=nj, nk=nk, values=values)
        cls.mesh = extract_isosurface(cls.field, 1.0)

    def test_all_records_nondifferentiable(self):
        self.assertGreater(len(self.mesh.intersections), 0)
        for rec in self.mesh.intersections:
            self.assertFalse(rec.sensitivity.differentiable)
            self.assertIn(
                rec.sensitivity.reason,
                (NONDIFF_ENDPOINT_ON_ISO, NONDIFF_ZERO_DENOMINATOR),
            )
            self.assertIsNone(rec.sensitivity.d_point_d_va)
            self.assertIsNone(rec.sensitivity.d_point_d_vb)
            self.assertIsNone(rec.sensitivity.d_point_d_iso)

    def test_both_nondiff_reasons_occur(self):
        reasons = {rec.sensitivity.reason for rec in self.mesh.intersections}
        # i=0/2 与 i=1 层之间的穿越边：一端恰 iso；
        # i=1 层内奇偶节点对边：两端都恰 iso 且分母为零。
        self.assertIn(NONDIFF_ENDPOINT_ON_ISO, reasons)
        self.assertIn(NONDIFF_ZERO_DENOMINATOR, reasons)

    def test_zero_denom_edges_are_midpoints(self):
        for rec in self.mesh.intersections:
            if rec.sensitivity.reason != NONDIFF_ZERO_DENOMINATOR:
                continue
            wa, wb = rec.weights
            self.assertEqual(wa, 0.5)
            self.assertEqual(wb, 0.5)


class SafeRampAllDifferentiableTest(unittest.TestCase):
    """通用斜坡（无节点恰 iso、分母均为整数差 1）全部可微且权重合法。"""

    @classmethod
    def setUpClass(cls):
        values = ramp_values(5, 4, 3)
        field = ScalarField3D(ni=5, nj=4, nk=3, values=values)
        cls.mesh = extract_isosurface(field, 4.5)

    def test_every_record_differentiable_and_weights_valid(self):
        self.assertGreater(len(self.mesh.intersections), 0)
        for rec in self.mesh.intersections:
            self.assertTrue(rec.sensitivity.differentiable)
            wa, wb = rec.weights
            self.assertGreater(wa, 0.0)
            self.assertLess(wa, 1.0)
            self.assertGreater(wb, 0.0)
            self.assertLess(wb, 1.0)
            self.assertAlmostEqual(wa + wb, 1.0, places=15)

    def test_empty_mesh_has_empty_records(self):
        field = ScalarField3D(ni=2, nj=2, nk=2, values=(0.0,) * 8)
        mesh = extract_isosurface(field, 5.0)
        self.assertEqual(mesh.intersections, ())


class NodeImpactQueryTest(unittest.TestCase):
    """intersections_for_node：一个输入节点变化影响哪些交点。"""

    @classmethod
    def setUpClass(cls):
        cls.values = ramp_values(3, 3, 3)
        cls.field = ScalarField3D(ni=3, nj=3, nk=3, values=cls.values)
        cls.mesh = extract_isosurface(cls.field, 2.5)

    def test_query_matches_incident_edge_keys(self):
        node = self.field.node_id(1, 1, 0)  # v=2
        impacted = self.mesh.intersections_for_node(node)
        incident = {
            rec.edge_key
            for rec in self.mesh.intersections
            if node in rec.edge_key
        }
        self.assertEqual({rec.edge_key for rec in impacted}, incident)
        self.assertGreaterEqual(len(incident), 2)

    def test_unrelated_node_impacts_no_shared_records(self):
        # (2,2,2) v=6：它只作为角点 7 属于一个立方体，与其余 7 个角点
        # （含体对角线到 (1,1,1) v=3）构成全部四面体边；这些端点值均 >=3
        # > iso=2.5，因此无任何穿越边。
        node = self.field.node_id(2, 2, 2)
        self.assertEqual(self.mesh.intersections_for_node(node), ())

    def test_perturbing_node_moves_exactly_its_intersections(self):
        # 实际改变一个节点：只有以它为端点的交点坐标发生变化。
        node = self.field.node_id(2, 1, 0)  # v=3
        values2 = list(self.values)
        values2[node] += 0.002
        mesh2 = extract_isosurface(
            ScalarField3D(ni=3, nj=3, nk=3, values=tuple(values2)), 2.5
        )
        before = {v.edge_key: v.point for v in self.mesh.vertices}
        after = {v.edge_key: v.point for v in mesh2.vertices}
        self.assertEqual(set(before), set(after))  # 小扰动下拓扑不变
        for key in before:
            moved = any(
                abs(after[key][d] - before[key][d]) > 1e-12 for d in range(3)
            )
            if node in key:
                self.assertTrue(moved, f"incident intersection {key} did not move")
            else:
                self.assertFalse(moved, f"non-incident intersection {key} moved")


if __name__ == "__main__":
    unittest.main()
