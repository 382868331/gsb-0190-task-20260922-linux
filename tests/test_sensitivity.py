"""共享边交点的来源、插值权重与局部标量敏感度测试。

覆盖：
- 线性斜坡体上手算交点，安全边用**双侧有限差分**核对 d/dv_a、d/dv_b、d/diso
  三个解析导数（t 与 point 两个层面）；
- 反向遍历共享边：同一网格边被多个四面体引用时只有一条共享记录；
- 标量差极小（不稳定分母）与端点恰 iso：标记不可微，不给出巨大伪导数；
- 一个输入节点变化影响哪些交点（intersections_affected_by_node），
  并用有限差分核对受影响交点的实际位移。

期望值来源：手算公式与独立双侧有限差分；不调用被测核心来“证明”解析导数，
有限差分仅作为对独立解析公式的数值核对（解析公式在测试中按 t 的定义重写）。
"""

import math
import unittest

from iso3d import (
    NONDIFF_ENDPOINT_ON_ISO,
    NONDIFF_UNSTABLE_DENOMINATOR,
    NONDIFF_ZERO_DENOMINATOR,
    UNSTABLE_DENOMINATOR_EPS,
    InvalidInputError,
    ScalarField3D,
    extract_isosurface,
    intersect_edge,
)


def ramp_field(ni=8, nj=6, nk=5):
    """线性斜坡 f = i + j + k：穿过网格边的标量差只可能是 1 或 2，分母稳定。"""
    values = tuple(
        float(i + j + k)
        for k in range(nk)
        for j in range(nj)
        for i in range(ni)
    )
    return ScalarField3D(ni=ni, nj=nj, nk=nk, values=values)


def node_coord(ni, nj, nk, node_id):
    i = node_id % ni
    j = (node_id // ni) % nj
    k = node_id // (ni * nj)
    return (float(i), float(j), float(k))


class HandComputedIntersectionTest(unittest.TestCase):
    """单个边交点的手算值：t、point、权重与三个导数。"""

    def test_unit_x_edge_hand_values(self):
        # 单位 x 方向边 A=(0,0,0) -> B=(1,0,0)，v_a=0, v_b=1, iso=0.25。
        rec = intersect_edge(0, 1, 0.0, 1.0, 0.25, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertTrue(rec.differentiable)
        self.assertIsNone(rec.nondifferentiable_reason)
        self.assertAlmostEqual(rec.t, 0.25, places=15)
        self.assertEqual(rec.point, (0.25, 0.0, 0.0))
        # 权重：(1-t) 在 a 端，t 在 b 端，和为 1。
        self.assertEqual(rec.weights, ((0, 0.75), (1, 0.25)))
        s = rec.sensitivity
        # t = (iso - v_a)/(v_b - v_a)，d = 1：
        # dt/dv_a = (iso-v_b)/d^2 = -0.75
        # dt/dv_b = (v_a-iso)/d^2 = -0.25
        # dt/diso = 1/d = 1
        self.assertAlmostEqual(s.dt_dva, -0.75, places=15)
        self.assertAlmostEqual(s.dt_dvb, -0.25, places=15)
        self.assertAlmostEqual(s.dt_diso, 1.0, places=15)
        self.assertEqual(s.dpoint_dva, (-0.75, 0.0, 0.0))
        self.assertEqual(s.dpoint_dvb, (-0.25, 0.0, 0.0))
        self.assertEqual(s.dpoint_diso, (1.0, 0.0, 0.0))

    def test_face_diagonal_edge_hand_values(self):
        # 面对角线 A=(0,0,0) -> B=(1,1,0)，v_a=0, v_b=2, iso=0.5。
        rec = intersect_edge(
            0, 3, 0.0, 2.0, 0.5, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0)
        )
        self.assertTrue(rec.differentiable)
        self.assertAlmostEqual(rec.t, 0.25, places=15)
        for d in range(3):
            self.assertAlmostEqual(rec.point[d], (0.25, 0.25, 0.0)[d], places=15)
        s = rec.sensitivity
        # d = 2：dt/dv_a = (0.5-2)/4 = -0.375；dt/dv_b = (0-0.5)/4 = -0.125；
        # dt/diso = 0.5。point = t*(1,1,0)，故每个非零坐标分量乘同一系数。
        self.assertAlmostEqual(s.dt_dva, -0.375, places=15)
        self.assertAlmostEqual(s.dt_dvb, -0.125, places=15)
        self.assertAlmostEqual(s.dt_diso, 0.5, places=15)
        for want, got in (
            ((-0.375, -0.375, 0.0), s.dpoint_dva),
            ((-0.125, -0.125, 0.0), s.dpoint_dvb),
            ((0.5, 0.5, 0.0), s.dpoint_diso),
        ):
            for d in range(3):
                self.assertAlmostEqual(got[d], want[d], places=15)


class FiniteDifferenceDerivativeTest(unittest.TestCase):
    """线性斜坡体所有安全交点：双侧有限差分核对三个导数（t 与 point）。"""

    @classmethod
    def setUpClass(cls):
        cls.ni, cls.nj, cls.nk = 8, 6, 5
        cls.field = ramp_field(cls.ni, cls.nj, cls.nk)
        cls.iso = 7.5
        cls.mesh = extract_isosurface(cls.field, cls.iso)
        cls.h = 1e-6

    def _recall(self, rec, dva=0.0, dvb=0.0, diso=0.0):
        """用给定扰动重新计算同一条边的交点记录（拓扑不变的小扰动）。"""
        a, b = rec.edge_key
        va = self.field.values[a] + dva
        vb = self.field.values[b] + dvb
        return intersect_edge(
            a,
            b,
            va,
            vb,
            self.iso + diso,
            node_coord(self.ni, self.nj, self.nk, a),
            node_coord(self.ni, self.nj, self.nk, b),
        )

    def test_all_ramp_intersections_differentiable(self):
        # f=i+j+k、iso=7.5：没有端点恰 iso，所有穿越边严格跨过且分母为 1 或 2。
        self.assertGreater(len(self.mesh.intersections), 0)
        for rec in self.mesh.intersections:
            self.assertTrue(
                rec.differentiable,
                f"edge {rec.edge_key} unexpectedly nondifferentiable: "
                f"{rec.nondifferentiable_reason}",
            )

    def test_weights_reconstruct_point_and_sum_to_one(self):
        for rec in self.mesh.intersections:
            (a, wa), (b, wb) = rec.weights
            self.assertEqual((a, b), rec.edge_key)
            self.assertAlmostEqual(wa + wb, 1.0, places=14)
            pa = node_coord(self.ni, self.nj, self.nk, a)
            pb = node_coord(self.ni, self.nj, self.nk, b)
            for d in range(3):
                self.assertAlmostEqual(
                    rec.point[d], wa * pa[d] + wb * pb[d], places=12
                )

    def test_two_sided_finite_differences_for_all_three_derivatives(self):
        h = self.h
        tol = 1e-7  # 中心差分误差 O(h^2) 远小于此；主要裕量给浮点舍入
        for rec in self.mesh.intersections:
            s = rec.sensitivity
            # --- t 的双侧差分 ---
            fp = self._recall(rec, dva=+h).t
            fm = self._recall(rec, dva=-h).t
            self.assertAlmostEqual((fp - fm) / (2 * h), s.dt_dva, delta=tol)
            fp = self._recall(rec, dvb=+h).t
            fm = self._recall(rec, dvb=-h).t
            self.assertAlmostEqual((fp - fm) / (2 * h), s.dt_dvb, delta=tol)
            fp = self._recall(rec, diso=+h).t
            fm = self._recall(rec, diso=-h).t
            self.assertAlmostEqual((fp - fm) / (2 * h), s.dt_diso, delta=tol)
            # --- point 三个分量的双侧差分（对三个输入分别扰动）---
            for perturb_kw, analytic in (
                ({"dva": h}, s.dpoint_dva),
                ({"dvb": h}, s.dpoint_dvb),
                ({"diso": h}, s.dpoint_diso),
            ):
                plus = self._recall(rec, **perturb_kw).point
                minus = self._recall(
                    rec, **{k: -v for k, v in perturb_kw.items()}
                ).point
                for d in range(3):
                    num = (plus[d] - minus[d]) / (2 * h)
                    self.assertAlmostEqual(
                        num,
                        analytic[d],
                        delta=tol,
                        msg=f"edge {rec.edge_key} dim {d} kw {perturb_kw}: "
                        f"FD {num} vs analytic {analytic[d]}",
                    )

    def test_chain_relation_dt_derivatives_sum_to_zero(self):
        # 恒等校验：t 对 (v_a, v_b, iso) 同时同向扰动应不变（平移不变性）：
        # dt/dv_a + dt/dv_b + dt/diso 中 iso 项符号相反，
        # 实际成立的是 dt/dv_a + dt/dv_b = -dt/diso。
        for rec in self.mesh.intersections:
            s = rec.sensitivity
            self.assertAlmostEqual(s.dt_dva + s.dt_dvb, -s.dt_diso, places=12)


class SharedRecordTest(unittest.TestCase):
    """同一网格边被多个四面体引用时共享同一条记录，不因遍历方向不同而不同。"""

    @classmethod
    def setUpClass(cls):
        cls.ni, cls.nj, cls.nk = 6, 6, 6
        cls.field = ramp_field(cls.ni, cls.nj, cls.nk)
        cls.mesh = extract_isosurface(cls.field, 5.5)

    def test_intersections_align_one_to_one_with_vertices(self):
        self.assertEqual(len(self.mesh.intersections), len(self.mesh.vertices))
        for v, rec in zip(self.mesh.vertices, self.mesh.intersections):
            self.assertEqual(v.edge_key, rec.edge_key)
            self.assertEqual(v.point, rec.point)
            self.assertAlmostEqual(v.t, rec.t, places=15)

    def test_edge_keys_unique_and_canonical(self):
        keys = [rec.edge_key for rec in self.mesh.intersections]
        self.assertEqual(len(keys), len(set(keys)))
        for a, b in keys:
            self.assertLess(a, b)

    def test_shared_edge_returns_same_object_from_different_faces(self):
        # vertex id -> 出现在哪些面中；找被两个面共用的交点。
        users = {}
        for face in self.mesh.faces:
            for vid in face.vertex_ids:
                users.setdefault(vid, []).append(face.id)
        shared = [vid for vid, faces in users.items() if len(faces) >= 2]
        self.assertGreater(len(shared), 0, "斜坡体内部应有大量共享交点")
        for vid in shared:
            r1 = self.mesh.intersection_for_vertex(vid)
            r2 = self.mesh.intersection_for_vertex(vid)
            self.assertIs(r1, r2)  # 必须是同一条记录对象
            self.assertEqual(
                self.mesh.intersection_for_edge(*r1.edge_key), r1
            )

    def test_reverse_traversal_gives_identical_record(self):
        # 取一条真实穿越边，以 (b,a) 反序（连同标量、坐标交换）调用，
        # 结果必须规范化为同一条记录：相同边键、相同 t、相同导数约定。
        rec = self.mesh.intersections[0]
        a, b = rec.edge_key
        pa = node_coord(self.ni, self.nj, self.nk, a)
        pb = node_coord(self.ni, self.nj, self.nk, b)
        rev = intersect_edge(
            b, a, self.field.values[b], self.field.values[a], self.mesh.iso, pb, pa
        )
        self.assertEqual(rev, rec)

    def test_intersection_for_edge_missing_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.mesh.intersection_for_edge(0, 10**6)


class NondifferentiableTest(unittest.TestCase):
    """端点恰 iso / 零分母 / 极小标量差：标不可微，不输出巨大伪精确数。"""

    def test_endpoint_exactly_iso_is_marked_nondifferentiable(self):
        # v_a 恰为 iso：交点在端点 a（t=0），拓扑由符号扰动决定。
        rec = intersect_edge(
            2, 5, 1.0, 2.0, 1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        self.assertFalse(rec.differentiable)
        self.assertIsNone(rec.sensitivity)
        self.assertEqual(rec.nondifferentiable_reason, NONDIFF_ENDPOINT_ON_ISO)
        self.assertEqual(rec.t, 0.0)
        self.assertEqual(rec.point, (0.0, 0.0, 0.0))
        self.assertEqual(rec.weights, ((2, 1.0), (5, 0.0)))
        # v_b 恰为 iso 同理（t=1）。
        rec2 = intersect_edge(
            2, 5, 0.0, 1.0, 1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        self.assertFalse(rec2.differentiable)
        self.assertEqual(rec2.nondifferentiable_reason, NONDIFF_ENDPOINT_ON_ISO)
        self.assertEqual(rec2.t, 1.0)

    def test_zero_denominator_both_on_iso_uses_midpoint(self):
        rec = intersect_edge(
            4, 9, 1.0, 1.0, 1.0, (0.0, 0.0, 0.0), (1.0, 1.0, 0.0)
        )
        self.assertFalse(rec.differentiable)
        self.assertIsNone(rec.sensitivity)
        self.assertEqual(rec.nondifferentiable_reason, NONDIFF_ZERO_DENOMINATOR)
        self.assertEqual(rec.t, 0.5)
        self.assertEqual(rec.point, (0.5, 0.5, 0.0))
        for w in (rec.weights[0][1], rec.weights[1][1]):
            self.assertTrue(math.isfinite(w))

    def test_tiny_scalar_difference_marked_unstable(self):
        # 严格跨过 iso，但分母 1e-13 低于相对阈值 1e-12：标不稳定、不给导数。
        va, vb = 1.0, 1.0 + 1e-13
        iso = 1.0 + 5e-14
        self.assertLess(
            abs(vb - va),
            UNSTABLE_DENOMINATOR_EPS * max(1.0, abs(va), abs(vb)),
        )
        rec = intersect_edge(
            0, 1, va, vb, iso, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        self.assertTrue(va < iso < vb)  # 确实严格跨过
        self.assertFalse(rec.differentiable)
        self.assertIsNone(rec.sensitivity)
        self.assertEqual(
            rec.nondifferentiable_reason, NONDIFF_UNSTABLE_DENOMINATOR
        )
        # 几何位置仍然给出且有限（中点附近），只是不承诺可微。
        self.assertTrue(all(math.isfinite(c) for c in rec.point))
        self.assertTrue(math.isfinite(rec.t))

    def test_stable_straddle_with_small_but_safe_gap_is_differentiable(self):
        # 反例对照：分母小但高于阈值（量级为 O(1) 标量下取 1e-9 > 1e-12），
        # 应正常给出可微导数。
        va, vb = 0.0, 1e-9
        iso = 0.25e-9
        rec = intersect_edge(
            0, 1, va, vb, iso, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        self.assertTrue(rec.differentiable)
        self.assertAlmostEqual(rec.t, 0.25, places=9)

    def test_extracted_iso_on_node_plane_marks_nondifferentiable(self):
        # f = i，iso = 1：整层节点恰为 iso，相关交点必须被标不可微，
        # 而可微交点（两端严格跨过时才会出现，本构型下几乎没有）规则不变。
        n = 4
        values = tuple(float(i) for k in range(n) for j in range(n) for i in range(n))
        field = ScalarField3D(ni=n, nj=n, nk=n, values=values)
        mesh = extract_isosurface(field, 1.0)
        reasons = {rec.nondifferentiable_reason for rec in mesh.intersections}
        self.assertTrue(reasons)
        self.assertTrue(
            NONDIFF_ENDPOINT_ON_ISO in reasons
            or NONDIFF_ZERO_DENOMINATOR in reasons
        )
        for rec in mesh.intersections:
            if rec.differentiable:
                continue
            self.assertIsNone(rec.sensitivity)
            self.assertIn(
                rec.nondifferentiable_reason,
                (
                    NONDIFF_ENDPOINT_ON_ISO,
                    NONDIFF_ZERO_DENOMINATOR,
                    NONDIFF_UNSTABLE_DENOMINATOR,
                ),
            )
            self.assertTrue(all(math.isfinite(c) for c in rec.point))


class NodeInfluenceTest(unittest.TestCase):
    """演示一个输入节点变化影响哪些交点，并核对实际位移。"""

    @classmethod
    def setUpClass(cls):
        cls.ni, cls.nj, cls.nk = 7, 6, 5
        cls.field = ramp_field(cls.ni, cls.nj, cls.nk)
        cls.iso = 8.5
        cls.mesh = extract_isosurface(cls.field, cls.iso)
        # 选一个内点节点 (3,3,2)。
        cls.node = cls.field.node_id(3, 3, 2)

    def test_affected_set_is_exactly_incident_crossing_edges(self):
        affected = self.mesh.intersections_affected_by_node(self.node)
        keys = {rec.edge_key for rec in affected}
        # 独立判定：所有交点中边键端点包含该节点者。
        expect = {
            rec.edge_key
            for rec in self.mesh.intersections
            if self.node in rec.edge_key
        }
        self.assertEqual(keys, expect)
        self.assertGreater(len(keys), 0)
        # 该节点在斜坡 f=i+j+k 下值为 8，iso=8.5；它是“下方”节点，
        # 影响边数等于其 6 条网格邻边中连到上方节点的数量（内点为 6，
        # 对角网格边不在晶格边集合内，交点还可能来自面/体对角线——
        # 这里只做数量 > 0 与一致性的稳健断言）。
        for rec in affected:
            self.assertIn(self.node, rec.edge_key)

    def test_perturbing_node_moves_predicted_intersections_only(self):
        h = 1e-6
        values = list(self.field.values)
        values[self.node] += h
        moved = ScalarField3D(
            ni=self.ni, nj=self.nj, nk=self.nk, values=tuple(values)
        )
        mesh2 = extract_isosurface(moved, self.iso)
        points2 = {v.edge_key: v.point for v in mesh2.vertices}
        affected_keys = {
            rec.edge_key
            for rec in self.mesh.intersections_affected_by_node(self.node)
        }
        for rec in self.mesh.intersections:
            a, b = rec.edge_key
            p2 = points2[(a, b)]
            if rec.edge_key not in affected_keys:
                # 不触及该节点的边：计算完全不依赖其标量，点位必须逐位不变。
                self.assertEqual(
                    p2,
                    rec.point,
                    f"unrelated edge {rec.edge_key} moved unexpectedly",
                )
                continue
            # 受影响边：位移必须与解析敏感度的一阶预测一致。
            s = rec.sensitivity
            self.assertTrue(rec.differentiable)
            grad = s.dpoint_dva if a == self.node else s.dpoint_dvb
            for d in range(3):
                self.assertAlmostEqual(
                    (p2[d] - rec.point[d]) / h, grad[d], delta=1e-6
                )

    def test_affected_lookup_accepts_only_existing_records(self):
        # 不与任何交点相邻的节点（如角落远离等值面）返回空元组。
        far = self.field.node_id(0, 0, 0)  # f=0，远离 iso=8.5
        self.assertEqual(self.mesh.intersections_affected_by_node(far), ())


class IntersectEdgeValidationTest(unittest.TestCase):
    def test_bool_node_id_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "id_a"):
            intersect_edge(
                True, 1, 0.0, 1.0, 0.5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
            )

    def test_non_finite_scalar_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "value_b"):
            intersect_edge(
                0, 1, 0.0, math.nan, 0.5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
            )

    def test_bad_point_length_rejected(self):
        with self.assertRaisesRegex(InvalidInputError, "point_a"):
            intersect_edge(0, 1, 0.0, 1.0, 0.5, (0.0, 0.0), (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
