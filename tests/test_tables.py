"""分解表与面片表的结构性测试（依据几何/组合学独立核验）。"""
from __future__ import annotations

import unittest

from mtiso.core import (
    CUBE_CORNERS,
    CUBE_TETRAHEDRA,
    MARCHING_TET_TABLE,
    TETRA_EDGES,
)


def _det3(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


class DecompositionTableTests(unittest.TestCase):
    def test_six_tetrahedra_explicit(self):
        self.assertEqual(len(CUBE_TETRAHEDRA), 6)
        for tet in CUBE_TETRAHEDRA:
            self.assertEqual(len(tet), 4)
            self.assertEqual(len(set(tet)), 4)
            for c in tet:
                self.assertIsInstance(c, int)
                self.assertIn(c, range(8))

    def test_unified_main_diagonal(self):
        # 每个四面体都含统一主对角线角点 0 与 6。
        for tet in CUBE_TETRAHEDRA:
            self.assertIn(0, tet)
            self.assertIn(6, tet)

    def test_six_non_diagonal_cube_edges_covered_once(self):
        # 不含角点 0/6 的 6 条立方体棱必须恰好在一个四面体中出现一次。
        middle_edges = {(1, 2), (2, 3), (3, 7), (4, 7), (4, 5), (1, 5)}
        seen = []
        for tet in CUBE_TETRAHEDRA:
            for i in range(4):
                for j in range(i + 1, 4):
                    e = tuple(sorted((tet[i], tet[j])))
                    if e in middle_edges:
                        seen.append(e)
        self.assertEqual(sorted(seen), sorted(middle_edges))

    def test_volumes_partition_unit_cube(self):
        signs = []
        volumes = []
        for tet in CUBE_TETRAHEDRA:
            p = [CUBE_CORNERS[c] for c in tet]
            d = _det3(
                (p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]),
                (p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]),
                (p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]),
            )
            self.assertEqual(abs(d), 1)  # 每个四面体体积恰为 1/6
            signs.append(1 if d > 0 else -1)
            volumes.append(abs(d) / 6.0)
        self.assertEqual(len(set(signs)), 1)  # 绕序统一
        self.assertAlmostEqual(sum(volumes), 1.0)  # 铺满单位立方体

    def test_face_diagonals_consistent_across_shared_cube_faces(self):
        # 独立枚举：每个四面体在立方体 6 个边界面上贡献的三角形边，
        # 汇总后每个面应被一条共享对角线剖成两个三角形；
        # 且相邻立方体（平移一个单位）在共享面上的对角线一致。
        faces = {
            "x0": (0, (0, 3, 7, 4)),
            "x1": (1, (1, 2, 6, 5)),
            "y0": (1, (0, 1, 5, 4)),
            "y1": (1, (3, 2, 6, 7)),
            "z0": (2, (0, 1, 2, 3)),
            "z1": (2, (4, 5, 6, 7)),
        }
        face_triangulation = {}
        for name, (axis, corners) in faces.items():
            tris = []
            for tet in CUBE_TETRAHEDRA:
                on = [c for c in tet if c in corners]
                if len(on) == 3:  # 四面体的一个三角形面落在该立方体面上
                    tris.append(tuple(sorted(on)))
            self.assertEqual(len(tris), 2, name)
            diag = tuple(sorted(set(tris[0]) & set(tris[1])))
            self.assertEqual(len(diag), 2, name)  # 两三角形恰共享一条对角线
            edge_union = set(tris[0]) | set(tris[1])
            self.assertEqual(edge_union, set(corners))
            face_triangulation[name] = (axis, corners, diag)

        # 低面平移后必须等于对应高面的剖分：
        #   z0 + z单位 -> 相邻立方体的 z1 面；逐对核对对角端点坐标。
        def corner_coord(c):
            return CUBE_CORNERS[c]

        for low, high, axis in (("z0", "z1", 2), ("y0", "y1", 1), ("x0", "x1", 0)):
            _, lc, ld = face_triangulation[low]
            _, hc, hd = face_triangulation[high]
            low_translated = tuple(
                tuple(corner_coord(c)[k] + (1 if k == axis else 0) for k in range(3))
                for c in ld
            )
            high_coords = {corner_coord(c) for c in hd}
            self.assertEqual(set(low_translated), high_coords)


class TetraEdgeTableTests(unittest.TestCase):
    def test_six_tetra_edges(self):
        self.assertEqual(len(TETRA_EDGES), 6)
        for i, (a, b) in enumerate(TETRA_EDGES):
            self.assertLess(a, b)
            self.assertIn(a, range(4))
            self.assertIn(b, range(4))

    def test_table_entries_use_only_crossing_edges(self):
        # 对 16 种符号配置：表中三角形引用的局部边必须恰为跨立边
        # （两端 above/below 分类不同），且整体构成 0/1/2 个三角形。
        for mask in range(16):
            above = {i: bool(mask & (1 << i)) for i in range(4)}
            crossing = {
                en
                for en, (a, b) in enumerate(TETRA_EDGES)
                if above[a] != above[b]
            }
            entry = MARCHING_TET_TABLE[mask]
            used = {en for tri in entry for en in tri}
            self.assertEqual(used, crossing, f"mask={mask:04b}")
            n_above = mask.bit_count()
            if n_above in (0, 4):
                self.assertEqual(entry, ())
            elif n_above in (1, 3):
                self.assertEqual(len(entry), 1)
                self.assertEqual(len(entry[0]), 3)
            else:
                # 2 above / 2 below：四边形截交，恰 2 个三角形，共享
                # 对角线的两个端点（对应 2 个跨立四面体边号，各出现 2 次），
                # 另 2 条环边各出现 1 次；共恰用 4 条跨立边。
                self.assertEqual(len(entry), 2)
                s0, s1 = set(entry[0]), set(entry[1])
                self.assertEqual(len(s0 & s1), 2)
                self.assertEqual(len(s0 | s1), 4)
                from collections import Counter
                cnt = Counter(e for tri in entry for e in tri)
                self.assertEqual(sorted(cnt.values()), [1, 1, 2, 2])

    def test_table_covers_all_masks(self):
        self.assertEqual(set(MARCHING_TET_TABLE), set(range(16)))


if __name__ == "__main__":
    unittest.main()
