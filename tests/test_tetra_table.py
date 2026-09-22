"""TETRA_TABLE 分解表自身的几何性质测试（独立于核心算法）。"""

import unittest

from iso3d import CUBE_CORNERS, TETRA_TABLE


def _det3(a, b, c):
    """三行向量行列式。"""
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


class TetraTableTest(unittest.TestCase):
    def test_six_tetrahedra(self):
        self.assertEqual(len(TETRA_TABLE), 6)

    def test_all_share_main_diagonal(self):
        # 每个四面体都必须包含主对角线端点 0 和 7。
        for tet in TETRA_TABLE:
            self.assertIn(0, tet)
            self.assertIn(7, tet)

    def test_each_tetrahedron_volume_is_one_sixth(self):
        total = 0.0
        for tet in TETRA_TABLE:
            p = [CUBE_CORNERS[c] for c in tet]
            u = tuple(p[1][d] - p[0][d] for d in range(3))
            v = tuple(p[2][d] - p[0][d] for d in range(3))
            w = tuple(p[3][d] - p[0][d] for d in range(3))
            volume = abs(_det3(u, v, w)) / 6.0
            self.assertAlmostEqual(volume, 1.0 / 6.0, places=15)
            total += volume
        self.assertAlmostEqual(total, 1.0, places=15)

    def test_each_cube_face_has_exactly_one_diagonal(self):
        # 收集所有四面体边，筛出立方体 6 个面上的面对角线。
        edges = set()
        for tet in TETRA_TABLE:
            for a in range(4):
                for b in range(a + 1, 4):
                    edges.add(frozenset((tet[a], tet[b])))

        def differing_axes(c1, c2):
            return {d for d in range(3) if c1[d] != c2[d]}

        face_diag_count = {axis: 0 for axis in range(3)}
        for edge in edges:
            c1, c2 = (CUBE_CORNERS[c] for c in edge)
            diff = differing_axes(c1, c2)
            if len(diff) == 2:
                # 面对角线：位于剩下那个坐标轴对应的两个面上。
                face_axis = ({0, 1, 2} - diff).pop()
                face_diag_count[face_axis] += 1
            else:
                # 只允许立方体棱（1 个坐标不同）或体对角线（3 个不同）。
                self.assertIn(len(diff), (1, 3))
        # 每个坐标轴对应一对相对面，共应恰好 2 条面对角线（每面一条）。
        for axis in range(3):
            self.assertEqual(face_diag_count[axis], 2, f"axis {axis}")


if __name__ == "__main__":
    unittest.main()
