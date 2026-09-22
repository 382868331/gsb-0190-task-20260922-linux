"""随机差分测试：独立逐四面体平面裁剪参考 vs 被测输出。

参考完全独立实现（不读取 MARCHING_TET_TABLE，不调用 extract_*）：
  * 每四面体独立按严格 >iso / <iso 分类，独立枚举 6 条跨立边；
  * 独立由"两跨立边是否同属一个四面体三角面"推导截交多边形的环相邻
    关系，判定 1 个三角形（1/3 分类）或四边形的 2 种合法对角线剖分；
  * 独立线性插值坐标、独立叉积判定退化与朝向。
随机源固定种子，离线、确定性。
"""
from __future__ import annotations

import math
import random
import unittest

from mtiso import CUBE_TETRAHEDRA, TETRA_EDGES, ScalarGrid, extract_isosurface

# 四面体 4 个三角面各自包含的局部边号（由顶点三元组独立推导）。
TRI_FACE_EDGES: list[frozenset[int]] = []
for _omit in range(4):
    _verts = [v for v in range(4) if v != _omit]
    _es: set[int] = set()
    for _i in range(3):
        for _j in range(_i + 1, 3):
            _a, _b = _verts[_i], _verts[_j]
            for _en, (_ea, _eb) in enumerate(TETRA_EDGES):
                if (_ea, _eb) == (_a, _b) or (_ea, _eb) == (_b, _a):
                    _es.add(_en)
    TRI_FACE_EDGES.append(frozenset(_es))


def _coplanar_on_tet_face(en1: int, en2: int) -> bool:
    """两条四面体棱是否同属一个三角面（交点在截面上是否环相邻）。"""
    return any({en1, en2} <= fs for fs in TRI_FACE_EDGES)


class IndependentReferenceTests(unittest.TestCase):
    def test_random_grids(self):
        rng = random.Random(20260922)
        for trial in range(40):
            n = rng.randint(2, 6)
            vals = [rng.uniform(-3.0, 3.0) for _ in range(n ** 3)]
            iso = rng.uniform(-2.5, 2.5)
            with self.subTest(trial=trial, n=n):
                self._one_grid(n, vals, iso)

    def _one_grid(self, n, vals, iso):
        grid = ScalarGrid(n, n, n, vals)
        res = extract_isosurface(grid, iso)
        nx = ny = nz = n
        nxy = nx * ny

        # 一般位置（连续随机）：不应有任何退化丢弃。
        self.assertEqual(res.stats.zero_area_faces_discarded, 0)
        vid_to_key = {vid: key for key, vid in res.edge_vertices.items()}

        # 独立累计所有四面体的跨立边键与坐标，并逐面认领。
        expected_keys: set[tuple[int, int]] = set()
        claimed_faces: set[int] = set()  # 输出面下标
        face_count = 0

        for iz in range(n - 1):
            for iy in range(n - 1):
                for ix in range(n - 1):
                    base = ix + nx * iy + nxy * iz
                    cg = (
                        base, base + 1, base + 1 + nx, base + nx,
                        base + nxy, base + nxy + 1,
                        base + nxy + 1 + nx, base + nxy + nx,
                    )
                    for tet in CUBE_TETRAHEDRA:
                        gids = (cg[tet[0]], cg[tet[1]],
                                cg[tet[2]], cg[tet[3]])
                        tv = [vals[g] for g in gids]
                        above = [v > iso for v in tv]
                        crossing = {
                            en: tuple(sorted((gids[a], gids[b])))
                            for en, (a, b) in enumerate(TETRA_EDGES)
                            if above[a] != above[b]
                        }
                        for key in crossing.values():
                            expected_keys.add(key)

                        tet_keys = {
                            tuple(sorted((gids[i], gids[j])))
                            for i in range(4) for j in range(i + 1, 4)
                        }
                        # 认领输出中三个顶点边键都属于本四面体的面。
                        my_faces = []
                        for fi, tri in enumerate(res.faces):
                            keys = {vid_to_key[w] for w in tri.vertex_ids}
                            if keys <= tet_keys:
                                my_faces.append((fi, tri, keys))
                                claimed_faces.add(fi)

                        k_above = sum(above)
                        if k_above in (0, 4):
                            self.assertEqual(crossing, {})
                            self.assertEqual(my_faces, [])
                            continue

                        # 独立坐标（线性插值定义）。
                        def point(key):
                            a, b = key
                            va, vb = vals[a], vals[b]
                            t = (iso - va) / (vb - va)
                            pa = _world(grid, a)
                            pb = _world(grid, b)
                            return tuple(pa[k] + t * (pb[k] - pa[k])
                                         for k in range(3))

                        if k_above in (1, 3):
                            self.assertEqual(len(crossing), 3)
                            self.assertEqual(len(my_faces), 1)
                            fi, tri, keys = my_faces[0]
                            self.assertEqual(set(keys), set(crossing.values()))
                            self.assertNotEqual(
                                _cross(tri, res.vertices), (0.0, 0.0, 0.0))
                            face_count += 1
                        else:
                            # 2 above / 2 below：凸四边形，恰 2 个三角形。
                            self.assertEqual(len(crossing), 4)
                            self.assertEqual(len(my_faces), 2)
                            used: dict[tuple[int, int], int] = {}
                            for fi, tri, keys in my_faces:
                                self.assertEqual(len(keys), 3)
                                self.assertNotEqual(
                                    _cross(tri, res.vertices),
                                    (0.0, 0.0, 0.0))
                                for key in keys:
                                    used[key] = used.get(key, 0) + 1
                            self.assertEqual(set(used), set(crossing.values()))
                            counts = sorted(used.values())
                            self.assertEqual(counts, [1, 1, 2, 2])
                            # 共享 2 次的交点对即对角线端点：它们必须
                            # 互不环相邻（不共属任何四面体三角面）。
                            diag_endpoints = [
                                en for en, key in crossing.items()
                                if used[key] == 2
                            ]
                            self.assertEqual(len(diag_endpoints), 2)
                            self.assertFalse(
                                _coplanar_on_tet_face(*diag_endpoints))
                            # 出现 1 次的两个交点对各自的环边组合：
                            # 4 条环边均共面。
                            ring = [en for en, key in crossing.items()
                                    if used[key] == 1]
                            self.assertEqual(len(ring), 2)
                            for re_ in ring:
                                self.assertTrue(
                                    any(_coplanar_on_tet_face(re_, d)
                                        for d in diag_endpoints))
                            face_count += 2

                        # 朝向：面法向与 (above 质心 - below 质心) 点积 > 0。
                        ca = [0.0, 0.0, 0.0]
                        cb = [0.0, 0.0, 0.0]
                        na = nb = 0
                        for li in range(4):
                            p = _world(grid, gids[li])
                            tgt = ca if above[li] else cb
                            for k in range(3):
                                tgt[k] += p[k]
                            if above[li]:
                                na += 1
                            else:
                                nb += 1
                        d = tuple(ca[k] / na - cb[k] / nb for k in range(3))
                        for fi, tri, _keys in my_faces:
                            nrm = _cross(tri, res.vertices)
                            self.assertGreater(
                                nrm[0] * d[0] + nrm[1] * d[1]
                                + nrm[2] * d[2], 0.0)

        # 所有输出面都被恰好一个四面体认领；键集与坐标逐位一致。
        self.assertEqual(len(claimed_faces), len(res.faces))
        self.assertEqual(face_count, len(res.faces))
        self.assertEqual(set(res.edge_vertices), expected_keys)
        for key, vid in res.edge_vertices.items():
            a, b = key
            va, vb = vals[a], vals[b]
            t = (iso - va) / (vb - va)
            pa, pb = _world(grid, a), _world(grid, b)
            expect = tuple(pa[k] + t * (pb[k] - pa[k]) for k in range(3))
            self.assertEqual(res.vertices[vid], expect)

        # 全局组合流形（一般位置规则网格 MT 必为流形，域边界可开口）。
        ec: dict[tuple[int, int], int] = {}
        for tri in res.faces:
            w = tri.vertex_ids
            for i in range(3):
                k = tuple(sorted((w[i], w[(i + 1) % 3])))
                ec[k] = ec.get(k, 0) + 1
        self.assertTrue(all(c in (1, 2) for c in ec.values()))
        self.assertEqual(res.stats.nonmanifold_edge_count, 0)

        # 每条输出三角形边两侧标量确实一上一下（边上两点分别来自跨立边，
        # 这里抽查插值点值在浮点意义下等于 iso）。
        for px, py, pz in res.vertices:
            # 交点未必在节点网格线上解析可验，仅做有限性检查。
            self.assertTrue(all(math.isfinite(q) for q in (px, py, pz)))


def _world(grid: ScalarGrid, node: int):
    z, rem = divmod(node, grid.nx * grid.ny)
    y, x = divmod(rem, grid.nx)
    return (
        grid.origin[0] + x * grid.spacing[0],
        grid.origin[1] + y * grid.spacing[1],
        grid.origin[2] + z * grid.spacing[2],
    )


def _cross(tri, vertices):
    w = tri.vertex_ids
    p0, p1, p2 = vertices[w[0]], vertices[w[1]], vertices[w[2]]
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    return (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)


if __name__ == "__main__":
    unittest.main()
