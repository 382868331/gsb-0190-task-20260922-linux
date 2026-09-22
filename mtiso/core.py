"""mtiso — 体素等值面的共享边交点提取（marching tetrahedra）。

公开 API 见 :data:`__all__`。本模块只依赖 Python 3.14 标准库。

契约摘要（详见 README）:
  * 网格节点数上限 16 x 16 x 16（每维 2..16）；值按 x + nx*(y + ny*z) 行主序存储。
  * 每个立方体沿统一主对角线 (局部角点 0 -> 6) 剖分为 6 个四面体（见
    :data:`CUBE_TETRAHEDRA`），所有交点以全局网格边键 (min_node_id, max_node_id)
    共享，同一条网格边只插值一次。
  * 值恰等于 iso 的节点按全局节点 ID 奇偶作符号扰动决定拓扑归属（偶 ID 归
    "标量增大侧"/above），几何位置不变（交点仍是线性插值结果）。
  * 三角形绕序统一校正为面朝标量增大方向；几何零面积三角形（叉积精确为 0）
    丢弃并记录在结果中。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = [
    "MAX_GRID_N",
    "MIN_GRID_N",
    "CUBE_CORNERS",
    "CUBE_TETRAHEDRA",
    "TETRA_EDGES",
    "MARCHING_TET_TABLE",
    "ScalarGrid",
    "Triangle",
    "DiscardedFace",
    "MeshStats",
    "MeshResult",
    "extract_isosurface",
]

#: 每维网格节点数上限（题面给定的接口边界）。
MAX_GRID_N = 16
#: 每维网格节点数下限（少于 2 个节点不存在任何体素）。
MIN_GRID_N = 2

#: 立方体 8 个局部角点的 (dx, dy, dz)，角点编号 0..7。
#:
#:        7--------6
#:       /|       /|
#:      4--------5 |
#:      | 3------|-2
#:      |/       |/
#:      0--------1
CUBE_CORNERS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),  # 0
    (1, 0, 0),  # 1
    (1, 1, 0),  # 2
    (0, 1, 0),  # 3
    (0, 0, 1),  # 4
    (1, 0, 1),  # 5
    (1, 1, 1),  # 6
    (0, 1, 1),  # 7
)

#: 立方体 -> 6 四面体分解表（统一主对角线 0-6）。
#:
#: 每个四元组是 :data:`CUBE_CORNERS` 的局部角点号；每个四面体都含主对角
#: 0-6，且对应 6 条不与角点 0/6 相接的立方体棱之一（1-2, 2-3, 3-7, 7-4,
#: 4-5, 5-1）。六个四面体体积各为 1/6，互不重叠且铺满单位立方体；在立方
#: 体间共享面上两侧剖分对角线一致（统一分解，不随位置翻转）。
CUBE_TETRAHEDRA: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 6),  # 棱 1-2
    (0, 2, 3, 6),  # 棱 2-3
    (0, 3, 7, 6),  # 棱 3-7
    (0, 7, 4, 6),  # 棱 7-4
    (0, 4, 5, 6),  # 棱 4-5
    (0, 5, 1, 6),  # 棱 5-1
)

#: 四面体局部 4 顶点之间的 6 条边（局部边号 -> 局部顶点对）。
TETRA_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),  # 边 0
    (0, 2),  # 边 1
    (0, 3),  # 边 2
    (1, 2),  # 边 3
    (1, 3),  # 边 4
    (2, 3),  # 边 5
)

#: 四面体面片表。4 位掩码 bit i=1 表示局部顶点 i 分类为 above（标量 >= iso
#: 且按符号扰动归 above）。每项给出 0、1 或 2 个三角形，三角形以三个
#: :data:`TETRA_EDGES` 局部边号表示（交点在这些边上）。
#:
#: 掩码 0/15 无面；1-above 与 3-above（单 below）各 1 个三角形；2-above
#: 为四边形截交，剖成 2 个三角形（共用 1-4 项所列对角线）。3-above 项复用
#: 其互补（单 below）情形的跨立边连接结构；最终绕序由提取阶段的几何定向
#: 步骤统一校正为面朝 above，故表中只要求连接结构正确。
MARCHING_TET_TABLE: Mapping[int, Sequence[Sequence[int]]] = {
    0: (),
    15: (),
    # --- 恰 1 个 above 顶点 ---
    1: ((0, 1, 2),),          # above {0}: 边 01,02,03
    2: ((0, 3, 4),),          # above {1}: 边 01,12,13
    4: ((1, 3, 5),),          # above {2}: 边 02,12,23
    8: ((2, 4, 5),),          # above {3}: 边 03,13,23
    # --- 恰 2 个 above 顶点：四边形截交 -> 2 三角形（环形序）---
    3: ((1, 2, 4), (1, 4, 3)),    # above {0,1}: 02,03,13 | 02,13,12
    5: ((0, 3, 5), (0, 5, 2)),    # above {0,2}: 01,12,23 | 01,23,03
    9: ((0, 4, 5), (0, 5, 1)),    # above {0,3}: 01,13,23 | 01,23,02
    6: ((0, 1, 5), (0, 5, 4)),    # above {1,2}: 01,02,23 | 01,23,13
    10: ((3, 5, 2), (3, 2, 0)),   # above {1,3}: 12,23,03 | 12,03,01
    12: ((1, 3, 4), (1, 4, 2)),   # above {2,3}: 02,12,13 | 02,13,03
    # --- 恰 3 个 above 顶点（单 below）：跨立边即与该 below 顶点相接的 3 条
    #     四面体棱；绕序由提取阶段的几何定向步骤统一校正 ---
    14: ((0, 1, 2),),         # below {0}: 边 01,02,03
    13: ((0, 3, 4),),         # below {1}: 边 01,12,13
    11: ((1, 3, 5),),         # below {2}: 边 02,12,23
    7: ((2, 4, 5),),          # below {3}: 边 03,13,23
}


def _reject_bool(name: str, value: object) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name}: bool is not accepted as an integer parameter "
                        f"(got {value!r})")


def _finite_number(name: str, value: object) -> float:
    """校验有限实数输入；bool 按整数参数规则拒绝，NaN/Infinity 拒绝。"""
    _reject_bool(name, value)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name}: expected a real number, got "
                        f"{type(value).__name__}")
    f = float(value)
    if math.isnan(f):
        raise ValueError(f"{name}: NaN is not allowed")
    if math.isinf(f):
        raise ValueError(f"{name}: infinity is not allowed")
    return f


def _bounded_int(name: str, value: object, lo: int, hi: int) -> int:
    _reject_bool(name, value)
    if not isinstance(value, int):
        raise TypeError(f"{name}: expected int, got {type(value).__name__}")
    if not lo <= value <= hi:
        raise ValueError(f"{name}: expected integer in [{lo}, {hi}], got {value}")
    return value


@dataclass(frozen=True)
class ScalarGrid:
    """规则轴对齐标量节点网格。

    :param nx, ny, nz: 每维节点数，各为 ``[2, 16]`` 的整数（拒绝 bool）。
    :param values: 节点标量，长度须恰为 ``nx*ny*nz``，按
        ``id = ix + nx*(iy + ny*iz)`` 行主序；元素须为有限实数
        （拒绝 NaN/Infinity 与 bool），错误消息带一维下标与 (ix,iy,iz)。
    :param origin: 网格原点（节点 (0,0,0) 的世界坐标），默认 (0,0,0)。
    :param spacing: 每维步长（各分量须严格为正的有限实数），默认 (1,1,1)。
    """

    nx: int
    ny: int
    nz: int
    values: tuple[float, ...]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        nx = _bounded_int("nx", self.nx, MIN_GRID_N, MAX_GRID_N)
        ny = _bounded_int("ny", self.ny, MIN_GRID_N, MAX_GRID_N)
        nz = _bounded_int("nz", self.nz, MIN_GRID_N, MAX_GRID_N)

        if isinstance(self.values, (str, bytes)) or not isinstance(
            self.values, Sequence
        ):
            raise TypeError("values: expected a sequence of real numbers")
        expected = nx * ny * nz
        if len(self.values) != expected:
            raise ValueError(
                f"values: length {len(self.values)} does not match "
                f"nx*ny*nz = {expected}"
            )
        out: list[float] = [0.0] * expected
        for i, raw in enumerate(self.values):
            _reject_bool(f"values[{i}]", raw)
            if not isinstance(raw, (int, float)):
                raise TypeError(
                    f"values[{i}]: expected real number, got "
                    f"{type(raw).__name__}"
                )
            f = float(raw)
            if math.isnan(f) or math.isinf(f):
                izz, rem = divmod(i, nx * ny)
                iyy, ixx = divmod(rem, nx)
                kind = "NaN" if math.isnan(f) else "infinity"
                raise ValueError(
                    f"values[{i}] (node ix={ixx}, iy={iyy}, iz={izz}): "
                    f"{kind} is not allowed"
                )
            out[i] = f

        if len(self.origin) != 3:
            raise ValueError("origin: expected exactly 3 components")
        if len(self.spacing) != 3:
            raise ValueError("spacing: expected exactly 3 components")
        origin = tuple(
            _finite_number(f"origin[{i}]", c) for i, c in enumerate(self.origin)
        )
        spacing = tuple(
            _finite_number(f"spacing[{i}]", c) for i, c in enumerate(self.spacing)
        )
        for i, s in enumerate(spacing):
            if not s > 0.0:
                raise ValueError(
                    f"spacing[{i}]: expected strictly positive spacing, got {s}"
                )

        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "nz", nz)
        object.__setattr__(self, "values", tuple(out))
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)

    def node_id(self, ix: int, iy: int, iz: int) -> int:
        return ix + self.nx * (iy + self.ny * iz)


@dataclass(frozen=True)
class Triangle:
    """一个等值面三角形，``vertex_ids`` 为 :class:`MeshResult` 顶点下标。

    绕序使其外法线（右手定则）指向标量增大方向。
    """

    vertex_ids: tuple[int, int, int]


@dataclass(frozen=True)
class DiscardedFace:
    """被丢弃三角形的可定位记录。"""

    voxel: tuple[int, int, int]
    tetrahedron: int  # 0..5，对应 CUBE_TETRAHEDRA 行
    triangle: int     # 该四面体内的三角形序号（0 或 1）
    edge_keys: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    reason: str       # 当前只有 "zero_area"


@dataclass(frozen=True)
class MeshStats:
    """网格与等值面的组合/几何统计（计数为精确整数，面积为浮点和）。"""

    voxel_count: int
    tetrahedron_count: int
    tetrahedra_with_surface: int
    triangles_generated: int       # 丢弃前生成的三角形数
    faces_kept: int
    zero_area_faces_discarded: int
    level_tetrahedra_skipped: int  # 4 顶点恰全部等于 iso 的四面体数（等值集为整个四面体，非二维面，跳过）
    vertex_count: int
    face_count: int
    boundary_edge_count: int       # 恰被 1 个三角形引用的网格边（开口边）
    nonmanifold_edge_count: int    # 被 >=3 个三角形引用的网格边
    surface_vertices_on_domain_boundary: int  # 落在网格域边界面上的等值面顶点
    surface_area: float            # 所有保留三角形面积之和


@dataclass(frozen=True)
class MeshResult:
    """提取结果。

    :param vertices: 交点世界坐标，下标即顶点 ID。
    :param faces: 保留的三角形（朝标量增大方向）。
    :param discarded_faces: 几何零面积而丢弃的三角形记录。
    :param edge_vertices: 全局网格边键 (min_node_id,max_node_id) -> 顶点 ID；
        同一条网格边只对应一个顶点，跨体素/四面体共享。
    :param boundary_edge_keys: 开口边（恰 1 面引用）的边键。
    :param nonmanifold_edge_keys: 非流形边（>=3 面引用）的边键。
    :param stats: 统计汇总。
    """

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[Triangle, ...]
    discarded_faces: tuple[DiscardedFace, ...]
    edge_vertices: Mapping[tuple[int, int], int]
    boundary_edge_keys: tuple[tuple[int, int], ...]
    nonmanifold_edge_keys: tuple[tuple[int, int], ...]
    stats: MeshStats


def _classified_above(node_id: int, value: float, iso: float) -> bool:
    """符号扰动拓扑分类：值恰为 iso 时按全局节点 ID 奇偶归属。

    偶数 ID 归 above（标量增大侧），奇数 ID 归 below。该结果只决定连边
    拓扑；交点几何位置另由真实线性插值给出，不受扰动影响。
    """
    if value > iso:
        return True
    if value < iso:
        return False
    return (node_id & 1) == 0


def extract_isosurface(grid: ScalarGrid, iso: object) -> MeshResult:
    """在 ``grid`` 上提取 ``iso`` 等值面（纯函数：不修改输入，无部分失败状态）。

    :raises TypeError: 输入类型非法（含 bool 充当整数）。
    :raises ValueError: 数值为 NaN/Infinity 或尺寸越界等（消息可定位）。
    """
    if not isinstance(grid, ScalarGrid):
        raise TypeError(
            f"grid: expected ScalarGrid, got {type(grid).__name__}"
        )
    level = _finite_number("iso", iso)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    values = grid.values
    ox, oy, oz = grid.origin
    sx, sy, sz = grid.spacing
    nxy = nx * ny
    total_nodes = nxy * nz

    # 预计算全部节点世界坐标：保证同一节点经任意网格边引用时坐标逐位一致。
    positions: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * total_nodes
    kid = 0
    for iz in range(nz):
        pz = oz + iz * sz
        for iy in range(ny):
            py = oy + iy * sy
            for ix in range(nx):
                positions[kid] = (ox + ix * sx, py, pz)
                kid += 1

    edge_vertices: dict[tuple[int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []

    def edge_vertex(ga: int, gb: int) -> int:
        if ga < gb:
            a, b = ga, gb
        else:
            a, b = gb, ga
        key = (a, b)
        vid = edge_vertices.get(key)
        if vid is not None:
            return vid

        va, vb = values[a], values[b]
        pa, pb = positions[a], positions[b]
        if va == level:
            if vb == level:
                # 双端恰等 iso：线性参数不确定。按符号扰动的极限约定，
                # 交点取该边上被扰动判为 above 的端点（几何仍是网格节点）。
                t = 0.0 if (a & 1) == 0 else 1.0
            else:
                t = 0.0
        elif vb == level:
            t = 1.0
        else:
            t = (level - va) / (vb - va)

        if t == 0.0:
            pos = pa
        elif t == 1.0:
            pos = pb
        else:
            pos = (pa[0] + t * (pb[0] - pa[0]),
                   pa[1] + t * (pb[1] - pa[1]),
                   pa[2] + t * (pb[2] - pa[2]))

        vid = len(vertices)
        edge_vertices[key] = vid
        vertices.append(pos)
        return vid

    faces: list[Triangle] = []
    discarded: list[DiscardedFace] = []
    tetra_with_surface = 0
    level_tet_skipped = 0
    triangles_generated = 0

    voxel_count = (nx - 1) * (ny - 1) * (nz - 1)

    for iz in range(nz - 1):
        base_z = iz * nxy
        for iy in range(ny - 1):
            base_y = base_z + iy * nx
            for ix in range(nx - 1):
                base = base_y + ix
                corner_gid = (
                    base,                    # 0 (0,0,0)
                    base + 1,                # 1 (1,0,0)
                    base + 1 + nx,           # 2 (1,1,0)
                    base + nx,               # 3 (0,1,0)
                    base + nxy,              # 4 (0,0,1)
                    base + nxy + 1,          # 5 (1,0,1)
                    base + nxy + 1 + nx,     # 6 (1,1,1)
                    base + nxy + nx,         # 7 (0,1,1)
                )

                for ti, tet in enumerate(CUBE_TETRAHEDRA):
                    g0 = corner_gid[tet[0]]
                    g1 = corner_gid[tet[1]]
                    g2 = corner_gid[tet[2]]
                    g3 = corner_gid[tet[3]]
                    gids = (g0, g1, g2, g3)
                    v0, v1, v2, v3 = values[g0], values[g1], values[g2], values[g3]

                    a0 = _classified_above(g0, v0, level)
                    a1 = _classified_above(g1, v1, level)
                    a2 = _classified_above(g2, v2, level)
                    a3 = _classified_above(g3, v3, level)
                    # 4 顶点真实值恰全部等于 iso：等值集是整个四面体（三维），
                    # 不存在二维等值面片；跳过并计数，不依赖扰动后的掩码。
                    if v0 == level and v1 == level and v2 == level and v3 == level:
                        level_tet_skipped += 1
                        continue
                    mask = (int(a0) | (int(a1) << 1)
                            | (int(a2) << 2) | (int(a3) << 3))
                    tri_edges = MARCHING_TET_TABLE[mask]
                    if not tri_edges:
                        continue
                    tetra_with_surface += 1

                    above_flags = (a0, a1, a2, a3)
                    ax_ = ay_ = az_ = 0.0
                    bx_ = by_ = bz_ = 0.0
                    na = nb = 0
                    for li in range(4):
                        px, py, pz = positions[gids[li]]
                        if above_flags[li]:
                            ax_ += px; ay_ += py; az_ += pz; na += 1
                        else:
                            bx_ += px; by_ += py; bz_ += pz; nb += 1
                    # above 质心 - below 质心；四面体内场线性，该向量与梯度
                    # 点积恒为正（v_above 的均值 >= iso > v_below 的均值）。
                    dx_ = ax_ / na - bx_ / nb
                    dy_ = ay_ / na - by_ / nb
                    dz_ = az_ / na - bz_ / nb

                    for tk, edge_nums in enumerate(tri_edges):
                        triangles_generated += 1
                        vids: list[int] = []
                        ekeys: list[tuple[int, int]] = []
                        for en in edge_nums:
                            la, lb = TETRA_EDGES[en]
                            ga, gb = gids[la], gids[lb]
                            vids.append(edge_vertex(ga, gb))
                            ekeys.append(
                                (ga, gb) if ga < gb else (gb, ga)
                            )

                        p0 = vertices[vids[0]]
                        p1 = vertices[vids[1]]
                        p2 = vertices[vids[2]]
                        ux = p1[0] - p0[0]
                        uy = p1[1] - p0[1]
                        uz = p1[2] - p0[2]
                        vvx = p2[0] - p0[0]
                        vvy = p2[1] - p0[1]
                        vvz = p2[2] - p0[2]
                        nx_ = uy * vvz - uz * vvy
                        ny_ = uz * vvx - ux * vvz
                        nz_ = ux * vvy - uy * vvx

                        # 几何零面积：叉积分量精确全 0（重合/共线），不设容差。
                        if nx_ == 0.0 and ny_ == 0.0 and nz_ == 0.0:
                            discarded.append(DiscardedFace(
                                voxel=(ix, iy, iz),
                                tetrahedron=ti,
                                triangle=tk,
                                edge_keys=(ekeys[0], ekeys[1], ekeys[2]),
                                reason="zero_area",
                            ))
                            continue

                        if nx_ * dx_ + ny_ * dy_ + nz_ * dz_ < 0.0:
                            vids[1], vids[2] = vids[2], vids[1]
                            ekeys[1], ekeys[2] = ekeys[2], ekeys[1]
                        faces.append(Triangle((vids[0], vids[1], vids[2])))

    # 组合边统计（三角形边，键为顶点 ID 对）。
    edge_face_count: dict[tuple[int, int], int] = {}
    for tri in faces:
        w = tri.vertex_ids
        for i in range(3):
            aa, bb = w[i], w[(i + 1) % 3]
            ek = (aa, bb) if aa < bb else (bb, aa)
            edge_face_count[ek] = edge_face_count.get(ek, 0) + 1

    boundary_keys = tuple(k for k, c in edge_face_count.items() if c == 1)
    nonmanifold_keys = tuple(k for k, c in edge_face_count.items() if c >= 3)

    # 落在网格域 6 个边界面上的等值面顶点（坐标精确比较：插值点严格位于
    # 网格边内部，不可能误等于边界平面坐标）。
    x_lo, y_lo, z_lo = ox, oy, oz
    x_hi = ox + (nx - 1) * sx
    y_hi = oy + (ny - 1) * sy
    z_hi = oz + (nz - 1) * sz
    boundary_vertices = 0
    for px, py, pz in vertices:
        if (px == x_lo or px == x_hi or py == y_lo or py == y_hi
                or pz == z_lo or pz == z_hi):
            boundary_vertices += 1

    area = 0.0
    for tri in faces:
        w = tri.vertex_ids
        p0 = vertices[w[0]]
        p1 = vertices[w[1]]
        p2 = vertices[w[2]]
        ux = p1[0] - p0[0]
        uy = p1[1] - p0[1]
        uz = p1[2] - p0[2]
        vvx = p2[0] - p0[0]
        vvy = p2[1] - p0[1]
        vvz = p2[2] - p0[2]
        cx = uy * vvz - uz * vvy
        cy = uz * vvx - ux * vvz
        cz = ux * vvy - uy * vvx
        area += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)

    stats = MeshStats(
        voxel_count=voxel_count,
        tetrahedron_count=6 * voxel_count,
        tetrahedra_with_surface=tetra_with_surface,
        triangles_generated=triangles_generated,
        faces_kept=len(faces),
        zero_area_faces_discarded=len(discarded),
        level_tetrahedra_skipped=level_tet_skipped,
        vertex_count=len(vertices),
        face_count=len(faces),
        boundary_edge_count=len(boundary_keys),
        nonmanifold_edge_count=len(nonmanifold_keys),
        surface_vertices_on_domain_boundary=boundary_vertices,
        surface_area=area,
    )

    return MeshResult(
        vertices=tuple(vertices),
        faces=tuple(faces),
        discarded_faces=tuple(discarded),
        edge_vertices=dict(edge_vertices),
        boundary_edge_keys=boundary_keys,
        nonmanifold_edge_keys=nonmanifold_keys,
        stats=stats,
    )
