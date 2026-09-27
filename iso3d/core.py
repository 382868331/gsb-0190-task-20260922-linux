"""体素等值面的共享边交点提取核心实现（marching tetrahedra）。

坐标系：节点 (i, j, k) 的坐标即 (i, j, k)，晶格单位立方体。
每个立方体沿主对角线 (0,0,0)-(1,1,1) 统一分解为 6 个四面体（见 TETRA_TABLE）。
所有四面体边的交点按全局网格边键（升序节点 ID 对）共享。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "CUBE_CORNERS",
    "MAX_DIM",
    "MAX_NODES",
    "TETRA_TABLE",
    "DegenerateFace",
    "Face",
    "InvalidInputError",
    "IsosurfaceMesh",
    "MeshStats",
    "ScalarField3D",
    "Vertex",
    "canonical_edge_key",
    "extract_isosurface",
    "perturbed_below",
    "update_isosurface",
]

# ---------------------------------------------------------------------------
# 公开常量
# ---------------------------------------------------------------------------

MAX_DIM = 16
"""单个坐标轴方向允许的最大节点数（含端点）。"""

MAX_NODES = MAX_DIM ** 3
"""标量场允许的最大节点总数 16^3 = 4096。"""

CUBE_CORNERS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
"""立方体 8 个角点的局部偏移，角点编号 0..7 按 (i + 2j + 4k) 顺序。"""

TETRA_TABLE: Tuple[Tuple[int, int, int, int], ...] = (
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 2, 3, 7),
    (0, 2, 6, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
)
"""立方体沿主对角线 0-7（即 (0,0,0)-(1,1,1)）的 6 四面体分解表。

每行是一个四面体的 4 个局部角点编号（见 CUBE_CORNERS）。第 n 个四面体
对应坐标轴的一种排列：{0, e_a, e_a+e_b, 7}。所有四面体共享体对角线
0-7；每个立方体面只被一条面内对角线剖分，且相邻立方体在同一共享面上
产生同一条对角线，因此四面体网格是保形的（conforming），跨单元交点
可按全局边键共享。
"""

_TET_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
)
"""四面体 4 个顶点（局部编号 0..3）构成的 6 条边。"""


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------

class InvalidInputError(ValueError):
    """输入违反公开契约时抛出。

    message 总是包含可定位的参数名（以及能定位时的节点索引），
    例如 ``"values[37]: NaN is not allowed"``。
    """

    def __init__(self, parameter: str, message: str) -> None:
        self.parameter = parameter
        super().__init__(f"{parameter}: {message}")


# ---------------------------------------------------------------------------
# 校验辅助
# ---------------------------------------------------------------------------

def _check_axis(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(name, f"must be an int, got {type(value).__name__}")
    if value < 1:
        raise InvalidInputError(name, f"must be >= 1, got {value}")
    if value > MAX_DIM:
        raise InvalidInputError(name, f"must be <= {MAX_DIM}, got {value}")
    return value


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(name, f"must be a real number, got {type(value).__name__}")
    f = float(value)
    if math.isnan(f):
        raise InvalidInputError(name, "NaN is not allowed")
    if math.isinf(f):
        raise InvalidInputError(name, "Infinity is not allowed")
    return f


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScalarField3D:
    """规则晶格上的标量场。

    values 为长度 ni*nj*nk 的扁平序列，行优先排列：
    节点 (i, j, k) 的值为 values[(k * nj + j) * ni + i]。
    节点 (i, j, k) 的坐标即 (i, j, k)（晶格单位）。
    """

    ni: int
    nj: int
    nk: int
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        ni = _check_axis("ni", self.ni)
        nj = _check_axis("nj", self.nj)
        nk = _check_axis("nk", self.nk)
        if ni * nj * nk > MAX_NODES:
            raise InvalidInputError(
                "values",
                f"total node count {ni}*{nj}*{nk} = {ni * nj * nk} exceeds {MAX_NODES}",
            )
        vals = self.values
        if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
            raise InvalidInputError(
                "values", f"must be a sequence of real numbers, got {type(vals).__name__}"
            )
        expected = ni * nj * nk
        if len(vals) != expected:
            raise InvalidInputError(
                "values", f"expected {expected} values for grid {ni}x{nj}x{nk}, got {len(vals)}"
            )
        checked: List[float] = []
        for idx, v in enumerate(vals):
            checked.append(_finite_float(f"values[{idx}]", v))
        object.__setattr__(self, "values", tuple(checked))

    def node_id(self, i: int, j: int, k: int) -> int:
        """节点 (i, j, k) 的全局 ID（行优先）。"""
        return (k * self.nj + j) * self.ni + i

    def value_at(self, i: int, j: int, k: int) -> float:
        """节点 (i, j, k) 的标量值。"""
        return self.values[self.node_id(i, j, k)]

    @classmethod
    def from_nested(cls, nested: Sequence[Sequence[Sequence[float]]]) -> "ScalarField3D":
        """由 nested[k][j][i] 形式的嵌套序列构造（k 最外层）。"""
        if isinstance(nested, (str, bytes)) or not isinstance(nested, Sequence):
            raise InvalidInputError("nested", "must be a nested sequence nested[k][j][i]")
        nk = len(nested)
        if nk == 0:
            raise InvalidInputError("nested", "must contain at least one k-slice")
        nj = len(nested[0])
        if nj == 0:
            raise InvalidInputError("nested", "k-slices must contain at least one j-row")
        ni = len(nested[0][0])
        if ni == 0:
            raise InvalidInputError("nested", "j-rows must contain at least one value")
        flat: List[float] = []
        for k in range(nk):
            rows = nested[k]
            if len(rows) != nj:
                raise InvalidInputError(
                    "nested", f"ragged input: nested[{k}] has {len(rows)} rows, expected {nj}"
                )
            for j in range(nj):
                row = rows[j]
                if len(row) != ni:
                    raise InvalidInputError(
                        "nested",
                        f"ragged input: nested[{k}][{j}] has {len(row)} values, expected {ni}",
                    )
                flat.extend(row)
        return cls(ni=ni, nj=nj, nk=nk, values=tuple(flat))


@dataclass(frozen=True)
class Vertex:
    """等值面顶点：一条全局网格边上的线性插值交点。"""

    id: int
    point: Tuple[float, float, float]
    edge_key: Tuple[int, int]
    t: float
    """插值参数：point = (1-t)*P(edge_key[0]) + t*P(edge_key[1])。"""


@dataclass(frozen=True)
class Face:
    """等值面三角形，顶点按右手定则定向，法向指向标量增大一侧。"""

    id: int
    vertex_ids: Tuple[int, int, int]
    area: float


@dataclass(frozen=True)
class DegenerateFace:
    """被丢弃的几何零面积三角形记录。"""

    cell: Tuple[int, int, int]
    tet_index: int
    edge_keys: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class MeshStats:
    """提取结果的统计信息。"""

    iso: float
    node_count: int
    cell_count: int
    tetrahedron_count: int
    vertex_count: int
    face_count: int
    degenerate_face_count: int
    boundary_edge_count: int
    """只被 1 个三角形使用的网格边数（开边界）。"""
    nonmanifold_edge_count: int
    """被 3 个及以上三角形使用的网格边数。"""
    total_area: float


@dataclass(frozen=True)
class IsosurfaceMesh:
    """等值面提取结果。"""

    iso: float
    vertices: Tuple[Vertex, ...]
    faces: Tuple[Face, ...]
    stats: MeshStats
    degenerate_faces: Tuple[DegenerateFace, ...]
    boundary_edges: Tuple[Tuple[int, int], ...]
    """按边键升序排列的边界边（恰好被 1 个面使用）。"""
    nonmanifold_edges: Tuple[Tuple[int, int], ...]
    """按边键升序排列的非流形边（被 3 个及以上面使用）。"""
    edge_use_count: Dict[Tuple[int, int], int] = field(compare=False)
    """每条网格边（端点顶点 ID 对，升序）被多少个三角形使用。"""

    def total_area(self) -> float:
        """所有保留三角形面积之和（与 stats.total_area 一致）。"""
        return self.stats.total_area


# ---------------------------------------------------------------------------
# 公开辅助
# ---------------------------------------------------------------------------

def canonical_edge_key(id_a: int, id_b: int) -> Tuple[int, int]:
    """两个全局节点 ID 的规范边键（升序对）。"""
    return (id_a, id_b) if id_a < id_b else (id_b, id_a)


def perturbed_below(value: float, iso: float, node_id: int) -> bool:
    """拓扑判定：节点是否位于等值面“下方”（标量较小一侧）。

    value < iso 为下方；value > iso 为上方；value == iso 时不改动几何，
    仅按全局节点 ID 的奇偶作确定性符号扰动：偶数 ID 视为下方，奇数 ID
    视为上方。该规则对整个网格一致，保证跨单元拓扑判定一致。
    """
    if value < iso:
        return True
    if value > iso:
        return False
    return node_id % 2 == 0


# ---------------------------------------------------------------------------
# 内部向量辅助
# ---------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _tri_area(p, q, r):
    return 0.5 * _norm(_cross(_sub(q, p), _sub(r, p)))


def _oriented_loop(pts, loop, ref_point):
    """调整 loop 的环绕方向，使其法向指向 ref_point 一侧。

    loop 是顶点坐标下标列表；返回新的下标列表（可能整体反转）。
    若三角形退化（零面积），返回顺序未定义但确定。
    """
    p0, p1, p2 = (pts[loop[0]], pts[loop[1]], pts[loop[2]])
    normal = _cross(_sub(p1, p0), _sub(p2, p0))
    if _dot(normal, _sub(ref_point, p0)) < 0.0:
        return [loop[0], loop[2], loop[1]]
    return loop


def _node_coord(node_id: int, ni: int, nj: int) -> Tuple[float, float, float]:
    i = node_id % ni
    j = (node_id // ni) % nj
    k = node_id // (ni * nj)
    return (float(i), float(j), float(k))


def _make_vertex(
    key: Tuple[int, int],
    values: Tuple[float, ...],
    ni: int,
    nj: int,
    iso: float,
    vid: int,
) -> Vertex:
    """在全局网格边 key 上按线性插值构造等值面顶点。"""
    id_a, id_b = key
    va = values[id_a]
    vb = values[id_b]
    denom = vb - va
    if denom == 0.0:
        # 仅当两端值都恰等于 iso 时发生（否则两端同侧，不会成为穿越边）。
        # 取中点以保持确定性；拓扑已由符号扰动决定。
        t = 0.5
    else:
        t = (iso - va) / denom
    pa = _node_coord(id_a, ni, nj)
    pb = _node_coord(id_b, ni, nj)
    point = (
        pa[0] + t * (pb[0] - pa[0]),
        pa[1] + t * (pb[1] - pa[1]),
        pa[2] + t * (pb[2] - pa[2]),
    )
    return Vertex(id=vid, point=point, edge_key=key, t=t)


def _assemble_mesh(
    iso: float,
    ni: int,
    nj: int,
    nk: int,
    vertices: List[Vertex],
    faces: List[Face],
    degenerate: List[DegenerateFace],
) -> IsosurfaceMesh:
    """由顶点/三角形/退化记录装配 IsosurfaceMesh 并计算全部统计。"""
    edge_use: Dict[Tuple[int, int], int] = {}
    for f in faces:
        vids = f.vertex_ids
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (vids[a], vids[b]) if vids[a] < vids[b] else (vids[b], vids[a])
            edge_use[key] = edge_use.get(key, 0) + 1
    boundary_edges = tuple(sorted(key for key, c in edge_use.items() if c == 1))
    nonmanifold_edges = tuple(sorted(key for key, c in edge_use.items() if c >= 3))
    total_area = sum(f.area for f in faces)
    cell_count = (ni - 1) * (nj - 1) * (nk - 1)
    stats = MeshStats(
        iso=iso,
        node_count=ni * nj * nk,
        cell_count=cell_count,
        tetrahedron_count=cell_count * len(TETRA_TABLE),
        vertex_count=len(vertices),
        face_count=len(faces),
        degenerate_face_count=len(degenerate),
        boundary_edge_count=len(boundary_edges),
        nonmanifold_edge_count=len(nonmanifold_edges),
        total_area=total_area,
    )
    return IsosurfaceMesh(
        iso=iso,
        vertices=tuple(vertices),
        faces=tuple(faces),
        stats=stats,
        degenerate_faces=tuple(degenerate),
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        edge_use_count=dict(edge_use),
    )


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

def _process_cubes(cubes, values, ni, nj, iso, get_vertex, vertex_point, emit_triangle) -> None:
    """对给定立方体序列执行 marching tetrahedra，结果经回调汇出。

    get_vertex(edge_key) -> 顶点 ID（调用方负责按边键共享/去重）；
    vertex_point(vid) -> 该顶点的坐标；
    emit_triangle((v0, v1, v2), cell, tet_index) 接收一个定向三角形
    （退化判定与记录由调用方在回调内完成）。
    """

    def nid(i: int, j: int, k: int) -> int:
        return (k * nj + j) * ni + i

    def ncoord(node_id: int) -> Tuple[float, float, float]:
        i = node_id % ni
        j = (node_id // ni) % nj
        k = node_id // (ni * nj)
        return (float(i), float(j), float(k))

    for cell in cubes:
        i, j, k = cell
        corner_ids = tuple(
            nid(i + di, j + dj, k + dk) for (di, dj, dk) in CUBE_CORNERS
        )
        below = tuple(perturbed_below(values[c], iso, c) for c in corner_ids)
        for tet_index, tet in enumerate(TETRA_TABLE):
            tet_nodes = tuple(corner_ids[c] for c in tet)
            tet_below = tuple(below[c] for c in tet)
            nb = sum(tet_below)
            if nb == 0 or nb == 4:
                continue
            crossings: List[Tuple[int, int]] = []
            for ea, eb in _TET_EDGES:
                if tet_below[ea] != tet_below[eb]:
                    crossings.append(
                        canonical_edge_key(tet_nodes[ea], tet_nodes[eb])
                    )
            # K4 二染色下穿越边数只可能是 3 或 4。
            if len(crossings) == 3:
                vids = tuple(get_vertex(key) for key in crossings)
                pts = [vertex_point(v) for v in vids]
                ref_local = 0 if not tet_below[0] else next(
                    idx for idx in range(1, 4) if not tet_below[idx]
                )
                ref_point = ncoord(tet_nodes[ref_local])
                order = _oriented_loop(pts, [0, 1, 2], ref_point)
                emit_triangle(
                    (vids[order[0]], vids[order[1]], vids[order[2]]),
                    cell,
                    tet_index,
                )
            else:  # 4 条穿越边：一个四边形，切成两个三角形
                entries = [(get_vertex(key), key) for key in crossings]
                pts = [vertex_point(v) for v, _ in entries]
                centroid = (
                    sum(p[0] for p in pts) / 4.0,
                    sum(p[1] for p in pts) / 4.0,
                    sum(p[2] for p in pts) / 4.0,
                )
                above_local = [idx for idx in range(4) if not tet_below[idx]]
                below_local = [idx for idx in range(4) if tet_below[idx]]
                above_mid = tuple(
                    sum(ncoord(tet_nodes[idx])[d] for idx in above_local)
                    / len(above_local)
                    for d in range(3)
                )
                below_mid = tuple(
                    sum(ncoord(tet_nodes[idx])[d] for idx in below_local)
                    / len(below_local)
                    for d in range(3)
                )
                ref = _sub(above_mid, below_mid)
                if _norm(ref) == 0.0:
                    ref = (1.0, 0.0, 0.0)
                # 构造与 ref 垂直的平面基 (u, w)。
                axis = min(
                    range(3),
                    key=lambda d: (abs(ref[d]), d),
                )
                basis = [0.0, 0.0, 0.0]
                basis[axis] = 1.0
                u = _cross(ref, tuple(basis))
                u = tuple(c / _norm(u) for c in u)
                w = _cross(ref, u)

                def angle(idx: int) -> float:
                    rel = _sub(pts[idx], centroid)
                    return math.atan2(_dot(rel, w), _dot(rel, u))

                order = sorted(
                    range(4), key=lambda idx: (angle(idx), entries[idx][1])
                )
                ref_point = above_mid
                loop = _oriented_loop(pts, list(order), ref_point)
                emit_triangle(
                    (entries[loop[0]][0], entries[loop[1]][0], entries[loop[2]][0]),
                    cell,
                    tet_index,
                )
                emit_triangle(
                    (entries[loop[0]][0], entries[loop[2]][0], entries[loop[3]][0]),
                    cell,
                    tet_index,
                )

def extract_isosurface(field: ScalarField3D, iso: float) -> IsosurfaceMesh:
    """从标量场提取 iso 等值面（marching tetrahedra）。

    参数
    ----
    field : ScalarField3D
        规则晶格标量场，每维 1..16 个节点，总数 <= 4096。
    iso : int 或 float
        等值。必须为有限实数；bool 被拒绝。

    返回
    ----
    IsosurfaceMesh
        顶点（按全局网格边键共享）、定向三角形与统计信息。
        合法但处处不穿越 iso 的场返回空网格（不抛异常）。

    抛出
    ----
    InvalidInputError
        任何输入违反契约时；此时不产生任何部分结果。
    """
    if not isinstance(field, ScalarField3D):
        raise InvalidInputError(
            "field", f"must be a ScalarField3D, got {type(field).__name__}"
        )
    iso_f = _finite_float("iso", iso)

    ni, nj, nk = field.ni, field.nj, field.nk
    values = field.values

    vertices: List[Vertex] = []
    vertex_by_edge: Dict[Tuple[int, int], int] = {}

    def get_vertex(key: Tuple[int, int]) -> int:
        vid = vertex_by_edge.get(key)
        if vid is not None:
            return vid
        vid = len(vertices)
        vertices.append(_make_vertex(key, values, ni, nj, iso_f, vid))
        vertex_by_edge[key] = vid
        return vid

    faces: List[Face] = []
    degenerate: List[DegenerateFace] = []

    def emit_triangle(vids: Tuple[int, int, int], cell, tet_index) -> None:
        p0, p1, p2 = (vertices[vids[0]].point, vertices[vids[1]].point, vertices[vids[2]].point)
        area = _tri_area(p0, p1, p2)
        if area == 0.0:
            degenerate.append(
                DegenerateFace(
                    cell=cell,
                    tet_index=tet_index,
                    edge_keys=tuple(vertices[v].edge_key for v in vids),
                )
            )
            return
        faces.append(Face(id=len(faces), vertex_ids=tuple(vids), area=area))

    _process_cubes(
        (
            (i, j, k)
            for k in range(nk - 1)
            for j in range(nj - 1)
            for i in range(ni - 1)
        ),
        values,
        ni,
        nj,
        iso_f,
        get_vertex,
        lambda vid: vertices[vid].point,
        emit_triangle,
    )

    return _assemble_mesh(iso_f, ni, nj, nk, vertices, faces, degenerate)


# ---------------------------------------------------------------------------
# 增量更新
# ---------------------------------------------------------------------------

def _run_cubes(cubes, values, ni, nj, iso):
    """在指定立方体集合上局部提取，返回 (vertices, triangles, degenerates)。

    vertices：局部 Vertex 列表（局部 ID 从 0 连续编号）；
    triangles：定向三角形列表，每项为三个顶点所在全局边键的有序三元组；
    degenerates：被丢弃的零面积面记录。
    """
    vertices: List[Vertex] = []
    vertex_by_edge: Dict[Tuple[int, int], int] = {}

    def get_vertex(key: Tuple[int, int]) -> int:
        vid = vertex_by_edge.get(key)
        if vid is not None:
            return vid
        vid = len(vertices)
        vertices.append(_make_vertex(key, values, ni, nj, iso, vid))
        vertex_by_edge[key] = vid
        return vid

    triangles: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []
    degenerates: List[DegenerateFace] = []

    def emit_triangle(vids: Tuple[int, int, int], cell, tet_index) -> None:
        p0, p1, p2 = (vertices[vids[0]].point, vertices[vids[1]].point, vertices[vids[2]].point)
        if _tri_area(p0, p1, p2) == 0.0:
            degenerates.append(
                DegenerateFace(
                    cell=cell,
                    tet_index=tet_index,
                    edge_keys=tuple(vertices[v].edge_key for v in vids),
                )
            )
            return
        triangles.append(tuple(vertices[v].edge_key for v in vids))

    _process_cubes(
        cubes,
        values,
        ni,
        nj,
        iso,
        get_vertex,
        lambda vid: vertices[vid].point,
        emit_triangle,
    )
    return vertices, triangles, degenerates


def update_isosurface(
    old_field: ScalarField3D,
    old_mesh: IsosurfaceMesh,
    new_field: ScalarField3D,
    changed_nodes=None,
    iso=None,
) -> IsosurfaceMesh:
    """在标量场局部修改后增量更新等值面，而不是重新提取整个表面。

    只对包含变化节点的立方体重新执行 marching tetrahedra；未受影响
    区域的顶点与三角形对象（及其 ID）原样保留，方便上层继续引用。
    共享边上的交点按全局边键去重，修改区域边缘不会产生裂缝；新增/
    消失的三角形（包括值恰好跨过 iso 的拓扑变化）由受影响立方体的
    局部重提取自然处理。返回网格的几何与对 new_field 完整重提取一致。

    参数
    ----
    old_field / new_field : ScalarField3D
        修改前后的标量场，维度必须相同。
    old_mesh : IsosurfaceMesh
        对 old_field 的提取结果（extract_isosurface 或本函数的输出）。
    changed_nodes : 可选，节点 ID 序列
        值发生变化的节点（行优先扁平索引）。None 时通过逐节点比较
        old_field 与 new_field 自动求差。
    iso : 可选
        等值。默认沿用 old_mesh.iso；若显式给出不同的等值，所有交点
        几何都会变化，无法保留身份，此时退化为对 new_field 的完整重提取。

    返回
    ----
    IsosurfaceMesh
        新的冻结网格对象。顶点和面 ID 在更新后可能不再连续：未受影响
        部分保留原 ID，新增部分从旧最大 ID 之后继续编号。

    抛出
    ----
    InvalidInputError
        任何输入违反契约时。本函数为纯函数：所有校验在计算之前完成，
        失败时不修改 old_mesh（其本身不可变），旧结果仍可继续使用。
    """
    if not isinstance(old_field, ScalarField3D):
        raise InvalidInputError(
            "old_field", f"must be a ScalarField3D, got {type(old_field).__name__}"
        )
    if not isinstance(new_field, ScalarField3D):
        raise InvalidInputError(
            "new_field", f"must be a ScalarField3D, got {type(new_field).__name__}"
        )
    if not isinstance(old_mesh, IsosurfaceMesh):
        raise InvalidInputError(
            "old_mesh", f"must be an IsosurfaceMesh, got {type(old_mesh).__name__}"
        )
    ni, nj, nk = old_field.ni, old_field.nj, old_field.nk
    if (new_field.ni, new_field.nj, new_field.nk) != (ni, nj, nk):
        raise InvalidInputError(
            "new_field",
            f"dimensions {new_field.ni}x{new_field.nj}x{new_field.nk} do not match "
            f"old_field {ni}x{nj}x{nk}",
        )
    if iso is None:
        iso_f = old_mesh.iso
    else:
        iso_f = _finite_float("iso", iso)
        if iso_f != old_mesh.iso:
            # 等值变化时所有交点几何都随之改变，身份无法保留；
            # 退化为完整重提取（几何同样与全量结果一致）。
            return extract_isosurface(new_field, iso_f)

    node_count = ni * nj * nk
    if changed_nodes is None:
        changed = [
            idx
            for idx in range(node_count)
            if old_field.values[idx] != new_field.values[idx]
        ]
    else:
        if isinstance(changed_nodes, (str, bytes)) or not isinstance(
            changed_nodes, Sequence
        ):
            raise InvalidInputError(
                "changed_nodes",
                f"must be a sequence of node ids, got {type(changed_nodes).__name__}",
            )
        changed = []
        seen = set()
        for pos, idx in enumerate(changed_nodes):
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise InvalidInputError(
                    f"changed_nodes[{pos}]",
                    f"must be an int, got {type(idx).__name__}",
                )
            if idx < 0 or idx >= node_count:
                raise InvalidInputError(
                    f"changed_nodes[{pos}]",
                    f"node id {idx} out of range [0, {node_count})",
                )
            if idx not in seen:
                seen.add(idx)
                changed.append(idx)

    # 受影响立方体：包含任一变化节点的所有立方体。共享边两个端点中任一
    # 值变化时，包含该边的所有立方体都在受影响集合内，因此边缘交点会被
    # 两侧一致地重建，不会留下裂缝。
    affected = set()
    for idx in changed:
        i = idx % ni
        j = (idx // ni) % nj
        k = idx // (ni * nj)
        for ck in (k - 1, k):
            if not 0 <= ck <= nk - 2:
                continue
            for cj in (j - 1, j):
                if not 0 <= cj <= nj - 2:
                    continue
                for ci in (i - 1, i):
                    if 0 <= ci <= ni - 2:
                        affected.add((ci, cj, ck))
    affected = sorted(affected)

    # 受影响区域在新旧场上的局部提取。旧场结果用于精确识别需要从
    # old_mesh 中移除的三角形（按顶点边键三元组的多重集匹配）。
    _old_local_verts, old_tris, _old_degen = _run_cubes(
        affected, old_field.values, ni, nj, iso_f
    )
    new_local_verts, new_tris, new_degen = _run_cubes(
        affected, new_field.values, ni, nj, iso_f
    )

    old_vertices_by_id: Dict[int, Vertex] = {v.id: v for v in old_mesh.vertices}
    remove: Dict[Tuple[Tuple[int, int], ...], int] = {}
    for triple in old_tris:
        remove[triple] = remove.get(triple, 0) + 1
    kept_faces: List[Face] = []
    for f in old_mesh.faces:
        try:
            triple = tuple(old_vertices_by_id[vid].edge_key for vid in f.vertex_ids)
        except KeyError:
            raise InvalidInputError(
                "old_mesh",
                f"face {f.id} references unknown vertex id; "
                "old_mesh is inconsistent, cannot update incrementally",
            ) from None
        if remove.get(triple, 0):
            remove[triple] -= 1
        else:
            kept_faces.append(f)
    if any(remove.values()):
        raise InvalidInputError(
            "old_mesh",
            "does not match old_field on the affected cells; "
            "cannot update incrementally",
        )

    # 顶点拼接：几何未变的交点复用旧 Vertex 对象（保留原 ID）；
    # 新出现的交点从旧最大 ID 之后继续编号。
    kept_vertex_ids = {vid for f in kept_faces for vid in f.vertex_ids}
    old_vertex_by_edge = {v.edge_key: v for v in old_mesh.vertices}
    final_vertices: Dict[int, Vertex] = {
        vid: old_vertices_by_id[vid] for vid in kept_vertex_ids
    }
    next_vid = max(old_vertices_by_id, default=-1) + 1
    local_to_final: Dict[int, int] = {}
    for lv in new_local_verts:
        old_v = old_vertex_by_edge.get(lv.edge_key)
        if old_v is not None and old_v.point == lv.point and old_v.t == lv.t:
            final_vertices[old_v.id] = old_v
            local_to_final[lv.id] = old_v.id
        else:
            while next_vid in final_vertices:
                next_vid += 1
            final_vertices[next_vid] = Vertex(
                id=next_vid, point=lv.point, edge_key=lv.edge_key, t=lv.t
            )
            local_to_final[lv.id] = next_vid
            next_vid += 1

    # 三角形拼接：未受影响的旧面原样保留；新面用最终顶点 ID 重新装配。
    final_vertex_points = {vid: v.point for vid, v in final_vertices.items()}
    edge_to_final = {v.edge_key: vid for vid, v in final_vertices.items()}
    next_fid = max((f.id for f in old_mesh.faces), default=-1) + 1
    new_faces: List[Face] = []
    for triple in new_tris:
        vids = tuple(edge_to_final[key] for key in triple)
        area = _tri_area(
            final_vertex_points[vids[0]],
            final_vertex_points[vids[1]],
            final_vertex_points[vids[2]],
        )
        new_faces.append(Face(id=next_fid, vertex_ids=vids, area=area))
        next_fid += 1

    affected_set = set(affected)
    kept_degenerate = [
        d for d in old_mesh.degenerate_faces if d.cell not in affected_set
    ]
    vertices_out = [final_vertices[vid] for vid in sorted(final_vertices)]
    faces_out = kept_faces + new_faces
    degenerate_out = kept_degenerate + new_degen
    return _assemble_mesh(iso_f, ni, nj, nk, vertices_out, faces_out, degenerate_out)
