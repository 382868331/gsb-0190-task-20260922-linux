"""体素等值面的共享边交点提取核心实现（marching tetrahedra）。

坐标系：节点 (i, j, k) 的坐标即 (i, j, k)，晶格单位立方体。
每个立方体沿主对角线 (0,0,0)-(1,1,1) 统一分解为 6 个四面体（见 TETRA_TABLE）。
所有四面体边的交点按全局网格边键（升序节点 ID 对）共享。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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
    "IsosurfaceExtractor",
    "IsosurfaceMesh",
    "MeshStats",
    "ScalarField3D",
    "Vertex",
    "canonical_edge_key",
    "extract_isosurface",
    "perturbed_below",
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


def _interpolate_edge(
    values: Sequence[float], iso: float, ni: int, nj: int, key: Tuple[int, int]
) -> Tuple[Tuple[float, float, float], float]:
    """全局网格边 key 上的线性插值交点与参数 t（纯函数，确定性）。

    两端值都恰等于 iso 时（分母为 0）取中点 t = 0.5；拓扑已由符号扰动决定。
    """
    id_a, id_b = key
    va = values[id_a]
    vb = values[id_b]
    denom = vb - va
    t = 0.5 if denom == 0.0 else (iso - va) / denom
    pa = _node_coord(id_a, ni, nj)
    pb = _node_coord(id_b, ni, nj)
    point = (
        pa[0] + t * (pb[0] - pa[0]),
        pa[1] + t * (pb[1] - pa[1]),
        pa[2] + t * (pb[2] - pa[2]),
    )
    return point, t


def _cell_primitives(
    values: Sequence[float],
    iso: float,
    ni: int,
    nj: int,
    ci: int,
    cj: int,
    ck: int,
) -> Tuple[Tuple[int, Tuple[Tuple[int, int], ...], Tuple[Tuple[Tuple[int, int], ...], ...], Tuple[Tuple[Tuple[int, int], ...], ...]], ...]:
    """计算单个立方体穿越 iso 的全部四面体图元（纯函数）。

    返回 tuple，每项对应一个有穿越的四面体：
    (tet_index, crossings, triangles, degenerates)，其中

    - crossings：按 _TET_EDGES 顺序的穿越边键（全局节点 ID 升序对）；
    - triangles：定向后的边键三元组（法向指向标量增大侧，非零面积）；
    - degenerates：几何零面积（精确叉积为零）的边键三元组，应被丢弃并记录。
    """

    def nid(i: int, j: int, k: int) -> int:
        return (k * nj + j) * ni + i

    corner_ids = tuple(
        nid(ci + di, cj + dj, ck + dk) for (di, dj, dk) in CUBE_CORNERS
    )
    below = tuple(perturbed_below(values[c], iso, c) for c in corner_ids)

    point_cache: Dict[Tuple[int, int], Tuple[float, float, float]] = {}

    def point_of(key: Tuple[int, int]) -> Tuple[float, float, float]:
        p = point_cache.get(key)
        if p is None:
            p, _ = _interpolate_edge(values, iso, ni, nj, key)
            point_cache[key] = p
        return p

    results = []
    for tet_index, tet in enumerate(TETRA_TABLE):
        tet_nodes = tuple(corner_ids[c] for c in tet)
        tet_below = tuple(below[c] for c in tet)
        nb = sum(tet_below)
        if nb == 0 or nb == 4:
            continue
        crossings: List[Tuple[int, int]] = []
        for ea, eb in _TET_EDGES:
            if tet_below[ea] != tet_below[eb]:
                crossings.append(canonical_edge_key(tet_nodes[ea], tet_nodes[eb]))
        # K4 二染色下穿越边数只可能是 3 或 4。
        triangles: List[Tuple[Tuple[int, int], ...]] = []
        degenerates: List[Tuple[Tuple[int, int], ...]] = []
        if len(crossings) == 3:
            pts = [point_of(key) for key in crossings]
            ref_local = 0 if not tet_below[0] else next(
                idx for idx in range(1, 4) if not tet_below[idx]
            )
            ref_point = _node_coord(tet_nodes[ref_local], ni, nj)
            order = _oriented_loop(pts, [0, 1, 2], ref_point)
            tri = (crossings[order[0]], crossings[order[1]], crossings[order[2]])
            if _tri_area(pts[order[0]], pts[order[1]], pts[order[2]]) == 0.0:
                degenerates.append(tri)
            else:
                triangles.append(tri)
        else:  # 4 条穿越边：一个四边形，切成两个三角形
            pts = [point_of(key) for key in crossings]
            centroid = (
                sum(p[0] for p in pts) / 4.0,
                sum(p[1] for p in pts) / 4.0,
                sum(p[2] for p in pts) / 4.0,
            )
            above_local = [idx for idx in range(4) if not tet_below[idx]]
            below_local = [idx for idx in range(4) if tet_below[idx]]
            above_mid = tuple(
                sum(_node_coord(tet_nodes[idx], ni, nj)[d] for idx in above_local)
                / len(above_local)
                for d in range(3)
            )
            below_mid = tuple(
                sum(_node_coord(tet_nodes[idx], ni, nj)[d] for idx in below_local)
                / len(below_local)
                for d in range(3)
            )
            ref = _sub(above_mid, below_mid)
            if _norm(ref) == 0.0:
                ref = (1.0, 0.0, 0.0)
            # 构造与 ref 垂直的平面基 (u, w)。
            axis = min(range(3), key=lambda d: (abs(ref[d]), d))
            basis = [0.0, 0.0, 0.0]
            basis[axis] = 1.0
            u = _cross(ref, tuple(basis))
            u = tuple(c / _norm(u) for c in u)
            w = _cross(ref, u)

            def angle(idx: int) -> float:
                rel = _sub(pts[idx], centroid)
                return math.atan2(_dot(rel, w), _dot(rel, u))

            order = sorted(range(4), key=lambda idx: (angle(idx), crossings[idx]))
            loop = _oriented_loop(pts, list(order), above_mid)
            for tri_idx in ((loop[0], loop[1], loop[2]), (loop[0], loop[2], loop[3])):
                tri = (crossings[tri_idx[0]], crossings[tri_idx[1]], crossings[tri_idx[2]])
                if _tri_area(pts[tri_idx[0]], pts[tri_idx[1]], pts[tri_idx[2]]) == 0.0:
                    degenerates.append(tri)
                else:
                    triangles.append(tri)
        results.append((tet_index, tuple(crossings), tuple(triangles), tuple(degenerates)))
    return tuple(results)


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

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
        point, t = _interpolate_edge(values, iso_f, ni, nj, key)
        vid = len(vertices)
        vertices.append(Vertex(id=vid, point=point, edge_key=key, t=t))
        vertex_by_edge[key] = vid
        return vid

    faces: List[Face] = []
    degenerate: List[DegenerateFace] = []
    edge_use: Dict[Tuple[int, int], int] = {}

    for k in range(nk - 1):
        for j in range(nj - 1):
            for i in range(ni - 1):
                cell = (i, j, k)
                for tet_index, crossings, triangles, degenerates in _cell_primitives(
                    values, iso_f, ni, nj, i, j, k
                ):
                    vids_for = {key: get_vertex(key) for key in crossings}
                    for tri in triangles:
                        vids = (vids_for[tri[0]], vids_for[tri[1]], vids_for[tri[2]])
                        p0, p1, p2 = (vertices[v].point for v in vids)
                        faces.append(
                            Face(id=len(faces), vertex_ids=vids, area=_tri_area(p0, p1, p2))
                        )
                        for a, b in ((0, 1), (1, 2), (2, 0)):
                            ekey = (
                                (vids[a], vids[b])
                                if vids[a] < vids[b]
                                else (vids[b], vids[a])
                            )
                            edge_use[ekey] = edge_use.get(ekey, 0) + 1
                    for tri in degenerates:
                        degenerate.append(
                            DegenerateFace(cell=cell, tet_index=tet_index, edge_keys=tri)
                        )

    boundary_edges = tuple(sorted(key for key, c in edge_use.items() if c == 1))
    nonmanifold_edges = tuple(sorted(key for key, c in edge_use.items() if c >= 3))
    total_area = sum(f.area for f in faces)
    node_count = ni * nj * nk
    cell_count = (ni - 1) * (nj - 1) * (nk - 1)
    stats = MeshStats(
        iso=iso_f,
        node_count=node_count,
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
        iso=iso_f,
        vertices=tuple(vertices),
        faces=tuple(faces),
        stats=stats,
        degenerate_faces=tuple(degenerate),
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        edge_use_count=dict(edge_use),
    )


# ---------------------------------------------------------------------------
# 增量更新
# ---------------------------------------------------------------------------

class IsosurfaceExtractor:
    """有状态增量提取器：体数据局部修改后只重算受影响的单元。

    用法::

        ext = IsosurfaceExtractor(field, iso)
        mesh0 = ext.mesh()
        mesh1 = ext.update({node_index: new_value, ...})

    契约
    ----
    - 受影响单元 = 任意被修改节点邻接的立方体（每方向 -1..0，裁剪到网格内），
      只有这些单元的四面体图元被重新计算；其余单元的面与顶点原样保留。
    - 顶点身份：顶点挂在全局网格边键上。未受影响边上的顶点 id、坐标不变；
      端点值变化但仍穿越的边保留原顶点 id（坐标原地刷新）；新出现的穿越边
      分配从未使用过的新 id；消失的边其顶点 id 退役且永不复用。面 id 同理。
      因此上层持有的未受影响顶点/面 id 在更新后仍然有效。
    - 共享边一致性：交点按全局边键唯一存储，修改区域边缘的共享边交点只
      更新一次，所有引用它的面（无论属于哪个单元）自动看到同一坐标，
      不会产生裂缝。
    - 原子性：update 先完成全部校验与新图元计算，再一次性提交内部状态；
      校验或计算失败时抛出异常，旧状态完全不变，旧网格可继续使用。
    - 几何一致性：更新后的几何（顶点坐标集合与三角形集合）与对修改后的
      场调用 extract_isosurface 全量重提取逐位一致；仅顶点/面 id 与
      排列顺序可能不同。

    注意：mesh() 返回的 IsosurfaceMesh 中，Vertex.id / Face.id 是稳定身份，
    不一定是 vertices / faces 元组中的下标；请按 .id 建立索引，不要直接用
    id 当下标（全量 extract_isosurface 的结果仍保证 id == 下标）。
    """

    def __init__(self, field: ScalarField3D, iso: float) -> None:
        if not isinstance(field, ScalarField3D):
            raise InvalidInputError(
                "field", f"must be a ScalarField3D, got {type(field).__name__}"
            )
        self._iso = _finite_float("iso", iso)
        self._ni, self._nj, self._nk = field.ni, field.nj, field.nk
        self._values: List[float] = list(field.values)
        self._vertices: Dict[int, Vertex] = {}
        self._vertex_by_edge: Dict[Tuple[int, int], int] = {}
        self._vertex_ref: Dict[int, int] = {}
        self._faces: Dict[int, Face] = {}
        self._edge_use: Dict[Tuple[int, int], int] = {}
        self._cell_vertices: Dict[Tuple[int, int, int], Tuple[int, ...]] = {}
        self._cell_faces: Dict[Tuple[int, int, int], Tuple[int, ...]] = {}
        self._cell_degenerate: Dict[Tuple[int, int, int], Tuple[DegenerateFace, ...]] = {}
        self._next_vid = 0
        self._next_fid = 0
        for k in range(self._nk - 1):
            for j in range(self._nj - 1):
                for i in range(self._ni - 1):
                    prims = _cell_primitives(
                        self._values, self._iso, self._ni, self._nj, i, j, k
                    )
                    if prims:
                        self._add_cell((i, j, k), prims)

    @property
    def iso(self) -> float:
        """当前等值。"""
        return self._iso

    @property
    def dims(self) -> Tuple[int, int, int]:
        """网格维度 (ni, nj, nk)。"""
        return (self._ni, self._nj, self._nk)

    # -- 内部状态操作 -------------------------------------------------

    def _get_vertex(self, key: Tuple[int, int]) -> int:
        point, t = _interpolate_edge(self._values, self._iso, self._ni, self._nj, key)
        vid = self._vertex_by_edge.get(key)
        if vid is None:
            vid = self._next_vid
            self._next_vid += 1
            self._vertex_by_edge[key] = vid
            self._vertex_ref[vid] = 0
        # 坐标按当前值原地刷新：端点未变时结果逐位相同，端点变化时所有
        # 引用该顶点的面都属于受影响单元、随即被重建，因此始终一致。
        self._vertices[vid] = Vertex(id=vid, point=point, edge_key=key, t=t)
        return vid

    def _add_cell(self, cell, prims) -> None:
        fids: List[int] = []
        degens: List[DegenerateFace] = []
        cell_vids = set()
        for tet_index, crossings, triangles, degenerates in prims:
            vids_for = {key: self._get_vertex(key) for key in crossings}
            cell_vids.update(vids_for.values())
            for tri in triangles:
                vids = (vids_for[tri[0]], vids_for[tri[1]], vids_for[tri[2]])
                p0, p1, p2 = (self._vertices[v].point for v in vids)
                fid = self._next_fid
                self._next_fid += 1
                self._faces[fid] = Face(
                    id=fid, vertex_ids=vids, area=_tri_area(p0, p1, p2)
                )
                fids.append(fid)
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    ekey = (
                        (vids[a], vids[b]) if vids[a] < vids[b] else (vids[b], vids[a])
                    )
                    self._edge_use[ekey] = self._edge_use.get(ekey, 0) + 1
            for tri in degenerates:
                degens.append(
                    DegenerateFace(cell=cell, tet_index=tet_index, edge_keys=tri)
                )
        # 顶点引用计数按“包含该穿越边的单元”计（与全量提取的顶点存在性
        # 语义一致：只要任一单元把它列为穿越边，顶点就存在）。
        for vid in cell_vids:
            self._vertex_ref[vid] += 1
        self._cell_vertices[cell] = tuple(sorted(cell_vids))
        if fids:
            self._cell_faces[cell] = tuple(fids)
        if degens:
            self._cell_degenerate[cell] = tuple(degens)

    def _remove_cell(self, cell, keep_vids=frozenset()) -> None:
        for fid in self._cell_faces.pop(cell, ()):
            face = self._faces.pop(fid)
            vids = face.vertex_ids
            for a, b in ((0, 1), (1, 2), (2, 0)):
                ekey = (vids[a], vids[b]) if vids[a] < vids[b] else (vids[b], vids[a])
                count = self._edge_use[ekey] - 1
                if count == 0:
                    del self._edge_use[ekey]
                else:
                    self._edge_use[ekey] = count
        for vid in self._cell_vertices.pop(cell, ()):
            ref = self._vertex_ref[vid] - 1
            if ref == 0 and vid not in keep_vids:
                vertex = self._vertices.pop(vid)
                del self._vertex_by_edge[vertex.edge_key]
                del self._vertex_ref[vid]
            else:
                self._vertex_ref[vid] = ref
        self._cell_degenerate.pop(cell, None)

    # -- 公开接口 -----------------------------------------------------

    def update(self, changes) -> IsosurfaceMesh:
        """应用局部体数据修改并返回更新后的网格。

        参数
        ----
        changes : Mapping[int, float] 或 (index, value) 对的可迭代对象
            节点扁平索引 -> 新标量值。索引必须在 [0, ni*nj*nk) 内且不得
            重复；值必须为有限实数。空修改为合法无操作。

        抛出
        ----
        InvalidInputError
            任何修改项非法时；此时内部状态完全不变，旧网格仍可使用。
        """
        parsed = self._parse_changes(changes)
        if not parsed:
            return self.mesh()
        ni, nj, nk = self._ni, self._nj, self._nk
        new_values = list(self._values)
        for idx, value in parsed.items():
            new_values[idx] = value
        affected = set()
        for idx in parsed:
            i = idx % ni
            j = (idx // ni) % nj
            k = idx // (ni * nj)
            for dk in (-1, 0):
                for dj in (-1, 0):
                    for di in (-1, 0):
                        ci, cj, ck = i + di, j + dj, k + dk
                        if 0 <= ci < ni - 1 and 0 <= cj < nj - 1 and 0 <= ck < nk - 1:
                            affected.add((ci, cj, ck))
        # 纯计算阶段：任何异常都在提交前抛出，内部状态不变。
        new_prims = {
            cell: _cell_primitives(new_values, self._iso, ni, nj, cell[0], cell[1], cell[2])
            for cell in sorted(affected)
        }
        # 更新后仍然穿越的边保留其顶点 id（身份尽量保留）。
        keep_vids = set()
        for prims in new_prims.values():
            for _, crossings, _, _ in prims:
                for key in crossings:
                    vid = self._vertex_by_edge.get(key)
                    if vid is not None:
                        keep_vids.add(vid)
        # 提交阶段：仅字典增删，不会失败。
        self._values = new_values
        for cell in sorted(affected):
            self._remove_cell(cell, keep_vids)
        for cell in sorted(affected):
            if new_prims[cell]:
                self._add_cell(cell, new_prims[cell])
        return self.mesh()

    def _parse_changes(self, changes) -> Dict[int, float]:
        if isinstance(changes, Mapping):
            items = list(changes.items())
        elif isinstance(changes, (str, bytes)):
            raise InvalidInputError(
                "changes",
                "must be a mapping or an iterable of (index, value) pairs, "
                f"got {type(changes).__name__}",
            )
        else:
            try:
                items = [tuple(pair) for pair in changes]
            except TypeError:
                raise InvalidInputError(
                    "changes",
                    "must be a mapping or an iterable of (index, value) pairs, "
                    f"got {type(changes).__name__}",
                ) from None
        node_count = self._ni * self._nj * self._nk
        parsed: Dict[int, float] = {}
        for item in items:
            if len(item) != 2:
                raise InvalidInputError(
                    "changes", f"each change must be an (index, value) pair, got {item!r}"
                )
            idx, value = item
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise InvalidInputError(
                    "changes", f"node index must be an int, got {type(idx).__name__}"
                )
            if not 0 <= idx < node_count:
                raise InvalidInputError(
                    "changes", f"node index {idx} out of range [0, {node_count})"
                )
            if idx in parsed:
                raise InvalidInputError("changes", f"duplicate node index {idx}")
            parsed[idx] = _finite_float(f"changes[{idx}]", value)
        return parsed

    def mesh(self) -> IsosurfaceMesh:
        """当前状态的网格快照（O(网格规模)）。"""
        ni, nj, nk = self._ni, self._nj, self._nk
        vertices = tuple(self._vertices[vid] for vid in sorted(self._vertices))
        faces = tuple(self._faces[fid] for fid in sorted(self._faces))
        degenerate: List[DegenerateFace] = []
        for cell in sorted(self._cell_degenerate, key=lambda c: (c[2], c[1], c[0])):
            degenerate.extend(self._cell_degenerate[cell])
        boundary_edges = tuple(
            sorted(key for key, c in self._edge_use.items() if c == 1)
        )
        nonmanifold_edges = tuple(
            sorted(key for key, c in self._edge_use.items() if c >= 3)
        )
        cell_count = (ni - 1) * (nj - 1) * (nk - 1)
        stats = MeshStats(
            iso=self._iso,
            node_count=ni * nj * nk,
            cell_count=cell_count,
            tetrahedron_count=cell_count * len(TETRA_TABLE),
            vertex_count=len(vertices),
            face_count=len(faces),
            degenerate_face_count=len(degenerate),
            boundary_edge_count=len(boundary_edges),
            nonmanifold_edge_count=len(nonmanifold_edges),
            total_area=sum(f.area for f in faces),
        )
        return IsosurfaceMesh(
            iso=self._iso,
            vertices=vertices,
            faces=faces,
            stats=stats,
            degenerate_faces=tuple(degenerate),
            boundary_edges=boundary_edges,
            nonmanifold_edges=nonmanifold_edges,
            edge_use_count=dict(self._edge_use),
        )
