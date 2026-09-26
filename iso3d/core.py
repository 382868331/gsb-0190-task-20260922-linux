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
    "BandMesh",
    "BandStats",
    "DegenerateFace",
    "Face",
    "InvalidInputError",
    "IsosurfaceMesh",
    "MeshStats",
    "ScalarField3D",
    "Vertex",
    "canonical_edge_key",
    "extract_band",
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


@dataclass(frozen=True)
class BandStats:
    """数值带提取结果的统计信息。"""

    low: float
    high: float
    zero_thickness: bool
    """low == high 的零厚度特殊结果：只有单层等值面，不代表有体积的实体。"""
    node_count: int
    cell_count: int
    tetrahedron_count: int
    vertex_count: int
    face_count: int
    cap_face_count: int
    """网格外边界盖片三角形数（低面/高面等值面之外的封口面）。"""
    degenerate_face_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    total_area: float
    signed_volume: float
    """有向三角网的体积积分 sum(dot(p0, cross(p1, p2))) / 6。

    对封闭网格（boundary_edge_count == 0）即数值带区域的体积；
    零厚度结果恒为 0.0（不是实体，不报告体积）。
    """


@dataclass(frozen=True)
class BandMesh:
    """数值带 low <= f <= high 的边界提取结果。

    法向统一朝向数值带外侧：高面（f == high）指向标量增大方向，
    低面（f == low）指向标量减小方向，盖片指向网格外侧。
    """

    low: float
    high: float
    zero_thickness: bool
    vertices: Tuple[Vertex, ...]
    faces: Tuple[Face, ...]
    stats: BandStats
    degenerate_faces: Tuple[DegenerateFace, ...]
    """被丢弃的零面积面；盖片产生的记录 tet_index 为 -1。"""
    boundary_edges: Tuple[Tuple[int, int], ...]
    nonmanifold_edges: Tuple[Tuple[int, int], ...]
    edge_use_count: Dict[Tuple[int, int], int] = field(compare=False)

    def signed_volume(self) -> float:
        """有向体积积分（与 stats.signed_volume 一致）。"""
        return self.stats.signed_volume

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
    registry = _VertexRegistry(field)
    vertices = registry.vertices

    faces: List[Face] = []
    degenerate: List[DegenerateFace] = []
    edge_use: Dict[Tuple[int, int], int] = {}

    def emit_triangle(vids: Tuple[int, int, int], cell, tet_index) -> None:
        p0, p1, p2 = (registry.points[vids[0]], registry.points[vids[1]], registry.points[vids[2]])
        area = _tri_area(p0, p1, p2)
        if area == 0.0:
            degenerate.append(
                DegenerateFace(
                    cell=cell,
                    tet_index=tet_index,
                    edge_keys=tuple(registry.vertices[v].edge_key for v in vids),
                )
            )
            return
        faces.append(Face(id=len(faces), vertex_ids=tuple(vids), area=area))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (vids[a], vids[b]) if vids[a] < vids[b] else (vids[b], vids[a])
            edge_use[key] = edge_use.get(key, 0) + 1

    _march_level(field, iso_f, True, registry, emit_triangle)

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
# 共享的四面体遍历（供 extract_isosurface / extract_band 复用）
# ---------------------------------------------------------------------------

class _VertexRegistry:
    """按 (全局网格边键, 等值层) 共享的交点顶点注册表。

    同一条网格边、同一个等值层上的交点全局唯一；不同等值层（低面/高面）
    即使落在同一条边上也注册为不同顶点，保证两层等值面不会错误合并。
    盖片落在网格节点上的顶点以 edge_key=(nid, nid), t=0.0 注册。
    """

    def __init__(self, field: ScalarField3D) -> None:
        self._field = field
        self.vertices: List[Vertex] = []
        self.points: List[Tuple[float, float, float]] = []
        self._by_key: Dict[Tuple[Tuple[int, int], float], int] = {}
        self._node_ids: Dict[int, int] = {}

    def node_coord(self, node_id: int) -> Tuple[float, float, float]:
        ni, nj = self._field.ni, self._field.nj
        i = node_id % ni
        j = (node_id // ni) % nj
        k = node_id // (ni * nj)
        return (float(i), float(j), float(k))

    def _append(self, point, edge_key, t) -> int:
        vid = len(self.vertices)
        self.vertices.append(Vertex(id=vid, point=point, edge_key=edge_key, t=t))
        self.points.append(point)
        return vid

    def get(self, key: Tuple[int, int], level: float) -> int:
        """边 key 上 level 等值层的插值交点（共享身份）。"""
        rkey = (key, level)
        vid = self._by_key.get(rkey)
        if vid is not None:
            return vid
        values = self._field.values
        id_a, id_b = key
        va = values[id_a]
        vb = values[id_b]
        denom = vb - va
        if denom == 0.0:
            # 仅当两端值都恰等于等值层时发生（否则两端同侧，不会成为穿越边）。
            # 取中点以保持确定性；拓扑已由符号扰动决定。
            t = 0.5
        else:
            t = (level - va) / denom
        pa = self.node_coord(id_a)
        pb = self.node_coord(id_b)
        point = (
            pa[0] + t * (pb[0] - pa[0]),
            pa[1] + t * (pb[1] - pa[1]),
            pa[2] + t * (pb[2] - pa[2]),
        )
        vid = self._append(point, key, t)
        self._by_key[rkey] = vid
        return vid

    def get_node(self, node_id: int) -> int:
        """网格节点本身作为顶点（盖片区域内部包含整个节点时使用）。"""
        vid = self._node_ids.get(node_id)
        if vid is None:
            vid = self._append(self.node_coord(node_id), (node_id, node_id), 0.0)
            self._node_ids[node_id] = vid
        return vid


def _march_level(
    field: ScalarField3D,
    iso_f: float,
    normal_toward_above: bool,
    registry: _VertexRegistry,
    emit_triangle,
) -> None:
    """对单个等值层遍历全部四面体并发射三角形。

    registry 负责按 (边键, 等值层) 共享/去重交点顶点；
    emit_triangle((v0, v1, v2), cell, tet_index) 处理定向后的三角形。
    normal_toward_above 为 True 时法向指向标量增大一侧，False 时相反。
    """
    ni, nj, nk = field.ni, field.nj, field.nk
    values = field.values

    def nid(i: int, j: int, k: int) -> int:
        return (k * nj + j) * ni + i

    def ncoord(node_id: int) -> Tuple[float, float, float]:
        i = node_id % ni
        j = (node_id // ni) % nj
        k = node_id // (ni * nj)
        return (float(i), float(j), float(k))

    def get_vertex(key: Tuple[int, int]) -> int:
        return registry.get(key, iso_f)

    def edge_mid(key: Tuple[int, int]) -> Tuple[float, float, float]:
        # 边中点（半整数坐标，精确）：用于环绕排序与法向定向。
        # 交点可能落在距节点 1 ulp 处形成 ~1e-16 的薄片三角形，
        # 直接用交点几何定向在数值上不稳定；中点保持组合结构，
        # 定向结果对任意非退化四面体都是确定的。
        pa = ncoord(key[0])
        pb = ncoord(key[1])
        return (
            0.5 * (pa[0] + pb[0]),
            0.5 * (pa[1] + pb[1]),
            0.5 * (pa[2] + pb[2]),
        )

    for k in range(nk - 1):
        for j in range(nj - 1):
            for i in range(ni - 1):
                corner_ids = tuple(
                    nid(i + di, j + dj, k + dk) for (di, dj, dk) in CUBE_CORNERS
                )
                below = tuple(
                    perturbed_below(values[c], iso_f, c) for c in corner_ids
                )
                cell = (i, j, k)
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
                        pts = [edge_mid(key) for key in crossings]
                        if normal_toward_above:
                            ref_local = 0 if not tet_below[0] else next(
                                idx for idx in range(1, 4) if not tet_below[idx]
                            )
                        else:
                            ref_local = 0 if tet_below[0] else next(
                                idx for idx in range(1, 4) if tet_below[idx]
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
                        pts = [edge_mid(key) for _, key in entries]
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
                        ref_point = above_mid if normal_toward_above else below_mid
                        loop = list(order)
                        q0, q1, q2 = pts[loop[0]], pts[loop[1]], pts[loop[2]]
                        qn = _cross(_sub(q1, q0), _sub(q2, q0))
                        if _dot(qn, _sub(ref_point, q0)) < 0.0:
                            loop.reverse()
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


# ---------------------------------------------------------------------------
# 数值带提取（extract_band）
# ---------------------------------------------------------------------------

# 网格 6 个外边界面：(轴向, 侧, 外法向, 该面的两个三角形（局部角点编号）)。
# 三角形划分与 TETRA_TABLE 在立方体面上诱导的面内对角线一致，
# 因此盖片与四面体网格保形，跨单元不会重复生成。
_BOUNDARY_FACES: Tuple[
    Tuple[int, int, Tuple[float, float, float], Tuple[int, int, int], Tuple[int, int, int]],
    ...,
] = (
    (0, 0, (-1.0, 0.0, 0.0), (0, 2, 6), (0, 6, 4)),
    (0, 1, (1.0, 0.0, 0.0), (1, 3, 7), (1, 7, 5)),
    (1, 0, (0.0, -1.0, 0.0), (0, 1, 5), (0, 5, 4)),
    (1, 1, (0.0, 1.0, 0.0), (2, 3, 7), (2, 7, 6)),
    (2, 0, (0.0, 0.0, -1.0), (0, 1, 3), (0, 3, 2)),
    (2, 1, (0.0, 0.0, 1.0), (4, 5, 7), (4, 7, 6)),
)

# 盖片多边形顶点标记：("node", 节点ID) 或 ("x", 边键, 等值层)。


def _clip_polygon(poly, level, keep_above):
    """用半空间 f >= level（keep_above=True）或 f <= level 裁剪凸多边形。

    poly 元素为 (point, value, tag)；返回同结构的新多边形。
    几何用真实标量值线性插值；f == level 的顶点统一判为内侧（归属一致，
    相邻三角形不会对同一区域重复生成盖片）。
    """
    out = []
    count = len(poly)
    for idx in range(count):
        cur = poly[idx]
        nxt = poly[(idx + 1) % count]
        cur_in = cur[1] >= level if keep_above else cur[1] <= level
        nxt_in = nxt[1] >= level if keep_above else nxt[1] <= level
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            denom = nxt[1] - cur[1]
            t = (level - cur[1]) / denom  # 值不同才可穿越，denom != 0
            point = tuple(cur[0][d] + t * (nxt[0][d] - cur[0][d]) for d in range(3))
            # 确定该线段所在的原始三角形边（用于共享交点身份）。
            cur_tag, nxt_tag = cur[2], nxt[2]
            if cur_tag[0] == "node" and nxt_tag[0] == "node":
                seg_key = canonical_edge_key(cur_tag[1], nxt_tag[1])
            elif cur_tag[0] == "x":
                seg_key = cur_tag[1]
            else:
                seg_key = nxt_tag[1]
            out.append((point, float(level), ("x", seg_key, level)))
    return out


def extract_band(field: ScalarField3D, low: float, high: float) -> BandMesh:
    """提取数值带 low <= f <= high 的区域边界（双层等值面 + 外边界盖片）。

    结果由三部分组成，法向统一朝向数值带外侧：

    1. 低面：f == low 的等值面，法向指向标量减小一侧；
    2. 高面：f == high 的等值面，法向指向标量增大一侧；
    3. 盖片：数值带区域与网格外边界相交处的封口三角形，
       法向指向网格外侧。盖片按外边界面的四面体对角线剖分，
       与内部四面体网格保形；每个边界面三角形只属于一个立方体，
       且 f == low / f == high 的顶点统一判为带内，相邻单元不会
       重复生成盖片。

    低面与高面使用同一四面体分解与共享交点身份（按 (边键, 等值层)
    注册），即使同一条网格边同时穿越 low 与 high，两层顶点也是
    不同的插值点，不会错误合并。

    low == high 是零厚度特殊结果：只返回该值的单层等值面
    （与 extract_isosurface 一致），zero_thickness=True，不生成盖片，
    signed_volume 恒为 0.0——它不是有体积的实体。

    参数
    ----
    field : ScalarField3D
        规则晶格标量场，约束同 extract_isosurface。
    low, high : int 或 float
        数值带下/上限，必须为有限实数且 low <= high；bool 被拒绝。

    返回
    ----
    BandMesh
        顶点、定向三角形、盖片计数与有向体积积分等统计。
        合法但数值带处处为空的场返回空网格（不抛异常）。
        某一维只有 1 个节点时网格没有任何立方体，返回空网格。

    抛出
    ----
    InvalidInputError
        任何输入违反契约时（含 low > high）；此时不产生任何部分结果。
    """
    if not isinstance(field, ScalarField3D):
        raise InvalidInputError(
            "field", f"must be a ScalarField3D, got {type(field).__name__}"
        )
    low_f = _finite_float("low", low)
    high_f = _finite_float("high", high)
    if low_f > high_f:
        raise InvalidInputError(
            "high", f"must be >= low ({low_f}), got {high_f}"
        )

    ni, nj, nk = field.ni, field.nj, field.nk
    values = field.values
    zero_thickness = low_f == high_f

    registry = _VertexRegistry(field)
    faces: List[Face] = []
    degenerate: List[DegenerateFace] = []
    cap_face_count = 0

    def emit_triangle(vids: Tuple[int, int, int], cell, tet_index) -> bool:
        p0, p1, p2 = (registry.points[vids[0]], registry.points[vids[1]], registry.points[vids[2]])
        area = _tri_area(p0, p1, p2)
        if area == 0.0:
            degenerate.append(
                DegenerateFace(
                    cell=cell,
                    tet_index=tet_index,
                    edge_keys=tuple(registry.vertices[v].edge_key for v in vids),
                )
            )
            return False
        faces.append(Face(id=len(faces), vertex_ids=tuple(vids), area=area))
        return True

    if zero_thickness:
        # 零厚度：单层等值面，法向指向标量增大一侧，不伪装成实体。
        _march_level(field, low_f, True, registry, emit_triangle)
    else:
        # 低面：法向指向标量减小一侧（数值带外侧）。
        _march_level(field, low_f, False, registry, emit_triangle)
        # 高面：法向指向标量增大一侧（数值带外侧）。
        _march_level(field, high_f, True, registry, emit_triangle)

        def nid(i: int, j: int, k: int) -> int:
            return (k * nj + j) * ni + i

        dims = (ni, nj, nk)
        for axis, side, outward, tri_a, tri_b in _BOUNDARY_FACES:
            n_axis = dims[axis]
            if n_axis < 2:
                continue  # 该方向没有立方体（整个网格无单元，结果为空）
            outer = 0 if side == 0 else n_axis - 2  # 边界立方体的索引
            ranges = [range(d - 1) for d in dims]
            ranges[axis] = (outer,)
            for c0 in ranges[0]:
                for c1 in ranges[1]:
                    for c2 in ranges[2]:
                        i, j, k = c0, c1, c2
                        corner_ids = tuple(
                            nid(i + di, j + dj, k + dk) for (di, dj, dk) in CUBE_CORNERS
                        )
                        cell = (i, j, k)
                        for tri in (tri_a, tri_b):
                            node_ids = tuple(corner_ids[c] for c in tri)
                            # 先用精确的整数节点坐标把三角形定向为法向朝外；
                            # 半平面裁剪保持凸多边形的环绕方向，因此后续
                            # 盖片多边形与扇形剖分自动继承正确朝向，无需再
                            # 依赖（可能在 ulp 级薄片上失稳的）交点几何判定。
                            n0, n1, n2 = (registry.node_coord(x) for x in node_ids)
                            tri_normal = _cross(_sub(n1, n0), _sub(n2, n0))
                            if _dot(tri_normal, outward) < 0.0:
                                node_ids = (node_ids[0], node_ids[2], node_ids[1])
                            poly = [
                                (
                                    registry.node_coord(node),
                                    values[node],
                                    ("node", node),
                                )
                                for node in node_ids
                            ]
                            poly = _clip_polygon(poly, low_f, True)
                            poly = _clip_polygon(poly, high_f, False)
                            if len(poly) < 3:
                                continue
                            vids = []
                            for point, _value, tag in poly:
                                if tag[0] == "node":
                                    vids.append(registry.get_node(tag[1]))
                                else:
                                    vids.append(registry.get(tag[1], tag[2]))
                            for idx in range(1, len(vids) - 1):
                                if emit_triangle(
                                    (vids[0], vids[idx], vids[idx + 1]), cell, -1
                                ):
                                    cap_face_count += 1

    # 统一归属：节点值恰等于 low/high 时，3D 等值层会在相邻边上注册
    # t=0/t=1 的层顶点，而盖片把该节点注册为节点顶点——几何上是同一点。
    # 按精确坐标合并顶点身份（节点坐标为整数、t=0/1 插值结果精确，
    # 不同等值层在同一条边上的 t 不同，不会被误并），避免邻单元重复盖面
    # 或盖片与等值面之间出现假边界。
    canon: Dict[Tuple[float, float, float], int] = {}
    remap: List[int] = []
    merged_vertices: List[Vertex] = []
    for v in registry.vertices:
        vid = canon.get(v.point)
        if vid is None:
            vid = len(merged_vertices)
            canon[v.point] = vid
            merged_vertices.append(Vertex(id=vid, point=v.point, edge_key=v.edge_key, t=v.t))
        remap.append(vid)
    if len(merged_vertices) != len(registry.vertices):
        faces = [
            Face(
                id=idx,
                vertex_ids=tuple(remap[v] for v in face.vertex_ids),
                area=face.area,
            )
            for idx, face in enumerate(faces)
        ]

    # 统一归属（面级）：标量恰等于阈值时，扰动薄片可能使相邻四面体
    # 生成几何完全相同的面（顶点三元组一致）。数值带边界是单层曲面，
    # 按无向顶点三元组归并：反向面成对抵消（零体积薄片壳），
    # 同向面只保留一个。
    by_triple: Dict[frozenset, List[Face]] = {}
    for face in faces:
        by_triple.setdefault(frozenset(face.vertex_ids), []).append(face)
    if any(len(group) > 1 for group in by_triple.values()):
        deduped: List[Face] = []
        for group in by_triple.values():
            if len(group) == 1:
                deduped.append(group[0])
                continue
            buckets: Dict[Tuple[int, int, int], List[Face]] = {}
            for face in group:
                a, b, c = face.vertex_ids
                canonical = min(((a, b, c), (b, c, a), (c, a, b)))
                buckets.setdefault(canonical, []).append(face)
            oriented = sorted(buckets.values(), key=len, reverse=True)
            net = len(oriented[0]) - (len(oriented[1]) if len(oriented) > 1 else 0)
            if net > 0:
                deduped.append(oriented[0][0])
        faces = [
            Face(id=idx, vertex_ids=face.vertex_ids, area=face.area)
            for idx, face in enumerate(deduped)
        ]

    edge_use: Dict[Tuple[int, int], int] = {}
    for face in faces:
        vids = face.vertex_ids
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (vids[a], vids[b]) if vids[a] < vids[b] else (vids[b], vids[a])
            edge_use[key] = edge_use.get(key, 0) + 1

    signed_volume = 0.0
    if not zero_thickness:
        for face in faces:
            p0, p1, p2 = (merged_vertices[v].point for v in face.vertex_ids)
            signed_volume += _dot(p0, _cross(p1, p2)) / 6.0

    boundary_edges = tuple(sorted(key for key, c in edge_use.items() if c == 1))
    nonmanifold_edges = tuple(sorted(key for key, c in edge_use.items() if c >= 3))
    total_area = sum(f.area for f in faces)
    node_count = ni * nj * nk
    cell_count = (ni - 1) * (nj - 1) * (nk - 1)
    stats = BandStats(
        low=low_f,
        high=high_f,
        zero_thickness=zero_thickness,
        node_count=node_count,
        cell_count=cell_count,
        tetrahedron_count=cell_count * len(TETRA_TABLE),
        vertex_count=len(merged_vertices),
        face_count=len(faces),
        cap_face_count=cap_face_count,
        degenerate_face_count=len(degenerate),
        boundary_edge_count=len(boundary_edges),
        nonmanifold_edge_count=len(nonmanifold_edges),
        total_area=total_area,
        signed_volume=signed_volume,
    )
    return BandMesh(
        low=low_f,
        high=high_f,
        zero_thickness=zero_thickness,
        vertices=tuple(merged_vertices),
        faces=tuple(faces),
        stats=stats,
        degenerate_faces=tuple(degenerate),
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        edge_use_count=dict(edge_use),
    )
