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
    "IntersectionRecord",
    "InvalidInputError",
    "IsosurfaceMesh",
    "MeshStats",
    "NONDIFF_ENDPOINT_ON_ISO",
    "NONDIFF_UNSTABLE_DENOMINATOR",
    "NONDIFF_ZERO_DENOMINATOR",
    "ScalarField3D",
    "ScalarSensitivity",
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

NONDIFF_ZERO_DENOMINATOR = "zero_denominator"
"""不可微原因：插值分母 vb-va 为零（两端标量值相等且恰为 iso）。"""

NONDIFF_ENDPOINT_ON_ISO = "endpoint_on_iso"
"""不可微原因：至少一个端点标量值恰等于 iso（拓扑分支点）。"""

NONDIFF_UNSTABLE_DENOMINATOR = "unstable_denominator"
"""不可微原因：分母虽非零但极小，解析导数数值不稳定。"""

# |vb-va| 小于此阈值时认为分母“不稳定”，不输出可能巨大的解析导数。
# 取 ~2^-24，远大于双精度舍入噪声但能拦住“标量差极小”的病态边；
# 该阈值只决定是否标注不可微，不影响几何交点本身的计算。
_DENOM_EPS = 2.0 ** -24

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
class ScalarSensitivity:
    """一个交点位置对三个标量输入的解析偏导（当前拓扑固定时）。

    设边端点为 A=edge_key[0]、B=edge_key[1]，坐标 P_A、P_B，
    标量值 v_a、v_b，交点 P = P_A + t (P_B - P_A)，
    t = (iso - v_a) / (v_b - v_a)。则对 v_a、v_b、iso 三个独立
    标量输入的解析导数为：

        dt/dv_a  = (iso - v_b) / (v_b - v_a)^2
        dt/dv_b  = (v_a - iso) / (v_b - v_a)^2
        dt/diso  = 1 / (v_b - v_a)
        dP/d(*)  = (P_B - P_A) * dt/d(*)

    differentiable 为 False 时三种情形之一（reason）：

    - ``NONDIFF_ZERO_DENOMINATOR``：v_b == v_a（两端值相等且恰为 iso，
      中点为约定值，对输入无导数可言）；
    - ``NONDIFF_ENDPOINT_ON_ISO``：v_a == iso 或 v_b == iso，交点恰在
      端点上，是符号扰动的拓扑分支点，仅在“当前拓扑不变”的单侧扰动
      下才有意义，故整体标为不可微；
    - ``NONDIFF_UNSTABLE_DENOMINATOR``：|v_b-v_a| 极小，解析公式数值
      不稳定，不输出巨大的伪精确数（几何交点仍正常给出）。

    不可微时三个导数元组均为 None，而非 0 或虚假大数。

    这些导数只在**当前拓扑不变的小扰动**下有效：它们描述交点沿同一条
    网格边的滑动；一旦扰动使端点越过 iso 改变符号（穿越边集合变化），
    会有交点产生/消失，导数不能跨拓扑延拓。
    """

    differentiable: bool
    d_point_d_va: Tuple[float, float, float]
    d_point_d_vb: Tuple[float, float, float]
    d_point_d_iso: Tuple[float, float, float]
    """交点坐标对 v_a、v_b、iso 的解析偏导；不可微时为 None。"""
    reason: str
    """不可微原因代码；可微时为空字符串。"""


@dataclass(frozen=True)
class IntersectionRecord:
    """一个共享交点的完整溯源记录（一条全局网格边对应唯一一条记录）。

    - ``vertex_id`` / ``edge_key``：交点顶点 ID 与其来源网格边（两端
      全局节点 ID 的升序对）。同一条网格边被任意多个四面体/面引用时，
      来源与敏感度只在此记录中生成一次，绝不按面重复。
    - ``weights``：对两端节点的线性插值权重
      ``(w_a, w_b) = (1-t, t)``，``w_a + w_b == 1``，
      ``point = w_a*P_A + w_b*P_B``。
    - ``sensitivity``：交点位置对两端标量值与 iso 的解析敏感度。
    """

    vertex_id: int
    edge_key: Tuple[int, int]
    weights: Tuple[float, float]
    sensitivity: ScalarSensitivity


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
    intersections: Tuple[IntersectionRecord, ...] = field(compare=False)
    """每个共享交点一条的溯源记录（来源边、插值权重、标量敏感度）。

    与 vertices 一一对应：``intersections[v].vertex_id == v``；
    同一网格边无论被多少四面体/面引用都只有一条记录。
    """
    edge_use_count: Dict[Tuple[int, int], int] = field(compare=False)
    """每条网格边（端点顶点 ID 对，升序）被多少个三角形使用。"""

    def intersection_for_vertex(self, vertex_id: int) -> IntersectionRecord:
        """按顶点 ID 取该共享交点的溯源记录。"""
        return self.intersections[vertex_id]

    def intersections_for_node(self, node_id: int) -> Tuple[IntersectionRecord, ...]:
        """所有以 node_id 为来源端点的交点记录（一个输入节点影响哪些交点）。"""
        return tuple(
            rec
            for rec in self.intersections
            if rec.edge_key[0] == node_id or rec.edge_key[1] == node_id
        )

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
        每个共享交点在 ``intersections`` 中有唯一溯源记录：来源网格边、
        插值权重，以及（可微时）交点位置对两端标量值与 iso 的解析偏导；
        端点恰 iso 或分母不稳定/为零时标记为不可微，不输出伪精确大数。
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

    def nid(i: int, j: int, k: int) -> int:
        return (k * nj + j) * ni + i

    def ncoord(node_id: int) -> Tuple[float, float, float]:
        i = node_id % ni
        j = (node_id // ni) % nj
        k = node_id // (ni * nj)
        return (float(i), float(j), float(k))

    vertices: List[Vertex] = []
    vertex_by_edge: Dict[Tuple[int, int], int] = {}
    records: List[IntersectionRecord] = []

    def get_vertex(key: Tuple[int, int]) -> int:
        vid = vertex_by_edge.get(key)
        if vid is not None:
            return vid
        id_a, id_b = key
        va = values[id_a]
        vb = values[id_b]
        denom = vb - va
        if denom == 0.0:
            # 仅当两端值都恰等于 iso 时发生（否则两端同侧，不会成为穿越边）。
            # 取中点以保持确定性；拓扑已由符号扰动决定。
            t = 0.5
        else:
            t = (iso_f - va) / denom
        pa = ncoord(id_a)
        pb = ncoord(id_b)
        edge_vec = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        point = (
            pa[0] + t * edge_vec[0],
            pa[1] + t * edge_vec[1],
            pa[2] + t * edge_vec[2],
        )
        # 标量敏感度：仅对“严格跨过 iso 且分母稳定”的边输出解析导数。
        if denom == 0.0:
            sensitivity = ScalarSensitivity(
                differentiable=False,
                d_point_d_va=None,
                d_point_d_vb=None,
                d_point_d_iso=None,
                reason=NONDIFF_ZERO_DENOMINATOR,
            )
        elif va == iso_f or vb == iso_f:
            # 交点恰在端点上：符号扰动的拓扑分支点，整体标为不可微。
            sensitivity = ScalarSensitivity(
                differentiable=False,
                d_point_d_va=None,
                d_point_d_vb=None,
                d_point_d_iso=None,
                reason=NONDIFF_ENDPOINT_ON_ISO,
            )
        elif abs(denom) < _DENOM_EPS:
            # 标量差极小：解析公式会给出巨大的伪精确数，标为不可微。
            sensitivity = ScalarSensitivity(
                differentiable=False,
                d_point_d_va=None,
                d_point_d_vb=None,
                d_point_d_iso=None,
                reason=NONDIFF_UNSTABLE_DENOMINATOR,
            )
        else:
            d2 = denom * denom

            def scale(s):
                return (s * edge_vec[0], s * edge_vec[1], s * edge_vec[2])

            sensitivity = ScalarSensitivity(
                differentiable=True,
                d_point_d_va=scale((iso_f - vb) / d2),
                d_point_d_vb=scale((va - iso_f) / d2),
                d_point_d_iso=scale(1.0 / denom),
                reason="",
            )
        vid = len(vertices)
        vertices.append(Vertex(id=vid, point=point, edge_key=key, t=t))
        records.append(
            IntersectionRecord(
                vertex_id=vid,
                edge_key=key,
                weights=(1.0 - t, t),
                sensitivity=sensitivity,
            )
        )
        vertex_by_edge[key] = vid
        return vid

    faces: List[Face] = []
    degenerate: List[DegenerateFace] = []
    edge_use: Dict[Tuple[int, int], int] = {}

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
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (vids[a], vids[b]) if vids[a] < vids[b] else (vids[b], vids[a])
            edge_use[key] = edge_use.get(key, 0) + 1

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
                        pts = [vertices[v].point for v in vids]
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
                        pts = [vertices[v].point for v, _ in entries]
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
        intersections=tuple(records),
        edge_use_count=dict(edge_use),
    )
