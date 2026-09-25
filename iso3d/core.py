"""体素等值面的共享边交点提取核心实现（marching tetrahedra）。

坐标系：节点 (i, j, k) 的坐标即 (i, j, k)，晶格单位立方体。
每个立方体沿主对角线 (0,0,0)-(1,1,1) 统一分解为 6 个四面体（见 TETRA_TABLE）。
所有四面体边的交点按全局网格边键（升序节点 ID 对）共享；同一条网格边
（无论被多少个四面体引用、遍历方向如何）只生成一条交点记录，其来源边、
插值权重与标量敏感度随之共享，绝不按面重复生成。

交点几何：t = (iso - v_a) / (v_b - v_a)，point = (1-t) P_a + t P_b，
其中 (a, b) 始终是规范边键（节点 ID 升序）的第一、第二端点。
严格跨过 iso 且分母稳定时，敏感度给出 point（及 t）对 v_a、v_b、iso 的
解析导数；端点恰 iso（拓扑由符号扰动决定）或分母为零/不稳定时标记为
不可微，不输出数值上无意义的“伪精确”巨大导数。敏感度只在**当前拓扑
（穿越/不穿越判定）保持不变**的小扰动下有效。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CUBE_CORNERS",
    "MAX_DIM",
    "MAX_NODES",
    "TETRA_TABLE",
    "UNSTABLE_DENOMINATOR_EPS",
    "NONDIFF_ENDPOINT_ON_ISO",
    "NONDIFF_ZERO_DENOMINATOR",
    "NONDIFF_UNSTABLE_DENOMINATOR",
    "DegenerateFace",
    "EdgeIntersection",
    "Face",
    "InvalidInputError",
    "IsosurfaceMesh",
    "MeshStats",
    "PointSensitivity",
    "ScalarField3D",
    "Vertex",
    "canonical_edge_key",
    "extract_isosurface",
    "intersect_edge",
    "perturbed_below",
]

# ---------------------------------------------------------------------------
# 公开常量
# ---------------------------------------------------------------------------

MAX_DIM = 16
"""单个坐标轴方向允许的最大节点数（含端点）。"""

MAX_NODES = MAX_DIM ** 3
"""标量场允许的最大节点总数 16^3 = 4096。"""

UNSTABLE_DENOMINATOR_EPS = 1e-12
"""分母 |v_b - v_a| 的相对稳定性阈值。

当 |v_b - v_a| <= UNSTABLE_DENOMINATOR_EPS * max(1, |v_a|, |v_b|) 时，
该边视为分母不稳定：交点仍按线性插值给出几何位置，但敏感度标记为
不可微（NONDIFF_UNSTABLE_DENOMINATOR），不输出被极小分母放大的、
数值上不可信的“伪精确”导数。
"""

NONDIFF_ENDPOINT_ON_ISO = "endpoint_on_iso"
"""不可微原因：至少一个端点标量恰等于 iso（拓扑由符号扰动决定）。"""

NONDIFF_ZERO_DENOMINATOR = "zero_denominator"
"""不可微原因：两端标量相等（插值分母严格为 0），交点取中点。"""

NONDIFF_UNSTABLE_DENOMINATOR = "unstable_denominator"
"""不可微原因：标量差极小、分母数值不稳定。"""

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
class PointSensitivity:
    """交点对两端点标量值与 iso 的局部解析敏感度（雅可比）。

    交点位于边 (a, b) 上（a < b 为节点 ID），t = (iso - v_a)/(v_b - v_a)，
    point = (1-t) P_a + t P_b。在当前拓扑（穿越判定）保持不变的小扰动下：

        dt/dv_a  = (iso - v_b) / (v_b - v_a)^2
        dt/dv_b  = (v_a - iso) / (v_b - v_a)^2
        dt/diso  = 1 / (v_b - v_a)
        dpoint/d* = (P_b - P_a) * dt/d*

    对沿坐标轴的单位晶格边（P_b - P_a 恰为一个单位向量），point 的导数
    只有一个非零分量，数值上等于 t 的相应导数。

    该导数仅适用于**固定拓扑**：扰动若改变任一端点相对 iso 的位置（穿越
    翻转、交点出现/消失），交点不再连续可微，不得外推到跨拓扑情形。
    """

    dt_dva: float
    """t 对规范边键第一端点标量 v_a 的导数。"""
    dt_dvb: float
    """t 对规范边键第二端点标量 v_b 的导数。"""
    dt_diso: float
    """t 对等值 iso 的导数。"""
    dpoint_dva: Tuple[float, float, float]
    """point 对 v_a 的梯度。"""
    dpoint_dvb: Tuple[float, float, float]
    """point 对 v_b 的梯度。"""
    dpoint_diso: Tuple[float, float, float]
    """point 对 iso 的梯度。"""


@dataclass(frozen=True)
class EdgeIntersection:
    """一条全局网格边上的（唯一）交点记录：来源、权重与敏感度共享。

    同一条网格边被多个四面体/面引用时，``extract_isosurface`` 保证所有
    引用拿到的是**同一个** EdgeIntersection 对象（按 edge_key 去重），
    其来源端点、插值权重与解析导数全局唯一，不会因遍历方向或所属面不同
    而产生不同答案。

    字段
    ----
    edge_key:
        来源网格边：升序全局节点 ID 对 (a, b)，a < b。
    point / t:
        交点坐标与插值权重（参数）。``weights`` 为两端点的凸组合权重
        ``((a 端权重 1-t), (b 端权重 t))``，与几何一一对应，且和为 1。
    differentiable:
        敏感度是否可解析给出。True 时 ``sensitivity`` 为 PointSensitivity；
        False 时 ``sensitivity`` 为 None 且 ``nondifferentiable_reason``
        为 NONDIFF_* 常量之一。
    nondifferentiable_reason:
        不可微原因（可微时为 None）：端点恰 iso、分母为零或分母不稳定。
    """

    edge_key: Tuple[int, int]
    point: Tuple[float, float, float]
    t: float
    weights: Tuple[Tuple[int, float], Tuple[int, float]]
    """((节点 a, 权重 1-t), (节点 b, 权重 t))：来源节点与插值权重。"""
    differentiable: bool
    sensitivity: Optional[PointSensitivity]
    nondifferentiable_reason: Optional[str]


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
    intersections: Tuple[EdgeIntersection, ...] = ()
    """按顶点 id（即首次出现顺序）排列的共享边交点记录，与 vertices 一一对应。

    ``intersections[v.id]`` 即顶点 v 的来源边、插值权重与局部标量敏感度。
    同一条网格边在整个网格中只有一条记录，被任意多个四面体引用都共享它。
    """
    edge_use_count: Dict[Tuple[int, int], int] = field(compare=False, default_factory=dict)
    """每条网格边（端点顶点 ID 对，升序）被多少个三角形使用。"""

    def total_area(self) -> float:
        """所有保留三角形面积之和（与 stats.total_area 一致）。"""
        return self.stats.total_area

    def intersection_for_vertex(self, vertex_id: int) -> EdgeIntersection:
        """取某个等值面顶点对应的共享边交点记录。"""
        return self.intersections[vertex_id]

    def intersection_for_edge(self, id_a: int, id_b: int) -> EdgeIntersection:
        """按来源网格边的两个全局节点 ID 取共享交点记录；不存在则 KeyError。"""
        key = canonical_edge_key(id_a, id_b)
        for rec in self.intersections:
            if rec.edge_key == key:
                return rec
        raise KeyError(key)

    def intersections_affected_by_node(self, node_id: int) -> Tuple[EdgeIntersection, ...]:
        """源节点 node_id 的标量发生（不改变拓扑的）小扰动时，受影响的交点。

        即所有来源边端点包含 node_id 的共享交点记录。返回的每条记录的
        敏感度给出该节点标量对交点位置的局部导数；这是“一个输入节点
        变化影响哪些交点”的直接答案，不做整块/整体场的传播。
        """
        return tuple(
            rec
            for rec in self.intersections
            if rec.edge_key[0] == node_id or rec.edge_key[1] == node_id
        )


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


def intersect_edge(
    id_a: int,
    id_b: int,
    value_a: float,
    value_b: float,
    iso: float,
    point_a: Sequence[float],
    point_b: Sequence[float],
) -> EdgeIntersection:
    """计算一条网格边上 iso 交点的来源、权重与局部标量敏感度。

    这是不依赖网格遍历的纯函数（被 ``extract_isosurface`` 内部复用）：
    给定规范边端点 (id_a, id_b)（无需预先排序，内部规范化）、两端标量、
    iso 与两端几何坐标，返回唯一的 :class:`EdgeIntersection` 记录。

    可微性判定
    ----------
    - 两端**严格跨过** iso 且分母 |v_b - v_a| 数值稳定：可微，返回解析
      导数（见 :class:`PointSensitivity`）。
    - 任一端点恰等于 iso：拓扑由符号扰动决定，标记
      ``NONDIFF_ENDPOINT_ON_ISO``。
    - v_a == v_b（分母严格为 0；仅可能两端都恰等于 iso）：交点取中点，
      标记 ``NONDIFF_ZERO_DENOMINATOR``（优先于端点原因）。
    - 标量差极小、分母低于相对稳定性阈值：标记
      ``NONDIFF_UNSTABLE_DENOMINATOR``，不输出被放大的巨大伪导数。

    不可微时 ``point``/``t`` 仍按与提取器一致的规则给出（分母为 0 取
    中点；端点恰 iso 取 t=0/1），仅 ``sensitivity`` 为 None。

    抛出
    ----
    InvalidInputError
        节点 ID 非 int（含 bool）、标量/iso/坐标为 NaN 或 ±Infinity、
        坐标不是长度 3 的序列时。
    """
    for name, node_id in (("id_a", id_a), ("id_b", id_b)):
        if isinstance(node_id, bool) or not isinstance(node_id, int):
            raise InvalidInputError(
                name, f"must be an int node id, got {type(node_id).__name__}"
            )
    _finite_float("value_a", value_a)
    _finite_float("value_b", value_b)
    iso_f = _finite_float("iso", iso)
    for name, p in (("point_a", point_a), ("point_b", point_b)):
        if not isinstance(p, Sequence) or isinstance(p, (str, bytes)) or len(p) != 3:
            raise InvalidInputError(
                name, "must be a length-3 sequence of real coordinates"
            )
        for d in range(3):
            _finite_float(f"{name}[{d}]", p[d])
    key = canonical_edge_key(id_a, id_b)
    if key != (id_a, id_b):
        # 规范化：把传反的端点交换到与边键一致的顺序。
        id_a, id_b = key
        value_a, value_b = value_b, value_a
        point_a, point_b = point_b, point_a
    pa = (float(point_a[0]), float(point_a[1]), float(point_a[2]))
    pb = (float(point_b[0]), float(point_b[1]), float(point_b[2]))
    va, vb = float(value_a), float(value_b)
    denom = vb - va
    delta = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])

    reason: Optional[str]
    sensitivity: Optional[PointSensitivity]
    if denom == 0.0:
        t = 0.5
        reason = NONDIFF_ZERO_DENOMINATOR
        sensitivity = None
    else:
        t = (iso_f - va) / denom
        if va == iso_f or vb == iso_f:
            reason = NONDIFF_ENDPOINT_ON_ISO
            sensitivity = None
        elif abs(denom) <= UNSTABLE_DENOMINATOR_EPS * max(
            1.0, abs(va), abs(vb)
        ):
            reason = NONDIFF_UNSTABLE_DENOMINATOR
            sensitivity = None
        else:
            reason = None
            inv_d = 1.0 / denom
            dt_dva = (iso_f - vb) * inv_d * inv_d
            dt_dvb = (va - iso_f) * inv_d * inv_d
            dt_diso = inv_d

            def scale_dt(dt: float):
                return (delta[0] * dt, delta[1] * dt, delta[2] * dt)

            sensitivity = PointSensitivity(
                dt_dva=dt_dva,
                dt_dvb=dt_dvb,
                dt_diso=dt_diso,
                dpoint_dva=scale_dt(dt_dva),
                dpoint_dvb=scale_dt(dt_dvb),
                dpoint_diso=scale_dt(dt_diso),
            )

    point = (
        pa[0] + t * delta[0],
        pa[1] + t * delta[1],
        pa[2] + t * delta[2],
    )
    return EdgeIntersection(
        edge_key=key,
        point=point,
        t=t,
        weights=((id_a, 1.0 - t), (id_b, t)),
        differentiable=sensitivity is not None,
        sensitivity=sensitivity,
        nondifferentiable_reason=reason,
    )


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
    values = field.values

    def nid(i: int, j: int, k: int) -> int:
        return (k * nj + j) * ni + i

    def ncoord(node_id: int) -> Tuple[float, float, float]:
        i = node_id % ni
        j = (node_id // ni) % nj
        k = node_id // (ni * nj)
        return (float(i), float(j), float(k))

    vertices: List[Vertex] = []
    intersections: List[EdgeIntersection] = []
    vertex_by_edge: Dict[Tuple[int, int], int] = {}

    def get_vertex(key: Tuple[int, int]) -> int:
        vid = vertex_by_edge.get(key)
        if vid is not None:
            return vid
        id_a, id_b = key
        rec = intersect_edge(
            id_a,
            id_b,
            values[id_a],
            values[id_b],
            iso_f,
            ncoord(id_a),
            ncoord(id_b),
        )
        vid = len(vertices)
        vertices.append(
            Vertex(id=vid, point=rec.point, edge_key=key, t=rec.t)
        )
        intersections.append(rec)
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
        intersections=tuple(intersections),
        edge_use_count=dict(edge_use),
    )
