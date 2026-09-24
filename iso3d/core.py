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
    "ScalarBlock3D",
    "ScalarField3D",
    "Vertex",
    "canonical_edge_key",
    "extract_isosurface",
    "extract_isosurface_blocks",
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
class ScalarBlock3D:
    """全局规则晶格中的一个同分辨率相邻体块（子区域）。

    体块覆盖全局节点坐标
    ``[origin[0], origin[0]+ni-1] x [origin[1], origin[1]+nj-1]
    x [origin[2], origin[2]+nk-1]``（含端点，晶格单位），因此含
    ``(ni-1)*(nj-1)*(nk-1)`` 个完整立方体。

    values 为该体块局部行优先扁平序列：局部节点 (li, lj, lk)
    （均从 0 起）的值为 ``values[(lk*nj + lj)*ni + li]``，对应全局
    节点 (origin[0]+li, origin[1]+lj, origin[2]+lk)。

    体块只描述几何位置与标量数据；全局网格形状在
    :func:`extract_isosurface_blocks` 调用时统一给出。
    """

    origin: Tuple[int, int, int]
    ni: int
    nj: int
    nk: int
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.origin, tuple) or len(self.origin) != 3:
            raise InvalidInputError(
                "origin", f"must be a (oi, oj, ok) tuple, got {type(self.origin).__name__}"
            )
        for axis, o in zip(("oi", "oj", "ok"), self.origin):
            if isinstance(o, bool) or not isinstance(o, int):
                raise InvalidInputError(
                    "origin", f"{axis} must be an int, got {type(o).__name__}"
                )
            if o < 0:
                raise InvalidInputError("origin", f"{axis} must be >= 0, got {o}")
        ni = _check_axis("ni", self.ni)
        nj = _check_axis("nj", self.nj)
        nk = _check_axis("nk", self.nk)
        vals = self.values
        if isinstance(vals, (str, bytes)) or not isinstance(vals, Sequence):
            raise InvalidInputError(
                "values", f"must be a sequence of real numbers, got {type(vals).__name__}"
            )
        expected = ni * nj * nk
        if len(vals) != expected:
            raise InvalidInputError(
                "values", f"expected {expected} values for block {ni}x{nj}x{nk}, got {len(vals)}"
            )
        checked: List[float] = []
        for idx, v in enumerate(vals):
            checked.append(_finite_float(f"values[{idx}]", v))
        object.__setattr__(self, "values", tuple(checked))

    @property
    def end(self) -> Tuple[int, int, int]:
        """体块在各轴上的末节点全局坐标（含端点）。"""
        return (
            self.origin[0] + self.ni - 1,
            self.origin[1] + self.nj - 1,
            self.origin[2] + self.nk - 1,
        )

    def global_node_id(self, shape: Tuple[int, int, int], li: int, lj: int, lk: int) -> int:
        """局部节点 (li, lj, lk) 在给定全局形状下的全局节点 ID。"""
        gni, gnj, _gnk = shape
        i = self.origin[0] + li
        j = self.origin[1] + lj
        k = self.origin[2] + lk
        return (k * gnj + j) * gni + i


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


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

def extract_isosurface(field: ScalarField3D, iso: float) -> IsosurfaceMesh:
    """从单个标量场提取 iso 等值面（marching tetrahedra）。

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
    shape = (field.ni, field.nj, field.nk)
    whole = ScalarBlock3D(
        origin=(0, 0, 0),
        ni=field.ni,
        nj=field.nj,
        nk=field.nk,
        values=field.values,
    )
    return _extract_mesh(shape, iso_f, (whole,))


def extract_isosurface_blocks(
    blocks: Sequence["ScalarBlock3D"],
    iso: float,
    shape: Tuple[int, int, int] | None = None,
) -> IsosurfaceMesh:
    """从若干同分辨率相邻体块分块提取并拼接 iso 等值面。

    每个 :class:`ScalarBlock3D` 独立完成本块的 marching tetrahedra
    （节点身份用全局节点 ID、坐标用全局晶格坐标、符号扰动用全局 ID
    奇偶），随后只按**全局网格边键**合并接缝顶点：边键相同即为同一
    个交点，顶点 ID 由该边键在全局网格中的排名唯一确定，与切分方式
    无关。不使用任何坐标距离/容差猜测。

    接缝契约
    --------
    - 所有体块必须**无缝无叠地铺满**全局网格的全部立方体：每个立方
      体恰好属于一个体块（共享面上的节点由相邻块共同引用）。
    - 被多个体块引用的共享节点，其标量值必须**逐位相等**；共享采样
      点的唯一所有者按覆盖块中 ``origin`` 最大者确定（origin 相同
      时取输入序列中靠后者）。值不一致直接报错。
    - 接缝两侧对同一条网格边算出的交点因此必然逐位相同；合并时仍做
      **精确相等校验**，不假设、不容差。
    - 在上述契约下，本函数结果与把数据拼成一个 :class:`ScalarField3D`
      后调用 :func:`extract_isosurface` 的结果**完全相同**（顶点、
      面、定向、退化面、统计逐字段相等）。

    参数
    ----
    blocks : Sequence[ScalarBlock3D]
        非空的体块序列，铺满 shape 描述的全局网格。
    iso : int 或 float
        等值，必须为有限实数。
    shape : (gni, gnj, gnk)，可选
        全局网格每维节点数。省略时取所有体块包围盒
        ``(max(end)+1, ...)``；显式给出可及早发现末端缺层的铺法。
        全局网格同样受每维 <= 16、总节点 <= 4096 限制。

    抛出
    ----
    InvalidInputError
        任何输入违反契约时（含体块越界、立方体重叠/缺失、共享节点
        值不一致）；校验在任何计算前完成，不产生部分结果，也不修改
        输入体块。
    """
    if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Sequence):
        raise InvalidInputError(
            "blocks", f"must be a sequence of ScalarBlock3D, got {type(blocks).__name__}"
        )
    if len(blocks) == 0:
        raise InvalidInputError("blocks", "must contain at least one block")
    for idx, blk in enumerate(blocks):
        if not isinstance(blk, ScalarBlock3D):
            raise InvalidInputError(
                "blocks",
                f"blocks[{idx}] must be a ScalarBlock3D, got {type(blk).__name__}",
            )
    iso_f = _finite_float("iso", iso)

    if shape is None:
        resolved_shape = (
            max(b.origin[0] + b.ni for b in blocks),
            max(b.origin[1] + b.nj for b in blocks),
            max(b.origin[2] + b.nk for b in blocks),
        )
        for axis, dim in enumerate(resolved_shape):
            if dim > MAX_DIM:
                raise InvalidInputError(
                    "shape",
                    f"inferred global grid has {dim} nodes on axis {axis}, "
                    f"exceeds MAX_DIM {MAX_DIM}; pass an explicit shape if intended",
                )
        gni, gnj, gnk = resolved_shape
        if gni * gnj * gnk > MAX_NODES:
            raise InvalidInputError(
                "shape",
                f"inferred global grid {gni}x{gnj}x{gnk} has {gni * gnj * gnk} nodes, "
                f"exceeds {MAX_NODES}",
            )
    else:
        if not isinstance(shape, tuple) or len(shape) != 3:
            raise InvalidInputError(
                "shape", f"must be a (gni, gnj, gnk) tuple, got {type(shape).__name__}"
            )
        resolved_shape = tuple(
            _check_axis(name, v)
            for name, v in (
                ("shape[0]", shape[0]),
                ("shape[1]", shape[1]),
                ("shape[2]", shape[2]),
            )
        )
        gni, gnj, gnk = resolved_shape
        if gni * gnj * gnk > MAX_NODES:
            raise InvalidInputError(
                "shape",
                f"total node count {gni}*{gnj}*{gnk} = {gni * gnj * gnk} "
                f"exceeds {MAX_NODES}",
            )

    for idx, blk in enumerate(blocks):
        for axis, (end, dim) in enumerate(zip(blk.end, resolved_shape)):
            if end >= dim:
                raise InvalidInputError(
                    "blocks",
                    f"blocks[{idx}] exceeds global grid on axis {axis}: "
                    f"end coordinate {end} >= shape {dim}",
                )

    return _extract_mesh(resolved_shape, iso_f, tuple(blocks))


def _extract_mesh(shape, iso_f, blocks):
    """统一引擎：blocks 已通过公开入口校验。

    合并全局节点值（唯一所有者 + 逐位一致性）→ 立方体所有权铺法
    校验 → 每块独立 marching → 按全局边键精确合并 → 全局排名赋顶点
    ID、按 (k,j,i,tet,sub) 扫描序排列面片。
    """
    gni, gnj, gnk = shape

    def global_id(i, j, k):
        return (k * gnj + j) * gni + i

    def gcoord(node_id):
        i = node_id % gni
        j = (node_id // gni) % gnj
        k = node_id // (gni * gnj)
        return (float(i), float(j), float(k))

    # --- 1. 合并节点值：共享采样点唯一所有者，值必须逐位一致 ---------
    node_values: Dict[int, float] = {}
    node_owner: Dict[int, int] = {}
    for bi, blk in enumerate(blocks):
        oi, oj, ok = blk.origin
        for lk in range(blk.nk):
            k = ok + lk
            for lj in range(blk.nj):
                j = oj + lj
                base = (lk * blk.nj + lj) * blk.ni
                for li in range(blk.ni):
                    gid = global_id(oi + li, j, k)
                    val = blk.values[base + li]
                    if gid in node_values:
                        if node_values[gid] != val:
                            raise InvalidInputError(
                                "blocks",
                                f"shared node ({oi + li}, {j}, {k}) has inconsistent "
                                f"values: {node_values[gid]!r} from blocks[{node_owner[gid]}] "
                                f"vs {val!r} from blocks[{bi}]",
                            )
                        # 所有者：origin 最大者；origin 相同取输入序列靠后者。
                        if blk.origin >= blocks[node_owner[gid]].origin:
                            node_owner[gid] = bi
                    else:
                        node_values[gid] = val
                        node_owner[gid] = bi

    # --- 2. 立方体所有权：无缝无叠铺满 ------------------------------
    total_cells = (gni - 1) * (gnj - 1) * (gnk - 1)
    cell_owner: Dict[Tuple[int, int, int], int] = {}
    for bi, blk in enumerate(blocks):
        oi, oj, ok = blk.origin
        for dk in range(blk.nk - 1):
            for dj in range(blk.nj - 1):
                for di in range(blk.ni - 1):
                    cell = (oi + di, oj + dj, ok + dk)
                    if cell in cell_owner:
                        raise InvalidInputError(
                            "blocks",
                            f"cell {cell} is owned by both blocks[{cell_owner[cell]}] "
                            f"and blocks[{bi}]: blocks must not overlap",
                        )
                    cell_owner[cell] = bi
    if len(cell_owner) != total_cells:
        for k in range(gnk - 1):
            for j in range(gnj - 1):
                for i in range(gni - 1):
                    if (i, j, k) not in cell_owner:
                        raise InvalidInputError(
                            "blocks",
                            f"cell {(i, j, k)} is not covered by any block: "
                            "blocks must tile the whole grid",
                        )

    # --- 3. 每块独立 marching tetrahedra ----------------------------
    vertex_data: Dict[Tuple[int, int], Tuple[Tuple[float, float, float], float]] = {}
    face_records: List[Tuple[Tuple[int, int, int, int, int], Tuple[int, int, int]]] = []
    degen_records: List[Tuple[Tuple[int, int, int, int, int], Tuple[Tuple[int, int], ...]]] = []

    for bi, blk in enumerate(blocks):
        local_vertices: Dict[Tuple[int, int], Tuple[Tuple[float, float, float], float]] = {}

        def get_vertex(key):
            data = local_vertices.get(key)
            if data is not None:
                return data
            id_a, id_b = key
            va = node_values[id_a]
            vb = node_values[id_b]
            denom = vb - va
            if denom == 0.0:
                # 仅当两端值都恰等于 iso 时发生（否则两端同侧，不会成为穿越边）。
                t = 0.5
            else:
                t = (iso_f - va) / denom
            pa = gcoord(id_a)
            pb = gcoord(id_b)
            point = (
                pa[0] + t * (pb[0] - pa[0]),
                pa[1] + t * (pb[1] - pa[1]),
                pa[2] + t * (pb[2] - pa[2]),
            )
            data = (point, t)
            local_vertices[key] = data
            return data

        oi, oj, ok = blk.origin
        for lk in range(blk.nk - 1):
            for lj in range(blk.nj - 1):
                for li in range(blk.ni - 1):
                    ci, cj, ck = oi + li, oj + lj, ok + lk
                    corner_ids = tuple(
                        global_id(ci + di, cj + dj, ck + dk)
                        for (di, dj, dk) in CUBE_CORNERS
                    )
                    below = tuple(
                        perturbed_below(node_values[c], iso_f, c) for c in corner_ids
                    )
                    for tet_index, tet in enumerate(TETRA_TABLE):
                        tet_nodes = tuple(corner_ids[c] for c in tet)
                        tet_below = tuple(below[c] for c in tet)
                        nb = sum(tet_below)
                        if nb == 0 or nb == 4:
                            continue
                        crossings = []
                        for ea, eb in _TET_EDGES:
                            if tet_below[ea] != tet_below[eb]:
                                crossings.append(
                                    canonical_edge_key(tet_nodes[ea], tet_nodes[eb])
                                )

                        def emit(vkeys, sub, _ck=ck, _cj=cj, _ci=ci, _tet=tet_index):
                            pts = [get_vertex(key)[0] for key in vkeys]
                            if _tri_area(pts[0], pts[1], pts[2]) == 0.0:
                                degen_records.append(
                                    ((_ck, _cj, _ci, _tet, sub), tuple(vkeys))
                                )
                                return
                            face_records.append(
                                ((_ck, _cj, _ci, _tet, sub), tuple(vkeys))
                            )

                        # K4 二染色下穿越边数只可能是 3 或 4。
                        if len(crossings) == 3:
                            keys = crossings
                            pts = [get_vertex(key)[0] for key in keys]
                            ref_local = 0 if not tet_below[0] else next(
                                idx for idx in range(1, 4) if not tet_below[idx]
                            )
                            order = _oriented_loop(
                                pts, [0, 1, 2], gcoord(tet_nodes[ref_local])
                            )
                            emit(
                                (keys[order[0]], keys[order[1]], keys[order[2]]),
                                0,
                            )
                        else:  # 4 条穿越边：一个四边形，切成两个三角形
                            pts = [get_vertex(key)[0] for key in crossings]
                            centroid = (
                                sum(p[0] for p in pts) / 4.0,
                                sum(p[1] for p in pts) / 4.0,
                                sum(p[2] for p in pts) / 4.0,
                            )
                            above_local = [idx for idx in range(4) if not tet_below[idx]]
                            below_local = [idx for idx in range(4) if tet_below[idx]]
                            above_mid = tuple(
                                sum(gcoord(tet_nodes[idx])[d] for idx in above_local)
                                / len(above_local)
                                for d in range(3)
                            )
                            below_mid = tuple(
                                sum(gcoord(tet_nodes[idx])[d] for idx in below_local)
                                / len(below_local)
                                for d in range(3)
                            )
                            ref = _sub(above_mid, below_mid)
                            if _norm(ref) == 0.0:
                                ref = (1.0, 0.0, 0.0)
                            axis = min(range(3), key=lambda d: (abs(ref[d]), d))
                            basis = [0.0, 0.0, 0.0]
                            basis[axis] = 1.0
                            u = _cross(ref, tuple(basis))
                            u = tuple(c / _norm(u) for c in u)
                            w = _cross(ref, u)

                            def angle(idx):
                                rel = _sub(pts[idx], centroid)
                                return math.atan2(_dot(rel, w), _dot(rel, u))

                            order = sorted(
                                range(4), key=lambda idx: (angle(idx), crossings[idx])
                            )
                            loop = _oriented_loop(pts, list(order), above_mid)
                            v0, v1, v2, v3 = (crossings[loop[q]] for q in range(4))
                            emit((v0, v1, v2), 0)
                            emit((v0, v2, v3), 1)

        # 合并本块交点：只按全局边键去重；跨块副本必须逐位一致（不靠距离）。
        for key, data in local_vertices.items():
            if key in vertex_data:
                if vertex_data[key] != data:
                    raise InvalidInputError(
                        "blocks",
                        f"intersection on global grid edge {key} disagrees across "
                        "blocks: shared node values must be identical",
                    )
            else:
                vertex_data[key] = data

    # --- 4. 全局身份：顶点 ID = 全局边键排名；面片按单元扫描序排序 ----
    all_edge_keys = tuple(sorted(vertex_data))
    vertex_id_by_key = {key: idx for idx, key in enumerate(all_edge_keys)}
    vertices = [
        Vertex(id=idx, point=vertex_data[key][0], edge_key=key, t=vertex_data[key][1])
        for idx, key in enumerate(all_edge_keys)
    ]

    face_records.sort(key=lambda rec: rec[0])
    faces: List[Face] = []
    edge_use: Dict[Tuple[int, int], int] = {}
    total_area = 0.0
    for _sort_key, vkeys in face_records:
        vids = tuple(vertex_id_by_key[key] for key in vkeys)
        p0, p1, p2 = (vertices[v].point for v in vids)
        area = _tri_area(p0, p1, p2)
        total_area += area
        faces.append(Face(id=len(faces), vertex_ids=vids, area=area))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (vids[a], vids[b]) if vids[a] < vids[b] else (vids[b], vids[a])
            edge_use[key] = edge_use.get(key, 0) + 1

    degen_records.sort(key=lambda rec: rec[0])
    degenerate = tuple(
        DegenerateFace(
            cell=(ci, cj, ck),
            tet_index=tet_index,
            edge_keys=edge_keys,
        )
        for (_ck, _cj, _ci, tet_index, _sub), edge_keys in degen_records
        for (ci, cj, ck) in [(_ci, _cj, _ck)]
    )

    boundary_edges = tuple(sorted(key for key, c in edge_use.items() if c == 1))
    nonmanifold_edges = tuple(sorted(key for key, c in edge_use.items() if c >= 3))
    node_count = gni * gnj * gnk
    cell_count = total_cells
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
        degenerate_faces=degenerate,
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        edge_use_count=edge_use,
    )
