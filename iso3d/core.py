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
    "BandFace",
    "BandMesh",
    "BandStats",
    "BandVertex",
    "canonical_edge_key",
    "extract_band",
    "extract_isosurface",
    "perturbed_below",
    "signed_volume",
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


# ---------------------------------------------------------------------------

BAND_BELOW = -1
"""数值带分类：f < low。"""
BAND_INSIDE = 0
"""数值带分类：low <= f <= high（恰等于阈值统一归带内）。"""
BAND_ABOVE = 1
"""数值带分类：f > high。"""


@dataclass(frozen=True)
class BandVertex:
    """数值带边界顶点。

    kind 为 ``"edge"`` 时是网格边上的阈值交点（level 给出所属阈值，
    0=low、1=high，low/high 即使几何重合也保持不同身份）；
    kind 为 ``"node"`` 时是恰在带内的网格节点（盖片用）。
    """

    id: int
    point: Tuple[float, float, float]
    kind: str
    edge_key: Tuple[int, int] | None
    level: int | None
    t: float


@dataclass(frozen=True)
class BandFace:
    """数值带边界三角形。

    surface 为 ``"low"``（低阈值墙）、``"high"``（高阈值墙）或
    ``"cap"``（网格外边界盖片）。统一按朝外（数值带外侧）定向。
    """

    id: int
    vertex_ids: Tuple[int, int, int]
    area: float
    surface: str


@dataclass(frozen=True)
class BandStats:
    """数值带提取结果统计。"""

    low: float
    high: float
    thickness_zero: bool
    node_count: int
    cell_count: int
    tetrahedron_count: int
    vertex_count: int
    face_count: int
    low_wall_face_count: int
    high_wall_face_count: int
    cap_face_count: int
    degenerate_face_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    total_area: float
    signed_volume: float


@dataclass(frozen=True)
class BandMesh:
    """``extract_band`` 的定向三角网结果（法向统一朝数值带外侧）。"""

    low: float
    high: float
    thickness_zero: bool
    """low == high 时为 True：结果是零厚度等值面，不构成有体积实体。"""
    vertices: Tuple[BandVertex, ...]
    faces: Tuple[BandFace, ...]
    stats: BandStats
    degenerate_faces: Tuple[DegenerateFace, ...]
    boundary_edges: Tuple[Tuple[int, int], ...]
    nonmanifold_edges: Tuple[Tuple[int, int], ...]
    edge_use_count: Dict[Tuple[int, int], int] = field(compare=False)

    def total_area(self) -> float:
        return self.stats.total_area

    def signed_volume(self) -> float:
        """有向三角网的散度体积积分（与 stats.signed_volume 一致）。"""
        return self.stats.signed_volume


def signed_volume(vertices, faces) -> float:
    """对定向三角网计算闭合体积积分 (1/6) * sum a.(b x c)。

    vertices 为带 ``point`` 属性的序列；faces 的每项取 ``vertex_ids``。
    法向朝外时结果为正。开曲面得到的值没有体积意义。
    """
    total = 0.0
    for face in faces:
        va, vb, vc = (vertices[v].point for v in face.vertex_ids)
        total += _dot(va, _cross(vb, vc))
    return total / 6.0


def _band_caps(nid, ncoord, values, ni, nj, nk, low_f, high_f,
               vertices, vertex_by_key, get_edge_vertex, get_node_vertex,
               emit, emit_fan, degenerate):
    """在数值带与网格外边界相交处补盖片。

    对四面体化后落在网格 6 个外平面上的三角形（每条恰由一个四面体
    贡献，故全局不重不漏），裁剪到带内部分 low<=f<=high，再以朝外
    法向三角化。交点身份按 (网格边键, 阈值) 解析，与阈值墙共享；
    恰等于阈值的节点统一归属带内，相邻单元不会重复盖面。
    """
    def in_band(v: float) -> bool:
        return low_f <= v <= high_f

    seen_tri = set()

    def clip_boundary_tri(tri_nodes, cell, tet_index, outward):
        tri_key = tuple(sorted(tri_nodes))
        if tri_key in seen_tri:
            return
        seen_tri.add(tri_key)
        # 沿原三角形周长按参数累积事件点：带内节点与阈值交点。
        # 一条边可能同时穿过 low、high（中间整段在带外/带内），
        # 因此按边上插值参数 t 排序收集。
        ordered = []
        for ea in range(3):
            eb = (ea + 1) % 3
            na, nb = tri_nodes[ea], tri_nodes[eb]
            va, vb = values[na], values[nb]
            events = []
            if in_band(va):
                events.append((0.0, get_node_vertex(na)))
            for level, tag in ((low_f, 0), (high_f, 1)):
                lo, hi = (va, vb) if va < vb else (vb, va)
                if lo < level < hi:
                    key = canonical_edge_key(na, nb)
                    events.append(
                        ((level - va) / (vb - va),
                         get_edge_vertex(key, level, tag))
                    )
            events.sort(key=lambda item: item[0])
            for _, vid in events:
                if not ordered or ordered[-1] != vid:
                    ordered.append(vid)
        poly = []
        for vid in ordered:
            if not poly or poly[-1] != vid:
                poly.append(vid)
        if len(poly) > 1 and poly[0] == poly[-1]:
            poly.pop()
        if len(poly) < 3:
            if len(poly) >= 1:
                degenerate.append(DegenerateFace(
                    cell=cell, tet_index=tet_index,
                    edge_keys=tuple(vertices[v].edge_key for v in poly),
                ))
            return
        pts = [vertices[v].point for v in poly]
        emit_fan(poly, pts, outward, cell, tet_index, "cap")

    # 四面体面（4 选 3）；判断三个角点是否都落在某个网格外平面上。
    from itertools import combinations as _comb
    tet_faces_local = tuple(_comb(range(4), 3))

    for k in range(nk - 1):
        for j in range(nj - 1):
            for i in range(ni - 1):
                cell = (i, j, k)
                corner_ids = tuple(
                    nid(i + di, j + dj, k + dk) for (di, dj, dk) in CUBE_CORNERS
                )
                planes = []
                if i == 0: planes.append((0, 0, -1.0))
                if i == ni - 2: planes.append((0, 1, 1.0))
                if j == 0: planes.append((1, 0, -1.0))
                if j == nj - 2: planes.append((1, 1, 1.0))
                if k == 0: planes.append((2, 0, -1.0))
                if k == nk - 2: planes.append((2, 1, 1.0))
                if not planes:
                    continue
                for tet_index, tet in enumerate(TETRA_TABLE):
                    for lf in tet_faces_local:
                        glob = tuple(corner_ids[tet[li]] for li in lf)
                        for axis, fixed_i, sign in planes:
                            if all(CUBE_CORNERS[tet[li]][axis] == fixed_i for li in lf):
                                outward = tuple(sign if d == axis else 0.0 for d in range(3))
                                clip_boundary_tri(glob, cell, tet_index, outward)


def extract_band(field: ScalarField3D, low, high) -> BandMesh:
    """提取 ``low <= f <= high`` 数值带的封闭定向边界。

    边界由三部分组成：低阈值 ``low`` 等值面墙、高阈值 ``high``
    等值面墙，以及数值带与网格外边界相交处的盖片。所有三角形统一
    按**法向朝向数值带外侧**定向。

    标量恰等于阈值的节点统一归属带内（``low<=f``、``f<=high``），
    全局一致，避免相邻单元重复盖面。

    ``low == high`` 时为零厚度特殊结果：只输出 ``f == low`` 等值面
    （标记 ``thickness_zero``），不补盖片、不宣称有向体积。

    抛出 ``InvalidInputError``：field 类型错、low/high 非有限实数、
    bool、或 ``low > high``。
    """
    if not isinstance(field, ScalarField3D):
        raise InvalidInputError(
            "field", f"must be a ScalarField3D, got {type(field).__name__}"
        )
    low_f = _finite_float("low", low)
    high_f = _finite_float("high", high)
    if low_f > high_f:
        raise InvalidInputError(
            "low", f"must be <= high, got low={low_f} > high={high_f}"
        )
    return _band_extract(field, low_f, high_f, low_f == high_f)


def _band_extract(
    field: ScalarField3D,
    low_f: float,
    high_f: float,
    zero_thickness: bool,
):
    ni, nj, nk = field.ni, field.nj, field.nk
    values = field.values

    def nid(i: int, j: int, k: int) -> int:
        return (k * nj + j) * ni + i

    def ncoord(node_id: int) -> Tuple[float, float, float]:
        i = node_id % ni
        j = (node_id // ni) % nj
        k = node_id // (ni * nj)
        return (float(i), float(j), float(k))

    vertices: List[BandVertex] = []
    vertex_by_key: Dict[tuple, int] = {}

    def get_edge_vertex(key: Tuple[int, int], level_value: float, level_tag: int) -> int:
        vkey = ("e", key, level_tag)
        vid = vertex_by_key.get(vkey)
        if vid is not None:
            return vid
        id_a, id_b = key
        va, vb = values[id_a], values[id_b]
        denom = vb - va
        t = 0.5 if denom == 0.0 else (level_value - va) / denom
        pa, pb = ncoord(id_a), ncoord(id_b)
        point = (
            pa[0] + t * (pb[0] - pa[0]),
            pa[1] + t * (pb[1] - pa[1]),
            pa[2] + t * (pb[2] - pa[2]),
        )
        if t == 0.0:
            return get_node_vertex(id_a)
        if t == 1.0:
            return get_node_vertex(id_b)
        vid = len(vertices)
        vertices.append(BandVertex(id=vid, point=point, kind="edge",
                                   edge_key=key, level=level_tag, t=t))
        vertex_by_key[vkey] = vid
        return vid

    def get_node_vertex(node: int) -> int:
        vkey = ("n", node)
        vid = vertex_by_key.get(vkey)
        if vid is not None:
            return vid
        p = ncoord(node)
        vid = len(vertices)
        vertices.append(BandVertex(id=vid, point=p, kind="node",
                                   edge_key=None, level=None, t=0.0))
        vertex_by_key[vkey] = vid
        return vid

    faces: List[BandFace] = []
    degenerate: List[DegenerateFace] = []
    edge_use: Dict[Tuple[int, int], int] = {}

    def emit(vids, cell, tet_index, surface):
        p0 = vertices[vids[0]].point
        p1 = vertices[vids[1]].point
        p2 = vertices[vids[2]].point
        area = _tri_area(p0, p1, p2)
        if area == 0.0:
            degenerate.append(DegenerateFace(
                cell=cell, tet_index=tet_index,
                edge_keys=tuple(vertices[v].edge_key for v in vids),
            ))
            return
        faces.append(BandFace(id=len(faces), vertex_ids=tuple(vids),
                              area=area, surface=surface))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            x, y = vids[a], vids[b]
            ek = (x, y) if x < y else (y, x)
            edge_use[ek] = edge_use.get(ek, 0) + 1

    def emit_fan(poly_vids, pts, outward, cell, tet_index, surface):
        if len(poly_vids) < 3:
            return
        normal = _cross(_sub(pts[1], pts[0]), _sub(pts[2], pts[0]))
        rev = _dot(normal, outward) < 0.0
        m = len(poly_vids)
        for idx in range(1, m - 1):
            tri = (poly_vids[0], poly_vids[idx], poly_vids[idx + 1])
            if rev:
                tri = (tri[0], tri[2], tri[1])
            emit(tri, cell, tet_index, surface)

    # -- 阈值墙：对每个阈值做一次标准 marching tetrahedra ----------------
    # level_tag 0 = low（二分类：f < low 在带外，f >= low 在带内）
    # level_tag 1 = high（二分类：f <= high 在带内，f > high 在带外）
    # 墙的法向指向“带外”参考点：low 取下方节点，high 取上方节点。
    wall_specs = [(low_f, 0, "low")]
    if not zero_thickness:
        wall_specs.append((high_f, 1, "high"))

    for level_value, tag, surface_name in wall_specs:
        def inside_band(v: float) -> bool:
            return v >= level_value if tag == 0 else v <= level_value

        for k in range(nk - 1):
            for j in range(nj - 1):
                for i in range(ni - 1):
                    corner_ids = tuple(
                        nid(i + di, j + dj, k + dk) for (di, dj, dk) in CUBE_CORNERS
                    )
                    inside = tuple(inside_band(values[c]) for c in corner_ids)
                    cell = (i, j, k)
                    for tet_index, tet in enumerate(TETRA_TABLE):
                        tet_nodes = tuple(corner_ids[c] for c in tet)
                        tin = tuple(inside[c] for c in tet)
                        ni_in = sum(tin)
                        if ni_in == 0 or ni_in == 4:
                            continue
                        crossings = []
                        for ea, eb in _TET_EDGES:
                            if tin[ea] != tin[eb]:
                                crossings.append(
                                    canonical_edge_key(tet_nodes[ea], tet_nodes[eb])
                                )
                        if len(crossings) == 3:
                            vids = tuple(get_edge_vertex(key, level_value, tag)
                                         for key in crossings)
                            pts = [vertices[v].point for v in vids]
                            # 带外参考节点：low 墙取 f<low 的节点，high 墙取 f>high。
                            ref_local = next(idx for idx in range(4) if not tin[idx])
                            order = _oriented_loop(
                                pts, [0, 1, 2], ncoord(tet_nodes[ref_local])
                            )
                            emit((vids[order[0]], vids[order[1]], vids[order[2]]),
                                 cell, tet_index, surface_name)
                        else:
                            entries = [
                                (get_edge_vertex(key, level_value, tag), key)
                                for key in crossings
                            ]
                            pts = [vertices[v].point for v, _ in entries]
                            centroid = tuple(
                                sum(p[d] for p in pts) / 4.0 for d in range(3)
                            )
                            out_local = [idx for idx in range(4)
                                         if (not tin[idx] if tag == 0 else not tin[idx])]
                            in_local = [idx for idx in range(4) if tin[idx]]
                            out_mid = tuple(
                                sum(ncoord(tet_nodes[idx])[d] for idx in out_local)
                                / len(out_local) for d in range(3)
                            )
                            in_mid = tuple(
                                sum(ncoord(tet_nodes[idx])[d] for idx in in_local)
                                / len(in_local) for d in range(3)
                            )
                            ref = _sub(out_mid, in_mid)
                            if _norm(ref) == 0.0:
                                ref = (1.0, 0.0, 0.0)
                            axis = min(range(3), key=lambda d: (abs(ref[d]), d))
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
                            loop = _oriented_loop(pts, list(order), out_mid)
                            emit((entries[loop[0]][0], entries[loop[1]][0],
                                  entries[loop[2]][0]), cell, tet_index, surface_name)
                            emit((entries[loop[0]][0], entries[loop[2]][0],
                                  entries[loop[3]][0]), cell, tet_index, surface_name)

    if not zero_thickness:
        _band_caps(nid, ncoord, values, ni, nj, nk, low_f, high_f,
                   vertices, vertex_by_key, get_edge_vertex, get_node_vertex,
                   emit, emit_fan, degenerate)

    boundary_edges = tuple(sorted(k for k, c in edge_use.items() if c == 1))
    nonmanifold_edges = tuple(sorted(k for k, c in edge_use.items() if c >= 3))
    total_area = sum(f.area for f in faces)
    vol = 0.0 if zero_thickness else signed_volume(vertices, faces)
    node_count = ni * nj * nk
    cell_count = (ni - 1) * (nj - 1) * (nk - 1)
    stats = BandStats(
        low=low_f, high=high_f, thickness_zero=zero_thickness,
        node_count=node_count, cell_count=cell_count,
        tetrahedron_count=cell_count * len(TETRA_TABLE),
        vertex_count=len(vertices), face_count=len(faces),
        low_wall_face_count=sum(1 for f in faces if f.surface == "low"),
        high_wall_face_count=sum(1 for f in faces if f.surface == "high"),
        cap_face_count=sum(1 for f in faces if f.surface == "cap"),
        degenerate_face_count=len(degenerate),
        boundary_edge_count=len(boundary_edges),
        nonmanifold_edge_count=len(nonmanifold_edges),
        total_area=total_area, signed_volume=vol,
    )
    return BandMesh(
        low=low_f, high=high_f, thickness_zero=zero_thickness,
        vertices=tuple(vertices), faces=tuple(faces), stats=stats,
        degenerate_faces=tuple(degenerate),
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        edge_use_count=dict(edge_use),
    )


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
    vertex_by_edge: Dict[Tuple[int, int], int] = {}

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
        point = (
            pa[0] + t * (pb[0] - pa[0]),
            pa[1] + t * (pb[1] - pa[1]),
            pa[2] + t * (pb[2] - pa[2]),
        )
        vid = len(vertices)
        vertices.append(Vertex(id=vid, point=point, edge_key=key, t=t))
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
        edge_use_count=dict(edge_use),
    )
