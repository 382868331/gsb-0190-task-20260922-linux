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
    "ScalarFieldBlock",
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
class ScalarFieldBlock:
    """位于全局规则晶格中某个轴对齐位置的标量场体块。

    field 的节点 (i, j, k)（局部索引）对应全局晶格节点
    ``(origin[0] + i, origin[1] + j, origin[2] + k)``。所有体块必须使用
    相同的单位晶格间距（即只给整数 origin，不支持缩放）。

    相邻体块允许在边界上共享节点：同分辨率体块对齐拼接时，接缝节点会被
    多块重复给出。合并按全局节点槽位 ``origin + 局部索引`` 进行（整数
    映射，不做任何浮点距离匹配）；每个全局槽位有唯一所有者，重复给出的
    值必须精确相等。
    """

    origin: Tuple[int, int, int]
    field: "ScalarField3D"

    def __post_init__(self) -> None:
        origin = self.origin
        if not isinstance(origin, tuple) or len(origin) != 3:
            raise InvalidInputError(
                "origin", f"must be a (oi, oj, ok) tuple of ints, got {type(origin).__name__}"
            )
        for axis, coord in zip(("oi", "oj", "ok"), origin):
            if isinstance(coord, bool) or not isinstance(coord, int):
                raise InvalidInputError(
                    "origin", f"{axis} must be an int, got {type(coord).__name__}"
                )
            if coord < 0:
                raise InvalidInputError("origin", f"{axis} must be >= 0, got {coord}")
        if not isinstance(self.field, ScalarField3D):
            raise InvalidInputError(
                "field", f"must be a ScalarField3D, got {type(self.field).__name__}"
            )

    def global_coords(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """每轴的全局 ``[起点, 终点]`` 节点索引区间（终点含端点）。"""
        oi, oj, ok = self.origin
        f = self.field
        return (
            (oi, oi + f.ni - 1),
            (oj, oj + f.nj - 1),
            (ok, ok + f.nk - 1),
        )


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

def _run_extraction(
    ni: int, nj: int, nk: int, values: Sequence[float], iso_f: float
) -> IsosurfaceMesh:
    """在给定全局规则晶格上执行 marching tetrahedra。

    values 为行优先扁平序列（节点 (i,j,k) 的全局 ID 为
    ``(k*nj + j)*ni + i``）。调用方负责完成全部输入校验，本函数不再
    抛出 InvalidInputError。
    """

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
    return _run_extraction(field.ni, field.nj, field.nk, field.values, iso_f)


# ---------------------------------------------------------------------------
# 同分辨率相邻体块拼接
# ---------------------------------------------------------------------------

def _check_block_sequence(blocks: object) -> int:
    if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Sequence):
        raise InvalidInputError(
            "blocks", f"must be a sequence of ScalarFieldBlock, got {type(blocks).__name__}"
        )
    n = len(blocks)
    if n == 0:
        raise InvalidInputError("blocks", "must contain at least one block")
    return n


def _merge_blocks(
    blocks: Sequence[ScalarFieldBlock],
) -> Tuple[int, int, int, Tuple[float, ...]]:
    """把相邻体块合并为一张全局规则晶格标量场。

    合并只依据整数槽位映射 ``全局节点 = origin + 局部索引``：
    每个全局节点槽位有唯一所有者（第一个覆盖它的体块）；接缝节点被多个
    体块重复给出时，其标量值必须精确相等，否则拒绝。全程不使用坐标距离
    猜测重复点。

    全局节点总数同样受 MAX_NODES 约束。返回 (ni, nj, nk, values)。
    """
    # 先确定全局包围盒。
    xlo = ylo = zlo = 0
    xhi = yhi = zhi = -1
    for bi, block in enumerate(blocks):
        (ax, bx), (ay, by), (az, bz) = block.global_coords()
        if bi == 0:
            xlo, xhi = ax, bx
            ylo, yhi = ay, by
            zlo, zhi = az, bz
        else:
            xlo = min(xlo, ax)
            ylo = min(ylo, ay)
            zlo = min(zlo, az)
            xhi = max(xhi, bx)
            yhi = max(yhi, by)
            zhi = max(zhi, bz)
    # 构造时 origin >= 0；并集包围盒下界必须为 (0,0,0)：这样合并网格的
    # 节点坐标与行优先全局节点 ID 与“整块提取”完全一致（包括 iso 恰等
    # 节点值时按全局 ID 奇偶的符号扰动），拼接结果才能与整块逐面一致。
    if (xlo, ylo, zlo) != (0, 0, 0):
        raise InvalidInputError(
            "blocks",
            f"block union must be anchored at the lattice origin (0, 0, 0); "
            f"got lower corner ({xlo}, {ylo}, {zlo}) — subtract the minimum "
            f"origin from all blocks",
        )
    ni = xhi - xlo + 1
    nj = yhi - ylo + 1
    nk = zhi - zlo + 1
    total = ni * nj * nk
    if total > MAX_NODES:
        raise InvalidInputError(
            "blocks",
            f"merged grid {ni}x{nj}x{nk} = {total} nodes exceeds {MAX_NODES}",
        )

    # owner[(k*nj + j)*ni + i] = 首个覆盖该槽位的体块号；
    # 值为 -1 表示尚未被任何体块覆盖。
    owner: List[int] = [-1] * total

    def slot(gi: int, gj: int, gk: int) -> int:
        return (gk * nj + gj) * ni + gi

    merged: List[float] = [0.0] * total
    for bi, block in enumerate(blocks):
        oi, oj, ok = block.origin
        f = block.field
        for dk in range(f.nk):
            gk = ok + dk
            for dj in range(f.nj):
                gj = oj + dj
                row_base = (dk * f.nj + dj) * f.ni
                for di in range(f.ni):
                    gi = oi + di
                    s = slot(gi, gj, gk)
                    value = f.values[row_base + di]
                    prev = owner[s]
                    if prev < 0:
                        owner[s] = bi
                        merged[s] = value
                    elif merged[s] != value:
                        raise InvalidInputError(
                            "blocks",
                            f"conflicting values at shared node ({gi}, {gj}, {gk}): "
                            f"block {prev} gives {merged[s]!r}, block {bi} gives {value!r}",
                        )
                    # 值精确相等：唯一所有者保持不变，重复槽位被接受。

    uncovered = owner.count(-1) if total else 0
    if uncovered:
        # 找到第一个空洞用于定位。
        s = owner.index(-1)
        gi = s % ni
        gj = (s // ni) % nj
        gk = s // (ni * nj)
        raise InvalidInputError(
            "blocks",
            f"blocks do not form a contiguous filled grid: node "
            f"({gi}, {gj}, {gk}) is not covered by any block",
        )
    return ni, nj, nk, tuple(merged)


def extract_isosurface_blocks(
    blocks: Sequence[ScalarFieldBlock], iso: float
) -> IsosurfaceMesh:
    """从若干同分辨率、按整数晶格对齐的相邻体块提取合并等值面。

    参数
    ----
    blocks : Sequence[ScalarFieldBlock]
        至少一个体块。每块用 ``origin`` 指定其在全局晶格中的节点原点，
        体块间可在边界面/边/线上共享节点（共享采样点）。
    iso : int 或 float
        等值，对所有体块相同。必须为有限实数；bool 被拒绝。

    合并规则
    --------
    - 全局节点槽位由 ``origin + 局部索引`` 的整数映射唯一确定，**不使用
      任何浮点距离匹配**；
    - 每个槽位有唯一所有者（第一个覆盖它的体块）；接缝节点被多块重复
      给出时，其标量值必须精确相等，否则抛出 InvalidInputError 并定位到
      全局坐标与冲突的两块；
    - 体块必须恰好铺满一个连续的轴对齐节点盒（无空洞、不错位），合并后
      网格节点总数不超过 MAX_NODES。

    返回
    ----
    IsosurfaceMesh
        与把所有体块拼成整块后调用 :func:`extract_isosurface` 的结果逐
        顶点、逐面（含顶点/面 ID）一致；接缝处不产生重复顶点或裂缝。

    抛出
    ----
    InvalidInputError
        任何输入违反契约时。校验在任何合并计算生效前完成，失败时不返回
        任何部分结果（本接口无持久状态，调用方传入的对象不被修改）。
    """
    n = _check_block_sequence(blocks)
    iso_f = _finite_float("iso", iso)
    # 逐块校验类型（ScalarFieldBlock.__post_init__ 已校验 origin 与 field），
    # 先完成全部校验再做合并，保证失败路径不产生部分结果。
    for bi in range(n):
        block = blocks[bi]
        if not isinstance(block, ScalarFieldBlock):
            raise InvalidInputError(
                f"blocks[{bi}]",
                f"must be a ScalarFieldBlock, got {type(block).__name__}",
            )
    ni, nj, nk, values = _merge_blocks(blocks)
    return _run_extraction(ni, nj, nk, values, iso_f)
