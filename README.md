# iso3d — 体素等值面的共享边交点提取（marching tetrahedra）

离线三维重建工程中的可复用模块：在规则体素晶格上用 marching tetrahedra
提取标量场等值面，所有交点按**全局网格边键**共享，输出定向三角形网格与
边界统计。每个共享交点同时附带**来源网格边、插值权重**与**局部标量敏感度**
（交点位置对两端标量值及 iso 的解析导数）。纯 Python 标准库实现，无任何
第三方依赖。

## 环境

- Python 3.14.7（已在该版本验证；仅使用标准库 `dataclasses` / `math` / `typing`）
- 固定解释器：`./.venv/bin/python`（软链到本机 Python 3.14.7）
- 安装 Python 后无需安装任何其他依赖；核心、测试、演示全部离线运行

## 快速开始

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 运行全部测试（69 个）
./.venv/bin/python demo.py                            # 运行演示（约 0.1 秒）
```

```python
from iso3d import ScalarField3D, extract_isosurface

# 16^3 满上限网格，f = 到中心距离平方，等值面为半径 4 的球面
n, c = 16, 7.5
values = tuple(
    (i - c) ** 2 + (j - c) ** 2 + (k - c) ** 2
    for k in range(n) for j in range(n) for i in range(n)
)
field = ScalarField3D(ni=n, nj=n, nk=n, values=values)
mesh = extract_isosurface(field, iso=16.0)
print(mesh.stats.total_area, mesh.stats.nonmanifold_edge_count)

# 取任一共享交点：来源边、插值权重与局部标量敏感度
rec = mesh.intersections[0]
(na, wa), (nb, wb) = rec.weights      # 来源节点与凸组合权重，wa + wb == 1
if rec.differentiable:
    s = rec.sensitivity               # 对 v_a、v_b、iso 的解析导数
    print(s.dt_dva, s.dt_dvb, s.dt_diso, s.dpoint_diso)
```

## 接口

### 输入

- `ScalarField3D(ni, nj, nk, values)`：规则晶格标量场。
  - `ni, nj, nk`：每维节点数，`1 <= n <= 16`（`MAX_DIM`），且
    `ni*nj*nk <= 4096`（`MAX_NODES` = 16³，本题的输入上限）。
  - `values`：长度 `ni*nj*nk` 的扁平实数序列，行优先排列：
    节点 `(i, j, k)` 的值为 `values[(k*nj + j)*ni + i]`。
    节点 `(i, j, k)` 的几何坐标即 `(i, j, k)`（晶格单位）。
  - `ScalarField3D.from_nested(nested)`：由 `nested[k][j][i]` 嵌套序列构造。
- `extract_isosurface(field, iso) -> IsosurfaceMesh`：提取等值面。
  `iso` 为有限实数。

### 输出（`IsosurfaceMesh`，冻结数据类）

- `vertices: tuple[Vertex, ...]`：每个顶点带 `point`（坐标）、
  `edge_key`（所在全局网格边，升序节点 ID 对）与插值参数 `t`。
  同一条网格边上的交点全局唯一、跨立方体/跨四面体共享。
- `intersections: tuple[EdgeIntersection, ...]`：与 `vertices` **一一对应**
  （`intersections[v.id]` 即顶点 v 的记录）。每条记录是该网格边的**唯一**
  记录，字段：
  - `edge_key`：来源网格边（升序节点 ID 对 `(a, b)`，a < b）；
  - `t` 与 `weights`：插值权重，`point = (1-t)P_a + t P_b`，
    `weights = ((a, 1-t), (b, t))`，两权重和恒为 1；
  - `differentiable` / `sensitivity` / `nondifferentiable_reason`：
    局部标量敏感度，见下节。
  - 查询方法：`intersection_for_vertex(vid)`、`intersection_for_edge(a, b)`
    （不存在抛 `KeyError`）、`intersections_affected_by_node(node_id)`
    （返回来源边端点含该节点的全部交点，即“一个输入节点变化影响哪些交点”）。
- `faces: tuple[Face, ...]`：三角形，`vertex_ids` 按右手定则定向，
  **法向指向标量增大方向**；`area` 为精确叉积面积。
- `stats: MeshStats`：节点/立方体/四面体/顶点/三角形计数、
  `boundary_edge_count`（仅被 1 个三角形使用的边）、
  `nonmanifold_edge_count`（被 ≥3 个三角形使用的边）、
  `degenerate_face_count`（被丢弃的零面积面数）、`total_area`。
- `degenerate_faces`：每条被丢弃零面积面的记录（所在立方体、四面体
  序号、三条边键）。
- `boundary_edges` / `nonmanifold_edges` / `edge_use_count`：边级明细。
- 合法但处处不穿越 `iso` 的场返回**空网格**（已证明无等值面），不抛异常。

### 局部标量敏感度（解析导数）

交点位于规范边 `(a, b)` 上，`t = (iso - v_a)/(v_b - v_a)`、
`point = (1-t)P_a + t P_b`。在**当前拓扑（穿越/不穿越判定）不变**的小
扰动下，`PointSensitivity` 给出解析雅可比：

```
dt/dv_a = (iso - v_b) / (v_b - v_a)^2
dt/dv_b = (v_a - iso) / (v_b - v_a)^2
dt/diso = 1 / (v_b - v_a)
dpoint/d(v_a 或 v_b 或 iso) = (P_b - P_a) * dt/d(...)
```

- 单位晶格棱上 `P_b - P_a` 是单位向量，`dpoint_*` 只有一个非零分量，
  数值上等于对应 `dt_*`；面对角线/体对角线上乘以位移向量。
- **可微性三态**（严格判定，不使用宽容差掩盖边界）：
  - 两端**严格跨过** iso 且 `|v_b-v_a|` 数值稳定 → `differentiable=True`，
    给出上述导数；
  - 任一端点标量**恰等于 iso**（拓扑由节点 ID 奇偶符号扰动决定）→
    `differentiable=False`，`nondifferentiable_reason="endpoint_on_iso"`；
  - `v_a == v_b`（分母严格为 0，交点取中点）→
    `"zero_denominator"`；
  - 标量差极小、`|v_b-v_a| <= 1e-12·max(1,|v_a|,|v_b|)`（常量
    `UNSTABLE_DENOMINATOR_EPS`）→ `"unstable_denominator"`。
    后三种情况下 `sensitivity is None`，**不输出被极小分母放大的巨大
    伪精确数**；几何 `point`/`t` 仍照常给出（与原行为一致）。
- 敏感度是**固定拓扑**下的局部线性化：扰动一旦使任一端点相对 iso 变号
  （交点出现/消失、穿越翻转），交点不再连续可微，**不承诺跨拓扑有效**。

### 独立纯函数

- `intersect_edge(id_a, id_b, value_a, value_b, iso, point_a, point_b)`：
  不依赖网格遍历，计算单条边的 `EdgeIntersection`。端点可任意顺序传入，
  内部规范化到升序边键，因此**反向遍历与正向遍历得到完全相同的记录**。
  非法输入（bool/非 int 节点 ID、NaN/Infinity、坐标非长度 3 序列）抛
  `InvalidInputError`。

### 公开常量与辅助

- `TETRA_TABLE`：6 四面体分解表（见下）。
- `CUBE_CORNERS`：立方体 8 角点局部偏移，角点编号按 `i + 2j + 4k`。
- `canonical_edge_key(id_a, id_b)`：规范边键（升序对）。
- `perturbed_below(value, iso, node_id)`：拓扑判定规则（见下）。
- `intersect_edge(...)`：单条边交点、权重与敏感度的纯函数（见上）。
- `UNSTABLE_DENOMINATOR_EPS = 1e-12`：分母相对稳定性阈值。
- `NONDIFF_ENDPOINT_ON_ISO` / `NONDIFF_ZERO_DENOMINATOR` /
  `NONDIFF_UNSTABLE_DENOMINATOR`：三种不可微原因常量。

## 四面体分解表

每个立方体沿主对角线 `(0,0,0)-(1,1,1)`（角点 0 与 7）统一分解为 6 个
四面体，第 n 个对应坐标轴的一种排列 `{0, e_a, e_a+e_b, 7}`：

```
TETRA_TABLE = (
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 2, 3, 7),
    (0, 2, 6, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
)
```

所有四面体共享体对角线 0-7；每个立方体面只被一条面内对角线剖分，且
相邻立方体在同一共享面上产生同一条对角线，因此四面体网格是保形的，
跨单元交点可按全局边键一致共享。`tests/test_tetra_table.py` 独立验证
每个四面体体积为 1/6、总体积为 1、每面恰好一条对角线。

## 关键规则

- **共享边键与唯一记录**：交点挂在四面体边上，边键为两端点**全局节点 ID
  的升序对**。同一条网格边（无论被多少个四面体、多少个面引用，也无论遍历
  方向如何）只生成**一条** `EdgeIntersection`，其来源端点、插值权重与
  敏感度随之共享，绝不按面重复生成不同答案。四面体边包括立方体棱、面内
  对角线与体对角线 0-7；对角线交点位于立方体面内/体内，同样按节点 ID 对
  共享（相邻立方体在同一条对角线上必然得到同一个键）。
- **iso 恰等于节点值**：仅作**拓扑判定**的确定性符号扰动，几何位置不变：
  全局节点 ID 为偶数视为“下方”（标量较小侧），奇数视为“上方”。
  若一条穿越边两端值都恰等于 iso（插值分母为 0），交点取该边中点。
- **零面积面**：面积精确为 0（叉积模长为 0）的三角形被丢弃并记录到
  `degenerate_faces`；不使用任何 epsilon 容差。注意丢弃退化面可能使
  组合上相邻的边变成边界边，这如实反映在 `boundary_edge_count` 中。
- **定向**：每个三角形生成后用该四面体的一个“上方”节点校验法向，
  保证法向指向标量增大一侧。

## 错误语义

所有非法输入抛出 `iso3d.InvalidInputError`（`ValueError` 子类），消息
包含可定位的参数名（及节点索引），例如 `values[37]: NaN is not allowed`：

- `ni/nj/nk`：拒绝 bool、非 int、`< 1`、`> 16`；
- `values`：拒绝非序列、长度不符、NaN、±Infinity、bool、非数值；
- `iso`：拒绝 bool、非数值、NaN、±Infinity；
- `field`：拒绝非 `ScalarField3D` 对象；
- `intersect_edge(...)`：节点 ID 拒绝 bool/非 int，标量、iso 与坐标
  拒绝 NaN/±Infinity，坐标必须是长度 3 的实数序列。

输入校验在任何计算之前完成，失败时不产生任何部分结果；本模块为纯函数
式接口，无可变全局状态。空结果（无穿越）是合法的“已证明无等值面”，
与错误路径严格区分。

## 误差口径

- 交点坐标为边上**线性插值**，浮点舍入量级（约 1e-16 相对）。
- 敏感度解析导数为 t 定义的闭式商数导数，本身只含浮点舍入；
  `tests/test_sensitivity.py` 在斜坡体全部可微交点上用**双侧中心差分**
  （h = 1e-6）独立核对 `dt/dv_a`、`dt/dv_b`、`dt/diso` 及 `point` 的
  三个分量，数值与解析之差在 1e-7 以内（中心差分截断误差 O(h²)）。
- 对**线性标量场**，提取的等值面是精确的平面截面：测试中 16³ 网格
  `f = i + j + k, iso = 7.5` 的总面积与独立解析截面面积比较，
  相对容差 1e-9（仅覆盖浮点累加舍入）。
- 对非线性场，等值面是分片线性逼近：面积随网格加密收敛到真实曲面
  面积，且对本实现恒不大于同分辨率的理论极限。测试仅做宽松合理性
  检查（球面积比在 0.75–1.0 之间），**精确口径以平面解析测试为准**。
- 退化判定使用精确零面积准则，不以宽容差掩盖边界情况。
- **敏感度仅在固定拓扑下成立**：端点相对 iso 变号即发生拓扑变化，
  此时交点不连续可微；恰 iso 与不稳定分母已显式标为不可微。

## 设计取舍

- **marching tetrahedra 而非 marching cubes**：四面体只有 4 个顶点，
  二染色下穿越边数只可能是 3 或 4，无 marching cubes 的多义性构型；
  配合符号扰动，拓扑判定完全确定。
- **节点 ID 奇偶扰动**：规则简单、全局一致、可复现，且不需要存储任何
  扰动状态；代价是 iso 恰好等于大片节点值时（如常量场取 iso 等于常量）
  会产生由扰动决定的薄片状 artifact——几何上仍是合法流形三角面，
  本模块如实输出而不静默“修正”。
- **全局边键共享**：顶点按边键去重，天然保证跨单元共享与网格水密
  （组合意义下每条边被 1 或 2 个三角形使用）。交点的来源、权重与
  敏感度挂在同一条去重记录上，反向遍历或跨面引用都拿到同一对象。
- **敏感度只在安全时给出**：严格跨过且分母稳定才输出解析导数；恰 iso
  与不稳定分母显式标不可微，宁可不承诺，也不输出 1/(极小分母) 这类
  数值上无意义的巨大“伪精确”数。
- **纯标准库**：不调用任何第三方同类实现；核心算法（分解、插值、
  定向、退化处理、敏感度解析导数）全部在本仓库内实现。

## 已知限制

- 仅支持规则等距晶格，坐标即整数索引（无 origin/spacing 变换；
  需要物理坐标时由调用方自行缩放）。
- 输入上限为每维 16 节点、总数 4096（本题接口边界），不是性能上限。
- 退化面被丢弃后，其邻边会在统计中表现为边界边；这是如实报告而非
  网格修复。本模块不做网格简化、法向估计或孔洞填补。
- 等值面恰与晶格平面重合时，扰动决定薄片归属，结果确定但可能与
  “直觉的一侧”不同；几何位置始终精确。
- 标量敏感度是**当前拓扑不变的小扰动**下的局部导数；`intersections_
  affected_by_node` 只回答“一个输入节点变化直接移动哪些交点”，
  不做分块拼接，也不优化/传播到整个体场。跨拓扑变化（交点出现/消失、
  穿越翻转）不承诺有效，相应记录已标为不可微。

## 仓库结构

```
iso3d/
  __init__.py          # 公开导出
  core.py              # 核心算法、数据结构与边交点敏感度
tests/
  _reference.py        # 独立参考实现（不 import iso3d）
  test_tetra_table.py  # 分解表几何性质
  test_extraction.py   # 平面解析、共享边、iso 落顶点、常量场、球场闭合性
  test_sensitivity.py  # 手算交点、双侧有限差分导数、反向共享边、
                       #   极小标量差/端点恰 iso 不可微、单节点影响
  test_validation.py   # 输入校验与错误语义
demo.py                # 固定输入演示（球面正常结果 + NaN 拒绝 + 敏感度）
README.md
```

运行方式（固定环境）：

```bash
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python demo.py     # 约 0.1 秒，远低于 90 秒预算
```

测试中的期望值来自 `tests/_reference.py` 的独立实现（解析平面截面面积、
独立穿越边枚举）与手算闭式公式，不调用被测核心；敏感度解析导数由测试中
独立书写的双侧有限差分核对。往返一致性（如顶点数与独立枚举的边键集合
相等）仅作为补充验证。
