# iso3d — 体素等值面的共享边交点提取（marching tetrahedra）

离线三维重建工程中的可复用模块：在规则体素晶格上用 marching tetrahedra
提取标量场等值面，所有交点按**全局网格边键**共享，输出定向三角形网格与
边界统计。支持把同一张网格切成若干**同分辨率相邻体块**分别提取后拼接：
接缝处共享采样点采用唯一所有者、交点顶点只由全局网格边键确定身份，
拼出的面与整块提取逐字段一致。纯 Python 标准库实现，无任何第三方依赖。

## 环境

- Python 3.14.7（已在该版本验证；仅使用标准库 `dataclasses` / `math` / `typing`）
- 安装 Python 后无需安装任何其他依赖；核心、测试、演示全部离线运行

## 快速开始

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 运行演示（约 0.2 秒）
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
```

分块提取与拼接：

```python
from iso3d import ScalarBlock3D, extract_isosurface_blocks

# 把 8x8x8 节点网格沿 i=3 的共享面切成两块（共享第 3 层节点）。
# block_a 覆盖全局节点 i=0..3，block_b 覆盖 i=3..7，接缝在 i=3。
block_a = ScalarBlock3D(origin=(0, 0, 0), ni=4, nj=8, nk=8, values=values_a)
block_b = ScalarBlock3D(origin=(3, 0, 0), ni=5, nj=8, nk=8, values=values_b)
mesh = extract_isosurface_blocks([block_a, block_b], iso=1.5, shape=(8, 8, 8))
# mesh 与把两块数据拼成一个 ScalarField3D 后 extract_isosurface 的结果
# 完全相同（顶点、面、定向、退化面、统计逐字段相等）。
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

### 分块输入与拼接

- `ScalarBlock3D(origin, ni, nj, nk, values)`：全局网格中的一个同分辨率
  相邻体块。
  - `origin=(oi, oj, ok)`：体块首节点的全局晶格坐标（非负整数）。
  - `ni, nj, nk`：体块每维节点数（1..16，同样受总节点上限约束）。
    体块覆盖全局节点 `[oi, oi+ni-1] x [oj, oj+nj-1] x [ok, ok+nk-1]`，
    含 `(ni-1)*(nj-1)*(nk-1)` 个完整立方体。
  - `values`：体块**局部**行优先扁平序列，局部节点 `(li,lj,lk)`
    （从 0 起）为 `values[(lk*nj + lj)*ni + li]`，对应全局节点
    `(oi+li, oj+lj, ok+lk)`。
  - 只读辅助：`block.end`（末节点全局坐标）、
    `block.global_node_id(shape, li, lj, lk)`。
- `extract_isosurface_blocks(blocks, iso, shape=None) -> IsosurfaceMesh`：
  分块提取并拼接。`blocks` 为非空 `ScalarBlock3D` 序列，`iso` 为有限
  实数；`shape=(gni,gnj,gnk)` 为全局网格节点数，省略时由所有体块的
  包围盒推断（推断结果仍受每维 ≤16、总节点 ≤4096 约束）。输出与
  `extract_isosurface` 相同的 `IsosurfaceMesh`。
  - **铺法契约**：体块必须无缝无叠地铺满全局网格的全部立方体——每个
    立方体恰好属于一个体块；只共享节点层（如 `ni=1` 的端层块）合法。
  - **接缝一致性**：被多个体块引用的共享节点，其值必须逐位相等。
    共享采样点的唯一所有者按覆盖块中 `origin` 最大者确定（origin
    相同时取输入序列靠后者）；顶点身份只看全局网格边键，**不使用
    任何坐标距离/容差**。
  - **等价性**：满足契约时，结果与拼成整块后调用
    `extract_isosurface` 完全相同，且与体块的输入顺序无关；
    顶点 ID 即其全局边键在全部穿越边键排序中的排名（切分无关）。

### 输出（`IsosurfaceMesh`，冻结数据类）

- `vertices: tuple[Vertex, ...]`：每个顶点带 `point`（坐标）、
  `edge_key`（所在全局网格边，升序节点 ID 对）与插值参数 `t`。
  同一条网格边上的交点全局唯一、跨立方体/跨四面体共享。
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

### 公开常量与辅助

- `TETRA_TABLE`：6 四面体分解表（见下）。
- `CUBE_CORNERS`：立方体 8 角点局部偏移，角点编号按 `i + 2j + 4k`。
- `canonical_edge_key(id_a, id_b)`：规范边键（升序对）。
- `perturbed_below(value, iso, node_id)`：拓扑判定规则（见下）。

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

- **共享边键**：交点挂在四面体边上，边键为两端点**全局节点 ID 的升序对**。
  四面体边包括立方体棱、面内对角线与体对角线 0-7；对角线交点位于立方体
  面内/体内，同样按节点 ID 对共享（相邻立方体在同一条对角线上必然得到
  同一个键）。
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
包含可定位的参数名（及节点/体块索引），例如 `values[37]: NaN is not allowed`：

- `ni/nj/nk`：拒绝 bool、非 int、`< 1`、`> 16`；
- `values`：拒绝非序列、长度不符、NaN、±Infinity、bool、非数值；
- `iso`：拒绝 bool、非数值、NaN、±Infinity；
- `field`：拒绝非 `ScalarField3D` 对象。

分块入口 `extract_isosurface_blocks` 额外拒绝（均报 `blocks` /
`shape` 参数，并给出可定位的体块下标、全局节点或立方体坐标）：

- `blocks`：非序列、空序列、含非 `ScalarBlock3D` 元素；
- `ScalarBlock3D` 的 `origin`：非三元 int 元组、负数、bool；
- `shape`：非三元 int 元组、bool/越界（≤0 或 >16）、总节点 >4096；
  省略 `shape` 时推断出的全局网格越界同样拒绝；
- 体块超出 `shape` 范围；
- 立方体**重叠**（两个体块都拥有同一单元）或**缺失**（存在无人拥有
  的单元，即铺法有缝）；
- 共享节点在不同体块中值**不一致**（如
  `blocks: shared node (1, 0, 0) has inconsistent values: ...`）。

输入校验在任何计算之前完成，失败时不产生任何部分结果；体块是冻结
数据类，失败也不会修改输入。本模块为纯函数式接口，无可变全局状态。
空结果（无穿越）是合法的“已证明无等值面”，与错误路径严格区分。

## 误差口径

- 交点坐标为边上**线性插值**，浮点舍入量级（约 1e-16 相对）。
- 对**线性标量场**，提取的等值面是精确的平面截面：测试中 16³ 网格
  `f = i + j + k, iso = 7.5` 的总面积与独立解析截面面积比较，
  相对容差 1e-9（仅覆盖浮点累加舍入）。
- 对非线性场，等值面是分片线性逼近：面积随网格加密收敛到真实曲面
  面积，且对本实现恒不大于同分辨率的理论极限。测试仅做宽松合理性
  检查（球面积比在 0.75–1.0 之间），**精确口径以平面解析测试为准**。
- 退化判定使用精确零面积准则，不以宽容差掩盖边界情况。

## 设计取舍

- **marching tetrahedra 而非 marching cubes**：四面体只有 4 个顶点，
  二染色下穿越边数只可能是 3 或 4，无 marching cubes 的多义性构型；
  配合符号扰动，拓扑判定完全确定。
- **节点 ID 奇偶扰动**：规则简单、全局一致、可复现，且不需要存储任何
  扰动状态；代价是 iso 恰好等于大片节点值时（如常量场取 iso 等于常量）
  会产生由扰动决定的薄片状 artifact——几何上仍是合法流形三角面，
  本模块如实输出而不静默“修正”。
- **全局边键共享**：顶点按边键去重，天然保证跨单元共享与网格水密
  （组合意义下每条边被 1 或 2 个三角形使用）。
- **分块接缝**：每块独立跑 marching 时，节点身份统一用**全局节点
  ID**（由 `origin + 局部索引` 与全局 `shape` 计算），坐标用全局
  晶格坐标，符号扰动用全局 ID 奇偶；因此接缝两侧对同一条边算出的
  交点逐位相同，合并时只按边键认亲并做精确相等校验，顶点 ID 由边
  键全局排名确定。全程不比较坐标、不用任何浮点距离去猜重复点。
- **纯标准库**：不调用任何第三方同类实现；核心算法（分解、插值、
  定向、退化处理）全部在本仓库内实现。

## 已知限制

- 仅支持规则等距晶格，坐标即整数索引（无 origin/spacing 变换；
  需要物理坐标时由调用方自行缩放）。
- 输入上限为每维 16 节点、总数 4096（本题接口边界），不是性能上限；
  分块后的**全局**网格同样受此上限约束（分块是切分计算与明确接缝
  所有权，不是放大输入规模）。
- 分块只支持**同分辨率**相邻体块、且必须铺满整个全局网格；不支持
  跨细化层级（粗/细网格）过渡、悬空部分覆盖或局部编辑增量更新。
- 退化面被丢弃后，其邻边会在统计中表现为边界边；这是如实报告而非
  网格修复。本模块不做网格简化、法向估计或孔洞填补。
- 等值面恰与晶格平面重合时，扰动决定薄片归属，结果确定但可能与
  “直觉的一侧”不同；几何位置始终精确。

## 仓库结构

```
iso3d/
  __init__.py          # 公开导出
  core.py              # 核心算法与数据结构（含分块提取/拼接引擎）
tests/
  _reference.py        # 独立参考实现（不 import iso3d）
  test_tetra_table.py  # 分解表几何性质
  test_extraction.py   # 平面解析、共享边、iso 落顶点、常量场、球场闭合性
  test_blocks.py       # 体块拼接：手算接缝、面/边/角缝等价性、独立枚举、失败语义
  test_validation.py   # 输入校验与错误语义
demo.py                # 固定输入演示（正常结果 + 分块拼接 + 两类实际拒绝）
README.md
```

测试中的期望值来自 `tests/_reference.py` 的独立实现（解析平面截面面积、
独立穿越边枚举），不调用被测核心；往返一致性（如顶点数与独立枚举的
边键集合相等）仅作为补充验证。分块测试的期望值是手算小例（共享棱/面
对角线的交点坐标与 t）、独立解析截面面积与独立穿越边枚举，而“拼接
== 整块”作为关系性质逐字段断言；另对 4x3x3 小网格枚举全部 16 种铺法
× 随机整数场 × 多个 iso（含 iso 落节点）做小规模穷举核对。
