# iso3d — 体素等值面的共享边交点提取（marching tetrahedra）

离线三维重建工程中的可复用模块：在规则体素晶格上用 marching tetrahedra
提取标量场等值面，所有交点按**全局网格边键**共享，输出定向三角形网格与
边界统计。纯 Python 标准库实现，无任何第三方依赖。

## 环境

- Python 3.14.7（已在该版本验证；仅使用标准库 `dataclasses` / `math` / `typing`）
- 安装 Python 后无需安装任何其他依赖；核心、测试、演示全部离线运行

## 快速开始

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 运行演示（约 0.1 秒）
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
- `IsosurfaceExtractor(field, iso)`：有状态增量提取器（见“增量更新”）。

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

## 增量更新

体数据只改了一小片时，用 `IsosurfaceExtractor` 避免全量重算：

```python
from iso3d import IsosurfaceExtractor

ext = IsosurfaceExtractor(field, iso=16.0)
mesh0 = ext.mesh()
mesh1 = ext.update({node_index: new_value, ...})  # 只重算受影响单元
```

- `update(changes)`：`changes` 为 `{扁平节点索引: 新值}` 映射或
  `(index, value)` 对序列；返回更新后的 `IsosurfaceMesh`。
  受影响单元 = 被修改节点邻接的立方体，只有这些单元的四面体图元被
  重新计算。
- **身份保留**：顶点挂在全局网格边键上。未受影响的顶点/面 id 不变；
  端点值变化但仍穿越的边保留原顶点 id（坐标原地刷新）；新穿越边分配
  从未使用过的新 id；消失的 id 退役且永不复用。上层持有的未受影响
  顶点/面 id 在更新后仍然有效。
- **共享边一致**：修改区域边缘的共享边交点按全局边键唯一存储、只更新
  一次，所有引用它的面自动看到同一坐标，不会产生裂缝；值跨越 iso 时
  新增/消失的三角形由受影响单元的重建自然处理。
- **几何一致**：更新后的几何（顶点坐标集合与三角形集合）与对修改后的
  场调用 `extract_isosurface` 全量重提取逐位一致，仅 id 与顺序可能
  不同（`tests/test_incremental.py` 逐位验证）。
- **失败原子性**：`update` 先完成全部校验与新图元计算，再一次提交；
  校验或计算失败时抛出 `InvalidInputError`，内部状态完全不变，旧网格
  可继续使用。
- 注意：增量更新后的网格中 `Vertex.id` / `Face.id` 是稳定身份，不一定
  等于 `vertices` / `faces` 元组下标，请按 `.id` 建立索引（全量
  `extract_isosurface` 的结果仍保证 id == 下标）。

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
包含可定位的参数名（及节点索引），例如 `values[37]: NaN is not allowed`：

- `ni/nj/nk`：拒绝 bool、非 int、`< 1`、`> 16`；
- `values`：拒绝非序列、长度不符、NaN、±Infinity、bool、非数值；
- `iso`：拒绝 bool、非数值、NaN、±Infinity；
- `field`：拒绝非 `ScalarField3D` 对象。
- `changes`（增量更新）：拒绝非映射/非键值对序列、bool 或非 int 索引、
  越界索引、重复索引、bool/非数值/NaN/±Infinity 的值。

输入校验在任何计算之前完成，失败时不产生任何部分结果；本模块为纯函数
式接口，无可变全局状态。空结果（无穿越）是合法的“已证明无等值面”，
与错误路径严格区分。

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
- **纯标准库**：不调用任何第三方同类实现；核心算法（分解、插值、
  定向、退化处理）全部在本仓库内实现。

## 已知限制

- 仅支持规则等距晶格，坐标即整数索引（无 origin/spacing 变换；
  需要物理坐标时由调用方自行缩放）。
- 输入上限为每维 16 节点、总数 4096（本题接口边界），不是性能上限。
- 退化面被丢弃后，其邻边会在统计中表现为边界边；这是如实报告而非
  网格修复。本模块不做网格简化、法向估计或孔洞填补。
- 等值面恰与晶格平面重合时，扰动决定薄片归属，结果确定但可能与
  “直觉的一侧”不同；几何位置始终精确。

## 仓库结构

```
iso3d/
  __init__.py          # 公开导出
  core.py              # 核心算法与数据结构
tests/
  _reference.py        # 独立参考实现（不 import iso3d）
  test_tetra_table.py  # 分解表几何性质
  test_extraction.py   # 平面解析、共享边、iso 落顶点、常量场、球场闭合性
  test_validation.py   # 输入校验与错误语义
  test_incremental.py  # 增量更新：几何一致性、身份保留、失败原子性
demo.py                # 固定输入演示（正常结果 + 增量更新 + 拒绝边界）
README.md
```

测试中的期望值来自 `tests/_reference.py` 的独立实现（解析平面截面面积、
独立穿越边枚举），不调用被测核心；往返一致性（如顶点数与独立枚举的
边键集合相等）仅作为补充验证。
