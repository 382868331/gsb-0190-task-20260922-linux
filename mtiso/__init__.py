"""mtiso — marching tetrahedra 等值面共享边交点提取（仅标准库）。"""
from __future__ import annotations

from .core import (
    CUBE_CORNERS,
    CUBE_TETRAHEDRA,
    DiscardedFace,
    MARCHING_TET_TABLE,
    MAX_GRID_N,
    MIN_GRID_N,
    MeshResult,
    MeshStats,
    ScalarGrid,
    TETRA_EDGES,
    Triangle,
    extract_isosurface,
)

__all__ = [
    "MAX_GRID_N",
    "MIN_GRID_N",
    "CUBE_CORNERS",
    "CUBE_TETRAHEDRA",
    "TETRA_EDGES",
    "MARCHING_TET_TABLE",
    "ScalarGrid",
    "Triangle",
    "DiscardedFace",
    "MeshStats",
    "MeshResult",
    "extract_isosurface",
]
