"""iso3d — 体素等值面的共享边交点提取（marching tetrahedra，纯标准库）。"""

from .core import (
    CUBE_CORNERS,
    MAX_DIM,
    MAX_NODES,
    TETRA_TABLE,
    DegenerateFace,
    Face,
    InvalidInputError,
    IsosurfaceMesh,
    MeshStats,
    ScalarField3D,
    Vertex,
    canonical_edge_key,
    extract_isosurface,
    perturbed_below,
)

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
    "canonical_edge_key",
    "extract_isosurface",
    "perturbed_below",
]
