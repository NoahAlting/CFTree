# src/reconstruction/trunk_estimation_tree.py

from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import rasterio
from rasterio import mask
import trimesh
from scipy.spatial import ConvexHull

def compute_trunk_base_from_dtm(crown_mesh: trimesh.Trimesh, dtm_path: Path | str) -> np.ndarray | None:
    """
    Return trunk base [bx, by, bz] (float64) or None if DTM has no valid support under the crown.
    """
    pts_xy = crown_mesh.vertices[:, :2]
    if pts_xy.shape[0] < 3:
        return None
    hull = ConvexHull(pts_xy)
    poly_xy = pts_xy[hull.vertices, :]
    from shapely.geometry import Polygon
    poly = Polygon(poly_xy)

    with rasterio.open(dtm_path) as src:
        out_img, out_transform = mask.mask(src, [poly], crop=True, filled=False)
        band = out_img[0]
        rows, cols = np.where(~band.mask)
        if rows.size == 0:
            return None
        xs, ys = rasterio.transform.xy(out_transform, rows, cols)
        coords = np.column_stack([xs, ys])
        center = crown_mesh.centroid[:2]
        idx = np.argmin(np.linalg.norm(coords - center, axis=1))
        return np.array([coords[idx, 0], coords[idx, 1], band.data[rows[idx], cols[idx]]], dtype=float)

def estimate_trunk_dimensions(CW_m: float, crown_median_z: float, trunk_base_z: float,
                              a=1.0, b=1.1, c=0.7) -> tuple[float, float, float]:
    """
    Returns (H, DBH_m, r_trunk_m). Applies simple allometry; no slenderness clamp by default.
    """
    if not (np.isfinite(CW_m) and np.isfinite(crown_median_z) and np.isfinite(trunk_base_z)):
        return (np.nan, np.nan, np.nan)
    H = float(crown_median_z - trunk_base_z)
    if H <= 0:
        return (H, np.nan, np.nan)
    DBH_m = float(a * (CW_m ** b) * (H ** c) / 100.0)
    r_trunk = 0.5 * DBH_m if np.isfinite(DBH_m) and DBH_m > 0 else np.nan
    return (H, DBH_m, r_trunk)

def build_trunk_geometry_lod3(crown_mesh: trimesh.Trimesh, trunk_base: np.ndarray,
                              r_trunk: float, crown_median_z: float, city: dict) -> tuple[dict | None, float | None]:
    """
    Builds a slanted cylinder (LoD3). Appends vertices to city['vertices'].
    Returns (trunk_geom, trunk_length_m).
    """
    if trunk_base is None or not (np.isfinite(r_trunk) and r_trunk > 0):
        return (None, None)
    top = np.array([crown_mesh.centroid[0], crown_mesh.centroid[1], crown_median_z], dtype=float)
    base = trunk_base.astype(float)
    axis = top - base
    L = float(np.linalg.norm(axis))
    if not np.isfinite(L) or L <= 0:
        return (None, None)

    cyl = trimesh.creation.cylinder(radius=r_trunk, height=L, sections=32)
    # align + translate
    from trimesh.geometry import align_vectors
    cyl.apply_translation([0.0, 0.0, -cyl.bounds[0, 2]])
    R = align_vectors([0, 0, 1], axis / L)
    cyl.apply_transform(R)
    cyl.apply_translation(base)

    Vc = cyl.vertices.astype(float)
    Fc = cyl.faces.astype(int)
    vbase = len(city["vertices"])
    city["vertices"].extend(Vc.tolist())
    shell = [[(f + vbase).tolist()] for f in Fc]
    trunk_geom = {"type": "Solid", "lod": 3.0, "boundaries": [shell]}
    return (trunk_geom, L)
