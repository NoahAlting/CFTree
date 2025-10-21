"""
src/vegetation_filter/HOMED_vegetation_filter.py

HOMED vegetation filter — isolates vegetation points from a clipped AHN tile.

Reads:
    data/<case>/tiles/<tile_id>/clipped.laz
Writes:
    data/<case>/tiles/<tile_id>/vegetation.laz
    data/<case>/tiles/<tile_id>/vegetation.xyz

Returns:
    {
        "tile_id": str,
        "status": "ok" | "skipped" | "failed",
        "output": Path | None,
    }
"""

from __future__ import annotations
import logging, copy, time
from pathlib import Path
import numpy as np
import laspy
from scipy.spatial import cKDTree
from scipy import ndimage as ndi


# ---------------------------------------------------------------------
# Tunable parameters 
# ---------------------------------------------------------------------
USE_AHN_CLASS_FILTER = False

CORE_B_SOR_K, CORE_B_SOR_SIGMA = 50, 0.9
CORE_C_SOR_K, CORE_C_SOR_SIGMA = 32, 1.0

USE_MIN_CLUSTER   = True
MINCLUSTER_RADIUS = 0.75
MINCLUSTER_MINPTS = 30

DILATE_RES_M        = 0.25
DILATE_RADIUS_M     = 0.6
DILATE_DZ_ABOVE     = 2.5
DILATE_DZ_BELOW     = 1.2
DILATE_USE_EC_GUARD = True


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def read_laz(path: Path) -> laspy.LasData:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return laspy.read(path)


def write_subset(src: laspy.LasData, mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = copy.deepcopy(src.header)
    sub = laspy.LasData(hdr)
    sub.points = src.points[mask]
    sub.write(out_path)


def write_xyz_from_mask(xyz_all: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, xyz_all[mask], fmt="%.6f")


def scaled_xyz(las: laspy.LasData) -> np.ndarray:
    s, o = las.header.scales, las.header.offsets
    return np.vstack((las.X * s[0] + o[0],
                      las.Y * s[1] + o[1],
                      las.Z * s[2] + o[2])).T


# ---------------------------------------------------------------------
# Core filtering helpers (unchanged logic)
# ---------------------------------------------------------------------
def sor_mask_for_subset(xyz_all: np.ndarray, base_mask: np.ndarray, k: int, sigma: float) -> np.ndarray:
    idx = np.where(base_mask)[0]
    out = np.zeros_like(base_mask, dtype=bool)
    if idx.size == 0:
        return out
    pts = xyz_all[idx]
    k_eff = min(max(2, k + 1), max(2, pts.shape[0]))
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k_eff)
    mean_nn = dists[:, 1:].mean(axis=1)
    mu = float(mean_nn.mean())
    sd = float(mean_nn.std(ddof=1)) if mean_nn.size > 1 else 0.0
    keep_local = mean_nn <= (mu + sigma * sd)
    out[idx[keep_local]] = True
    return out


def remove_small_components(xyz_all: np.ndarray, base_mask: np.ndarray,
                            radius: float = 0.6, min_pts: int = 60) -> np.ndarray:
    idx = np.where(base_mask)[0]
    if idx.size == 0:
        return base_mask.copy()
    sub = xyz_all[idx]
    tree = cKDTree(sub)
    visited = np.zeros(len(sub), dtype=bool)
    keep = np.zeros_like(base_mask, dtype=bool)

    for i in range(len(sub)):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            comp.append(j)
            for k in tree.query_ball_point(sub[j], r=radius):
                if not visited[k]:
                    stack.append(k)
        if len(comp) >= min_pts:
            keep[idx[np.array(comp)]] = True
    return keep


def _to_grid_idx(xy: np.ndarray, xmin: float, ymin: float, res: float):
    ix = np.floor((xy[:, 0] - xmin) / res).astype(int)
    iy = np.floor((xy[:, 1] - ymin) / res).astype(int)
    return ix, iy


def dilated_xy_mask_from_core(xyz: np.ndarray, core_mask: np.ndarray,
                              res: float, radius: float):
    xy_all = xyz[:, :2]
    core_xy = xy_all[core_mask]
    if core_xy.size == 0:
        raise ValueError("Empty core; cannot build dilated mask.")
    margin = radius + 2 * res
    xmin = float(core_xy[:, 0].min() - margin)
    ymin = float(core_xy[:, 1].min() - margin)
    xmax = float(core_xy[:, 0].max() + margin)
    ymax = float(core_xy[:, 1].max() + margin)
    nx = int(np.ceil((xmax - xmin) / res)) + 1
    ny = int(np.ceil((ymax - ymin) / res)) + 1
    occ = np.zeros((ny, nx), dtype=bool)
    ix, iy = _to_grid_idx(core_xy, xmin, ymin, res)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    occ[iy, ix] = True
    dist = ndi.distance_transform_edt(~occ, sampling=(res, res))
    dil = dist <= radius
    return xmin, ymin, res, dil


def points_inside_dilated_mask(xyz: np.ndarray, xmin: float, ymin: float,
                               res: float, dil_grid: np.ndarray) -> np.ndarray:
    xy = xyz[:, :2]
    ny, nx = dil_grid.shape
    ix, iy = _to_grid_idx(xy, xmin, ymin, res)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    out = np.zeros(xy.shape[0], dtype=bool)
    ok = np.where(inside)[0]
    if ok.size:
        out[ok] = dil_grid[iy[ok], ix[ok]]
    return out


# ---------------------------------------------------------------------
# Main per-tile entry point (pipeline API)
# ---------------------------------------------------------------------
def filter_tile(tile_dir: Path, overwrite: bool = False) -> dict:
    """
    Run HOMED vegetation filter for one tile directory.

    Returns:
        {"tile_id": str, "status": "ok"|"skipped"|"failed", "output": Path|None}
    """
    tile_id = tile_dir.name
    input_path = tile_dir / "clipped.laz"
    out_laz = tile_dir / "vegetation.laz"
    out_xyz = tile_dir / "vegetation.xyz"

    if out_laz.exists() and not overwrite:
        logging.info(f"[{tile_id}] Skipping existing vegetation.laz (use --overwrite to redo)")
        return {"tile_id": tile_id, "status": "skipped", "output": out_laz}

    if not input_path.exists():
        logging.warning(f"[{tile_id}] Missing input clipped.laz")
        return {"tile_id": tile_id, "status": "missing_input", "output": None}

    t0 = time.perf_counter()
    try:
        las = read_laz(input_path)
        n = len(las.points)
        logging.info(f"[{tile_id}] Read clipped.laz with {n:,} points")
        for d in ("classification", "return_number", "number_of_returns"):
            if not hasattr(las, d):
                raise ValueError(f"LAS missing required dimension: {d}")

        cls = np.asarray(las.classification)
        rn  = np.asarray(las.return_number)
        nr  = np.asarray(las.number_of_returns)
        xyz = scaled_xyz(las)

        # Step 01: optional classification == 1
        m01 = (cls == 1) if USE_AHN_CLASS_FILTER else np.ones_like(cls, dtype=bool)

        # Step 03: echo-consistent early echoes
        m03b = m01 & (nr == 3) & (rn <= 2) & (rn < nr)
        m03c = m01 & (nr >= 4) & (rn <= 3) & (rn < nr)

        # Step 03b/c: SOR filters
        m03b_sor = sor_mask_for_subset(xyz, m03b, CORE_B_SOR_K, CORE_B_SOR_SIGMA)
        m03c_sor = sor_mask_for_subset(xyz, m03c, CORE_C_SOR_K, CORE_C_SOR_SIGMA)

        # Step 04: core mask (union)
        m_core = m03b_sor | m03c_sor
        if USE_MIN_CLUSTER and m_core.any():
            m_core = remove_small_components(
                xyz, m_core,
                radius=MINCLUSTER_RADIUS,
                min_pts=MINCLUSTER_MINPTS,
            )

        # Step 05: dilated mask with vertical guard
        core_idx = np.where(m_core)[0]
        cand_mask = m01 & (~m_core)
        if DILATE_USE_EC_GUARD:
            cand_mask &= (rn < nr)
        cand_idx = np.where(cand_mask)[0]

        xmin, ymin, res, dil_grid = dilated_xy_mask_from_core(xyz, m_core, DILATE_RES_M, DILATE_RADIUS_M)
        inside_xy = points_inside_dilated_mask(xyz[cand_idx], xmin, ymin, res, dil_grid)

        m_mask = np.zeros_like(m_core, dtype=bool)
        if inside_xy.any():
            keep_xy_idx = cand_idx[inside_xy]
            tree_core = cKDTree(xyz[core_idx])
            dist, nn = tree_core.query(
                xyz[keep_xy_idx],
                k=1,
                distance_upper_bound=max(DILATE_RADIUS_M, 0.01),
            )
            ok = np.isfinite(dist)
            if ok.any():
                zc = xyz[keep_xy_idx[ok], 2]
                zn = xyz[core_idx[nn[ok].astype(int)], 2]
                dz = zc - zn
                ok[ok] &= (dz <= DILATE_DZ_ABOVE) & (dz >= -DILATE_DZ_BELOW)
            m_mask[keep_xy_idx[ok]] = True

        # Step 06: final vegetation = core ∪ mask
        m_final = m_core | m_mask

        # Write outputs
        write_subset(las, m_final, out_laz)
        write_xyz_from_mask(xyz, m_final, out_xyz)

        kept = int(m_final.sum())
        dt = time.perf_counter() - t0
        logging.info(f"[{tile_id}] Filtered vegetation: {kept:,} pts ({dt:.1f}s)")
        return {"tile_id": tile_id, "status": "ok", "output": out_laz}

    except Exception as e:
        logging.warning(f"[{tile_id}] Vegetation filter failed: {e}")
        return {"tile_id": tile_id, "status": "failed", "output": None}
