#!/usr/bin/env python
"""
scripts/run_tree_reconstruction.py

Step 3 — 3D geometry reconstruction.
Parallel per tile; serial per tree.
"""

from __future__ import annotations
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import laspy
import numpy as np
from trimesh import load as load_mesh
from scipy.spatial import cKDTree
import trimesh
import gc
import os


from src.config import get_config, setup_logger
from src.reconstruction.alpha_wrap_tree import alpha_wrap_tree
from src.reconstruction.extract_tree_metrics import compute_tree_metrics
# later: from src.reconstruction.construct_geometry import construct_geometry
# later: from src.reconstruction.write_cityjson_tile import write_cityjson_tile

# Suppress verbose logs from dependencies
logging.getLogger("fiona").setLevel(logging.WARNING)
logging.getLogger("trimesh").setLevel(logging.WARNING)
logging.getLogger("rasterio").setLevel(logging.WARNING)


# ---------------------------------------------------------------------
# helper: warm up to speed up first calls
# ---------------------------------------------------------------------
def warmup_once():
    """Warm up cKDTree, Trimesh ray 'contains', and voxel engine (single core)."""
    pts = np.random.rand(2000, 3).astype(np.float32)
    q   = np.random.rand(200,  3).astype(np.float32)
    cKDTree(pts, compact_nodes=True, balanced_tree=True).query(q, k=1, workers=1)

    verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=int)
    m = trimesh.Trimesh(verts, faces, process=False)
    _ = m.contains(np.array([[0.1,0.1,0.1],[2,2,2]], dtype=float))
    _ = m.voxelized(0.2).fill()

    logging.info(f"[init] embree available: {trimesh.ray.has_embree}")



# ---------------------------------------------------------------------
# Tile worker
# ---------------------------------------------------------------------
def process_tile(tile_dir: Path, cfg: dict, overwrite: bool = False, keep_cache: bool = False) -> dict:
    """
    Process a single tile: run alpha wrapping, tree metric extraction, and prepare data
    for geometry construction and CityJSON export.
    """
    # ------------------------------------------------------------------
    # Thread control & warmup
    # ------------------------------------------------------------------
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })

    if not getattr(process_tile, "_warmed_up", False):
        warmup_once()
        process_tile._warmed_up = True

    # ------------------------------------------------------------------
    # Tile setup
    # ------------------------------------------------------------------
    tile_id = tile_dir.name
    logging.info(f"[{tile_id}] Starting reconstruction")

    cache_dir = tile_dir / "_cache"
    if cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    forest_path = tile_dir / "forest.laz"
    dtm_path = tile_dir / "clipped_dtm.tif"

    if not forest_path.exists():
        logging.warning(f"[{tile_id}] Missing forest.laz — skipping")
        return {"tile_id": tile_id, "status": "missing_input"}

    if not dtm_path.exists():
        logging.warning(f"[{tile_id}] Missing DTM — skipping")
        return {"tile_id": tile_id, "status": "missing_input"}

    # ------------------------------------------------------------------
    # Load forest points
    # ------------------------------------------------------------------
    with laspy.open(forest_path) as lf:
        las = lf.read()

    if "gtid" not in las.point_format.dimension_names:
        logging.error(f"[{tile_id}] LAS file missing 'gtid' field")
        return {"tile_id": tile_id, "status": "invalid_input"}

    unique_gtids = np.unique(las["gtid"])
    if len(unique_gtids) == 0:
        logging.warning(f"[{tile_id}] No GTIDs found — skipping tile")
        return {"tile_id": tile_id, "status": "empty_tile"}

    # For development/testing, limit number of trees processed
    unique_gtids = unique_gtids[:10]

    tile_trees = []

    # ------------------------------------------------------------------
    # Tree loop
    # ------------------------------------------------------------------
    for gtid in unique_gtids:
        logging.debug("="*40 + f"[{tile_id}] Processing GTID {gtid}")


        idxs = np.where(las["gtid"] == gtid)[0]
        if idxs.size < 50:
            logging.debug(f"[{tile_id}] GTID {gtid}: {idxs.size} pts < 50, skip")
            continue

        # Extract and localize point cloud
        pts = np.c_[las.x[idxs], las.y[idxs], las.z[idxs]]
        offset = pts.mean(axis=0)
        local_pts = pts - offset
        logging.info(f"[{tile_id}] GTID {gtid}: translating to local coordinates (offset={offset})")

        # Save temporary XYZ file for alpha wrapping
        xyz_path = cache_dir / f"tree_{gtid}.xyz"
        np.savetxt(xyz_path, local_pts, fmt="%.6f")

        # Alpha wrapping in local space
        res_alpha = alpha_wrap_tree(xyz_path, cache_dir, overwrite=True)
        if res_alpha["status"] != "ok":
            logging.warning(f"[{tile_id}] GTID {gtid}: alpha wrap failed")
            continue

        mesh_path = res_alpha["outputs"]["mesh_ply"]
        mesh = load_mesh(mesh_path)

        # Compute metrics (local→global handled internally)
        logging.info(f"[{tile_id}] GTID {gtid}: computing metrics")
        metrics = compute_tree_metrics(mesh, local_pts, dtm_path, offset)
        if metrics["status"] != "ok":
            logging.warning(f"[{tile_id}] GTID {gtid}: metric computation failed")
            continue

        # Free memory immediately after use
        del local_pts, pts, mesh, res_alpha
        gc.collect()

        # Compact log summary
        c = metrics["crown"]
        t = metrics["trunk"]
        logging.info(
            f"[{tile_id}] Tree {gtid}: "
            f"TRUNK(H={t['H_m']:.2f} m, DBH={t['DBH_m']:.3f} m, r={t['r_trunk']:.3f} m) | "
            f"CROWN(CW={c['CW_m']:.3f} m, r50={c['r50_m']:.3f} m, porosity={c['porosity']:.3f})"
        )

        tile_trees.append({
            "gtid": int(gtid),
            "offset": offset.tolist(),
            "metrics": metrics,
            "mesh_path": str(mesh_path)
        })

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    # TODO: construct_geometry(tile_trees)
    # TODO: write_cityjson_tile(tile_dir, tile_trees)

    if not keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    logging.info(f"[{tile_id}] Reconstruction complete ({len(tile_trees)} trees)")
    return {"tile_id": tile_id, "n_trees": len(tile_trees), "status": "ok"}


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run 3D tree reconstruction (Step 3)")
    parser.add_argument("--case", help="Case name (default from config if omitted)")
    parser.add_argument("--n-cores", type=int, help="Number of parallel cores (default from config)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    case = args.case or cfg["case"]
    n_cores = args.n_cores or cfg["default_cores"]

    setup_logger(case, "tree_reconstruction", level=args.log_level)

    # Suppress noisy logs
    for noisy in ["trimesh", "rasterio", "fiona", "shapely"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"Running reconstruction for case={case} with n_cores={n_cores}")

    tiles_root = cfg["data_root"] / case / "tiles"
    if not tiles_root.exists():
        logging.error(f"No tiles found at {tiles_root}")
        return

    tile_dirs = [p for p in tiles_root.iterdir() if p.is_dir()]
    if args.dry_run:
        logging.info(f"Dry run — found {len(tile_dirs)} tiles")
        return

    # Parallel execution
    results = []
    with ProcessPoolExecutor(max_workers=n_cores) as ex:
        futs = {ex.submit(process_tile, t, cfg, args.overwrite, args.keep_cache): t for t in tile_dirs}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            logging.info(f"[{res['tile_id']}] status={res['status']}")

    logging.info("All tiles processed.")


if __name__ == "__main__":
    main()
