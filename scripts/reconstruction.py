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
import json

from src.config import get_config, setup_logger
from src.reconstruction.alpha_wrap_tree import alpha_wrap_tree
from src.reconstruction.extract_tree_metrics import compute_tree_metrics
from src.reconstruction.construct_geometry import construct_lod3
from src.reconstruction.write_cityjson import init_cityjson, add_tree, finalize_cityjson


# ---------------------------------------------------------------------
# Warmup for libraries (helps avoid startup lag)
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

    logging.debug(f"[init] embree available: {trimesh.ray.has_embree}")


# ---------------------------------------------------------------------
# Tile worker
# ---------------------------------------------------------------------
def process_tile(
    tile_dir: Path,
    cfg: dict,
    overwrite: bool = False,
    keep_cache: bool = False,
    max_trees: int | None = None,
) -> dict:
    """
    Process a single tile: run alpha wrapping, tree metric extraction,
    geometry construction, and CityJSON export.
    """
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })

    if not getattr(process_tile, "_warmed_up", False):
        warmup_once()
        process_tile._warmed_up = True

    tile_id = tile_dir.name
    logging.info(f"[{tile_id}] Starting reconstruction")

    cache_dir = tile_dir / "_cache"
    if cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    forest_path = tile_dir / "forest.laz"
    dtm_path = tile_dir / "clipped_dtm.tif"
    cityjson_path = tile_dir / "trees_lod3.city.json"

    if not forest_path.exists() or not dtm_path.exists():
        logging.warning(f"[{tile_id}] Missing inputs — skipping tile")
        return {"tile_id": tile_id, "status": "missing_input"}

    if cityjson_path.exists() and not overwrite:
        logging.info(f"[{tile_id}] CityJSON already exists — skipping")
        return {"tile_id": tile_id, "status": "exists"}

    # Load point cloud
    with laspy.open(forest_path) as lf:
        las = lf.read()

    if "gtid" not in las.point_format.dimension_names:
        logging.error(f"[{tile_id}] LAS file missing 'gtid' field")
        return {"tile_id": tile_id, "status": "invalid_input"}

    unique_gtids = np.unique(las["gtid"])
    if len(unique_gtids) == 0:
        logging.warning(f"[{tile_id}] No GTIDs found — skipping tile")
        return {"tile_id": tile_id, "status": "empty_tile"}

    city = init_cityjson()
    processed = 0

    if max_trees: # limit number of trees for testing
        unique_gtids = unique_gtids[:max_trees]

    # log number of trees we are processing and say if there was a max_trees limit
    logging.info(f"[{tile_id}] Processing {len(unique_gtids)} trees" + (f" (limited to max_trees={max_trees})" if max_trees else ""))

    for gtid in unique_gtids:                      
        logging.debug("="*40 + f"[{tile_id}] Processing GTID {gtid}")

        idxs = np.where(las["gtid"] == gtid)[0]
        if idxs.size < 50:
            logging.debug(f"[{tile_id}] GTID {gtid}: {idxs.size} pts < 50, skip")
            continue

        # Extract & localize
        pts = np.c_[las.x[idxs], las.y[idxs], las.z[idxs]]
        offset = pts.mean(axis=0)
        local_pts = pts - offset
        logging.debug(f"[{tile_id}] GTID {gtid}: localize point cloud (offset={offset})")

        xyz_path = cache_dir / f"tree_{gtid}.xyz"
        np.savetxt(xyz_path, local_pts, fmt="%.6f")

        # Alpha wrap
        res_alpha = alpha_wrap_tree(xyz_path, cache_dir, overwrite=True)
        if res_alpha["status"] != "ok":
            logging.warning(f"[{tile_id}] GTID {gtid}: alpha wrap failed")
            continue

        mesh_path = res_alpha["outputs"]["mesh_ply"]
        mesh = load_mesh(mesh_path)

        # Metrics
        logging.debug(f"[{tile_id}] GTID {gtid}: computing metrics")
        metrics = compute_tree_metrics(mesh, local_pts, dtm_path, offset)
        if metrics["status"] != "ok":
            logging.warning(f"[{tile_id}] GTID {gtid}: metric computation failed")
            del mesh, local_pts
            continue

        # Construct geometry (LoD3 only for now)
        tree_geom = construct_lod3(mesh, metrics, offset, gtid=int(gtid), tile_id=tile_id)
        if not tree_geom["components"]:
            logging.warning(f"[{tile_id}] GTID {gtid}: no geometry constructed")
            del mesh, local_pts, res_alpha
            continue

        # Add to CityJSON
        add_tree(city, int(gtid), tree_geom["components"], offset, tree_geom["attributes"])

        processed += 1

        # Free memory
        del mesh, local_pts, res_alpha, tree_geom
        gc.collect()

    # Finalize CityJSON
    if processed > 0:
        city_final = finalize_cityjson(city)
        with open(cityjson_path, "w", encoding="utf-8") as f:
            json.dump(city_final, f, indent=2)
        logging.info(f"[{tile_id}] CityJSON written: {cityjson_path.name} ({processed} trees)")
    else:
        logging.warning(f"[{tile_id}] No trees processed — skipping CityJSON write.")

    if not keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    logging.info(f"[{tile_id}] Reconstruction complete ({processed} trees)")
    return {"tile_id": tile_id, "n_trees": processed, "status": "ok"}


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run 3D tree reconstruction (Step 3)")
    parser.add_argument("--case", type=str, help="Case name (default from config if omitted)")
    parser.add_argument("--n-cores", type=int, help="Number of parallel cores (default from config if omitted)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--max-trees", type=int, default=None, help="Limit number of trees per tile (for testing)")
    args = parser.parse_args()

    cfg = get_config()
    case = args.case or cfg["case"]
    n_cores = args.n_cores or cfg["default_cores"]

    setup_logger(case, "tree_reconstruction", level=args.log_level)

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

    results = []
    with ProcessPoolExecutor(max_workers=n_cores) as ex:
        futs = {
            ex.submit(process_tile, t, cfg, args.overwrite, args.keep_cache, args.max_trees): t
            for t in tile_dirs
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            logging.info(f"[{res['tile_id']}] status={res['status']}")

    logging.info("All tiles processed.")


if __name__ == "__main__":
    main()
