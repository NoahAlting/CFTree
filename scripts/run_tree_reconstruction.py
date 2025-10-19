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

from src.config import get_config, setup_logger
from src.reconstruction.alpha_wrap_tree import alpha_wrap_tree
from src.reconstruction.trunk_estimation_tree import compute_trunk_base_from_dtm, estimate_trunk_dimensions

# later: from src.reconstruction.write_cityjson import write_cityjson_tile

# ---------------------------------------------------------------------
# Tile worker
# ---------------------------------------------------------------------
def process_tile(tile_dir: Path, cfg: dict, overwrite=False, keep_cache=False) -> dict:
    tile_id = tile_dir.name
    logging.info(f"[{tile_id}] Starting reconstruction.")

    cache_dir = tile_dir / "_cache"
    if cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    forest_path = tile_dir / "forest.laz"
    if not forest_path.exists():
        logging.warning(f"[{tile_id}] Missing forest.laz — skipping.")
        return {"tile_id": tile_id, "status": "missing_input"}

    # Load forest points
    with laspy.open(forest_path) as lf:
        las = lf.read()
    gtid = las.points["gtid"]
    unique_gtids = np.unique(las["gtid"])
    take = unique_gtids[:10]  # test limit

    for gtid in take:
        idxs = np.where(las["gtid"] == gtid)[0]
        if idxs.size == 0:
            logging.debug(f"[{tile_id}] GTID {gtid}: no points, skip")
            continue

        xyz_path = cache_dir / f"tree_{gtid}.xyz"
        pts = np.c_[las.x[idxs], las.y[idxs], las.z[idxs]]
        # optional: skip tiny trees
        if pts.shape[0] < 50:
            logging.debug(f"[{tile_id}] GTID {gtid}: {pts.shape[0]} pts < 50, skip")
            continue
        np.savetxt(xyz_path, pts, fmt="%.6f")

        res_alpha = alpha_wrap_tree(xyz_path, cache_dir, overwrite=True)  # during dev
        if res_alpha["status"] != "ok":
            logging.warning(f"[{tile_id}] GTID {gtid}: alpha wrap failed")
            continue

    
        # dtm_path = tile_dir / "clipped_dtm.tif"

        # Trunk estimation
        mesh = load_mesh(res_alpha["outputs"]["mesh_ply"])
        # trunk_base = compute_trunk_base_from_dtm(mesh, dtm_path)
        # # Dummy CW / height values for now
        # CW_m = np.linalg.norm(mesh.bounding_box.extents[:2])
        # crown_median_z = np.median(mesh.vertices[:, 2])
        # H, DBH_m, r_trunk = estimate_trunk_dimensions(CW_m, crown_median_z, trunk_base[2] if trunk_base is not None else np.nan)
        # logging.info(f"[{tile_id}] Tree {tree_id}: H={H:.2f} m, DBH={DBH_m:.3f} m, r={r_trunk:.3f} m")

        # TODO: add to CityJSON collector

    # TODO: finalize and write per-tile CityJSON

    if not keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    logging.info(f"[{tile_id}] Reconstruction complete.")
    return {"tile_id": tile_id, "status": "ok"}


# ---------------------------------------------------------------------
# CLI entry
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

    # Apply defaults if not provided
    case = args.case or cfg["case"]
    n_cores = args.n_cores or cfg["default_cores"]

    setup_logger(case, "tree_reconstruction", level=args.log_level)
    logging.info(f"Running reconstruction for case={case} with n_cores={n_cores}")

    tiles_root = cfg["data_root"] / case / "tiles"
    if not tiles_root.exists():
        logging.error(f"No tiles found at {tiles_root}")
        return

    tile_dirs = [p for p in tiles_root.iterdir() if p.is_dir()]
    if args.dry_run:
        logging.info(f"Dry run — found {len(tile_dirs)} tiles.")
        return

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
