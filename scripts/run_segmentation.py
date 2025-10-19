#!/usr/bin/env python3
"""
scripts/run_tree_segmentation.py

Step 2: Vegetation filtering (HOMED) + Segmentation (TreeSeparation) + Forest ID generalization.

Example:
    nohup python -m scripts.run_tree_segmentation --case wippolder --n-cores 4 &
"""

import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.config import get_config, setup_logger
from src.vegetation_filter.HOMED_vegetation_filter import filter_tile
from src.segmentation.segment_tile import segment_tile
from src.segmentation.generalize_forest_ids import generalize_forest_ids


# ---------------------------------------------------------------------
# Per-tile worker (top-level for multiprocessing)
# ---------------------------------------------------------------------
def process_tile(tile_dir: Path, overwrite: bool = False) -> dict:
    """
    Run vegetation filtering + segmentation for one tile.
    Returns combined status dict.
    """
    tile_id = tile_dir.name
    try:
        # Step 1: vegetation filter
        result_filter = filter_tile(tile_dir, overwrite)
        if result_filter["status"] not in ("ok", "skipped"):
            return {"tile_id": tile_id, "status": f"veg_failed ({result_filter['status']})"}

        # Step 2: segmentation
        result_seg = segment_tile(tile_dir, overwrite)
        return {
            "tile_id": tile_id,
            "status": f"seg_{result_seg['status']}",
            "outputs": result_seg.get("outputs", {}),
        }

    except Exception as e:
        return {"tile_id": tile_id, "status": f"error: {e}", "outputs": {}}


# ---------------------------------------------------------------------
# Runner main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run vegetation filtering, segmentation, and forest ID generalization.")
    parser.add_argument("--case", type=str, help="Case name (default from config)")
    parser.add_argument("--n-cores", type=int, default=None, help="Number of parallel workers (default from config)")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--dry-run", action="store_true", help="List tiles only, no processing")
    args = parser.parse_args()

    # Load configuration
    cfg = get_config()
    case = args.case or cfg["case"]
    setup_logger(case, "tree_segmentation", args.log_level)

    # Resolve number of cores
    n_cores = args.n_cores or cfg["default_cores"]
    logging.info(f"Starting tree segmentation pipeline for case: {case}")
    logging.info(f"Parallel workers: {n_cores}")
    logging.info(f"Overwrite: {args.overwrite}")

    # Locate tiles
    tiles_root = cfg["data_root"] / case / "tiles"
    if not tiles_root.exists():
        logging.error(f"Tiles directory not found: {tiles_root}")
        return

    tile_dirs = sorted([p for p in tiles_root.iterdir() if (p / "clipped.laz").exists()])
    if not tile_dirs:
        logging.info("No clipped tiles found — nothing to process.")
        return

    logging.info(f"Found {len(tile_dirs)} tiles for case {case}")
    if args.dry_run:
        for t in tile_dirs:
            logging.info(f"[DRY RUN] Would process tile: {t.name}")
        return

    # ------------------------------------------------------------------
    # Parallel or serial execution
    # ------------------------------------------------------------------
    if n_cores > 1:
        logging.info(f"Running in parallel with {n_cores} cores.")
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            futures = {pool.submit(process_tile, td, args.overwrite): td.name for td in tile_dirs}
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    result = fut.result()
                    logging.info(f"[{tid}] {result['status'].upper()}")
                except Exception as e:
                    logging.warning(f"[{tid}] Exception: {e}")
    else:
        logging.info("Running in serial mode.")
        for td in tile_dirs:
            result = process_tile(td, args.overwrite)
            logging.info(f"[{td.name}] {result['status'].upper()}")

    # ------------------------------------------------------------------
    # Step 3: Forest generalization
    # ------------------------------------------------------------------
    try:
        out_forest_hulls = cfg["data_root"] / case / "forest_hulls.geojson"
        out_gtid_map = cfg["data_root"] / case / "gtid_map.csv"

        # Skip if already exists and overwrite not requested
        if (out_forest_hulls.exists() and out_gtid_map.exists()) and not args.overwrite:
            logging.info(
                f"Forest generalization outputs already exist — skipped "
                f"(use --overwrite to regenerate)."
            )
        else:
            logging.info("Starting forest ID generalization...")
            result_generalize = generalize_forest_ids(case, overwrite=args.overwrite)

            if result_generalize["status"] == "ok":
                logging.info(
                    f"Forest generalization complete: {result_generalize['n_trees']} trees → "
                    f"{result_generalize['outputs']['forest_hulls']} / "
                    f"{result_generalize['outputs']['gtid_map']}"
                )
            else:
                logging.warning(
                    f"Forest generalization returned status: {result_generalize['status']}"
                )

    except Exception as e:
        logging.error(f"Forest generalization failed: {e}")

    logging.info(f"Completed tree segmentation pipeline for case: {case}")



# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
