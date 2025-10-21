#!/usr/bin/env python3
import subprocess
import logging
import time
import argparse
import geopandas as gpd
from src.config import get_config, setup_logger
import os
import signal


def query_tile_intersection(cfg, case, buffer=20.0):
    """
    Returns the list of intersecting tile IDs for a given case.
    Replicates the logic from get_data.py.
    """
    aoi_path = cfg["case_root"] / case / "case_area.geojson"
    tiles_path = cfg["resources_dir"] / "AHN_subunits_GeoTiles" / "AHN_subunits_GeoTiles.shp"

    # Load AOI and buffer it
    aoi = gpd.read_file(aoi_path).to_crs(cfg["crs"])
    aoi["geometry"] = aoi.buffer(buffer)

    # Load GeoTiles and find intersections
    tiles = gpd.read_file(tiles_path).to_crs(cfg["crs"])
    intersecting = tiles[tiles.intersects(aoi.union_all())]
    tile_ids = intersecting["GT_AHNSUB"].unique().tolist()
    return tile_ids



def run_stage(name, cmd):
    """Run one pipeline stage and ensure cleanup of all workers if interrupted."""
    start = time.time()

    # Start subprocess in a new process group so we can kill all its children later
    process = subprocess.Popen(
        cmd,
        shell=True,
        preexec_fn=os.setsid,  # create new session (POSIX)
    )

    try:
        process.wait()
        if process.returncode != 0:
            logging.warning(f"{name} failed with exit code {process.returncode}")
    except KeyboardInterrupt:
        logging.warning(f"KeyboardInterrupt detected — terminating {name} and its workers...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        raise
    except Exception as e:
        logging.warning(f"{name} encountered error: {e}")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        raise

    elapsed = (time.time() - start) / 60
    logging.info(f"Finished {name} in {elapsed:.2f} min")


def main():
    start = time.time()  # <-- define start time

    # -------------------------------
    # Parse CLI arguments
    # -------------------------------
    parser = argparse.ArgumentParser(description="Run full CFTree pipeline (get_data → segmentation → reconstruction).")
    parser.add_argument("--case", type=str, help="Case name (default from config)")
    parser.add_argument("--n-cores", type=int, help="Number of parallel workers (default from config)")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--dry-run", action="store_true", help="Only list tiles to be processed")
    parser.add_argument("--buffer", type=float, default=20.0, help="Buffer distance around AOI (default 20m)")
    parser.add_argument("--max-trees", type=int, default=None, help="Limit number of trees per tile (for testing)")
    args = parser.parse_args()

    # -------------------------------
    # Load configuration
    # -------------------------------
    cfg = get_config()
    
    case = args.case if args.case is not None else cfg["case"]
    n_cores = args.n_cores if args.n_cores is not None else cfg["default_cores"]

    # -------------------------------
    # Setup main logger
    # -------------------------------
    log_path = setup_logger(case, "main", level="INFO")
    logger = logging.getLogger()  # <-- define logger

    # -------------------------------
    # Estimate tiles to process
    # -------------------------------
    try:
        tile_ids = query_tile_intersection(cfg, case, buffer=args.buffer)
        if tile_ids:
            logger.info(f"Case '{case}' requires {len(tile_ids)} tiles: {tile_ids}")
        else:
            logger.warning(f"Case '{case}' — no intersecting tiles found.")
    except Exception as e:
        logger.warning(f"Could not determine tile intersection: {e}")
        tile_ids = []

    # -------------------------------
    # Base command builder
    # -------------------------------
    base_cmd = f"--case {case} --n-cores {n_cores}"
    if args.overwrite:
        base_cmd += " --overwrite"
    if args.dry_run:
        base_cmd += " --dry-run"
    base_cmd += f" --log-level {args.log_level}"

    # -------------------------------
    # Stage command definitions
    # -------------------------------
    cmd_get_data = f"python -m scripts.get_data {base_cmd} --buffer {args.buffer}"
    cmd_segmentation = f"python -m scripts.segmentation {base_cmd}"
    cmd_reconstruction = f"python -m scripts.reconstruction {base_cmd}"
    if args.max_trees is not None:
        cmd_reconstruction += f" --max-trees {args.max_trees}" 

    # -------------------------------
    # Run stages sequentially
    # -------------------------------
    logger.info("\n" + "=" * 60 + "Starting full CFTree pipeline")
    logger.info("=" * 20 + " Stage 1: Data Acquisition")
    logger.info(f"buffer distance: {args.buffer} m")
    logger.info(f'running...\t ETA = ~ {len(tile_ids) / n_cores * 1.2} minutes (assuming ~1.2 min per tile)')
    run_stage("get_data", cmd_get_data)

    logger.info("=" * 20 + " Stage 2: Segmentation")
    logger.info(f'running...\t ETA = ~ {len(tile_ids) / n_cores * 0.0} minutes (assuming ~0.0 min per tile)')
    run_stage("segmentation", cmd_segmentation)

    logger.info("=" * 20 + " Stage 3: Reconstruction")
    logger.info(f"max trees per tile: {args.max_trees if args.max_trees is not None else 'unlimited'}")
    logger.info(f'running...\t ETA = ~ {len(tile_ids) / n_cores * 0.0} minutes (assuming ~0.0 min per tile)')
    run_stage("reconstruction", cmd_reconstruction)

    total_time = (time.time() - start) / 60
    logger.info("\n" + "=" * 60 + "Pipeline complete")
    logger.info(f"total elapsed time: {total_time:.2f} minutes")


if __name__ == "__main__":
    main()
