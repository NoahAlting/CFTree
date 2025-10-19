#!/usr/bin/env python3
"""
scripts/run_get_pointclouds.py

Full pipeline for downloading and clipping AHN geotiles for a case.

Example:
    nohup python -m scripts.run_get_pointclouds --n-cores 2 --case wippolder &
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import geopandas as gpd
from pathlib import Path

from src.config import get_config, setup_logger
from src.get_pointclouds.download_geotiles import download_tile
from src.get_pointclouds.clip_tile import clip_tile


# ---------------------------------------------------------------------
# Tile worker (must be top-level for multiprocessing)
# ---------------------------------------------------------------------
def process_tile(tile_id: str, output_dir: Path, base_url: str, overwrite: bool, aoi_path: Path) -> dict:
    """Download and immediately clip one tile."""
    try:
        result_dl = download_tile(tile_id, output_dir, base_url, overwrite)
        laz_path = result_dl.get("paths", {}).get("laz")

        if result_dl["status"] != "ok" or not laz_path or not Path(laz_path).exists():
            return {"tile_id": tile_id, "status": "download_failed"}

        result_clip = clip_tile(Path(laz_path), aoi_path)
        return {"tile_id": tile_id, "status": result_clip["status"]}
    except Exception as e:
        return {"tile_id": tile_id, "status": f"error: {e}"}


# ---------------------------------------------------------------------
# Runner main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run get_pointclouds pipeline for a case.")
    parser.add_argument("--case", type=str, help="Case name (default from config)")
    parser.add_argument("--n-cores", type=int, default=None, help="Number of parallel workers (default from config)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download tiles if they exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--dry-run", action="store_true", help="Only list tiles to be processed")
    parser.add_argument("--buffer", type=float, default=20.0, help="Buffer in meters around AOI")
    args = parser.parse_args()

    # Load configuration
    cfg = get_config()
    case = args.case or cfg["case"]
    setup_logger(case, "get_pointclouds", args.log_level)

    # Resolve number of cores
    n_cores = args.n_cores or cfg["default_cores"]
    logging.info(f"Starting get_pointclouds for case: {case}")
    logging.info(f"Parallel workers: {n_cores} (from {'CLI' if args.n_cores else 'config'})")
    logging.info(f"Buffer distance: {args.buffer} m")

    aoi_path = cfg["case_path"] / "city_bbox.geojson"
    buffered_aoi_path = cfg["case_path"] / "city_bbox_buffered.geojson"
    resources_dir = cfg["resources_dir"]
    output_dir = cfg["data_case_path"]

    # ------------------------------------------------------------------
    # Step 1: Load and buffer AOI
    # ------------------------------------------------------------------
    logging.info(f"Loading AOI from {aoi_path}")
    aoi = gpd.read_file(aoi_path).to_crs("EPSG:28992")
    aoi["geometry"] = aoi.buffer(args.buffer)
    aoi.to_file(buffered_aoi_path, driver="GeoJSON")
    logging.info(f"Buffered AOI saved to {buffered_aoi_path}")

    # ------------------------------------------------------------------
    # Step 2: Determine intersecting tiles
    # ------------------------------------------------------------------
    tiles = gpd.read_file(resources_dir / "AHN_subunits_GeoTiles.shp").to_crs("EPSG:28992")
    intersecting = tiles[tiles.intersects(aoi.union_all())]
    tile_ids = intersecting["GT_AHNSUB"].unique().tolist()

    if not tile_ids:
        logging.info("No intersecting tiles found.")
        return

    logging.info(f"Found {len(tile_ids)} intersecting tiles: {tile_ids if len(tile_ids) <= 10 else '...'}")

    if args.dry_run:
        logging.info("[DRY RUN] Exiting before downloads.")
        return

    # ------------------------------------------------------------------
    # Step 3: Per-tile pipeline (download → clip)
    # ------------------------------------------------------------------
    base_url = "https://geotiles.citg.tudelft.nl/AHN5_T"

    if n_cores > 1:
        logging.info(f"Running {len(tile_ids)} tiles in parallel using {n_cores} cores.")
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            futures = {
                pool.submit(process_tile, tid, output_dir, base_url, args.overwrite, buffered_aoi_path): tid
                for tid in tile_ids
            }
            for f in as_completed(futures):
                tid = futures[f]
                try:
                    result = f.result()
                    logging.info(f"[{tid}] {result['status'].upper()}")
                except Exception as e:
                    logging.warning(f"[{tid}] Exception: {e}")
    else:
        logging.info("Running serial mode.")
        for tid in tile_ids:
            result = process_tile(tid, output_dir, base_url, args.overwrite, buffered_aoi_path)
            logging.info(f"[{tid}] {result['status'].upper()}")

    logging.info(f"Completed get_pointclouds for case: {case}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
