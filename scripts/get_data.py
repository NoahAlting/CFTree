# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

#!/usr/bin/env python3
"""
scripts/run_get_data.py

Full pipeline for downloading and clipping AHN geotiles for a case.

Example:
    nohup python -m scripts.run_get_data --n-cores 2 --case wippolder &
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import geopandas as gpd
from pathlib import Path

from src.config import get_config, setup_logger
from src.get_data.download_geotiles import download_tile
from src.get_data.clip_tile import clip_tile
from src.get_data.extract_dtm import compute_tile_dtm


# ---------------------------------------------------------------------
# Tile worker (must be top-level for multiprocessing)
# ---------------------------------------------------------------------
def process_tile(tile_id: str, output_dir: Path, base_url: str, overwrite: bool, aoi_path: Path) -> dict:
    """Download, clip, and compute DTM for one tile."""
    try:
        # ---------------------------
        # 1. Download raw tile
        # ---------------------------
        result_dl = download_tile(tile_id, output_dir, base_url, overwrite=overwrite)
        laz_path = result_dl.get("paths", {}).get("laz")

        if result_dl["status"] != "ok" or not laz_path or not Path(laz_path).exists():
            return {"tile_id": tile_id, "status": "download_failed"}

        # ---------------------------
        # 2. Clip tile to AOI
        # ---------------------------
        result_clip = clip_tile(Path(laz_path), aoi_path, overwrite=overwrite)
        if result_clip["status"] != "ok":
            return {"tile_id": tile_id, "status": result_clip["status"]}

        clipped_path = result_clip.get("paths", {}).get("clipped")
        if not clipped_path or not Path(clipped_path).exists():
            logging.warning(f"[{tile_id}] Clipped tile missing — skipping DTM generation.")
            return {"tile_id": tile_id, "status": "clip_failed"}

        # ---------------------------
        # 3. Compute DTM
        # ---------------------------
        dtm_path = Path(clipped_path).with_name("clipped_dtm.tif")
        if not dtm_path.exists() or overwrite:
            try:
                logging.info(f"[{tile_id}] Computing DTM from clipped tile...")
                compute_tile_dtm(Path(clipped_path), dtm_path, ground_only=True)
                status = "ok"
            except Exception as e:
                logging.warning(f"[{tile_id}] DTM generation failed: {e}")
                status = "dtm_failed"
        else:
            logging.debug(f"[{tile_id}] DTM already exists — skipped.")
            status = "ok"

        # ---------------------------
        # 4. Return tile summary
        # ---------------------------
        return {
            "tile_id": tile_id,
            "status": status,
            "paths": {
                "raw": str(laz_path),
                "clipped": str(clipped_path),
                "dtm": str(dtm_path),
            },
        }

    except Exception as e:
        logging.exception(f"[{tile_id}] Unexpected error: {e}")
        return {"tile_id": tile_id, "status": f"error: {e}"}


# ---------------------------------------------------------------------
# Runner main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run get_data pipeline for a case.")
    parser.add_argument("--case", type=str, help="Case name (default from config)")
    parser.add_argument("--n-cores", type=int, default=None, help="Number of parallel workers (default from config)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download tiles if they exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--dry-run", action="store_true", help="Only list tiles to be processed")
    parser.add_argument("--buffer", type=float, default=20.0, help="Buffer in meters around AOI")
    args = parser.parse_args()

    # Load configuration
    cfg = get_config(case_name=args.case, n_cores=args.n_cores)
    case = cfg["case"]
    n_cores = cfg["default_cores"]

    setup_logger(case, "get_data", args.log_level)

    logging.info(f"Starting get_data for case: {case}")
    logging.info(f"Parallel workers: {n_cores} (from {'CLI' if args.n_cores else 'config'})")
    logging.info(f"Buffer distance: {args.buffer} m")

    aoi_path = cfg["case_path"] / "case_area.geojson"
    buffered_aoi_path = cfg["case_path"] / "case_area_buffered.geojson"
    resources_dir = cfg["resources_dir"] 
    output_dir = cfg["data_case_path"]

    # ------------------------------------------------------------------
    # Step 1: Load and buffer AOI
    # ------------------------------------------------------------------
    logging.info(f"Loading AOI from {aoi_path}")
    logging.info(f"CRS: {cfg['crs']}")
    aoi = gpd.read_file(aoi_path).to_crs(cfg["crs"])
    aoi["geometry"] = aoi.buffer(args.buffer)
    aoi.to_file(buffered_aoi_path, driver="GeoJSON")
    logging.info(f"Buffered AOI saved to {buffered_aoi_path}")

    # ------------------------------------------------------------------
    # Step 2: Determine intersecting tiles
    # ------------------------------------------------------------------
    tiles = gpd.read_file(resources_dir / "AHN_subunits_GeoTiles" / "AHN_subunits_GeoTiles.shp").to_crs(cfg["crs"])
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

    logging.info(f"Completed get_data for case: {case}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
