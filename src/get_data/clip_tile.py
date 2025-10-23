# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/clip_tile.py

import subprocess
import logging
from pathlib import Path


def clip_tile(laz_path: Path, aoi_path: Path, output_dir: Path | None = None, overwrite: bool = False) -> dict:
    """
    Clip a single LAZ file using PDAL through the robust bash script.

    Returns
    -------
    dict
        {
            "tile_id": str,
            "status": "ok" | "failed" | "missing_input" | "missing_aoi" | "missing_script",
            "paths": {"clipped": Path | None}
        }
    """

    script_path = Path(__file__).parent / "tiles_clipper_robust.sh"
    tile_id = laz_path.parent.name
    output_dir = output_dir or laz_path.parent
    clipped_path = output_dir / "clipped.laz"

    # --- Validate prerequisites ---
    if not script_path.exists():
        logging.error(f"[{tile_id}] Clipping script not found: {script_path}")
        return {"tile_id": tile_id, "status": "missing_script", "paths": {"clipped": None}}

    if not laz_path.exists():
        logging.error(f"[{tile_id}] Input LAZ not found: {laz_path}")
        return {"tile_id": tile_id, "status": "missing_input", "paths": {"clipped": None}}

    if not aoi_path.exists():
        logging.error(f"[{tile_id}] AOI file not found: {aoi_path}")
        return {"tile_id": tile_id, "status": "missing_aoi", "paths": {"clipped": None}}

    # --- Skip existing clipped tile ---
    if clipped_path.exists() and not overwrite:
        logging.info(f"[{tile_id}] Skipping existing clipped tile")
        return {"tile_id": tile_id, "status": "ok", "paths": {"clipped": clipped_path}}

    # --- Run clipping ---
    logging.info(f"[{tile_id}] Clipping raw tile → {clipped_path.name}")
    try:
        subprocess.run(
            [
                "bash",
                str(script_path),
                str(laz_path),
                str(aoi_path),
                str(clipped_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if clipped_path.exists():
            logging.info(f"[{tile_id}] Clipped successfully → {clipped_path}")
            return {"tile_id": tile_id, "status": "ok", "paths": {"clipped": clipped_path}}
        else:
            logging.warning(f"[{tile_id}] Clipping completed but file missing: {clipped_path}")
            return {"tile_id": tile_id, "status": "failed", "paths": {"clipped": None}}

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore").strip()
        logging.warning(f"[{tile_id}] Clipping failed: {stderr}")
        return {"tile_id": tile_id, "status": "failed", "paths": {"clipped": None}}

    except Exception as e:
        logging.exception(f"[{tile_id}] Unexpected clipping error: {e}")
        return {"tile_id": tile_id, "status": f"error: {e}", "paths": {"clipped": None}}
