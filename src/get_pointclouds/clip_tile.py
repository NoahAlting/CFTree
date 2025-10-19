"""
src/get_pointclouds/clip_tile.py

Clip a single downloaded AHN tile using PDAL via the robust bash script.
"""

import subprocess
import logging
from pathlib import Path


def clip_tile(laz_path: Path, aoi_path: Path, output_dir: Path | None = None) -> dict:
    """
    Clip a single LAZ file using PDAL through the robust bash script.

    Parameters
    ----------
    laz_path : Path
        Path to the downloaded .laz file (e.g., data/wippolder/tiles/37EN2_11/raw.laz)
    aoi_path : Path
        Path to the buffered AOI GeoJSON (e.g., cases/wippolder/city_bbox_buffered.geojson)
    output_dir : Path, optional
        Directory to store the clipped file (defaults to same folder as laz_path)

    Returns
    -------
    dict
        { 'tile_id': str, 'status': 'clipped'|'failed', 'output': Path }
    """
    script_path = Path(__file__).parent / "tiles_clipper_robust.sh"
    if not script_path.exists():
        logging.error(f"Clipping script not found: {script_path}")
        return {"tile_id": laz_path.stem, "status": "missing_script", "output": None}

    if not laz_path.exists():
        logging.error(f"Input LAZ not found: {laz_path}")
        return {"tile_id": laz_path.stem, "status": "missing_input", "output": None}

    if not aoi_path.exists():
        logging.error(f"AOI file not found: {aoi_path}")
        return {"tile_id": laz_path.stem, "status": "missing_aoi", "output": None}

    # Determine output path
    tile_dir = laz_path.parent
    tile_id = tile_dir.name
    output_dir = output_dir or tile_dir
    clipped_path = output_dir / "clipped.laz"

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
        logging.info(f"[{tile_id}] Clipped successfully → {clipped_path}")
        return {"tile_id": tile_id, "status": "clipped", "output": clipped_path}

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore").strip()
        logging.warning(f"[{tile_id}] Clipping failed: {stderr}")
        return {"tile_id": tile_id, "status": "failed", "output": None}
