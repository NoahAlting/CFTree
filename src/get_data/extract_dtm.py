# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/extract_dtm.py


import pdal
import json
import logging
from pathlib import Path
from src.config import get_config


def compute_tile_dtm(clipped_las: Path, dtm_out: Path,
                     resolution: float = 0.5,
                     rigidness: int = 3,
                     iterations: int = 500,
                     ground_only: bool = True) -> Path:
    """Compute DTM from a clipped .laz file using PDAL CSF + GDAL writer."""

    cfg = get_config()
    crs = cfg["crs"]

    # Build PDAL pipeline dynamically
    pipeline_def = [
        str(clipped_las),
        {
            "type": "filters.csf",
            "resolution": resolution,
            "rigidness": rigidness,
            "iterations": iterations,
        },
    ]

    if ground_only:
        pipeline_def.append({
            "type": "filters.range",
            "limits": "Classification[2:2]",
        })

    pipeline_def.append({
        "type": "writers.gdal",
        "filename": str(dtm_out),
        "resolution": resolution,
        "output_type": "min",
        "nodata": -9999,
        "override_srs": crs,
    })

    # Run pipeline
    logging.debug(f"Running PDAL DTM pipeline on {clipped_las}")
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    try:
        pipeline.execute()
        logging.info(f"DTM written to {dtm_out}")
    except RuntimeError as e:
        logging.error(f"PDAL pipeline failed for {clipped_las}: {e}")
        raise

    return dtm_out


# For manual test:
if __name__ == "__main__":
    compute_tile_dtm(
        Path("data/wippolder/tiles/37EN2_11/clipped.laz"),
        Path("data/wippolder/tiles/37EN2_11/clipped_dtm.tif")
    )
