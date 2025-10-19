import pdal
import json

# --------------------------
# PDAL pipeline: CSF → ground points → rasterize (min)
# --------------------------
pipeline_json = """
[
  "clipped.laz",
  {
    "type": "filters.csf",
    "resolution": 0.5,
    "rigidness": 3,
    "iterations": 500
  },
  {
    "type": "filters.range",
    "limits": "Classification[2:2]"
  },
  {
    "type": "writers.gdal",
    "filename": "clipped_dtm.tif",
    "resolution": 0.5,
    "output_type": "min",
    "nodata": -9999,
    "override_srs": "EPSG:28992"
  }
]
"""

pipeline_def = json.loads(pipeline_json)
pipeline = pdal.Pipeline(json.dumps(pipeline_def))
pipeline.execute()

print("Done: clipped_dtm.tif created with CRS=EPSG:28992")
