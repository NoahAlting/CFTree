# Scripts — Pipeline Execution

This folder contains all high-level **orchestration scripts** for running the CFD-ready urban tree reconstruction pipeline.  
Each script represents one stage of the workflow and can be executed independently or sequentially.

## Full Pipeline instructions:

### Input Requirements:
Each case must be initialized by case folder holding a polygon of the desired area to process in geojson format. 
For testing, an example neighbourhood of the city of Delft is provided: `/cases/wippolder/city_bbox.geojson`.

Global, case-specific configurations can be set in `/src/config.py` and is automatically propagated to all modules.
``` python
# ---------------------------------------------------------------------
# Case configurations used throughout the pipeline
# ---------------------------------------------------------------------
CASE_CONFIGURATIONS = {
    "case_root": Path("cases"),             # user case input directory
    "data_root": Path("data"),              # data storage root (large files)
    "resources_dir": Path("resources/AHN_subunits_GeoTiles"),
    "case": "wippolder",                    # test case
    "default_cores": 2,                     # Global default for parallelization
    "crs": "EPSG:28992",                    # Amersfoort / RD New
}
```

Before running it is necessary to build the cpp executables at `src/segmentation/TreeSeparation` and `src/reconstruction/AlphaWrap`.

> **NOTE:** The pipeline supports city-scale processing. Large cases may contain many tiles and will therefore produce large files.
It is recommended to make sure enough disk space is available at `data_root`.

### Execution Order:

1. **Data acquisition** → `get_data.py`  
2. **Tree segmentation** → `segmentation.py`  
3. **Geometry reconstruction** → `reconstruction.py`

Each stage reads and writes to `data/<case>/tiles/<tile_id>/`, ensuring reproducibility and modularity.
Logs for each run are written to:  `cases/<case_name>/logs/<step_name>.log`

### Example Full Pipeline:
``` bash
python -m scripts.get_data --case wippolder --n-cores 4
python -m scripts.segmentation --case wippolder --n-cores 8
python -m scripts.reconstruction --case wippolder --n-cores 16
```
All stages are independent, enabling partial reruns or debugging at any point.
Logs and outputs are structured per case for traceability.

### Shared CLI Parameters

All scripts accept the following common flags:

| Flag | Type | Description |
|------|------|--------------|
| `--case` | `str` | Name of the case folder under `cases/` (default from config). |
| `--n-cores` | `int` | Number of CPU cores to use. Parallel execution via `ProcessPoolExecutor`. |
| `--overwrite` | `flag` | Force regeneration of existing outputs. |
| `--log-level` | `str` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`). |
| `--dry-run` | `flag` | List tasks without executing them. |



## 1. Data Acquisition — `get_data.py`

Downloads and prepares AHN5 tiles for the selected case.  
Performs download, clipping, and DTM generation.

### Main steps:
1. Buffer the case polygon (`city_bbox.geojson`).
2. Find intersecting AHN5 tiles.
3. Download raw `.laz` files from [TU Delft GeoTiles server](https://geotiles.citg.tudelft.nl/AHN5_T).
4. Clip tiles to AOI and compute `clipped_dtm.tif`.

### Outputs:
per tile:
``` bash
raw.laz                 # raw ALS point cloud tile downloaded
raw.lax                 # spatial index file
clipped.laz             # point cloud clipped to case polygon
clipped_dtm.tif         # DTM raster of clipped tile
```


### Optional flags:
| Flag | Type | Description |
|------|------|-------------|
| `--buffer` | `float` | Buffer distance in meters around AOI (default 20m). |

### Example:
```bash
python -m scripts.get_data --case wippolder --n-cores 4 --buffer 10
```


## 2. Tree Segmentation — `segmentation.py`
Applies vegetation filtering and tree segmentation, producing per-tree point clusters and harmonized IDs.

### Main steps:

1. Vegetation filtering using HOMED algorithm.
2. Tree segmentation via modified TreeSeparation (C++)
3. Forest ID generalization across all tiles.

### Outputs:
per tile:
``` bash
vegetation.laz          # filtered vegetation point cloud in LAS format
vegetation.xyz          # filtered vegetation point cloud in XYZ format used for segmentation
segmentation.xyz        # segmented tree clusters 
forest.laz              # segmented tree clusters with unified gtid attribute
```

per case:
``` bash
forest_hulls.geojson    # 2D projected convex hulls of tree clusters
gtid_map.csv            # case index registry
```

### Example:
``` bash
python -m scripts.tree_segmentation --case wippolder --n-cores 8
```

## 3. Geometry Reconstruction — `reconstruction.py`
Generates watertight 3D tree geometries (crown + trunk) for CFD analysis.

### Main steps:
1. Load `forest.laz` and `clipped_dtm.tif` per tile.
2. Compute morphological metrics per tree.
3. Reconstruct each tree geometry in `LoD3.B`
4. Export tree geometries and attributes to `CityJSON` per tile.

### Outputs:
per tile:
``` bash
trees_lod3.city.json    # final output file, ready for CFD-use
_cache/ (temporary)     # temporary per-tree point cloud in local coordinates, removed after processing

```

### Optional flags:
| Flag           | Type   | Description                                   |
| -------------- | ------ | --------------------------------------------- |
| `--keep-cache` | `flag` | Keep intermediate per-tree cached files.      |
| `--max-trees`  | `int`  | Limit number of trees per tile (for testing). |

### Example:
```bash
python -m scripts.tree_reconstruction --case wippolder --n-cores 16 --keep-cache
```

