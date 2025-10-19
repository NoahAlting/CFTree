# 🌳 CFTree — Get Point Clouds Pipeline

This repository contains the **`run_get_pointclouds`** pipeline, which automates the process of **downloading and clipping AHN LiDAR tiles** based on a case-specific area of interest (AOI).  
It is the first step in the broader **CFTree digital twin workflow** for generating 3D tree models for CFD simulations.

---

## 🧭 What the Script Does

`scripts/run_get_pointclouds.py` performs the following steps:

1. **Load AOI**  
   Reads the case polygon from  
   `cases/<case>/city_bbox.geojson`.

2. **Buffer the AOI**  
   Expands the polygon (default: 20 meters) to ensure all relevant tiles are included.  
   Saved as `city_bbox_buffered.geojson`.

3. **Find Intersecting Tiles**  
   Uses the reference shapefile  
   `resources/AHN_subunits_GeoTiles.shp`  
   to determine which AHN tiles intersect the buffered AOI.

4. **Parallel Download of Tiles**  
   For each intersecting tile:
   - Downloads the corresponding `.LAZ` and `.LAX` files from the AHN web service.
   - Stores them in  
     `data/<case>/tiles/<tile_id>/raw.laz` and `raw.lax`.

5. **Clip Tiles**  
   Each downloaded tile is clipped to the buffered AOI using PDAL (via a robust bash pipeline).  
   The output is saved as  
   `data/<case>/tiles/<tile_id>/clipped.laz`.

6. **Logging**  
   All operations are logged in  
   `cases/<case>/logs/get_pointclouds.log`  
   with UTC timestamps and a session header.

---

## ⚙️ Configuration

All global settings (paths, default case, and core count) are defined in `src/config.py`:

```python
DEFAULTS = {
    "case_root": Path("cases"),
    "data_root": Path("data"),
    "resources_dir": Path("resources/AHN_subunits_GeoTiles"),
    "case": "wippolder",
    "default_cores": 2,
}

```
## 🚀 How to Run

### Default run (uses config defaults)
``` bash
python -m scripts.run_get_pointclouds
```

Runs for:

- case = wippolder
- default_cores = 2
- buffer = 20 meters
- overwrite = False


### overwrite settings
``` bash
python -m scripts.run_get_pointclouds --case delft --n-cores 4 --buffer 50 --overwrite
```

### Dry-run mode (list intersecting tiles only)
``` bash
python -m scripts.run_get_pointclouds --dry-run
```

### Background (detached) run
``` bash
nohup python -m scripts.run_get_pointclouds > cases/wippolder/logs/run.out 2>&1 &
```


## 📂 Output Structure
After running the pipeline, your folders will look like:
``` bash
cases/wippolder/
├── city_bbox.geojson
├── city_bbox_buffered.geojson
└── logs/
    └── get_pointclouds.log

data/wippolder/tiles/
├── 37EN2_11/
│   ├── raw.laz
│   ├── raw.lax
│   └── clipped.laz
└── 37EN2_12/
    ├── raw.laz
    ├── raw.lax
    └── clipped.laz
```


## 🧩 Notes
- The clipping step uses PDAL through src/get_pointclouds/tiles_clipper_robust.sh.
- The resources directory (resources/AHN_subunits_GeoTiles) is static and read-only.
- Logs and buffered AOIs are not versioned (.gitignore excludes them).
- Future steps (segmentation, classification, reconstruction) will follow this same structure.