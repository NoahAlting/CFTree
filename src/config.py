"""
src/config.py
Central configuration for the CFTree pipeline.
"""

from pathlib import Path
import logging
from datetime import datetime, timezone

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

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
def setup_logger(case: str, logfile_name: str, level: str = "INFO") -> Path:
    """
    Set up a logger that writes to cases/<case>/logs/<logfile_name>.log.

    - Creates directories automatically.
    - Logs to both console and file.
    - Uses UTC ISO-8601 timestamps.
    - Adds a NEW SESSION banner at start.
    """
    case_root = CASE_CONFIGURATIONS["case_root"]
    case_path = case_root / case
    log_dir = case_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{logfile_name}.log"

    # Reset any prior logging config
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)sZ [%(levelname)s] [%(processName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

    # UTC timestamps
    logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()

    banner = f"\n=== NEW SESSION {datetime.now(timezone.utc).isoformat()}Z ==="
    logging.info(banner)
    logging.info(f"Logging to: {log_path}")
    return log_path


# ---------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------
def get_config() -> dict:
    """
    Return resolved configuration with canonical paths and compute settings.
    """
    cfg = CASE_CONFIGURATIONS.copy()

    case_name = cfg["case"]

    resolved = {
        "case_root": Path(cfg["case_root"]).expanduser().resolve(),
        "data_root": Path(cfg["data_root"]).expanduser().resolve(),
        "resources_dir": Path(cfg["resources_dir"]).expanduser().resolve(),
        "case": case_name,
        "default_cores": int(cfg.get("default_cores", 1)),
        "crs": cfg["crs"],
    }

    # Derived paths
    resolved["case_path"] = resolved["case_root"] / case_name
    resolved["data_case_path"] = resolved["data_root"] / case_name
    resolved["data_case_path"].mkdir(parents=True, exist_ok=True)

    return resolved


# ---------------------------------------------------------------------
# CLI inspection
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = get_config()
    print("Active CFTree configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
