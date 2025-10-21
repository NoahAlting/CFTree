"""
src/config.py
Central configuration for the CFTree pipeline.
"""

from pathlib import Path
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------
# Default case configurations
# ---------------------------------------------------------------------
DEFAULT_CONFIG = {
    "case_root": Path("cases"),             # user case input directory
    "data_root": Path("data"),              # data storage root (large files)
    "resources_dir": Path("resources"),
    "case": "wippolder",                    # default case
    "default_cores": 2,                     # global default for parallelization
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
    case_root = DEFAULT_CONFIG["case_root"]
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

    banner = "\n" + "="*40 + f" NEW SESSION {datetime.now(timezone.utc).isoformat()}Z" + "="*40
    logging.info(banner)
    logging.info(f"Logging to: {log_path}")
    return log_path


# ---------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------
def get_config(case_name: str | None = None, n_cores: int | None = None) -> dict:
    """
    Return resolved configuration with canonical paths and compute settings.

    Parameters
    ----------
    case_name : str, optional
        Case name to override default.
    n_cores : int, optional
        Number of cores to override default.

    If not provided, defaults to 'wippolder' and 2 cores.
    """
    cfg = DEFAULT_CONFIG.copy()

    # Override defaults if arguments are provided
    case_name = case_name or cfg["case"]
    n_cores = n_cores or cfg["default_cores"]

    resolved = {
        "case_root": Path(cfg["case_root"]).expanduser().resolve(),
        "data_root": Path(cfg["data_root"]).expanduser().resolve(),
        "resources_dir": Path(cfg["resources_dir"]).expanduser().resolve(),
        "case": case_name,
        "default_cores": int(n_cores),
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
