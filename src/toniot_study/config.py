"""Configuration and constants for the TON_IoT missing-data study."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Keyword (found in file name, lower-case) -> canonical device name.
DEVICE_KEYWORDS = {
    "fridge": "fridge",
    "garage": "garage_door",
    "gps": "gps_tracker",
    "modbus": "modbus",
    "motion": "motion_light",
    "thermostat": "thermostat",
    "weather": "weather",
    "combined": "combined",
}

# Columns that are metadata, not sensor readings.
TIME_COLS = {"date", "time", "ts", "timestamp", "datetime"}
LABEL_COL = "label"
TYPE_COL = "type"
META_COLS = TIME_COLS | {LABEL_COL, TYPE_COL}

ARCHIVE_EXTS = {".zip", ".rar", ".7z", ".gz", ".tgz", ".tar", ".bz2", ".xz"}


@dataclass
class AuditConfig:
    data_root: Path
    out_dir: Path
    # Max rows used for the ML separability probes (stratified sample).
    max_rows_probe: int = 200_000
    # Minimum samples per attack type to call it "usable" for evaluation.
    min_samples_per_attack: int = 500
    # Skip line-counting CSVs larger than this (MB); 0 = never skip.
    count_rows_max_mb: float = 0
    # Gap threshold = max(gap_factor * median interval, gap_min_seconds).
    gap_factor: float = 10.0
    gap_min_seconds: float = 60.0
    random_state: int = 42
    make_plots: bool = True
    # Also audit Train_Test_* IoT files (in addition to Processed_* files).
    audit_train_test: bool = True

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.out_dir = Path(self.out_dir)
