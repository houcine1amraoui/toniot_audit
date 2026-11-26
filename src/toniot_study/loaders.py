"""Loading TON_IoT telemetry CSVs while keeping raw quirks visible to the audit."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABEL_COL, META_COLS, TYPE_COL

TIME_FORMATS = [
    "%d-%b-%y %H:%M:%S",   # TON_IoT style: 31-Mar-19 12:36:52
    "%d-%b-%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
]
BOOL_TOKENS = {"true", "false", "0", "1"}


@dataclass
class LoadedTable:
    path: Path
    raw: pd.DataFrame                 # all columns as stripped-later strings
    df: pd.DataFrame                  # cleaned: timestamp, sensors, label, type
    sensor_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    time_method: str
    notes: list[str] = field(default_factory=list)


def norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")


def read_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path, dtype=str, keep_default_na=False, na_values=[],
        low_memory=False, encoding_errors="replace",
    )
    raw.columns = [norm_col(c) for c in raw.columns]
    # drop pandas "Unnamed: n" index columns
    raw = raw.loc[:, ~raw.columns.str.startswith("unnamed")]
    return raw


def parse_timestamp(raw: pd.DataFrame) -> tuple[pd.Series, str]:
    n = len(raw)
    best = (pd.Series(pd.NaT, index=raw.index), "none")
    best_rate = 0.0
    cands = []
    if "ts" in raw:
        t = pd.to_numeric(raw["ts"].str.strip(), errors="coerce")
        cands.append((pd.to_datetime(t, unit="s", errors="coerce"), "ts(unix)"))
    if "date" in raw and "time" in raw:
        s = raw["date"].str.strip() + " " + raw["time"].str.strip()
        for fmt in TIME_FORMATS:
            cands.append((pd.to_datetime(s, format=fmt, errors="coerce"), f"date+time[{fmt}]"))
        try:
            cands.append((pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True),
                          "date+time[mixed]"))
        except (TypeError, ValueError):
            pass
    for col in ("timestamp", "datetime"):
        if col in raw:
            try:
                cands.append((pd.to_datetime(raw[col].str.strip(), errors="coerce",
                                             format="mixed"), col))
            except (TypeError, ValueError):
                cands.append((pd.to_datetime(raw[col].str.strip(), errors="coerce"), col))
    for ser, name in cands:
        rate = ser.notna().mean() if n else 0
        if rate > best_rate:
            best, best_rate = (ser, name), rate
        if rate > 0.999:
            break
    return best


def _classify(s: pd.Series) -> str:
    ne = s[s != ""]
    if ne.empty:
        return "empty"
    low = set(ne.str.lower().unique()[:20])
    if ne.nunique() <= 2 and low <= BOOL_TOKENS | {"on", "off", "yes", "no"}:
        return "categorical"
    frac_num = pd.to_numeric(ne, errors="coerce").notna().mean()
    return "numeric" if frac_num >= 0.98 else "categorical"


def load_table(path: Path) -> LoadedTable:
    raw = read_raw(path)
    notes = []
    stripped = raw.apply(lambda c: c.str.strip())
    ts, method = parse_timestamp(raw)
    sensor_cols = [c for c in raw.columns if c not in META_COLS]
    num, cat = [], []
    df = pd.DataFrame({"timestamp": ts})
    for c in sensor_cols:
        kind = _classify(stripped[c])
        if kind == "numeric":
            df[c] = pd.to_numeric(stripped[c].replace("", np.nan), errors="coerce")
            num.append(c)
        else:
            v = stripped[c].str.lower().replace("", np.nan)
            vals = set(v.dropna().unique())
            if vals and vals <= BOOL_TOKENS and vals & {"true", "false"} and vals & {"0", "1"}:
                v = v.replace({"1": "true", "0": "false"})
                notes.append(f"{c}: mixed boolean encodings (true/false and 1/0) unified")
            df[c] = v
            cat.append(c)
    if LABEL_COL in raw:
        df[LABEL_COL] = pd.to_numeric(stripped[LABEL_COL], errors="coerce").astype("Int64")
    else:
        notes.append("no 'label' column")
    if TYPE_COL in raw:
        df[TYPE_COL] = stripped[TYPE_COL].str.lower().replace("", np.nan)
    else:
        notes.append("no 'type' column")
    return LoadedTable(path, raw, df, num + cat, num, cat, method, notes)
