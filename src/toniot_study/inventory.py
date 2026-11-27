"""Step 1 - Inventory: walk the dataset folder and describe every file."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd

from .config import ARCHIVE_EXTS, DEVICE_KEYWORDS, AuditConfig


def guess_device(name: str) -> str | None:
    n = name.lower()
    for kw, dev in DEVICE_KEYWORDS.items():
        if kw in n:
            return dev
    return None


def categorize(rel: Path) -> str:
    """Assign a TON_IoT component category from the relative path."""
    p = str(rel).lower().replace("\\", "/")
    name = rel.name.lower()
    ext = rel.suffix.lower()
    if ext in ARCHIVE_EXTS:
        return "archive"
    if ext in (".pcap", ".pcapng") or "pcap" in p:
        return "raw_pcap"
    if "groundtruth" in p.replace("_", "") or "securityevents" in p.replace("_", ""):
        return "ground_truth"
    if guess_device(name) is not None and ext in (".csv", ".xlsx", ".log", ".txt"):
        if "train_test" in p or "train-test" in p:
            return "iot_train_test"
        if "raw" in p or "original" in p:
            return "iot_raw"
        return "iot_processed"
    if "network" in p:
        if "train_test" in p:
            return "network_train_test"
        if "bro" in p.split("/")[1:2] or "_bro" in p or "zeek" in p:
            return "network_raw_zeek"
        if "raw" in p or "original" in p:
            return "network_raw"
        return "network_processed"
    if "linux" in p:
        return "linux_processed" if "processed" in p or "train_test" in p else "linux_raw_atop"
    if "windows" in p or name.startswith("win"):
        return "windows"
    if ext in (".pdf", ".docx", ".doc", ".md", ".xlsx", ".xls"):
        return "description" if ("description" in p or "stat" in p) else "documentation"
    if "description" in p or "stat" in p:
        return "description"
    return "other"


def count_csv_rows(path: Path, chunk: int = 1 << 24) -> int:
    """Fast newline count (data rows = lines - header)."""
    n = 0
    last = b""
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            n += b.count(b"\n")
            last = b[-1:]
    if last and last != b"\n":
        n += 1  # last line without trailing newline
    return max(n - 1, 0)


def read_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        try:
            return [c.strip() for c in next(csv.reader(f))]
        except StopIteration:
            return []


def build_inventory(cfg: AuditConfig) -> pd.DataFrame:
    root = cfg.data_root
    if not root.exists():
        raise FileNotFoundError(
            f"Data root not found: {root}\n"
            "In Colab, mount Drive first and check the folder name in MyDrive."
        )
    rows = []
    out_resolved = cfg.out_dir.resolve()
    for dirpath, dirnames, files in os.walk(root):
        # never inventory our own output folder if it sits inside the data root
        dirnames[:] = [d for d in dirnames if (Path(dirpath) / d).resolve() != out_resolved]
        for fn in files:
            full = Path(dirpath) / fn
            rel = full.relative_to(root)
            size = full.stat().st_size
            rec = {
                "rel_path": str(rel),
                "top_folder": rel.parts[0] if len(rel.parts) > 1 else ".",
                "file_name": fn,
                "ext": full.suffix.lower(),
                "size_mb": round(size / 1e6, 3),
                "category": categorize(rel),
                "device": guess_device(fn),
                "n_rows": None,
                "n_cols": None,
                "columns": None,
                "note": "",
            }
            if rec["ext"] == ".csv":
                hdr = read_header(full)
                rec["n_cols"] = len(hdr)
                rec["columns"] = "|".join(hdr)
                too_big = cfg.count_rows_max_mb and size / 1e6 > cfg.count_rows_max_mb
                if too_big:
                    rec["note"] = "row count skipped (size limit)"
                else:
                    try:
                        rec["n_rows"] = count_csv_rows(full)
                    except OSError as e:  # Drive hiccups
                        rec["note"] = f"row count failed: {e}"
            elif rec["category"] == "archive":
                rec["note"] = "compressed archive - extract before auditing"
            rows.append(rec)
    inv = pd.DataFrame(rows)
    if inv.empty:
        raise RuntimeError(f"No files found under {root}")
    return inv.sort_values(["category", "rel_path"]).reset_index(drop=True)


def summarize_inventory(inv: pd.DataFrame) -> pd.DataFrame:
    g = inv.groupby("category").agg(
        n_files=("rel_path", "count"),
        total_size_mb=("size_mb", "sum"),
        total_rows=("n_rows", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
    )
    return g.sort_values("total_size_mb", ascending=False).round(2)
