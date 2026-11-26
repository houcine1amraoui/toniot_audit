#!/usr/bin/env python
"""Inventory + suitability audit of the TON_IoT dataset.

Colab:
    python scripts/run_audit.py --data-root "/content/drive/MyDrive/TON_IoT" \
                                --out-dir   "/content/drive/MyDrive/TON_IoT_audit"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toniot_study import AuditConfig, run_audit, run_inventory  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/content/drive/MyDrive/TON_IoT")
    ap.add_argument("--out-dir", default="/content/drive/MyDrive/TON_IoT_audit")
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--no-train-test", action="store_true", help="skip Train_Test_* IoT files")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--max-rows-probe", type=int, default=200_000)
    ap.add_argument("--min-samples-per-attack", type=int, default=500)
    ap.add_argument("--count-rows-max-mb", type=float, default=0,
                    help="skip row counting for CSVs bigger than this (0 = count all)")
    a = ap.parse_args()
    cfg = AuditConfig(data_root=a.data_root, out_dir=a.out_dir,
                      max_rows_probe=a.max_rows_probe,
                      min_samples_per_attack=a.min_samples_per_attack,
                      count_rows_max_mb=a.count_rows_max_mb,
                      make_plots=not a.no_plots, audit_train_test=not a.no_train_test)
    inv = run_inventory(cfg)
    if not a.inventory_only:
        run_audit(cfg, inv)


if __name__ == "__main__":
    main()
