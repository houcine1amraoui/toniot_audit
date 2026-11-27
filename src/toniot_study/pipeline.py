"""Orchestrates inventory + audit and writes all outputs."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import audit as A
from . import report as R
from .config import AuditConfig, LABEL_COL, TYPE_COL
from .inventory import build_inventory, summarize_inventory
from .loaders import load_table


MISSING_IOT_MSG = (
    "NO IoT TELEMETRY FILES FOUND under {root}.\n"
    "The study needs TON_IoT's IoT sensor files (IoT_Fridge, IoT_Garage_Door, IoT_GPS_Tracker,\n"
    "IoT_Modbus, IoT_Motion_Light, IoT_Thermostat, IoT_Weather). Download at least one of:\n"
    "  Train_Test_datasets/Train_Test_IoT_dataset/   (small, balanced, ~7 CSVs)\n"
    "  Processed_datasets/Processed_IoT_dataset/     (full telemetry, ~7-8 CSVs)\n"
    "from https://research.unsw.edu.au/projects/toniot-datasets and put them under the data root."
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_inventory(cfg: AuditConfig) -> pd.DataFrame:
    log(f"Inventory of {cfg.data_root}")
    inv = build_inventory(cfg)
    summ = summarize_inventory(inv)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    inv.to_csv(cfg.out_dir / "inventory_files.csv", index=False)
    summ.to_csv(cfg.out_dir / "inventory_summary.csv")
    log(f"{len(inv)} files, {inv.size_mb.sum():.1f} MB")
    print(summ.to_string())
    arch = inv[inv.category == "archive"]
    if len(arch):
        log(f"WARNING: {len(arch)} archive(s) found - extract them, otherwise their content is not audited:")
        print("  " + "\n  ".join(arch.rel_path))
    return inv


def _audit_one(dev: str, path: Path, cfg: AuditConfig, out: Path):
    log(f"  loading {path.name}")
    t = load_table(path)
    a = {"file": str(path), "n_rows": len(t.df), "sensor_cols": t.sensor_cols,
         "numeric_cols": t.numeric_cols, "categorical_cols": t.categorical_cols,
         "load_notes": t.notes}
    a["quality"] = A.column_quality(t)
    a["labels"] = A.label_audit(t)
    a["time"] = A.time_audit(t, cfg)
    a["segments"] = A.attack_segments(t)
    a["mixing"] = A.temporal_mixing(t)
    a["duplicates"] = A.duplicate_audit(t)
    log(f"  probes on {dev} ...")
    a["probes"], a["ablation"] = A.separability_probes(t, cfg)
    a["checks"] = A.suitability_checks(dev, t, a, cfg)
    d = out / dev
    d.mkdir(parents=True, exist_ok=True)
    a["quality"].to_csv(d / "column_quality.csv", index=False)
    a["segments"].to_csv(d / "attack_segments.csv", index=False)
    a["mixing"].to_csv(d / "temporal_mixing.csv", index=False)
    a["ablation"].to_csv(d / "sensor_ablation.csv", index=False)
    if cfg.make_plots:
        R.plot_timeline(dev, t, d / "timeline.png")
        R.plot_sensor_by_class(dev, t, d / "sensor_by_class.png")
    return t, a


def _pick_files(inv: pd.DataFrame, category: str) -> dict[str, Path]:
    sub = inv[(inv.category == category) & (inv.ext == ".csv") & inv.device.notna()]
    files = {}
    for _, r in sub.iterrows():
        # if several files per device, keep the largest
        if r.device not in files or r.size_mb > files[r.device][1]:
            files[r.device] = (r.rel_path, r.size_mb)
    return {d: Path(p) for d, (p, _) in sorted(files.items())}


def run_audit(cfg: AuditConfig, inv: pd.DataFrame | None = None) -> dict:
    if inv is None:
        inv = run_inventory(cfg)
    results: dict = {"versions": {}}
    groups = [("processed", "iot_processed")]
    if cfg.audit_train_test:
        groups.append(("train_test", "iot_train_test"))
    tables: dict[str, dict] = {}
    if not inv.category.isin(["iot_processed", "iot_train_test"]).any():
        msg = MISSING_IOT_MSG.format(root=cfg.data_root)
        log(msg)
        results["missing_iot"] = msg
    for gname, cat in groups:
        files = _pick_files(inv, cat)
        if not files:
            log(f"No {cat} files found - skipping")
            continue
        log(f"Auditing {gname}: {list(files)}")
        out = cfg.out_dir / gname
        results[gname], tables[gname] = {}, {}
        for dev, rel in files.items():
            try:
                t, a = _audit_one(dev, cfg.data_root / rel, cfg, out)
                results[gname][dev] = a
                tables[gname][dev] = t
            except Exception as e:  # keep going on one bad file
                log(f"  ERROR on {dev}: {e!r}")
                results[gname][dev] = {"error": repr(e)}
        log(f"Cross-device alignment ({gname})")
        results[f"alignment_{gname}"] = A.cross_device_alignment(tables[gname])
    if "processed" in tables and "train_test" in tables:
        for dev in sorted(set(tables["processed"]) & set(tables["train_test"])):
            results["versions"][dev] = A.compare_versions(
                tables["processed"][dev], tables["train_test"][dev], cfg)
    # light summary of network train/test labels (not the study focus)
    net = inv[(inv.category == "network_train_test") & (inv.ext == ".csv")]
    results["network"] = {}
    for _, r in net.iterrows():
        try:
            hdr = [c.strip().lower() for c in str(r.columns).split("|")]
            use = [c for c in r.columns.split("|") if c.strip().lower() in (LABEL_COL, TYPE_COL)]
            if use:
                x = pd.read_csv(cfg.data_root / r.rel_path, usecols=use, dtype=str)
                x.columns = [c.strip().lower() for c in x.columns]
                results["network"][r.rel_path] = {
                    c: x[c].str.strip().value_counts().to_dict() for c in x.columns}
            results["network"].setdefault(r.rel_path, {})["n_cols"] = len(hdr)
        except Exception as e:
            results["network"][r.rel_path] = {"error": repr(e)}
    write_outputs(cfg, inv, results)
    return results


def write_outputs(cfg: AuditConfig, inv: pd.DataFrame, res: dict):
    out = cfg.out_dir
    checks = []
    for g in ("processed", "train_test"):
        for dev, a in res.get(g, {}).items():
            for c in a.get("checks", []):
                checks.append({"version": g, **c})
    chk = pd.DataFrame(checks)
    chk.to_csv(out / "suitability_checks.csv", index=False)

    # overview table per device
    ov = []
    for g in ("processed", "train_test"):
        for dev, a in res.get(g, {}).items():
            if "error" in a:
                ov.append({"version": g, "device": dev, "error": a["error"]})
                continue
            p, tm, lb = a["probes"], a["time"], a["labels"]
            ov.append({
                "version": g, "device": dev, "rows": a["n_rows"],
                "sensors": ", ".join(a["sensor_cols"]),
                "n_types": lb.get("n_types"), "attack_frac": lb.get("attack_fraction"),
                "span_days": tm.get("span_days"), "median_dt_s": tm.get("median_interval_s"),
                "nat_gaps": tm.get("n_natural_gaps"),
                "F1_values": p.get("values_only_type_macro_f1"),
                "F1_time": p.get("time_only_type_macro_f1"),
                "F1_chrono": (p.get("values_only_type_chrono_split") or {}).get("macro_f1"),
                "PASS": int((chk.query("version==@g and device==@dev").status == "PASS").sum()) if len(chk) else 0,
                "WARN": int((chk.query("version==@g and device==@dev").status == "WARN").sum()) if len(chk) else 0,
                "FAIL": int((chk.query("version==@g and device==@dev").status == "FAIL").sum()) if len(chk) else 0,
            })
    overview = pd.DataFrame(ov)
    overview.to_csv(out / "device_overview.csv", index=False)

    if cfg.make_plots:
        for g in ("processed", "train_test"):
            labs = {d: a["labels"] for d, a in res.get(g, {}).items() if "labels" in a}
            if labs:
                R.plot_class_counts(labs, out / f"class_counts_{g}.png")
            al = res.get(f"alignment_{g}", {})
            if isinstance(al.get("overlap_pct_matrix"), pd.DataFrame):
                R.plot_overlap(al["overlap_pct_matrix"], out / f"device_overlap_{g}.png")

    R.write_json({k: v for k, v in res.items()}, out / "audit_full.json")
    (out / "AUDIT_REPORT.md").write_text(render_markdown(cfg, inv, res, overview, chk))
    log(f"Done. Report: {out / 'AUDIT_REPORT.md'}")


def render_markdown(cfg, inv, res, overview, chk) -> str:
    L = ["# TON_IoT – Inventory & Suitability Audit",
         f"Data root: `{cfg.data_root}`  \nGenerated: {time.strftime('%Y-%m-%d %H:%M')}\n",
         "Study: *smart-home / IoT cyber-attack detection under missing sensor data*.\n",
         "## 1. Inventory", R.md_table(summarize_inventory(inv).reset_index()),
         "Telemetry files found:",
         R.md_table(inv[inv.category.str.startswith("iot")][
             ["category", "device", "rel_path", "size_mb", "n_rows", "columns"]]),
         ]
    if res.get("missing_iot"):
        L += ["> **STOP – " + res["missing_iot"].replace("\n", "  \n> ") + "**\n"]
    arch = inv[inv.category == "archive"]
    if len(arch):
        L += ["> **Archives present and not audited:** " + ", ".join(arch.rel_path) + "\n"]
    L += ["## 2. Device overview",
          "F1_values = values-only decision-tree macro-F1 on `type` (3-fold CV); "
          "F1_time = same using only time features (leakage probe); "
          "F1_chrono = values-only, train on first 70% of time, test on last 30%.\n",
          R.md_table(overview),
          "## 3. Suitability checks"]
    if len(chk):
        L.append(R.md_table(chk[chk.status != "PASS"], max_rows=200) if (chk.status != "PASS").any()
                 else "All checks PASS.\n")
        L.append("Full list in `suitability_checks.csv`.\n")
    for g in ("processed", "train_test"):
        if g not in res:
            continue
        L.append(f"## 4. Per-device details ({g})")
        for dev, a in res[g].items():
            L.append(f"### {dev}")
            if "error" in a:
                L.append(f"ERROR: `{a['error']}`\n")
                continue
            L.append(f"File `{a['file']}` – {a['n_rows']:,} rows; sensors: "
                     f"numeric {a['numeric_cols']}, categorical {a['categorical_cols']}")
            if a["load_notes"]:
                L.append("Load notes: " + "; ".join(a["load_notes"]))
            L.append("\n**Classes:** " + ", ".join(f"{k}={v:,}" for k, v in a["labels"].get("type_counts", {}).items()))
            L.append(f"\n**Time:** " + ", ".join(f"{k}={v}" for k, v in a["time"].items()))
            L.append(f"\n**Duplicates:** " + ", ".join(f"{k}={v}" for k, v in a["duplicates"].items()))
            L.append("\n**Column quality**\n" + R.md_table(a["quality"]))
            L.append("**Temporal mixing** (how much normal data falls inside each attack window)\n"
                     + R.md_table(a["mixing"]))
            seg = a["segments"]
            if len(seg):
                s = seg.groupby("type").agg(n_segments=("n_rows", "size"), rows=("n_rows", "sum"),
                                            median_seg_rows=("n_rows", "median"),
                                            median_seg_min=("duration_min", "median")).reset_index()
                L.append("**Contiguous segments**\n" + R.md_table(s))
            p = {k: v for k, v in a["probes"].items()}
            L.append("**Separability probes:** " + ", ".join(f"{k}={v}" for k, v in p.items()))
            L.append("\n**Sensor ablation** (drop one sensor -> values-only macro-F1)\n"
                     + R.md_table(a["ablation"]))
            L.append(f"![timeline]({g}/{dev}/timeline.png)\n")
    for g in ("processed", "train_test"):
        al = res.get(f"alignment_{g}")
        if not al or "skipped" in al:
            continue
        L += [f"## 5. Cross-device alignment ({g})",
              f"Union of seconds: {al['n_union_seconds']:,}; seconds where *all* devices report: "
              f"{al['pct_seconds_all_devices_present']}%",
              "\nDevices reporting per second: " + str(al["devices_reporting_per_second"]),
              "\nOverlap matrix (% of row-device seconds seen by column-device):\n",
              R.md_table(al["overlap_pct_matrix"].reset_index().rename(columns={"index": "device"})),
              "Label agreement on shared seconds:\n", R.md_table(al["label_agreement"]),
              f"![overlap](device_overlap_{g}.png)\n"]
    if res.get("versions"):
        L.append("## 6. Processed vs Train_Test versions")
        rows = [{"device": d, **{k: v for k, v in x.items() if not isinstance(v, (dict, list))}}
                for d, x in res["versions"].items()]
        L.append(R.md_table(pd.DataFrame(rows)))
    if res.get("network"):
        L.append("## 7. Network train/test (label summary only)")
        for f, v in res["network"].items():
            L.append(f"- `{f}`: {v}")
    L += ["\n## 8. How to read this for the study",
          "- **C2 WARN (single channel)**: per-device files hold 1–4 channels; partial-feature "
          "missingness is richest on weather/modbus/combined; for 1–2 channel devices simulate "
          "record dropout or build a merged multi-device view (section 5).",
          "- **C9 WARN (time leakage)**: if time alone predicts the attack type, never use "
          "timestamps as features and prefer chronological or attack-window-aware splits; "
          "otherwise missing-data results will be inflated.",
          "- **C11 WARN (saturated baseline)**: near-perfect baselines leave little room to "
          "measure degradation; report per-class F1 and use harder splits.",
          "- Natural gaps (section 2/time) are real missingness: keep them separate from the "
          "missingness you inject, or remove gap periods from the clean baseline."]
    return "\n".join(L) + "\n"
