"""Step 2 - Audit: data quality, labels, time structure, separability probes,
cross-device alignment and suitability checks for the missing-data study."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .config import LABEL_COL, TYPE_COL, AuditConfig
from .loaders import LoadedTable

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------
# 1. Raw column quality
# ----------------------------------------------------------------------------
def column_quality(t: LoadedTable) -> pd.DataFrame:
    rows = []
    for c in t.raw.columns:
        r = t.raw[c]
        s = r.str.strip()
        clean = t.df[c] if c in t.df else None
        rec = {
            "column": c,
            "role": ("sensor_numeric" if c in t.numeric_cols else
                     "sensor_categorical" if c in t.categorical_cols else "meta"),
            "n_empty": int((s == "").sum()),
            "pct_empty": round(100 * (s == "").mean(), 3),
            "n_padded_whitespace": int((r != s).sum()),
            "n_unique_raw": int(r.nunique()),
            "n_unique_clean": int(clean.nunique()) if clean is not None else int(s.nunique()),
            "top_values": "; ".join(f"{k!r}:{v}" for k, v in s.value_counts().head(5).items()),
        }
        if c in t.numeric_cols:
            x = t.df[c]
            ne = s[s != ""]
            rec.update(
                n_unparseable=int(pd.to_numeric(ne, errors="coerce").isna().sum()),
                min=x.min(), max=x.max(), mean=round(x.mean(), 4), std=round(x.std(), 4),
                pct_zero=round(100 * (x == 0).mean(), 3),
            )
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. Labels
# ----------------------------------------------------------------------------
def label_audit(t: LoadedTable) -> dict:
    df = t.df
    out: dict = {}
    if TYPE_COL in df:
        vc = df[TYPE_COL].value_counts(dropna=False)
        out["type_counts"] = {str(k): int(v) for k, v in vc.items()}
        out["n_types"] = int(df[TYPE_COL].nunique())
    if LABEL_COL in df:
        out["label_counts"] = {str(k): int(v) for k, v in df[LABEL_COL].value_counts(dropna=False).items()}
        out["attack_fraction"] = round(float((df[LABEL_COL] == 1).mean()), 4)
    if TYPE_COL in df and LABEL_COL in df:
        is_norm = df[TYPE_COL].eq("normal")
        bad = ((df[LABEL_COL] == 0) & ~is_norm) | ((df[LABEL_COL] == 1) & is_norm)
        out["label_type_inconsistent_rows"] = int(bad.sum())
    return out


# ----------------------------------------------------------------------------
# 3. Time structure
# ----------------------------------------------------------------------------
def time_audit(t: LoadedTable, cfg: AuditConfig) -> dict:
    ts = t.df["timestamp"]
    out = {"time_method": t.time_method,
           "pct_unparsed_time": round(100 * ts.isna().mean(), 4)}
    v = ts.dropna()
    if v.empty:
        return out
    order_ok = (v.diff().dropna() >= pd.Timedelta(0)).mean()
    srt = v.sort_values()
    uniq = srt.drop_duplicates()
    d = uniq.diff().dropna().dt.total_seconds()
    med = float(d.median()) if len(d) else np.nan
    thr = max(cfg.gap_factor * med, cfg.gap_min_seconds) if len(d) else np.nan
    gaps = d[d > thr] if len(d) else d
    per_ts = srt.value_counts()
    out.update(
        start=str(srt.iloc[0]), end=str(srt.iloc[-1]),
        span_days=round((srt.iloc[-1] - srt.iloc[0]).total_seconds() / 86400, 3),
        n_distinct_days=int(srt.dt.normalize().nunique()),
        pct_rows_in_file_order=round(100 * float(order_ok), 2),
        n_unique_timestamps=int(len(uniq)),
        rows_sharing_timestamp=int((per_ts[per_ts > 1]).sum()),
        max_rows_per_timestamp=int(per_ts.max()),
        median_interval_s=med,
        p95_interval_s=float(d.quantile(0.95)) if len(d) else np.nan,
        max_interval_s=float(d.max()) if len(d) else np.nan,
        gap_threshold_s=thr,
        n_natural_gaps=int(len(gaps)),
        total_gap_hours=round(float(gaps.sum()) / 3600, 3),
        pct_span_in_gaps=round(100 * float(gaps.sum()) /
                               max((srt.iloc[-1] - srt.iloc[0]).total_seconds(), 1), 2),
    )
    return out


def attack_segments(t: LoadedTable) -> pd.DataFrame:
    """Contiguous runs of the same `type` in time order."""
    df = t.df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    if TYPE_COL not in df or df.empty:
        return pd.DataFrame()
    seg = (df[TYPE_COL] != df[TYPE_COL].shift()).cumsum()
    g = df.groupby(seg).agg(type=(TYPE_COL, "first"), start=("timestamp", "min"),
                            end=("timestamp", "max"), n_rows=("timestamp", "size"))
    g["duration_min"] = (g["end"] - g["start"]).dt.total_seconds() / 60
    return g.reset_index(drop=True)


def temporal_mixing(t: LoadedTable) -> pd.DataFrame:
    """For each attack type: how much normal traffic falls inside its time window.
    Low values = attacks live in their own time windows -> time leakage risk."""
    df = t.df.dropna(subset=["timestamp"])
    if TYPE_COL not in df:
        return pd.DataFrame()
    normal = df.loc[df[TYPE_COL] == "normal", "timestamp"]
    rows = []
    for typ, g in df[df[TYPE_COL] != "normal"].groupby(TYPE_COL):
        lo, hi = g["timestamp"].min(), g["timestamp"].max()
        inside = normal.between(lo, hi)
        rows.append({"type": typ, "n": len(g), "window_start": lo, "window_end": hi,
                     "window_hours": round((hi - lo).total_seconds() / 3600, 2),
                     "normal_rows_inside_window": int(inside.sum()),
                     "pct_of_normal_inside": round(100 * inside.mean(), 2) if len(normal) else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. Duplicates
# ----------------------------------------------------------------------------
def duplicate_audit(t: LoadedTable) -> dict:
    df = t.df
    key = ["timestamp"] + t.sensor_cols
    out = {"exact_duplicate_rows": int(df.duplicated().sum()),
           "duplicate_feature_rows": int(df.duplicated(subset=key).sum())}
    if TYPE_COL in df:
        k = pd.util.hash_pandas_object(df[key], index=False)
        nun = df.groupby(k)[TYPE_COL].transform("nunique")
        out["rows_with_conflicting_type"] = int((nun > 1).sum())
        # same sensor values (ignoring time) with different labels -> ambiguity
        if t.sensor_cols:
            kv = pd.util.hash_pandas_object(df[t.sensor_cols], index=False)
            nun_v = df.groupby(kv)[TYPE_COL].transform("nunique")
            out["pct_rows_value_ambiguous"] = round(100 * float((nun_v > 1).mean()), 2)
            out["n_distinct_value_vectors"] = int(kv.nunique())
    return out


# ----------------------------------------------------------------------------
# 5. Separability probes (fast decision trees)
# ----------------------------------------------------------------------------
def _sample(df: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    if len(df) <= cfg.max_rows_probe:
        return df
    frac = cfg.max_rows_probe / len(df)
    parts = []
    for _, g in df.groupby(TYPE_COL):
        k = min(len(g), max(1000, int(round(len(g) * frac))))
        parts.append(g.sample(k, random_state=cfg.random_state))
    return pd.concat(parts)


def _value_features(df: pd.DataFrame, t: LoadedTable, drop: str | None = None) -> pd.DataFrame:
    feats = []
    for c in t.numeric_cols:
        if c == drop:
            continue
        x = df[c]
        feats.append(x.fillna(-999).rename(c))
        if x.isna().any():
            feats.append(x.isna().astype(int).rename(f"{c}__isna"))
    cats = [c for c in t.categorical_cols if c != drop]
    if cats:
        feats.append(pd.get_dummies(df[cats], dummy_na=True, dtype=int))
    return pd.concat(feats, axis=1) if feats else pd.DataFrame(index=df.index)


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    t0 = ts.min()
    return pd.DataFrame({"t_seconds": (ts - t0).dt.total_seconds(),
                         "hour": ts.dt.hour, "dow": ts.dt.dayofweek}, index=df.index)


def _cv_f1(X: pd.DataFrame, y: pd.Series, cfg: AuditConfig, average: str) -> float:
    if X.shape[1] == 0 or y.nunique() < 2:
        return np.nan
    if y.value_counts().min() < 3:
        keep = y.map(y.value_counts()) >= 3
        X, y = X[keep], y[keep]
    skf = StratifiedKFold(3, shuffle=True, random_state=cfg.random_state)
    scores = []
    for tr, te in skf.split(X, y):
        m = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
                                   random_state=cfg.random_state)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(f1_score(y.iloc[te], m.predict(X.iloc[te]), average=average))
    return round(float(np.mean(scores)), 4)


def _chrono_f1(X: pd.DataFrame, y: pd.Series, order: pd.Series, cfg: AuditConfig) -> dict:
    idx = order.sort_values(kind="stable").index
    X, y = X.loc[idx], y.loc[idx]
    cut = int(0.7 * len(y))
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    out = {"classes_only_in_train": sorted(set(ytr) - set(yte)),
           "classes_only_in_test": sorted(set(yte) - set(ytr))}
    if X.shape[1] == 0 or ytr.nunique() < 2 or yte.empty:
        out["macro_f1"] = np.nan
        return out
    m = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=cfg.random_state)
    m.fit(X.iloc[:cut], ytr)
    out["macro_f1"] = round(float(f1_score(yte, m.predict(X.iloc[cut:]), average="macro")), 4)
    return out


def separability_probes(t: LoadedTable, cfg: AuditConfig) -> tuple[dict, pd.DataFrame]:
    df = t.df.dropna(subset=["timestamp"])
    if TYPE_COL not in df or LABEL_COL not in df or df[TYPE_COL].nunique() < 2:
        return {"skipped": "needs label+type with >=2 classes"}, pd.DataFrame()
    df = _sample(df.dropna(subset=[TYPE_COL, LABEL_COL]), cfg)
    y_bin = df[LABEL_COL].astype(int)
    y_typ = df[TYPE_COL].astype(str)
    Xv, Xt = _value_features(df, t), _time_features(df)
    res = {
        "n_rows_used": int(len(df)),
        "values_only_binary_f1": _cv_f1(Xv, y_bin, cfg, "binary"),
        "values_only_type_macro_f1": _cv_f1(Xv, y_typ, cfg, "macro"),
        "time_only_binary_f1": _cv_f1(Xt, y_bin, cfg, "binary"),
        "time_only_type_macro_f1": _cv_f1(Xt, y_typ, cfg, "macro"),
        "values_plus_time_type_macro_f1": _cv_f1(pd.concat([Xv, Xt], axis=1), y_typ, cfg, "macro"),
        "values_only_type_chrono_split": _chrono_f1(Xv, y_typ, df["timestamp"], cfg),
    }
    # Per-sensor ablation: preview of how much each sensor matters
    abl = []
    base = res["values_only_type_macro_f1"]
    for c in t.sensor_cols:
        f = _cv_f1(_value_features(df, t, drop=c), y_typ, cfg, "macro")
        abl.append({"dropped_sensor": c, "type_macro_f1": f,
                    "delta_vs_all_sensors": round(f - base, 4) if pd.notna(f) and pd.notna(base) else np.nan})
    return res, pd.DataFrame(abl)


# ----------------------------------------------------------------------------
# 6. Cross-device alignment (for building a multi-sensor view)
# ----------------------------------------------------------------------------
def cross_device_alignment(tables: dict[str, LoadedTable]) -> dict:
    devs = {d: t for d, t in tables.items() if d != "combined"}
    if len(devs) < 2:
        return {"skipped": "need >=2 device files"}
    sets = {d: set(t.df["timestamp"].dropna().dt.floor("s")) for d, t in devs.items()}
    names = sorted(sets)
    ov = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            inter = len(sets[a] & sets[b])
            ov.loc[a, b] = round(100 * inter / max(len(sets[a]), 1), 2)
    union = set().union(*sets.values())
    cover = pd.Series(0, index=pd.Index(sorted(union)))
    for s in sets.values():
        cover.loc[list(s)] += 1
    dist = cover.value_counts().sort_index()
    # label agreement on shared seconds
    agree = []
    labs = {}
    for d, t in devs.items():
        if LABEL_COL in t.df:
            x = t.df.dropna(subset=["timestamp"]).assign(sec=lambda z: z.timestamp.dt.floor("s"))
            labs[d] = x.groupby("sec")[LABEL_COL].max()
    for i, a in enumerate(sorted(labs)):
        for b in sorted(labs)[i + 1:]:
            j = labs[a].to_frame("a").join(labs[b].to_frame("b"), how="inner")
            if len(j):
                agree.append({"dev_a": a, "dev_b": b, "shared_seconds": len(j),
                              "pct_label_agree": round(100 * float((j.a == j.b).mean()), 2)})
    return {
        "overlap_pct_matrix": ov,  # row device: % of its seconds also seen by column device
        "n_union_seconds": len(union),
        "devices_reporting_per_second": {int(k): int(v) for k, v in dist.items()},
        "pct_seconds_all_devices_present": round(100 * float((cover == len(sets)).mean()), 3),
        "label_agreement": pd.DataFrame(agree),
    }


# ----------------------------------------------------------------------------
# 7. Processed vs Train_Test comparison
# ----------------------------------------------------------------------------
def _row_hashes(raw: pd.DataFrame, cols: list[str]) -> set:
    norm = raw[cols].apply(lambda c: c.astype(str).str.strip().str.lower())
    return set(pd.util.hash_pandas_object(norm, index=False).tolist())


def compare_versions(proc: LoadedTable, tt: LoadedTable, cfg: AuditConfig) -> dict:
    common = [c for c in proc.raw.columns if c in tt.raw.columns]
    samp = tt.raw.sample(min(5000, len(tt.raw)), random_state=cfg.random_state)
    hp = _row_hashes(proc.raw, common)
    hs = _row_hashes(samp, common)
    la, lb = label_audit(proc), label_audit(tt)
    return {"rows_processed": len(proc.raw), "rows_train_test": len(tt.raw),
            "common_columns": common,
            "pct_train_test_rows_found_in_processed": round(100 * len(hs & hp) / max(len(hs), 1), 2),
            "attack_fraction_processed": la.get("attack_fraction"),
            "attack_fraction_train_test": lb.get("attack_fraction"),
            "types_processed": la.get("type_counts"), "types_train_test": lb.get("type_counts")}


# ----------------------------------------------------------------------------
# 8. Suitability checks
# ----------------------------------------------------------------------------
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def suitability_checks(dev: str, t: LoadedTable, a: dict, cfg: AuditConfig) -> list[dict]:
    chk = []

    def add(cid, crit, status, evidence):
        chk.append({"device": dev, "check": cid, "criterion": crit,
                    "status": status, "evidence": evidence})

    ns = len(t.sensor_cols)
    add("C1", "Has sensor value columns", PASS if ns else FAIL, f"{ns} sensor cols: {t.sensor_cols}")
    add("C2", ">=2 sensor channels (partial missingness possible)",
        PASS if ns >= 2 else WARN,
        f"{ns} channel(s)" + ("" if ns >= 2 else " - only whole-record dropout can be simulated"))
    pu = a["time"].get("pct_unparsed_time", 100)
    add("C3", "Timestamps parse (>99%)", PASS if pu < 1 else (WARN if pu < 10 else FAIL),
        f"{pu}% unparsed via {a['time'].get('time_method')}")
    q = a["quality"]
    sens = q[q.role.str.startswith("sensor")]
    nat = float(sens.pct_empty.max()) if len(sens) else 0.0
    add("C4", "Low natural missingness in sensors (<5%) -> clean baseline",
        PASS if nat < 5 else WARN, f"max {nat:.2f}% empty in a sensor column")
    tc = a["labels"].get("type_counts", {})
    small = {k: v for k, v in tc.items() if k != "normal" and v < cfg.min_samples_per_attack}
    add("C5", f"Every attack type has >= {cfg.min_samples_per_attack} rows",
        PASS if tc and not small else WARN if tc else FAIL,
        f"small classes: {small}" if small else f"{len(tc) - 1 if tc else 0} attack types OK")
    inc = a["labels"].get("label_type_inconsistent_rows", 0)
    add("C6", "label and type consistent", PASS if inc == 0 else WARN, f"{inc} inconsistent rows")
    conf = a["duplicates"].get("rows_with_conflicting_type", 0)
    pc = 100 * conf / max(len(t.df), 1)
    add("C7", "Few conflicting duplicates (<1%)", PASS if pc < 1 else WARN, f"{conf} rows ({pc:.2f}%)")
    p = a["probes"]
    if "skipped" not in p:
        v = p["values_only_type_macro_f1"]
        add("C8", "Sensor values carry attack signal (values-only macro-F1 >= 0.6)",
            PASS if v >= 0.6 else WARN if v >= 0.35 else FAIL,
            f"values-only macro-F1={v}; binary F1={p['values_only_binary_f1']}")
        tm = p["time_only_type_macro_f1"]
        add("C9", "No strong time leakage (time-only macro-F1 < 0.9 and < values-only)",
            PASS if (tm < 0.9 and tm <= v) else WARN,
            f"time-only macro-F1={tm} vs values-only={v} -> use time-aware splits & drop time features"
            if not (tm < 0.9 and tm <= v) else f"time-only macro-F1={tm}")
        ch = p["values_only_type_chrono_split"]
        miss = ch["classes_only_in_train"] + ch["classes_only_in_test"]
        add("C10", "Chronological split keeps all classes on both sides",
            PASS if not miss else WARN,
            f"chrono macro-F1={ch['macro_f1']}; classes absent on one side: {miss}")
        hr = v if v == v else 0
        add("C11", "Headroom: baseline not saturated (values-only macro-F1 < 0.99)",
            PASS if hr < 0.99 else WARN,
            f"{hr} - if ~1.0 degradation from missingness may still be measurable but check easy-label artefacts")
    return chk
