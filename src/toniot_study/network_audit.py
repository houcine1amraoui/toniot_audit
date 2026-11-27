"""Network-flow audit (TON_IoT Processed_Network_dataset + train_test_network).

One streaming pass over all flows (chunked, memory-safe) collects:
  schema, class counts, time windows, Zeek '-' missingness per column/group/class,
  IP purity (leakage), duplicate/conflict statistics, and a uniform per-class
  sample (bottom-k random keys) used for the ML probes:
  - which feature groups carry the attack signal (id, time, conn, dns, ssl, http, weird)
  - how much each group's loss hurts detection (preview of missing-data sensitivity)
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .config import AuditConfig
from . import report as R

MISSING_TOKENS = ["-", "", "(empty)"]
ID_COLS = ["src_ip", "src_port", "dst_ip", "dst_port"]
CONN_COLS = ["proto", "service", "duration", "src_bytes", "dst_bytes", "conn_state",
             "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]
PROTO_GROUPS = ["dns", "ssl", "http", "weird"]
BEHAV_GROUPS = ["conn"] + PROTO_GROUPS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_cols(cols):
    return [str(c).replace("﻿", "").strip().lower().replace(" ", "_") for c in cols]


def group_of(c: str) -> str:
    if c in ("label", "type"):
        return "target"
    if c == "ts":
        return "time"
    if c in ID_COLS:
        return "id"
    if c in CONN_COLS:
        return "conn"
    for g in PROTO_GROUPS:
        if c.startswith(g + "_"):
            return g
    return "other"


# ----------------------------------------------------------------------------
# Streaming accumulator
# ----------------------------------------------------------------------------
class NetAccumulator:
    def __init__(self, sample_per_class: int, seed: int = 42, hash_cols_tt: list | None = None):
        self.cap = sample_per_class
        self.rng = np.random.default_rng(seed)
        self.cols: list[str] | None = None
        self.schema: dict[str, list[str]] = {}
        self.extra_cols: dict[str, list[str]] = {}
        self.n = 0
        self.file_counts: dict[str, Counter] = {}
        self.type_counts = Counter()
        self.label_type = Counter()
        self.ts_min, self.ts_max = {}, {}
        self.day_type = Counter()
        self.bad_ts = 0
        self.miss_total = Counter()
        self.miss_by_type: dict[str, Counter] = defaultdict(Counter)
        self.group_present_by_type: dict[str, Counter] = defaultdict(Counter)
        self.src_ip_type = Counter()
        self.dst_ip_type = Counter()
        self.h_behav, self.t_codes, self.h_tt = [], [], []
        self.type_index: dict[str, int] = {}
        self.hash_cols_tt = hash_cols_tt
        self.samples: dict[str, pd.DataFrame] = {}

    # -- helpers --
    def _tcode(self, types: pd.Series) -> np.ndarray:
        for t in types.unique():
            self.type_index.setdefault(t, len(self.type_index))
        return types.map(self.type_index).to_numpy(np.int16)

    def _sample(self, chunk: pd.DataFrame):
        u = self.rng.random(len(chunk))
        chunk = chunk.assign(_u=u)
        for t, g in chunk.groupby("type", sort=False):
            cur = self.samples.get(t)
            g = g if cur is None else pd.concat([cur, g], ignore_index=True)
            if len(g) > self.cap:
                g = g.nsmallest(self.cap, "_u")
            self.samples[t] = g

    def add_file(self, path: Path, chunksize: int):
        name = path.name
        self.file_counts[name] = Counter()
        reader = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[],
                             chunksize=chunksize, encoding="utf-8-sig",
                             encoding_errors="replace", low_memory=False)
        for i, ch in enumerate(reader):
            ch.columns = norm_cols(ch.columns)
            ch = ch.loc[:, ~ch.columns.str.startswith("unnamed")]
            if self.cols is None:
                self.cols = list(ch.columns)
            if i == 0:
                self.schema[name] = list(ch.columns)
                extra = [c for c in ch.columns if c not in self.cols]
                if extra:
                    self.extra_cols[name] = extra
            ch = ch.reindex(columns=self.cols, fill_value="-")
            self._process(ch, name)
        log(f"    {name}: {sum(self.file_counts[name].values()):,} rows")

    def _process(self, ch: pd.DataFrame, fname: str):
        ch["type"] = ch["type"].str.strip().str.lower()
        ch["label"] = ch["label"].str.strip()
        self.n += len(ch)
        vc = ch["type"].value_counts()
        self.file_counts[fname].update(vc.to_dict())
        self.type_counts.update(vc.to_dict())
        self.label_type.update(ch.groupby(["label", "type"]).size().to_dict())
        feat = [c for c in self.cols if c not in ("label", "type")]
        miss = ch[feat].isin(MISSING_TOKENS)
        self.miss_total.update(miss.sum().to_dict())
        mb = miss.groupby(ch["type"]).sum()
        for t, row in mb.iterrows():
            self.miss_by_type[t].update(row.to_dict())
        # protocol-group presence: any field not missing and not "0"
        for g in PROTO_GROUPS:
            gc = [c for c in feat if group_of(c) == g]
            if gc:
                pres = ~(ch[gc].isin(MISSING_TOKENS + ["0"])).all(axis=1)
                self.group_present_by_type[g].update(pres.groupby(ch["type"]).sum().to_dict())
        if "ts" in ch:
            ts = pd.to_numeric(ch["ts"], errors="coerce")
            self.bad_ts += int(ts.isna().sum())
            agg = ts.groupby(ch["type"]).agg(["min", "max"])
            for t, r in agg.iterrows():
                self.ts_min[t] = min(self.ts_min.get(t, np.inf), r["min"])
                self.ts_max[t] = max(self.ts_max.get(t, -np.inf), r["max"])
            day = pd.to_datetime(ts, unit="s", errors="coerce").dt.strftime("%Y-%m-%d")
            self.day_type.update(pd.DataFrame({"d": day, "t": ch["type"]}).value_counts().to_dict())
        for col, ctr in (("src_ip", self.src_ip_type), ("dst_ip", self.dst_ip_type)):
            if col in ch:
                ctr.update(ch.groupby([col, "type"]).size().to_dict())
        behav = [c for c in feat if group_of(c) in BEHAV_GROUPS]
        self.h_behav.append(pd.util.hash_pandas_object(ch[behav], index=False).to_numpy())
        self.t_codes.append(self._tcode(ch["type"]))
        if self.hash_cols_tt:
            hc = [c for c in self.hash_cols_tt if c in ch]
            self.h_tt.append(pd.util.hash_pandas_object(ch[hc], index=False).to_numpy())
        self._sample(ch)

    # -- finalisation --
    def sample_df(self) -> pd.DataFrame:
        return pd.concat(self.samples.values(), ignore_index=True).drop(columns="_u")

    def duplicates(self) -> dict:
        h = np.concatenate(self.h_behav)
        t = np.concatenate(self.t_codes)
        df = pd.DataFrame({"h": h, "t": t})
        dup = int(df.duplicated("h").sum())
        nun = df.groupby("h")["t"].transform("nunique")
        conf = nun > 1
        inv = {v: k for k, v in self.type_index.items()}
        conf_by_type = df.loc[conf, "t"].map(inv).value_counts().to_dict()
        # rows whose behavioural vector is shared with normal -> unavoidable confusion
        return {"rows": len(df), "duplicate_behaviour_rows": dup,
                "pct_duplicate_behaviour_rows": round(100 * dup / max(len(df), 1), 2),
                "n_distinct_behaviour_vectors": int(df.h.nunique()),
                "rows_in_conflicting_vectors": int(conf.sum()),
                "pct_rows_in_conflicting_vectors": round(100 * float(conf.mean()), 3),
                "conflicting_rows_by_type": {str(k): int(v) for k, v in conf_by_type.items()}}

    def ip_purity(self, ctr: Counter) -> pd.DataFrame:
        s = pd.Series(ctr)
        if s.empty:
            return pd.DataFrame()
        df = s.rename("n").reset_index()
        df.columns = ["ip", "type", "n"]
        per_ip = df.groupby("ip").agg(n_types=("type", "nunique"), total=("n", "sum"))
        normal_ips = set(df.loc[df.type == "normal", "ip"])
        rows = []
        for t, g in df.groupby("type"):
            pure = g[g.ip.map(per_ip.n_types) == 1].n.sum()
            no_norm = g[~g.ip.isin(normal_ips)].n.sum()
            top = g.nlargest(3, "n")
            rows.append({"type": t, "rows": int(g.n.sum()), "n_ips": int(g.ip.nunique()),
                         "pct_rows_from_single_class_ips": round(100 * pure / g.n.sum(), 2),
                         "pct_rows_from_ips_never_normal": round(100 * no_norm / g.n.sum(), 2),
                         "top_ips": "; ".join(f"{r.ip}:{r.n}" for r in top.itertuples())})
        return pd.DataFrame(rows).sort_values("rows", ascending=False)


# ----------------------------------------------------------------------------
# Probes on the uniform per-class sample
# ----------------------------------------------------------------------------
def encode(sample: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    X, kinds = {}, {}
    for c in cols:
        s = sample[c]
        miss = s.isin(MISSING_TOKENS)
        num = pd.to_numeric(s.where(~miss), errors="coerce")
        nonmiss = (~miss).sum()
        if nonmiss and num.notna().sum() / nonmiss >= 0.98:
            X[c] = num.fillna(-1).astype(float)
            kinds[c] = "numeric"
        else:
            X[c] = pd.Series(pd.factorize(s)[0], index=s.index).astype(float)
            kinds[c] = "categorical"
    return pd.DataFrame(X, index=sample.index), kinds


def cv_scores(X, y, seed, folds=3, depth=20):
    if X.shape[1] == 0:
        return {"macro_f1": np.nan}
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    mf, wf, bf = [], [], []
    yb = (y != "normal").astype(int)
    imp = np.zeros(X.shape[1])
    for tr, te in skf.split(X, y):
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=seed)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        mf.append(f1_score(y.iloc[te], p, average="macro"))
        wf.append(f1_score(y.iloc[te], p, average="weighted"))
        bf.append(f1_score(yb.iloc[te], (p != "normal").astype(int)))
        imp += m.feature_importances_
    return {"macro_f1": round(float(np.mean(mf)), 4), "weighted_f1": round(float(np.mean(wf)), 4),
            "binary_f1": round(float(np.mean(bf)), 4),
            "_importance": pd.Series(imp / folds, index=X.columns)}


def per_class_chrono(X, y, ts, seed, depth=20, frac=0.7):
    """Train on the earliest 70% of every class, test on the latest 30%."""
    order = pd.DataFrame({"y": y, "ts": ts}).sort_values("ts", kind="stable")
    rank = order.groupby("y").cumcount()
    size = order.groupby("y")["y"].transform("size")
    is_tr = (rank < (frac * size)).reindex(X.index)
    m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=seed)
    m.fit(X[is_tr], y[is_tr])
    p = m.predict(X[~is_tr])
    return {"macro_f1": round(float(f1_score(y[~is_tr], p, average="macro")), 4),
            "binary_f1": round(float(f1_score((y[~is_tr] != "normal").astype(int),
                                              (p != "normal").astype(int))), 4)}


def run_probes(sample: pd.DataFrame, cols: list[str], seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y = sample["type"].astype(str)
    groups = {g: [c for c in cols if group_of(c) == g]
              for g in ["id", "time", "conn"] + PROTO_GROUPS}
    X, kinds = encode(sample, [c for g in groups.values() for c in g])
    behav = [c for g in BEHAV_GROUPS for c in groups[g]]
    sets = {"id_only": groups["id"], "time_only": groups["time"], "conn_only": groups["conn"],
            "behaviour (conn+dns+ssl+http+weird)": behav,
            "behaviour + id": behav + groups["id"],
            "all (incl. time)": behav + groups["id"] + groups["time"]}
    res = {}
    imp = None
    for name, cs in sets.items():
        log(f"    probe: {name}")
        r = cv_scores(X[cs], y, seed)
        if name.startswith("behaviour ("):
            imp = r.get("_importance")
        r.pop("_importance", None)
        res[name] = r
    ts = pd.to_numeric(sample.get("ts", pd.Series(np.nan, index=sample.index)), errors="coerce")
    if ts.notna().any():
        res["behaviour, per-class chronological 70/30"] = per_class_chrono(X[behav], y, ts, seed)
    base = res["behaviour (conn+dns+ssl+http+weird)"]["macro_f1"]
    abl = []
    for g in BEHAV_GROUPS:
        if not groups[g]:
            continue
        keep = [c for c in behav if c not in groups[g]]
        r = cv_scores(X[keep], y, seed)
        r.pop("_importance", None)
        abl.append({"group_removed": g, "n_features_removed": len(groups[g]),
                    "macro_f1": r["macro_f1"], "binary_f1": r["binary_f1"],
                    "delta_macro_f1": round(r["macro_f1"] - base, 4)})
    imp_df = (imp.sort_values(ascending=False).rename("importance").reset_index()
              .rename(columns={"index": "feature"}) if imp is not None else pd.DataFrame())
    if len(imp_df):
        imp_df["group"] = imp_df.feature.map(group_of)
        imp_df["kind"] = imp_df.feature.map(kinds)
    return res, pd.DataFrame(abl), imp_df


# ----------------------------------------------------------------------------
# Checks + report
# ----------------------------------------------------------------------------
def net_checks(acc, dup, ipt, probes, abl, cfg) -> pd.DataFrame:
    c = []

    def add(i, crit, st, ev):
        if "=nan" in str(ev):
            st, ev = "N/A", str(ev) + " (not measurable for this file, e.g. no ts column)"
        c.append({"check": i, "criterion": crit, "status": st, "evidence": ev})
    add("N1", "Schema identical across files", "PASS" if not acc.extra_cols and
        len({tuple(v) for v in acc.schema.values()}) == 1 else "WARN",
        f"extra columns: {acc.extra_cols}" if acc.extra_cols else
        f"{len({tuple(v) for v in acc.schema.values()})} distinct schema(s)")
    bad = sum(v for (lab, t), v in acc.label_type.items()
              if (lab == "0") != (t == "normal"))
    add("N2", "label and type consistent", "PASS" if bad == 0 else "WARN", f"{bad} inconsistent rows")
    small = {t: v for t, v in acc.type_counts.items() if v < cfg.min_samples_per_attack}
    add("N3", f"Every class has >= {cfg.min_samples_per_attack} rows",
        "PASS" if not small else "WARN", f"small: {small}" if small else "all OK")
    b = probes.get("behaviour (conn+dns+ssl+http+weird)", {}).get("macro_f1", np.nan)
    add("N4", "Behavioural features carry signal (macro-F1 >= 0.6)",
        "PASS" if b >= 0.6 else "FAIL", f"behaviour-only macro-F1={b}")
    i = probes.get("id_only", {}).get("macro_f1", np.nan)
    add("N5", "Identifier leakage (IP/port-only macro-F1 < 0.8)",
        "PASS" if i < 0.8 else "WARN",
        f"id-only macro-F1={i} -> exclude src/dst IP and ports from models" if i >= 0.8 else f"id-only={i}")
    t = probes.get("time_only", {}).get("macro_f1", np.nan)
    add("N6", "Time leakage (ts-only macro-F1 < 0.8)", "PASS" if t < 0.8 else "WARN",
        f"time-only macro-F1={t} -> exclude ts; use time-aware evaluation" if t >= 0.8 else f"time-only={t}")
    ch = probes.get("behaviour, per-class chronological 70/30", {}).get("macro_f1", np.nan)
    add("N7", "Generalises to later traffic (per-class chrono macro-F1 >= 0.6)",
        "PASS" if ch >= 0.6 else "WARN", f"chrono macro-F1={ch} vs random-CV {b}")
    add("N8", "Few conflicting duplicates (<1% of rows)",
        "PASS" if dup["pct_rows_in_conflicting_vectors"] < 1 else "WARN",
        f"{dup['pct_rows_in_conflicting_vectors']}% rows share a behaviour vector with another class")
    if len(abl):
        worst = abl.delta_macro_f1.min()
        add("N9", "Losing a feature group measurably hurts detection (some delta <= -0.05)",
            "PASS" if worst <= -0.05 else "WARN",
            "; ".join(f"-{r.group_removed}: {r.delta_macro_f1:+.3f}" for r in abl.itertuples()))
    multi_day = pd.Series(acc.day_type).rename("n").reset_index()
    if len(multi_day):
        multi_day.columns = ["day", "type", "n"]
        nd = multi_day.groupby("type").day.nunique()
        one = nd[nd < 2].index.tolist()
        add("N10", "Each class appears on >= 2 days (time-block splits possible)",
            "PASS" if not one else "WARN", f"single-day classes: {one}" if one else f"days per class: {nd.to_dict()}")
    if len(ipt):
        att = ipt[ipt.type != "normal"]
        p = float((att.pct_rows_from_ips_never_normal * att.rows).sum() / max(att.rows.sum(), 1))
        add("N11", "Attack traffic not trivially identified by source IP",
            "PASS" if p < 50 else "WARN", f"{p:.1f}% of attack rows come from IPs never seen in normal traffic")
    return pd.DataFrame(c)


def write_network_outputs(name, acc, out: Path, cfg, inv_rows=None, tt_found=None):
    out.mkdir(parents=True, exist_ok=True)
    log(f"  finalising {name}")
    counts = pd.DataFrame(acc.file_counts).fillna(0).astype(int).T
    counts.to_csv(out / "class_counts_per_file.csv")
    tc = pd.Series(acc.type_counts).sort_values(ascending=False)
    win = pd.DataFrame({"rows": tc,
                        "first": pd.to_datetime(pd.Series(acc.ts_min), unit="s"),
                        "last": pd.to_datetime(pd.Series(acc.ts_max), unit="s")})
    win["span_hours"] = ((win["last"] - win["first"]).dt.total_seconds() / 3600).round(2)
    win.index.name = "type"
    win.to_csv(out / "class_time_windows.csv")
    days = pd.Series(acc.day_type)
    if len(days):
        days = days.rename("n").reset_index()
        days.columns = ["day", "type", "n"]
        days = days.pivot_table(index="day", columns="type", values="n", fill_value=0).astype(int)
        days.to_csv(out / "class_by_day.csv")
    feat = [c for c in acc.cols if c not in ("label", "type")]
    miss = pd.DataFrame({"group": [group_of(c) for c in feat],
                         "pct_missing_all": [round(100 * acc.miss_total[c] / acc.n, 2) for c in feat]},
                        index=feat)
    for t in tc.index:
        miss[f"pct_missing_{t}"] = [round(100 * acc.miss_by_type[t][c] / acc.type_counts[t], 1) for c in feat]
    miss.index.name = "column"
    miss.to_csv(out / "column_missingness.csv")
    gp = pd.DataFrame({g: {t: round(100 * acc.group_present_by_type[g][t] / acc.type_counts[t], 2)
                           for t in tc.index} for g in PROTO_GROUPS})
    gp.index.name = "type"
    gp.to_csv(out / "group_presence_pct_by_class.csv")
    dup = acc.duplicates()
    ipt = acc.ip_purity(acc.src_ip_type)
    ipt.to_csv(out / "src_ip_purity.csv", index=False)
    ipd = acc.ip_purity(acc.dst_ip_type)
    ipd.to_csv(out / "dst_ip_purity.csv", index=False)
    sample = acc.sample_df()
    log(f"  probes on uniform sample ({len(sample):,} rows, <= {acc.cap:,} per class)")
    probes, abl, imp = run_probes(sample, feat, cfg.random_state)
    abl.to_csv(out / "group_ablation.csv", index=False)
    imp.to_csv(out / "feature_importance_behaviour.csv", index=False)
    chk = net_checks(acc, dup, ipt, probes, abl, cfg)
    chk.to_csv(out / "network_checks.csv", index=False)
    summary = {"rows": acc.n, "files": list(acc.file_counts), "extra_cols": acc.extra_cols,
               "bad_ts": acc.bad_ts, "type_counts": dict(tc), "duplicates": dup,
               "probes": probes, "train_test_rows_found_in_processed_pct": tt_found}
    R.write_json(summary, out / "network_summary.json")
    if cfg.make_plots:
        _plots(out, days if isinstance(days, pd.DataFrame) else None, gp, miss)
    md = _markdown(name, acc, counts, win, days, miss, gp, dup, ipt, probes, abl, imp, chk, tt_found)
    (out / "NETWORK_REPORT.md").write_text(md)
    log(f"  report: {out / 'NETWORK_REPORT.md'}")
    return summary


def _plots(out, days, gp, miss):
    import matplotlib.pyplot as plt
    if days is not None and len(days):
        d = np.log10(days.T + 1)
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * d.shape[1] + 3), 0.45 * d.shape[0] + 2))
        im = ax.imshow(d.values, aspect="auto", cmap="magma")
        ax.set_yticks(range(d.shape[0]), d.index)
        ax.set_xticks(range(d.shape[1]), d.columns, rotation=60, ha="right", fontsize=8)
        fig.colorbar(im, label="log10(rows + 1)")
        ax.set_title("Flows per class per day")
        fig.tight_layout(); fig.savefig(out / "class_by_day.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(gp) + 2))
    im = ax.imshow(gp.values, aspect="auto", vmin=0, vmax=100, cmap="viridis")
    ax.set_yticks(range(len(gp)), gp.index); ax.set_xticks(range(gp.shape[1]), gp.columns)
    for i in range(gp.shape[0]):
        for j in range(gp.shape[1]):
            ax.text(j, i, f"{gp.values[i, j]:.0f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, label="% of flows with the protocol group present")
    ax.set_title("Structural presence of protocol fields by class")
    fig.tight_layout(); fig.savefig(out / "group_presence.png", dpi=110); plt.close(fig)


def _markdown(name, acc, counts, win, days, miss, gp, dup, ipt, probes, abl, imp, chk, tt_found):
    pr = pd.DataFrame(probes).T.reset_index().rename(columns={"index": "feature set"})
    L = [f"# TON_IoT network flows – audit ({name})",
         f"Rows: {acc.n:,} in {len(acc.file_counts)} file(s); unparseable ts: {acc.bad_ts:,}\n",
         "## Suitability checks", R.md_table(chk),
         "## Class counts and time windows", R.md_table(win.reset_index()),
         "![per day](class_by_day.png)\n" if isinstance(days, pd.DataFrame) else "",
         "## Where does the signal come from? (decision tree, 3-fold CV on uniform per-class sample)",
         R.md_table(pr),
         "## Missing-data preview: remove one feature group",
         R.md_table(abl),
         "## Top behavioural features", R.md_table(imp.head(20)),
         "## Structural missingness (Zeek '-')",
         "Protocol groups exist only for matching flows (DNS fields only for DNS, etc.). "
         "This *structural* absence is not sensor failure and must be kept distinct from injected missingness.\n",
         R.md_table(gp.reset_index()), "![presence](group_presence.png)\n",
         "Per-column missing % (overall):\n",
         R.md_table(miss[["group", "pct_missing_all"]].reset_index(), max_rows=60),
         "## Duplicates", ", ".join(f"{k}={v}" for k, v in dup.items()) + "\n",
         "## Source-IP purity (leakage)", R.md_table(ipt),
         "## Rows per file", R.md_table(counts.reset_index().rename(columns={"index": "file"}))]
    if acc.extra_cols:
        L.append(f"Extra columns dropped (not in first file's schema): {acc.extra_cols}\n")
    if tt_found is not None:
        L.append(f"## train_test_network vs processed\n{tt_found}% of train_test rows found in processed "
                 "(hash of common columns, excluding ts/http_referrer).\n")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def run_network_audit(cfg: AuditConfig, inv: pd.DataFrame, sample_per_class: int = 20_000,
                      chunksize: int = 500_000, max_files: int = 0) -> dict:
    out = cfg.out_dir / "network"
    proc = inv[(inv.category == "network_processed") & (inv.ext == ".csv")].copy()
    proc["num"] = proc.file_name.str.extract(r"(\d+)").astype(float)
    proc = proc.sort_values("num")
    if max_files:
        proc = proc.head(max_files)
    tt = inv[(inv.category == "network_train_test") & (inv.ext == ".csv")]
    res = {}
    tt_cols = None
    if len(tt):
        with open(cfg.data_root / tt.iloc[0].rel_path, encoding="utf-8-sig", errors="replace") as f:
            tt_cols = [c for c in norm_cols(f.readline().strip().split(","))
                       if c not in ("ts", "http_referrer")]
    if len(proc):
        log(f"Network audit: {len(proc)} processed file(s)")
        acc = NetAccumulator(sample_per_class, cfg.random_state, hash_cols_tt=tt_cols)
        for rel in proc.rel_path:
            acc.add_file(cfg.data_root / rel, chunksize)
        tt_found = None
        if len(tt):
            acc_tt = NetAccumulator(sample_per_class, cfg.random_state, hash_cols_tt=tt_cols)
            log("Network audit: train_test_network")
            acc_tt.add_file(cfg.data_root / tt.iloc[0].rel_path, chunksize)
            hp = np.concatenate(acc.h_tt)
            ht = np.concatenate(acc_tt.h_tt)
            tt_found = round(100 * float(np.isin(ht, hp).mean()), 2)
            res["train_test"] = write_network_outputs("train_test_network", acc_tt,
                                                      out / "train_test", cfg)
        res["processed"] = write_network_outputs("Processed_Network_dataset", acc,
                                                 out / "processed", cfg, tt_found=tt_found)
    elif len(tt):
        acc_tt = NetAccumulator(sample_per_class, cfg.random_state)
        acc_tt.add_file(cfg.data_root / tt.iloc[0].rel_path, chunksize)
        res["train_test"] = write_network_outputs("train_test_network", acc_tt, out / "train_test", cfg)
    else:
        log("No network CSVs found.")
    return res
