"""Plots and Markdown/JSON report writing."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import TYPE_COL  # noqa: E402


def md_table(df: pd.DataFrame, max_rows: int = 60, floatfmt: str = "{:.4g}") -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    d = df.head(max_rows).copy()

    def f(x):
        if isinstance(x, float):
            return "" if np.isnan(x) else floatfmt.format(x)
        return str(x).replace("|", "\\|").replace("\n", " ")
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(f(v) for v in r.values) + " |")
    if len(df) > max_rows:
        out.append(f"\n_... {len(df) - max_rows} more rows in CSV_")
    return "\n".join(out) + "\n"


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict(orient="records")
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, default=_jsonable, indent=2))


# ------------------------------- plots ------------------------------------
def plot_timeline(dev, t, path: Path, max_points=60_000):
    df = t.df.dropna(subset=["timestamp"])
    if TYPE_COL not in df or df.empty:
        return
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    types = sorted(df[TYPE_COL].dropna().unique(), key=lambda x: (x != "normal", x))
    ymap = {k: i for i, k in enumerate(types)}
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(types) + 1.5))
    ax.scatter(df["timestamp"], df[TYPE_COL].map(ymap), s=1, alpha=0.4,
               c=np.where(df[TYPE_COL] == "normal", "tab:blue", "tab:red"))
    ax.set_yticks(range(len(types)), types)
    ax.set_title(f"{dev}: when does each class occur? (sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_sensor_by_class(dev, t, path: Path, max_points=50_000):
    cols = t.numeric_cols[:4]
    if not cols or TYPE_COL not in t.df:
        return
    df = t.df.sample(min(len(t.df), max_points), random_state=0)
    types = sorted(df[TYPE_COL].dropna().unique(), key=lambda x: (x != "normal", x))
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4), squeeze=False)
    for ax, c in zip(axes[0], cols):
        data = [df.loc[df[TYPE_COL] == k, c].dropna() for k in types]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(types) + 1), types, rotation=60, fontsize=8)
        ax.set_title(c)
    fig.suptitle(f"{dev}: numeric sensor distributions by class")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_class_counts(all_labels: dict, path: Path):
    df = pd.DataFrame({d: v.get("type_counts", {}) for d, v in all_labels.items()}).fillna(0)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    df.T.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("rows")
    ax.set_title("Class composition per device file")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_overlap(ov: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ov.values.astype(float), vmin=0, vmax=100, cmap="viridis")
    ax.set_xticks(range(len(ov)), ov.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(ov)), ov.index)
    for i in range(len(ov)):
        for j in range(len(ov)):
            ax.text(j, i, f"{ov.values[i, j]:.0f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, label="% of row-device seconds also present in column-device")
    ax.set_title("Timestamp overlap between devices (1 s resolution)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
