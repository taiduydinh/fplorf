#!/usr/bin/env python3
"""Generate the complete reproducible output package for the FPL study.

The program is the public entry point for the computational workflow.  Starting
from the player--gameweek panel, it performs the following stages:

1. validate and standardize the raw data;
2. construct forecasting-model cost vectors using only pre-evaluation data;
3. solve the two-stage and joint roster-selection models;
4. evaluate static and fixed-squad sequential lineup policies over GW27--38;
5. run feature-regularization, Pareto, and availability sensitivity analyses;
6. regenerate every analytical main-paper and supplementary figure, while
   copying the author-designed workflow PDF without redrawing it; and
7. export tables, player-level rosters, provenance records, hashes, and a
   machine-readable completion marker.

Every internal forecasting, optimization, budget, and fixed-squad sequential
result is recomputed from the raw player--gameweek panel.  No historical results
CSV or saved roster snapshot is read.  The only external numerical input is the
four Santoro benchmark series, which is explicitly identified as externally
supplied comparison data.

Figures are designed at their intended manuscript width, use at least 8 pt
explicit text, and are exported as vector PDFs with ``dpi=300`` and
``bbox_inches='tight'``.  They contain no plot titles; captions belong in
LaTeX.  Legends are placed outside the data region whenever practical.  The
dense method-similarity matrix is designed for a landscape supplementary page.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fpl_paper_pipeline_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr, wilcoxon
except ImportError as exc:  # pragma: no cover
    raise SystemExit("SciPy is required; install requirements_fpl_pipeline.txt") from exc


TARGET_GW = 27
FINAL_GW = 38
BASE_XI_CAP = 83.5
RANDOM_SEED = 20260824

# The Springer manuscript has an effective full-text width of approximately
# 6.69 inches.  Figures are drawn at their intended printed width so that
# Overleaf does not shrink 8--10 pt labels into unreadably small text.
FULL_TEXT_WIDTH_IN = 6.70
LANDSCAPE_TEXT_WIDTH_IN = 9.70
BASE_FONT_SIZE_PT = 10.0
MIN_FIGURE_TEXT_PT = 8.0

FEATURES = [
    "starts",
    "ict_index",
    "expected_goals_conceded",
    "selected",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
]
FEATURE_LABELS = {
    "starts": "Starts",
    "ict_index": "ICT",
    "expected_goals_conceded": "xGC",
    "selected": "Selected",
    "expected_assists": "xA",
    "expected_goal_involvements": "xGI",
    "expected_goals": "xG",
    "value": "Value",
}
POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
POSITION_NAMES = {"GK": "Goalkeepers", "DEF": "Defenders", "MID": "Midfielders", "FWD": "Forwards"}

SANTORO_SCORES = {
    "Santoro (mean-MIP-random alg)": [45, 34, 36, 17, 17, 37, 31, 35, 23, 21, 17, 8],
    "Santoro (greedy alg)": [26, 40, 20, 7, 10, 23, 3, 7, 4, 23, 15, 13],
    "Santoro (mean-MIP-rank)": [52, 53, 36, 26, 22, 42, 40, 38, 28, 52, 25, 6],
    "Santoro (mean-MIP-online)": [78, 35, 44, 35, 82, 31, 66, 41, 46, 65, 47, 53],
}

def parse_args() -> argparse.Namespace:
    """Define command-line options and return the parsed configuration."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Recompute all FPL study results and figures from the raw dataset."
    )
    parser.add_argument("--data", type=Path, default=Path("merged_gw_2324.csv"))
    parser.add_argument("--output", type=Path, default=Path("fpl_paper_outputs_server"))
    parser.add_argument("--font", type=Path, default=here / "lmroman10-regular.otf")
    parser.add_argument(
        "--framework-figure",
        type=Path,
        default=here / "fpl_paper_outputs_final" / "main_figures" / "fig01_framework.pdf",
        help=(
            "Author-designed workflow PDF to copy as main figure 1. "
            "The pipeline validates and preserves this file but does not draw it."
        ),
    )
    parser.add_argument(
        "--analysis-core", dest="analysis_core", type=Path,
        default=here / "fpl_regularization_experiments.py",
        help="Module containing regularization and stability analyses.",
    )
    parser.add_argument(
        "--roster-design-runner", dest="roster_design_runner", type=Path,
        default=here / "fpl_roster_design_experiments.py",
        help="Script containing two-stage, joint-roster, Pareto, and availability analyses.",
    )
    parser.add_argument(
        "--full-internal-runner",
        type=Path,
        default=here / "fpl_internal_experiments.py",
        help="Script that recomputes the complete internal method screen.",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--simulation-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--shap-sample", type=int, default=2500)
    parser.add_argument(
        "--skip-computational-experiments", dest="skip_computational_experiments",
        action="store_true",
        help="Reuse previously completed experiment directories; intended for figure-layout diagnostics only.",
    )
    parser.add_argument(
        "--existing-roster-design-dir", dest="existing_roster_design_dir",
        type=Path, default=None,
        help="Completed roster-design output directory used only when computation is skipped.",
    )
    parser.add_argument("--existing-regularization-dir", type=Path, default=None,
                        help="Completed regularization output directory used only when computation is skipped.")
    parser.add_argument("--existing-full-internal-dir", type=Path, default=None,
                        help="Reuse a completed raw-data internal screen only for diagnostics.")
    parser.add_argument("--skip-shap", action="store_true", help="Installation test only; not for publication output.")
    parser.add_argument("--smoke-test", action="store_true", help="Fast dependency/layout test.")
    return parser.parse_args()


def log(message: str) -> None:
    """Write an immediately flushed progress message for server log monitoring."""
    print(message, flush=True)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_pdf(fig, path: Path, *, bottom: float | None = None, top: float | None = None) -> None:
    """Save a complete, tightly cropped PDF atomically and close the figure.

    A temporary file is validated before it replaces the destination, which
    prevents interrupted server runs from leaving a truncated PDF behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if bottom is not None or top is not None:
        fig.subplots_adjust(bottom=bottom if bottom is not None else 0.11, top=top if top is not None else 0.96)
    temporary = path.with_name(f".{path.stem}.writing.pdf")
    try:
        for attempt in range(2):
            if temporary.exists():
                temporary.unlink()
            fig.savefig(
                temporary,
                format="pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.04,
            )
            payload = temporary.read_bytes()
            if payload.startswith(b"%PDF-") and payload.rstrip().endswith(b"%%EOF"):
                temporary.replace(path)
                break
            if attempt == 1:
                raise RuntimeError(f"Matplotlib produced an incomplete PDF: {path}")
    finally:
        if temporary.exists():
            temporary.unlink()
        plt.close(fig)


def configure_style(font_path: Path) -> str:
    """Apply the shared publication style and return the selected font name."""
    font_name = "DejaVu Serif"
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_name, "DejaVu Serif"],
            "font.size": BASE_FONT_SIZE_PT,
            "axes.labelsize": BASE_FONT_SIZE_PT,
            "axes.titlesize": BASE_FONT_SIZE_PT,
            "figure.labelsize": BASE_FONT_SIZE_PT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.borderpad": 0.45,
            "legend.handlelength": 2.0,
            "legend.columnspacing": 1.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return font_name


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the player--gameweek panel used by every analysis."""
    required = {"name", "position", "team", "value", "GW", "total_points", "minutes", *FEATURES}
    data = pd.read_csv(path)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    data = data.copy()
    data["GW"] = pd.to_numeric(data["GW"], errors="raise").astype(int)
    for col in ["value", "total_points", "minutes", *FEATURES]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    if data["value"].median() > 20:
        data["value"] /= 10.0
    return data


def first_series(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return the first column matching ``name`` from a possibly duplicated frame."""
    mask = frame.columns == name
    if not mask.any():
        raise KeyError(f"Missing Results column: {name}")
    selected = frame.loc[:, mask]
    return selected.iloc[:, 0] if isinstance(selected, pd.DataFrame) else selected


def load_recomputed_results(path: Path) -> pd.DataFrame:
    """Load the 94-series internal score matrix and append external benchmarks."""
    results = pd.read_csv(path)
    if "GW" not in results.columns:
        raise ValueError("Fresh internal score matrix is missing the GW column.")
    expected_gameweeks = list(range(TARGET_GW, FINAL_GW + 1))
    if results["GW"].astype(int).tolist() != expected_gameweeks:
        raise ValueError("Fresh internal score matrix must cover GW27--38 exactly.")
    results = results.drop(columns="GW")
    if results.shape[1] != 94:
        raise ValueError(f"Expected 94 freshly recomputed internal series, found {results.shape[1]}.")
    if results.columns.duplicated().any() or results.isna().any().any():
        raise ValueError("Fresh internal score matrix has duplicate labels or missing values.")
    for label, values in SANTORO_SCORES.items():
        results[label] = values
    return results


def display_label(label: str) -> str:
    """Normalize method labels while preserving their scientific meaning."""
    label = str(label)
    label = label.replace("Exponantial", "Exponential")
    label = label.replace("Simulation (Non Parametric)", "Bootstrap simulation")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def short_label(label: str) -> str:
    """Create compact, consistent labels for legends and crowded axes."""
    label = display_label(label)
    label = re.sub(
        r"ARIMA\s*\((\d+),(\d+),(\d+),\s*Budget\s*=\s*(\d+(?:\.\d+)?)\)",
        r"ARIMA (\1,\2,\3) £\4m",
        label,
    )
    replacements = [
        ("Santoro (mean-MIP-random alg)", "Santoro: random"),
        ("Santoro (greedy alg)", "Santoro: greedy"),
        ("Santoro (mean-MIP-rank)", "Santoro: rank"),
        ("Santoro (mean-MIP-online)", "Santoro: online"),
        ("Monte Carlo Simulation", "Monte Carlo"),
        ("Exponential Smoothing", "Exp. smoothing"),
        ("Linear Regression", "Linear trend"),
        ("Sequential", "seq."),
        ("Hybrid ", "H. "),
        ("Robust ", "R. "),
        (" (Higher Total Points)", ""),
        (" (Lower Total Points)", ""),
        ("Avg.", "avg."),
        ("ICT Score", "ICT"),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    label = re.sub(r"\(Budget\s*=\s*(\d+(?:\.\d+)?)\)", r"£\1m", label)
    return label


def frame_for(results: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Return selected score series with normalized display labels."""
    return pd.DataFrame({label: pd.to_numeric(first_series(results, label), errors="coerce").fillna(0).to_numpy() for label in labels})


def available(results: pd.DataFrame, labels: list[str]) -> list[str]:
    """Filter a requested method list to series present in the result matrix."""
    return [label for label in labels if label in set(results.columns)]


def eda_figures(data: pd.DataFrame, figures: Path, shap_sample: int, seed: int, skip_shap: bool) -> None:
    """Generate descriptive feature plots and position-specific SHAP panels."""
    train = data.loc[data["GW"] < TARGET_GW].copy()
    corr_features = [*FEATURES, "value"]

    fig, axes = plt.subplots(2, 2, figsize=(FULL_TEXT_WIDTH_IN, 5.5), sharex=True)
    for ax, position in zip(axes.flat, POSITION_ORDER):
        subset = train.loc[train["position"] == position]
        corrs = subset[corr_features + ["total_points"]].corr(numeric_only=True)["total_points"].drop("total_points").sort_values()
        colors = np.where(corrs >= 0, "#d95f4e", "#4c78a8")
        ax.barh([FEATURE_LABELS.get(x, x) for x in corrs.index], corrs.values, color=colors)
        ax.axvline(0, color="0.25", linewidth=0.7)
        ax.text(0.97, 0.95, POSITION_NAMES[position], transform=ax.transAxes,
                ha="right", va="top", weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5})
        ax.grid(axis="x", alpha=0.22)
    fig.supxlabel("Pearson correlation with gameweek points")
    save_pdf(fig, figures / "fig02_position_correlations.pdf")

    matrix = train[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [FEATURE_LABELS[x] for x in FEATURES]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=MIN_FIGURE_TEXT_PT, color="white" if abs(value) > 0.55 else "black")
    fig.colorbar(image, ax=ax, shrink=0.82, label="Pearson correlation")
    save_pdf(fig, figures / "fig03_feature_correlation_heatmap.pdf")

    distribution_features = ["expected_assists", "expected_goal_involvements", "expected_goals", "expected_goals_conceded", "ict_index", "value"]
    colors = dict(zip(POSITION_ORDER, ["#4c78a8", "#59a14f", "#f28e2b", "#e15759"]))
    fig, axes = plt.subplots(2, 3, figsize=(FULL_TEXT_WIDTH_IN, 5.2))
    rng = np.random.default_rng(seed)
    for ax, feature in zip(axes.flat, distribution_features):
        arrays = [train.loc[train["position"] == p, feature].to_numpy(float) for p in POSITION_ORDER]
        parts = ax.violinplot(arrays, positions=np.arange(1, 5), showextrema=False, widths=0.78)
        for body, p in zip(parts["bodies"], POSITION_ORDER):
            body.set_facecolor(colors[p]); body.set_edgecolor(colors[p]); body.set_alpha(0.23)
        for xpos, arr, p in zip(range(1, 5), arrays, POSITION_ORDER):
            finite = arr[np.isfinite(arr)]
            if len(finite) > 350:
                finite = rng.choice(finite, 350, replace=False)
            jitter = rng.uniform(-0.14, 0.14, len(finite))
            ax.scatter(xpos + jitter, finite, s=3, alpha=0.12, color=colors[p], edgecolor="none")
            if len(arr):
                q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
                ax.plot([xpos, xpos], [q1, q3], color="0.15", linewidth=2.3)
                ax.scatter([xpos], [med], color="white", edgecolor="0.15", s=18, zorder=4)
        ax.set_xticks(range(1, 5), POSITION_ORDER)
        ax.set_ylabel(FEATURE_LABELS[feature])
        ax.grid(axis="y", alpha=0.2)
    save_pdf(fig, figures / "fig04_feature_distributions.pdf")

    shap_paths = {
        "GK": figures / "fig05a_shap_goalkeepers.pdf",
        "DEF": figures / "fig05b_shap_defenders.pdf",
        "MID": figures / "fig05c_shap_midfielders.pdf",
        "FWD": figures / "fig05d_shap_forwards.pdf",
    }
    if skip_shap:
        _linear_shap_fallback(train, shap_paths, shap_sample, seed)
    else:
        _xgboost_shap(train, shap_paths, shap_sample, seed)


def _xgboost_shap(train: pd.DataFrame, paths: dict[str, Path], sample_n: int, seed: int) -> None:
    """Fit the tree-based exploratory model and render SHAP summaries."""
    try:
        import shap
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit("Publication SHAP figures require xgboost and shap. Install requirements_fpl_pipeline.txt or use --skip-shap only for a smoke test.") from exc
    rng = np.random.default_rng(seed)
    for position, path in paths.items():
        subset = train.loc[train["position"] == position, FEATURES + ["total_points"]].dropna()
        if len(subset) > sample_n:
            subset = subset.iloc[rng.choice(len(subset), sample_n, replace=False)]
        X = subset[FEATURES]
        y = subset["total_points"]
        model = XGBRegressor(n_estimators=240, max_depth=3, learning_rate=0.035, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", random_state=seed, n_jobs=1)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        plt.figure(figsize=(6.2, 4.5))
        shap.summary_plot(values, X.rename(columns=FEATURE_LABELS), show=False, max_display=len(FEATURES), plot_size=None)
        fig = plt.gcf()
        for ax in fig.axes:
            ax.set_title("")
        save_pdf(fig, path)


def _linear_shap_fallback(train: pd.DataFrame, paths: dict[str, Path], sample_n: int, seed: int) -> None:
    """Smoke-test-only analytic linear SHAP approximation."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    for position, path in paths.items():
        subset = train.loc[train["position"] == position, FEATURES + ["total_points"]].dropna()
        if len(subset) > sample_n:
            subset = subset.iloc[rng.choice(len(subset), sample_n, replace=False)]
        scaler = StandardScaler().fit(subset[FEATURES])
        Xz = scaler.transform(subset[FEATURES])
        model = Ridge(alpha=1.0).fit(Xz, subset["total_points"])
        contributions = Xz * model.coef_
        order = np.argsort(np.mean(np.abs(contributions), axis=0))
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        for yi, j in enumerate(order):
            vals = contributions[:, j]
            colors = plt.cm.coolwarm((Xz[:, j] - Xz[:, j].min()) / (np.ptp(Xz[:, j]) + 1e-9))
            ax.scatter(vals, np.full(len(vals), yi) + rng.normal(0, 0.08, len(vals)), c=colors, s=5, alpha=0.35, edgecolor="none")
        ax.axvline(0, color="0.4", linewidth=0.7)
        ax.set_yticks(range(len(order)), [FEATURE_LABELS[FEATURES[j]] for j in order])
        ax.set_xlabel("Approximate feature contribution (smoke test only)")
        save_pdf(fig, path)


def panel_label(ax, label: str) -> None:
    """Place a consistent subpanel identifier just above an axis."""
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        weight="bold",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        zorder=10,
    )


def add_bottom_legend(
    fig,
    handles: list,
    labels: list[str],
    *,
    max_columns: int = 3,
) -> float:
    """Add a non-overlapping legend and return the required bottom margin.

    The margin grows with the number of legend rows.  This avoids the previous
    fixed-margin behavior, which could squeeze multi-row legends against axis
    labels when a figure was placed at manuscript width.
    """
    if not handles:
        return 0.12
    columns = max(1, min(max_columns, len(labels)))
    rows = math.ceil(len(labels) / columns)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=columns,
        frameon=True,
    )
    return min(0.14 + 0.055 * rows, 0.42)


def cumulative_plot(results: pd.DataFrame, labels: list[str], path: Path, *, ncol: int = 3) -> None:
    """Plot cumulative GW27--38 scores with the legend below the data panel."""
    labels = list(dict.fromkeys(available(results, labels)))
    if not labels:
        raise ValueError(f"No requested columns available for {path.name}")
    frame = frame_for(results, labels).cumsum()
    gameweeks = np.arange(TARGET_GW, TARGET_GW + len(frame))
    fig, ax = plt.subplots(figsize=(FULL_TEXT_WIDTH_IN, 4.35))
    for label in labels:
        ax.plot(gameweeks, frame[label], marker="o", markersize=3.2, linewidth=1.55, label=short_label(label))
    ax.set_xticks(gameweeks)
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative realized points")
    ax.grid(alpha=0.24)
    handles, legend_labels = ax.get_legend_handles_labels()
    bottom = add_bottom_legend(
        fig,
        handles,
        legend_labels,
        max_columns=ncol,
    )
    save_pdf(fig, path, bottom=bottom)


def method_similarity(results: pd.DataFrame, path: Path, top_n: int = 28) -> None:
    """Cluster leading methods by weekly-score correlation."""
    numeric = results.select_dtypes(include=[np.number]).copy()
    numeric = numeric.loc[:, ~numeric.columns.duplicated()]
    numeric = numeric[numeric.sum().sort_values(ascending=False).head(min(top_n, numeric.shape[1])).index]
    corr = numeric.corr(method="spearman").fillna(0).to_numpy()
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1)
    distance = (1 - corr) / 2
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    order = leaves_list(tree)
    values = np.abs(corr[order][:, order])
    labels = [short_label(numeric.columns[i]) for i in order]
    fig = plt.figure(figsize=(LANDSCAPE_TEXT_WIDTH_IN, 8.1))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.3, 8.7], hspace=0.03)
    ax_tree = fig.add_subplot(grid[0])
    dendrogram(tree, ax=ax_tree, no_labels=True, color_threshold=0)
    ax_tree.axis("off")
    ax = fig.add_subplot(grid[1])
    image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(
        range(len(labels)),
        labels=labels,
        rotation=90,
        fontsize=8.2,
    )
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8.2)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.01, label=r"$|$Spearman $\rho|$")
    fig.subplots_adjust(left=0.27, right=0.94, bottom=0.31, top=0.98)
    save_pdf(fig, path)


def budget_value(label: str, base: str) -> float | None:
    """Extract the XI budget encoded in a result-series label."""
    cleaned = label.replace(" Sequential", "")
    if cleaned == base:
        return BASE_XI_CAP
    match = re.search(r"Budget\s*=\s*(\d+(?:\.\d+)?)", cleaned)
    return float(match.group(1)) if match else None


def compact_budget_label(label: str, score: float | None = None) -> str:
    """Return a compact budget/policy label for crowded sensitivity figures.

    The method family is already stated in each figure caption, so repeating a
    long model name in every legend entry only reduces the effective font size.
    """
    match = re.search(r"Budget\s*=\s*(\d+(?:\.\d+)?)", str(label))
    budget = float(match.group(1)) if match else BASE_XI_CAP
    budget_text = f"£{budget:g}m"
    if "Sequential" in str(label):
        budget_text += " seq."
    if score is not None:
        budget_text += f" ({score:.0f})"
    return budget_text


def collect_budget_columns(results: pd.DataFrame, base: str, *, sequential: bool = False) -> list[str]:
    """Find and sort all available budget variants for one method family."""
    def belongs(text: str) -> bool:
        compact = text.replace(" Sequential", "")
        if base.startswith("ARIMA"):
            order = re.search(r"ARIMA\s*\((\d,\d,\d)", base)
            return bool(order and re.search(rf"ARIMA\s*\({re.escape(order.group(1))}(?:,|\))", compact))
        if base == "Monte Carlo Simulation":
            return compact.startswith("Monte Carlo")
        if base.startswith("Hybrid ICT"):
            return compact.startswith("Hybrid ICT Score")
        if base == "ICT Score":
            return compact.startswith("ICT Score") and not compact.startswith("Hybrid")
        return compact.startswith(base)

    selected = []
    for col in results.columns:
        text = str(col)
        is_seq = "Sequential" in text
        if is_seq != sequential:
            continue
        if belongs(text) and (text.replace(" Sequential", "") == base or "Budget" in text):
            selected.append(text)
    return sorted(set(selected), key=lambda x: -(budget_value(x, base) or -1))


def bump_plot(results: pd.DataFrame, columns: list[str], path: Path) -> None:
    """Plot week-by-week ranks for a set of budget-policy variants."""
    columns = list(dict.fromkeys(available(results, columns)))
    frame = frame_for(results, columns).cumsum()
    ranks = frame.rank(axis=1, method="min", ascending=False)
    gameweeks = np.arange(TARGET_GW, TARGET_GW + len(frame))
    figure_height = 4.45 + 0.055 * max(0, len(columns) - 7)
    fig, ax = plt.subplots(figsize=(FULL_TEXT_WIDTH_IN, figure_height))
    for col in columns:
        label = compact_budget_label(col, frame[col].iloc[-1])
        ax.plot(
            gameweeks,
            ranks[col],
            marker="o",
            markersize=3,
            linewidth=1.55,
            label=label,
        )
    ax.set_ylim(len(columns) + 0.5, 0.5)
    ax.set_yticks(range(1, len(columns) + 1))
    ax.set_xlim(gameweeks[0] - 0.2, gameweeks[-1] + 0.2)
    ax.set_xticks(gameweeks)
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative rank (1 = best)")
    ax.grid(axis="x", alpha=0.22)
    handles, labels = ax.get_legend_handles_labels()
    bottom = add_bottom_legend(fig, handles, labels, max_columns=4)
    save_pdf(fig, path, bottom=bottom)


def winner_lollipop(results: pd.DataFrame, columns: list[str], path: Path) -> None:
    """Summarize cumulative points across budget variants with a lollipop chart."""
    columns = list(dict.fromkeys(available(results, columns)))
    frame = frame_for(results, columns)
    labels = [compact_budget_label(column) for column in columns]
    palette = plt.cm.tab10(np.linspace(0, 1, len(columns)))
    winners = frame.to_numpy().argmax(axis=1)
    counts = np.bincount(winners, minlength=len(columns))
    totals = frame.sum().to_numpy()
    order = np.argsort(totals)
    figure_height = 4.25 + 0.14 * max(0, len(columns) - 5)
    fig = plt.figure(figsize=(FULL_TEXT_WIDTH_IN, figure_height))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.3, 4.7], hspace=0.07)
    strip = fig.add_subplot(grid[0])
    strip.imshow(winners[None, :], aspect="auto", cmap=ListedColormap(palette), vmin=-0.5, vmax=len(columns) - 0.5)
    strip.set_xticks(range(len(frame)), range(TARGET_GW, TARGET_GW + len(frame)))
    strip.xaxis.tick_top(); strip.set_yticks([]); strip.tick_params(axis="x", pad=1)
    strip.set_xlabel("Gameweek", labelpad=4); strip.xaxis.set_label_position("top")
    bar = fig.add_subplot(grid[1])
    ypos = np.arange(len(columns))
    bar.hlines(ypos, 0, totals[order], color="0.65", linewidth=1.2)
    bar.scatter(totals[order], ypos, c=palette[order], s=36, zorder=3)
    for x, y in zip(totals[order], ypos):
        bar.text(x + max(totals) * 0.012, y, f"{x:.0f}", va="center", fontsize=8.5)
    bar.set_yticks(ypos, [labels[i] for i in order])
    bar.set_xlabel("GW27--38 realized points")
    bar.grid(axis="x", alpha=0.22)
    handles = [Line2D([0], [0], marker="s", linestyle="", color=palette[i], label=f"{labels[i]} (wins {counts[i]})") for i in range(len(columns))]
    legend_labels = [handle.get_label() for handle in handles]
    bottom = add_bottom_legend(fig, handles, legend_labels, max_columns=3)
    save_pdf(fig, path, bottom=bottom, top=0.90)


def pareto_figure(roster_design_dir: Path, path: Path) -> None:
    """Render predicted XI-versus-bench Pareto frontiers by forecast method."""
    source = roster_design_dir / "reviewer1_comment2_pareto_frontier.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Pareto CSV not found: {source}")
    data = pd.read_csv(source)
    preferred_methods = ["Weighted average", "ARIMA(1,0,0)"]
    data = data.loc[data["method"].isin(preferred_methods)].copy()
    methods = [method for method in preferred_methods if method in set(data["method"])]
    if not methods:
        raise RuntimeError("No preferred forecasting method is available for the Pareto figure.")
    fig, axes_grid = plt.subplots(
        1,
        len(methods),
        figsize=(FULL_TEXT_WIDTH_IN, 3.9),
        squeeze=False,
    )
    axes = axes_grid.ravel()
    for ax, method in zip(axes, methods):
        group = data.loc[data["method"] == method].sort_values("xi_score", ascending=False)
        ax.plot(group["xi_score"], group["bench_score"], color="#4c78a8", linewidth=1.5)
        scatter = ax.scatter(group["xi_score"], group["bench_score"], c=100 * group["delta"], cmap="viridis", s=45, edgecolor="white", linewidth=0.6)
        for row in group.itertuples(index=False):
            x_offset = -24 if row.delta <= 0.01 + 1e-12 else 4
            ax.annotate(
                f"{100*row.delta:g}%",
                (row.xi_score, row.bench_score),
                xytext=(x_offset, 4),
                textcoords="offset points",
                fontsize=8.5,
                clip_on=False,
            )
        compromise = group.loc[group.get("is_compromise", False).astype(bool)] if "is_compromise" in group else group.iloc[[]]
        if not compromise.empty:
            ax.scatter(compromise["xi_score"], compromise["bench_score"], marker="*", s=170, color="#e15759", edgecolor="black", linewidth=0.7, label="Forecast-space compromise")
        ax.text(0.97, 0.96, method, transform=ax.transAxes,
                ha="right", va="top", weight="bold")
        ax.set_xlabel(r"Predicted XI + captain value, $F_{\mathrm{XI}}$")
        ax.margins(x=0.08, y=0.08)
        first_integer = math.ceil(float(group["xi_score"].min()))
        last_integer = math.floor(float(group["xi_score"].max()))
        if first_integer <= last_integer:
            ax.set_xticks(np.arange(first_integer, last_integer + 1))
        ax.grid(alpha=0.22)
    axes[0].set_ylabel(r"Predicted bench value, $F_{\mathrm{B}}$")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        bottom = add_bottom_legend(fig, handles, labels, max_columns=2)
    else:
        bottom = 0.16
    cbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label(r"Permitted XI loss, $100\delta$ (%)")
    save_pdf(fig, path, bottom=bottom)


def family_uplift(results: pd.DataFrame, path: Path) -> list[str]:
    """Compare the best series in each method family against a common baseline."""
    baseline_candidates = ["ARIMA (1,0,0) Sequential (Budget = 70)"]
    baseline = next((x for x in baseline_candidates if x in results), None)
    if baseline is None:
        raise KeyError("Baseline ARIMA sequential £70 column is missing.")
    candidates = [c for c in results.select_dtypes(include=[np.number]).columns if c != baseline and "Santoro" not in c]
    baseline_values = first_series(results, baseline).to_numpy(float)
    medians = {c: float(np.median(first_series(results, c).to_numpy(float) - baseline_values)) for c in candidates}
    chosen = sorted(candidates, key=medians.get, reverse=True)[:18]
    arrays = [first_series(results, c).to_numpy(float) - baseline_values for c in chosen]
    pvals = []
    for arr in arrays:
        try:
            pvals.append(float(wilcoxon(arr, zero_method="pratt").pvalue))
        except Exception:
            pvals.append(1.0)
    order = np.argsort(pvals)
    qvals = np.ones(len(pvals))
    ranked = np.minimum.accumulate((np.asarray(pvals)[order] * len(pvals) / np.arange(1, len(pvals) + 1))[::-1])[::-1]
    qvals[order] = np.clip(ranked, 0, 1)
    fig, ax = plt.subplots(figsize=(FULL_TEXT_WIDTH_IN, 7.0))
    parts = ax.violinplot(arrays, positions=np.arange(1, len(arrays) + 1), vert=False, showextrema=False)
    rng = np.random.default_rng(RANDOM_SEED)
    colors = plt.cm.tab20(np.linspace(0, 1, len(arrays)))
    for idx, (body, arr, color, q) in enumerate(zip(parts["bodies"], arrays, colors, qvals), start=1):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.25)
        ax.scatter(arr, idx + rng.uniform(-0.12, 0.12, len(arr)), color=color, s=14, alpha=0.45, edgecolor="none")
        ax.scatter(np.median(arr), idx, color=color, edgecolor="white", s=30, zorder=4)
        ax.text(max(max(a) for a in arrays) + 1.0, idx, f"q={q:.3f}", va="center", fontsize=8.0)
    ax.axvline(0, color="0.2", linestyle="--", linewidth=0.9)
    ax.set_yticks(range(1, len(chosen) + 1), [short_label(c) for c in chosen])
    ax.set_xlabel(f"Weekly points relative to {short_label(baseline)}")
    ax.grid(axis="x", alpha=0.22)
    save_pdf(fig, path)
    return chosen


def top_distributions(results: pd.DataFrame, selected: list[str], path: Path, top_k: int = 10) -> None:
    """Show weekly-score distributions for the leading cumulative methods."""
    selected = selected[:top_k]
    arrays = [first_series(results, c).to_numpy(float) for c in selected]
    fig, ax = plt.subplots(figsize=(FULL_TEXT_WIDTH_IN, 5.8))
    parts = ax.violinplot(arrays, positions=np.arange(1, len(arrays) + 1), vert=False, showextrema=False)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(arrays)))
    for idx, (body, arr, color) in enumerate(zip(parts["bodies"], arrays, colors), start=1):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.24)
        q05, med, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
        ax.plot([q05, q95], [idx, idx], color=color, linewidth=1.6)
        ax.scatter(arr, idx + rng.uniform(-0.12, 0.12, len(arr)), color=color, s=14, alpha=0.45, edgecolor="none")
        ax.scatter(med, idx, color=color, edgecolor="white", s=30, zorder=4)
    ax.set_yticks(range(1, len(selected) + 1), [short_label(c) for c in selected])
    ax.set_xlabel("Weekly realized points")
    ax.grid(axis="x", alpha=0.22)
    save_pdf(fig, path)


def external_benchmark(results: pd.DataFrame, path: Path) -> None:
    """Compare internal methods with the externally supplied Santoro series."""
    focal = ["ARIMA (1,0,0) Sequential (Budget = 70)", "Weighted Avg."]
    santoro = list(SANTORO_SCORES)
    labels = available(results, focal + santoro)
    frame = frame_for(results, labels)
    gameweeks = np.arange(TARGET_GW, TARGET_GW + len(frame))
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(FULL_TEXT_WIDTH_IN, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    for label in labels:
        top.plot(gameweeks, frame[label].cumsum(), marker="o", markersize=3, linewidth=1.55, label=short_label(label))
    top.set_ylabel("Cumulative realized points")
    top.grid(alpha=0.22)
    best_external = max(santoro, key=lambda c: frame[c].sum())
    reference = frame[best_external].to_numpy(float)
    for label in available(results, focal):
        bottom.plot(gameweeks, frame[label].to_numpy(float) - reference, marker="o", markersize=3, linewidth=1.4, label=f"{short_label(label)} - {short_label(best_external)}")
    bottom.axhline(0, color="0.25", linestyle="--", linewidth=0.8)
    bottom.set_xlabel("Gameweek")
    bottom.set_ylabel("Weekly difference")
    bottom.set_xticks(gameweeks)
    bottom.grid(alpha=0.22)
    handles1, labels1 = top.get_legend_handles_labels(); handles2, labels2 = bottom.get_legend_handles_labels()
    legend_bottom = add_bottom_legend(
        fig,
        handles1 + handles2,
        labels1 + labels2,
        max_columns=2,
    )
    save_pdf(fig, path, bottom=legend_bottom)


def generate_score_figures(results: pd.DataFrame, figures: Path, roster_design_dir: Path) -> None:
    """Generate the retained family, budget, and benchmark score figures."""
    method_similarity(results, figures / "fig06_method_similarity.pdf")
    cumulative_plot(results, ["Simple Avg.", "Weighted Avg.", "Exponential Smoothing", "Robust Simple Avg.", "Robust Weighted Avg.", "Robust Exponential Smoothing"], figures / "fig07a_averaging_robust.pdf", ncol=3)
    cumulative_plot(results, ["Simple Avg.", "Simulation (Non Parametric)", "Monte Carlo Simulation"], figures / "fig07b_simulation.pdf")
    cumulative_plot(results, ["ARIMA (0,0,1)", "ARIMA (1,0,0)", "ARIMA (0,1,1)", "ARIMA (1,0,1)", "ARIMA (1,1,0)", "ARIMA (1,1,1)", "ARIMA (2,1,1)", "ARIMA (1,1,2)", "ARIMA (2,1,2)"], figures / "fig08_arima_variants.pdf", ncol=3)
    cumulative_plot(results, ["ICT Score", "Robust ICT Score", "Involvement"], figures / "fig09_alternative_objectives.pdf")
    cumulative_plot(results, ["Simple Avg.", "Hybrid Simple Avg. 1:2 (Higher Total Points)", "Hybrid Simple Avg. 2:1 (Lower Total Points)", "Weighted Avg.", "Hybrid Weighted 1:2 (Higher Total Points)", "Hybrid Weighted 2:1 (Lower Total Points)", "Exponential Smoothing", "Hybrid Exponential Smoothing 1:2 (Higher Total Points)", "Hybrid Exponential Smoothing 2:1 (Lower Total Points)"], figures / "fig10a_hybrid_averaging.pdf", ncol=3)
    cumulative_plot(results, ["Simulation (Non Parametric)", "Hybrid Simulation (Non Parametric) 1:2 (Higher Total Points)", "Hybrid Simulation (Non Parametric) 2:1 (Lower Total Points)", "Monte Carlo Simulation", "Hybrid Monte Carlo 1:2 (Higher Total Points)", "Hybrid Monte Carlo 2:1 (Lower Total Points)"], figures / "fig10b_hybrid_simulation.pdf", ncol=3)
    cumulative_plot(results, ["ARIMA (0,0,1)", "Hybrid ARIMA 1:2 (Higher Total Points)", "Hybrid ARIMA 2:1 (Lower Total Points)"], figures / "fig11a_hybrid_arima.pdf")
    cumulative_plot(results, ["ICT Score", "Hybrid ICT Score 1:2 (Higher Total Points)", "Hybrid ICT Score 2:1 (Lower Total Points)", "Linear Regression", "Hybrid Linear Regression 1:2 (Higher Total Points)", "Hybrid Linear Regression 2:1 (Lower Total Points)"], figures / "fig11b_hybrid_ict_linear.pdf", ncol=3)

    bump_plot(results, collect_budget_columns(results, "Simple Avg."), figures / "fig12_simple_budget_ranks.pdf")
    bump_plot(results, collect_budget_columns(results, "Weighted Avg."), figures / "fig13_weighted_budget_ranks.pdf")
    arima100 = collect_budget_columns(results, "ARIMA (1,0,0)") + collect_budget_columns(results, "ARIMA (1,0,0)", sequential=True)
    arima101 = collect_budget_columns(results, "ARIMA (1,0,1)") + collect_budget_columns(results, "ARIMA (1,0,1)", sequential=True)
    bump_plot(results, arima100, figures / "fig14_arima100_sequential_budget_ranks.pdf")
    bump_plot(results, arima101, figures / "fig15_arima101_sequential_budget_ranks.pdf")

    winner_lollipop(results, collect_budget_columns(results, "ARIMA (0,0,1)"), figures / "fig16_arima001_budget_sensitivity.pdf")
    winner_lollipop(results, collect_budget_columns(results, "ARIMA (1,0,0)"), figures / "fig17_arima100_budget_sensitivity.pdf")
    winner_lollipop(results, collect_budget_columns(results, "ARIMA (1,0,1)"), figures / "fig18_arima101_budget_sensitivity.pdf")
    winner_lollipop(results, collect_budget_columns(results, "ICT Score"), figures / "fig19_ict_budget_sensitivity.pdf")
    hybrid_ict = available(results, ["Hybrid ICT Score 1:2 (Higher Total Points)"]) + collect_budget_columns(results, "Hybrid ICT Score")
    winner_lollipop(results, hybrid_ict, figures / "fig20_hybrid_ict_budget_sensitivity.pdf")
    monte_carlo = available(results, ["Monte Carlo Simulation"]) + collect_budget_columns(results, "Monte Carlo Simulation")
    winner_lollipop(results, monte_carlo, figures / "fig21_monte_carlo_budget_sensitivity.pdf")
    winner_lollipop(results, collect_budget_columns(results, "Simple Avg.", sequential=True), figures / "fig22_simple_sequential_budget_sensitivity.pdf")
    winner_lollipop(results, collect_budget_columns(results, "Weighted Avg.", sequential=True), figures / "fig23_weighted_sequential_budget_sensitivity.pdf")
    winner_lollipop(results, collect_budget_columns(results, "ARIMA (0,0,1)", sequential=True), figures / "fig24_arima001_sequential_budget_sensitivity.pdf")
    pareto_figure(roster_design_dir, figures / "fig25_pareto_frontier.pdf")
    cumulative_plot(results, ["ARIMA (1,0,0) Sequential (Budget = 70)", "Monte Carlo Simulation", "Weighted Avg.", "Hybrid Simple Avg. 1:2 (Higher Total Points)"], figures / "fig26_family_best_cumulative.pdf", ncol=2)
    top_methods = family_uplift(results, figures / "fig27_family_uplift.pdf")
    top_distributions(results, top_methods, figures / "fig28_top10_weekly_distributions.pdf")
    external_benchmark(results, figures / "fig29_external_benchmark.pdf")


def regularization_evidence_figure(
    data: pd.DataFrame,
    regularization_dir: Path,
    path: Path,
) -> None:
    """Consolidate redundancy, stability, and LASSO evidence into one figure."""
    correlations = pd.read_csv(
        regularization_dir / "reviewer1_comment3_feature_correlations.csv",
        index_col=0,
    ).loc[FEATURES, FEATURES]
    comparison = pd.read_csv(
        regularization_dir / "reviewer1_comments3_4_main_comparison.csv"
    ).set_index("model")
    lasso = pd.read_csv(
        regularization_dir / "reviewer1_comment4_lasso_feature_selection.csv"
    )
    model_order = [
        model for model in ["Full Ridge", "Reduced Ridge", "Parsimonious Ridge", "Full LASSO"]
        if model in comparison.index
    ]

    fig = plt.figure(figsize=(FULL_TEXT_WIDTH_IN, 6.9))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.08, 0.92],
        hspace=0.66,
        wspace=0.54,
    )
    corr_ax = fig.add_subplot(grid[0, 0])
    stability_ax = fig.add_subplot(grid[0, 1])
    lasso_ax = fig.add_subplot(grid[1, :])
    matrix = correlations.to_numpy(float)
    image = corr_ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [FEATURE_LABELS[feature] for feature in FEATURES]
    corr_ax.set_xticks(range(len(labels)), labels=labels, rotation=40, ha="right")
    corr_ax.set_yticks(range(len(labels)), labels=labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = matrix[row, col]
            corr_ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color="white" if abs(value) > 0.55 else "black",
            )
    panel_label(corr_ax, "(a)")
    fig.colorbar(image, ax=corr_ax, fraction=0.046, pad=0.04, label="Pearson correlation")

    x = np.arange(len(model_order))
    width = 0.36
    cost_stability = comparison.loc[model_order, "cost_spearman_mean"].to_numpy(float)
    xi_stability = comparison.loc[model_order, "xi_jaccard_mean"].to_numpy(float)
    stability_ax.bar(x - width / 2, cost_stability, width, label=r"Cost-vector $\rho$")
    stability_ax.bar(x + width / 2, xi_stability, width, label="XI Jaccard")
    stability_ax.set_xticks(
        x,
        [short_label(model) for model in model_order],
        rotation=24,
        ha="right",
        fontsize=8.5,
    )
    stability_ax.set_ylim(0, 1.05)
    stability_ax.set_ylabel("Bootstrap stability")
    stability_ax.grid(axis="y", alpha=0.22)
    panel_label(stability_ax, "(b)")
    handles, labels_b = stability_ax.get_legend_handles_labels()
    stability_ax.legend(
        handles,
        labels_b,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.43),
        ncol=2,
        frameon=True,
    )

    lasso_pivot = (
        lasso.pivot(index="position", columns="feature_label", values="selection_frequency")
        .reindex(index=POSITION_ORDER, columns=[FEATURE_LABELS[feature] for feature in FEATURES])
    )
    image_lasso = lasso_ax.imshow(lasso_pivot.to_numpy(float), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    lasso_ax.set_xticks(range(lasso_pivot.shape[1]), labels=lasso_pivot.columns, rotation=32, ha="right")
    lasso_ax.set_yticks(range(lasso_pivot.shape[0]), labels=lasso_pivot.index)
    for row in range(lasso_pivot.shape[0]):
        for col in range(lasso_pivot.shape[1]):
            value = lasso_pivot.iloc[row, col]
            lasso_ax.text(
                col,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8.0,
                color="white" if value >= 0.60 else "black",
            )
    panel_label(lasso_ax, "(c)")
    fig.colorbar(image_lasso, ax=lasso_ax, fraction=0.024, pad=0.025, label="LASSO selection frequency")
    save_pdf(fig, path, bottom=0.11, top=0.98)


def corrected_performance_figure(roster_design_dir: Path, regularization_dir: Path, path: Path) -> pd.DataFrame:
    """Plot controlled roster-design and regularization performance comparisons."""
    controlled = pd.read_csv(roster_design_dir / "reviewer1_comment2_weekly_scores.csv")
    controlled = controlled.loc[
        controlled["solution_id"].str.contains(r"joint_pareto \| delta=0\.000", regex=True)
    ].copy()
    controlled["model"] = controlled["method"].replace({"Weighted average": "Weighted average", "ARIMA(1,0,0)": "ARIMA(1,0,0)"})
    controlled = controlled[["model", "policy", "GW", "weekly_points", "cumulative_points"]]

    regularized = pd.read_csv(regularization_dir / "reviewer1_comments3_4_weekly_scores.csv")
    regularized = regularized[["model", "policy", "GW", "weekly_points", "cumulative_points"]]
    combined = pd.concat([controlled, regularized], ignore_index=True)
    combined.to_csv(path.parent.parent / "results" / "primary_model_weekly_scores.csv", index=False)

    order = ["Weighted average", "ARIMA(1,0,0)", "Full Ridge", "Reduced Ridge", "Parsimonious Ridge", "Full LASSO"]
    colors = dict(zip(order, plt.cm.tab10(np.linspace(0, 0.85, len(order)))))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_TEXT_WIDTH_IN, 3.75),
        sharey=True,
    )
    for panel_index, (ax, policy) in enumerate(zip(axes, ["static", "sequential"])):
        for model in order:
            group = combined.loc[(combined["model"] == model) & (combined["policy"] == policy)].sort_values("GW")
            if group.empty:
                continue
            ax.plot(group["GW"], group["cumulative_points"], marker="o", markersize=3.2, linewidth=1.55, color=colors[model], label=model)
        ax.set_xticks(range(TARGET_GW, FINAL_GW + 1))
        ax.set_xlabel("Gameweek")
        ax.grid(alpha=0.22)
        panel_label(ax, f"({chr(97 + panel_index)}) {policy.capitalize()}")
    axes[0].set_ylabel("Cumulative realized points")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    bottom = add_bottom_legend(fig, handles, legend_labels, max_columns=3)
    save_pdf(fig, path, bottom=bottom)
    return combined


def budget_policy_figure(roster_design_dir: Path, path: Path) -> None:
    """Compare XI caps and second-stage bench-budget rules under common forecasts."""
    summary = pd.read_csv(roster_design_dir / "reviewer1_comment2_summary.csv")
    subset = summary.loc[summary["design_family"] == "Two-stage"].copy()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_TEXT_WIDTH_IN, 3.75),
        sharey=True,
    )
    style = {
        ("fixed_split", "static"): ("#4C78A8", "-", "Fixed allocation - static"),
        ("fixed_split", "sequential"): ("#4C78A8", "--", "Fixed allocation - sequential"),
        ("actual_remainder", "static"): ("#F28E2B", "-", "Actual remainder - static"),
        ("actual_remainder", "sequential"): ("#F28E2B", "--", "Actual remainder - sequential"),
    }
    for index, (ax, method) in enumerate(zip(axes, ["Weighted average", "ARIMA(1,0,0)"])):
        part = subset.loc[subset["method"] == method]
        for (rule, policy), (color, linestyle, label) in style.items():
            group = part.loc[(part["bench_budget_rule"] == rule) & (part["policy"] == policy)].sort_values("xi_cap")
            if not group.empty:
                ax.plot(group["xi_cap"], group["realized_points"], marker="o", linewidth=1.6, linestyle=linestyle, color=color, label=label)
        ax.axvline(BASE_XI_CAP, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Starting-XI cap (£m)")
        ax.grid(alpha=0.22)
        panel_label(ax, f"({chr(97 + index)}) {method}")
    axes[0].set_ylabel("Realized GW27--38 points")
    handles, labels = axes[0].get_legend_handles_labels()
    bottom = add_bottom_legend(fig, handles, labels, max_columns=2)
    save_pdf(fig, path, bottom=bottom)


def corrected_external_benchmark(roster_design_dir: Path, path: Path) -> None:
    """Compare the joint ARIMA roster with externally supplied benchmark scores."""
    weekly = pd.read_csv(roster_design_dir / "reviewer1_comment2_weekly_scores.csv")
    primary = weekly.loc[
        (weekly["policy"] == "sequential")
        & weekly["solution_id"].str.contains(r"joint_pareto \| delta=0\.000", regex=True)
    ].copy()
    primary["label"] = primary["method"].map(
        {
            "Weighted average": "Joint weighted (seq.)",
            "ARIMA(1,0,0)": "Joint ARIMA(1,0,0) (seq.)",
        }
    )
    gameweeks = np.arange(TARGET_GW, FINAL_GW + 1)
    external = pd.DataFrame({"GW": gameweeks, **SANTORO_SCORES})
    strongest_external = "Santoro (mean-MIP-online)"

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(FULL_TEXT_WIDTH_IN, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    for label, group in primary.groupby("label", sort=False):
        group = group.sort_values("GW")
        top.plot(group["GW"], group["weekly_points"].cumsum(), marker="o", markersize=3, linewidth=1.7, label=label)
    for column in SANTORO_SCORES:
        top.plot(
            gameweeks,
            external[column].cumsum(),
            marker="o",
            markersize=2.8,
            linewidth=1.35,
            label=short_label(column),
        )
    top.set_ylabel("Cumulative realized points")
    top.grid(alpha=0.22)
    panel_label(top, "(a)")

    external_reference = external[strongest_external].to_numpy(float)
    for label, group in primary.groupby("label", sort=False):
        group = group.sort_values("GW")
        short_internal = "Weighted" if "weighted" in label.lower() else "ARIMA(1,0,0)"
        bottom.plot(
            group["GW"],
            group["weekly_points"].to_numpy(float) - external_reference,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{short_internal} - Santoro online",
        )
    bottom.axhline(0, color="0.25", linestyle="--", linewidth=0.8)
    bottom.set_xlabel("Gameweek")
    bottom.set_ylabel("Weekly difference")
    bottom.set_xticks(gameweeks)
    bottom.grid(alpha=0.22)
    panel_label(bottom, "(b)")
    handles_top, labels_top = top.get_legend_handles_labels()
    handles_bottom, labels_bottom = bottom.get_legend_handles_labels()
    legend_bottom = add_bottom_legend(
        fig,
        handles_top + handles_bottom,
        labels_top + labels_bottom,
        max_columns=2,
    )
    save_pdf(fig, path, bottom=legend_bottom)


def detailed_legacy_method_figure(results: pd.DataFrame, path: Path) -> None:
    """Retain a detailed fixed-split benchmark comparison for the supplement."""
    static_results = results.loc[:, [column for column in results.columns if "Sequential" not in str(column) and "Santoro" not in str(column)]]
    groups = [
        ["Simple Avg.", "Weighted Avg.", "Exponential Smoothing", "Robust Simple Avg.", "Robust Weighted Avg.", "Robust Exponential Smoothing"],
        ["Simple Avg.", "Simulation (Non Parametric)", "Monte Carlo Simulation", "ARIMA (0,0,1)", "ARIMA (1,0,0)", "ARIMA (1,0,1)"],
        ["ICT Score", "Robust ICT Score", "Involvement", "Linear Regression"],
        ["Hybrid Simple Avg. 1:2 (Higher Total Points)", "Hybrid Weighted 1:2 (Higher Total Points)", "Hybrid ARIMA 1:2 (Higher Total Points)", "Hybrid ICT Score 1:2 (Higher Total Points)", "Hybrid Linear Regression 1:2 (Higher Total Points)", "Hybrid Monte Carlo 1:2 (Higher Total Points)"],
    ]
    gameweeks = np.arange(TARGET_GW, TARGET_GW + len(static_results))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(LANDSCAPE_TEXT_WIDTH_IN, 7.5),
    )
    for index, (ax, requested) in enumerate(zip(axes.flat, groups)):
        labels = available(static_results, requested)
        frame = frame_for(static_results, labels).cumsum()
        for label in labels:
            ax.plot(gameweeks, frame[label], marker="o", markersize=2.8, linewidth=1.35, label=short_label(label))
        ax.set_xticks(gameweeks)
        ax.set_xlabel("Gameweek")
        ax.set_ylabel("Cumulative realized points")
        ax.grid(alpha=0.20)
        panel_label(ax, f"({chr(97 + index)})")
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.39),
            ncol=2,
            frameon=True,
            fontsize=8.0,
        )
    fig.subplots_adjust(hspace=0.48, wspace=0.22, bottom=0.10, top=0.98)
    save_pdf(fig, path)


def availability_figure(roster_design_dir: Path, path: Path) -> None:
    """Plot the effect of expected and uncertainty-penalized starting availability."""
    data = pd.read_csv(roster_design_dir / "reviewer1_comment5_availability_comparison.csv")
    methods = [method for method in ["Weighted average", "ARIMA(1,0,0)"] if method in set(data["method"])]
    designs = ["Unadjusted joint $\\delta=0$", "Expected availability", "Availability robust"]
    colors = ["#4C78A8", "#59A14F", "#E15759"]
    x = np.arange(len(methods))
    width = 0.24
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_TEXT_WIDTH_IN, 3.65),
    )
    for offset, (design, color) in enumerate(zip(designs, colors)):
        subset = data.loc[data["availability_design"] == design].set_index("method").reindex(methods)
        axes[0].bar(x + (offset - 1) * width, subset["realized_points"], width, color=color, label=design)
        axes[1].bar(x + (offset - 1) * width, subset["mean_squad_start_probability"], width, color=color, label=design)
    axes[0].set_xticks(x, methods)
    axes[0].set_ylabel("Realized GW27--38 points")
    axes[0].grid(axis="y", alpha=0.22)
    panel_label(axes[0], "(a)")
    axes[1].set_xticks(x, methods)
    axes[1].set_ylabel("Mean pre-GW27 starting probability")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.22)
    panel_label(axes[1], "(b)")
    handles, labels = axes[0].get_legend_handles_labels()
    bottom = add_bottom_legend(fig, handles, labels, max_columns=3)
    save_pdf(fig, path, bottom=bottom)


def export_score_results(
    results: pd.DataFrame,
    result_dir: Path,
    table_dir: Path,
    full_internal_dir: Path,
) -> None:
    """Export weekly scores, totals, source flags, and freshly solved rosters."""
    result_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    publication_screen = results.loc[:, [
        column for column in results.columns
        if "Sequential" not in str(column)
        and "Santoro" not in str(column)
    ]].copy()
    clean = publication_screen.copy()
    clean.insert(0, "GW", range(TARGET_GW, TARGET_GW + len(clean)))
    clean.to_csv(result_dir / "raw_recomputed_static_forecasting_screen.csv", index=False)
    cumulative = clean.copy()
    for col in cumulative.columns[1:]:
        cumulative[col] = pd.to_numeric(cumulative[col], errors="coerce").fillna(0).cumsum()
    cumulative.to_csv(result_dir / "raw_recomputed_static_forecasting_screen_cumulative.csv", index=False)
    totals = pd.DataFrame({"method": publication_screen.columns, "final_total": [pd.to_numeric(first_series(publication_screen, c), errors="coerce").fillna(0).sum() for c in publication_screen.columns]})
    totals["rank"] = totals["final_total"].rank(method="min", ascending=False).astype(int)
    totals["display_label"] = totals["method"].map(display_label)
    totals.sort_values(["rank", "method"]).to_csv(table_dir / "raw_recomputed_static_method_totals_and_ranks.csv", index=False)
    mapping = pd.DataFrame(
        {
            "stored_label": publication_screen.columns,
            "publication_label": [display_label(c) for c in publication_screen.columns],
        }
    )
    mapping.to_csv(result_dir / "terminology_mapping.csv", index=False)
    roster_path = full_internal_dir / "full_internal_rosters.csv"
    if not roster_path.is_file():
        raise FileNotFoundError(f"Freshly recomputed roster output is missing: {roster_path}")
    rosters = pd.read_csv(roster_path)
    roster_series = {
        "Weighted Avg.": "Weighted Average",
        "ARIMA (1,0,0, Budget = 70)": "ARIMA(1,0,0) Sequential £70m",
        "Monte Carlo Simulation": "Monte Carlo",
        "Hybrid Simple Avg. 1:2 (Higher Total Points)": "Hybrid Simple Average 1:2",
        "Linear Regression": "Linear Regression",
        "ICT Score": "ICT Score",
    }
    manuscript_rosters = rosters.loc[rosters["series"].isin(roster_series)].copy()
    manuscript_rosters["method"] = manuscript_rosters["series"].map(roster_series)
    manuscript_rosters = manuscript_rosters[
        ["method", "name", "team", "position", "value", "role", "is_captain", "is_vice_captain"]
    ]
    if len(manuscript_rosters) != 15 * len(roster_series):
        raise AssertionError("Expected six freshly recomputed 15-player manuscript rosters.")
    manuscript_rosters.to_csv(table_dir / "gameweek27_rosters_raw_recomputed.csv", index=False)
    shutil.copy2(roster_path, result_dir / "all_raw_recomputed_rosters.csv")


def import_module(path: Path, name: str):
    """Import a Python support module from an explicit filesystem path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_support_files(args: argparse.Namespace) -> None:
    """Confirm that the readable analysis modules are available beside the runner.

    Keeping the modules as ordinary Python files makes the public repository
    searchable, reviewable, and easy to maintain.  Command-line options can be
    used to point to alternative module locations when needed.
    """
    required = {
        "regularization analysis module": args.analysis_core,
        "roster-design analysis module": args.roster_design_runner,
        "internal experiment module": args.full_internal_runner,
        "author-designed workflow PDF": args.framework_figure,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required support files are missing. Keep the repository files together:\n"
            + "\n".join(missing)
        )


def read_valid_pdf(path: Path, label: str) -> bytes:
    """Read a complete PDF asset and reject missing or truncated input."""
    payload = path.read_bytes()
    if not payload.startswith(b"%PDF-") or not payload.rstrip().endswith(b"%%EOF"):
        raise ValueError(f"{label} is not a complete PDF: {path}")
    return payload


def run_supporting_experiments(args: argparse.Namespace, output: Path) -> tuple[Path, Path]:
    """Run roster-design, availability, and feature-regularization analyses."""
    roster_design_dir = output / "roster_design"
    regularization_dir = output / "feature_regularization"
    command = [sys.executable, "-u", str(args.roster_design_runner), "--data", str(args.data), "--output", str(roster_design_dir), "--workers", str(args.workers)]
    if args.smoke_test:
        # The complete five-method fixed-squad screen is still required because
        # the retained supplement contains all five sequential families.
        command.extend(["--pareto-deltas", "0", "0.05"])
    log("Running roster-design and Pareto experiment ...")
    subprocess.run(command, check=True)
    log("Running feature-regularization and stability experiment ...")
    core = import_module(args.analysis_core, "fpl_analysis_core")
    core.run_regularization_experiment(args.data, regularization_dir, 4 if args.smoke_test else args.bootstrap, args.seed)
    return roster_design_dir, regularization_dir


def run_full_internal_screen(
    args: argparse.Namespace,
    roster_design_dir: Path,
    output: Path,
) -> Path:
    """Recompute the complete internal method screen from the raw data panel."""
    full_internal_dir = output / "full_internal_raw_recomputation"
    command = [
        sys.executable,
        "-u",
        str(args.full_internal_runner),
        "--data",
        str(args.data),
        "--roster-design-dir",
        str(roster_design_dir),
        "--output",
        str(full_internal_dir),
        "--workers",
        str(args.workers),
        "--simulation-draws",
        str(20 if args.smoke_test else args.simulation_draws),
        "--seed",
        str(args.seed),
    ]
    if args.smoke_test:
        command.append("--smoke-test")
    log("Running complete 94-series raw-data internal experiment screen ...")
    subprocess.run(command, check=True)
    marker = full_internal_dir / "RUN_COMPLETED_SUCCESSFULLY.txt"
    if not marker.is_file():
        raise RuntimeError("Full internal raw-data screen did not create its success marker.")
    return full_internal_dir


def copy_controlled_tables(roster_design_dir: Path, regularization_dir: Path, table_dir: Path, result_dir: Path) -> None:
    """Copy analysis outputs into stable public table and result filenames."""
    for src_name, dst_name in [
        ("reviewer1_comment2_main_comparison.csv", "table01_joint_design_comparison.csv"),
        ("reviewer1_comment2_pareto_frontier.csv", "pareto_frontier.csv"),
        ("reviewer1_comment2_weekly_scores.csv", "joint_design_weekly_scores.csv"),
        ("reviewer1_comment2_rosters.csv", "joint_design_rosters.csv"),
        ("reviewer1_comment1_policy_audit.csv", "table_policy_audit.csv"),
        ("reviewer1_comment5_availability_comparison.csv", "table_availability_comparison.csv"),
        ("reviewer1_comment5_availability_rosters.csv", "availability_rosters.csv"),
    ]:
        src = roster_design_dir / src_name
        if src.is_file():
            shutil.copy2(src, (table_dir if dst_name.startswith("table") else result_dir) / dst_name)
    for src_name, dst_name in [
        ("reviewer1_comments3_4_main_comparison.csv", "table_regularization_comparison.csv"),
        ("reviewer1_comment3_cost_vector_stability.csv", "cost_vector_stability.csv"),
        ("reviewer1_comment4_lasso_feature_selection.csv", "lasso_feature_selection.csv"),
        ("reviewer1_comments3_4_weekly_scores.csv", "regularization_weekly_scores.csv"),
        ("reviewer1_comments3_4_rosters.csv", "regularization_rosters.csv"),
        ("reviewer1_comment1_regularization_policy_audit.csv", "table_regularization_policy_audit.csv"),
    ]:
        src = regularization_dir / src_name
        if src.is_file():
            shutil.copy2(src, (table_dir if dst_name.startswith("table") else result_dir) / dst_name)


def copy_full_internal_results(full_internal_dir: Path, result_dir: Path, table_dir: Path) -> None:
    """Validate and publish the raw-data recomputation outputs and provenance."""
    required = [
        "full_internal_score_matrix.csv",
        "full_internal_static_weekly_scores_long.csv",
        "full_internal_cost_vectors.csv",
        "full_internal_series_provenance.csv",
        "independent_controlled_crosscheck.csv",
        "run_config.json",
        "RUN_COMPLETED_SUCCESSFULLY.txt",
    ]
    for name in required:
        source = full_internal_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing fresh internal output: {source}")
        destination_name = (
            "full_internal_run_config.json" if name == "run_config.json" else name
        )
        shutil.copy2(source, result_dir / destination_name)
    provenance = pd.read_csv(full_internal_dir / "full_internal_series_provenance.csv")
    if len(provenance) != 94 or set(provenance["source_type"]) != {"raw-data recomputation"}:
        raise AssertionError("All 94 internal series must carry raw-data provenance.")
    summary = pd.read_csv(full_internal_dir / "full_internal_score_matrix.csv")
    totals = pd.DataFrame(
        {
            "series": summary.columns[1:],
            "GW27_38_total": [summary[column].sum() for column in summary.columns[1:]],
            "source_type": "raw-data recomputation",
        }
    ).sort_values("GW27_38_total", ascending=False)
    totals.to_csv(table_dir / "table_all_internal_series_raw_recomputed_totals.csv", index=False)


def build_primary_comparison_table(roster_design_dir: Path, regularization_dir: Path, table_dir: Path) -> pd.DataFrame:
    """Assemble the compact cross-analysis comparison table used in reporting."""
    controlled = pd.read_csv(roster_design_dir / "reviewer1_comment2_main_comparison.csv")
    controlled = controlled.loc[
        (controlled["design_family"] == "Joint Pareto")
        & np.isclose(controlled["delta"], 0.0, equal_nan=False)
    ]
    controlled_rows = controlled[["method", "policy", "realized_points", "total_autosubs", "substitute_points"]].copy()
    controlled_rows = controlled_rows.rename(columns={"method": "model"})
    controlled_rows["experiment"] = "Joint δ=0 roster design"

    regularized = pd.read_csv(regularization_dir / "reviewer1_comments3_4_main_comparison.csv")
    regularized_rows = []
    for row in regularized.itertuples(index=False):
        for policy, points_column in [("static", "static_points"), ("sequential", "sequential_points")]:
            regularized_rows.append(
                {
                    "model": row.model,
                    "policy": policy,
                    "realized_points": getattr(row, points_column),
                    "total_autosubs": np.nan,
                    "substitute_points": np.nan,
                    "experiment": "Regularization comparison",
                }
            )
    table = pd.concat([controlled_rows, pd.DataFrame(regularized_rows)], ignore_index=True)
    table = table.sort_values(["policy", "realized_points"], ascending=[True, False])
    table.to_csv(table_dir / "table_main_primary_model_comparison.csv", index=False)
    return table


RETAINED_SUPPLEMENTARY_FIGURES = {
    "fig03_feature_correlation_heatmap.pdf": "figS01_feature_correlation_heatmap.pdf",
    "fig04_feature_distributions.pdf": "figS02_feature_distributions.pdf",
    "fig05a_shap_goalkeepers.pdf": "figS03a_shap_goalkeepers.pdf",
    "fig05b_shap_defenders.pdf": "figS03b_shap_defenders.pdf",
    "fig05c_shap_midfielders.pdf": "figS03c_shap_midfielders.pdf",
    "fig05d_shap_forwards.pdf": "figS03d_shap_forwards.pdf",
    "fig06_method_similarity.pdf": "figS04_method_similarity.pdf",
    "fig07a_averaging_robust.pdf": "figS05a_averaging_robust.pdf",
    "fig07b_simulation.pdf": "figS05b_simulation.pdf",
    "fig09_alternative_objectives.pdf": "figS06_alternative_objectives.pdf",
    "fig10a_hybrid_averaging.pdf": "figS07a_hybrid_averaging.pdf",
    "fig10b_hybrid_simulation.pdf": "figS07b_hybrid_simulation.pdf",
    "fig11a_hybrid_arima.pdf": "figS08a_hybrid_arima.pdf",
    "fig11b_hybrid_ict_linear.pdf": "figS08b_hybrid_ict_linear.pdf",
    "fig12_simple_budget_ranks.pdf": "figS09_simple_budget_ranks.pdf",
    "fig13_weighted_budget_ranks.pdf": "figS10_weighted_budget_ranks.pdf",
    "fig14_arima100_sequential_budget_ranks.pdf": "figS11_arima100_sequential_budget_ranks.pdf",
    "fig15_arima101_sequential_budget_ranks.pdf": "figS12_arima101_sequential_budget_ranks.pdf",
    "fig16_arima001_budget_sensitivity.pdf": "figS13_arima001_budget_sensitivity.pdf",
    "fig17_arima100_budget_sensitivity.pdf": "figS14_arima100_budget_sensitivity.pdf",
    "fig18_arima101_budget_sensitivity.pdf": "figS15_arima101_budget_sensitivity.pdf",
    "fig19_ict_budget_sensitivity.pdf": "figS16_ict_budget_sensitivity.pdf",
    "fig20_hybrid_ict_budget_sensitivity.pdf": "figS17_hybrid_ict_budget_sensitivity.pdf",
    "fig21_monte_carlo_budget_sensitivity.pdf": "figS18_monte_carlo_budget_sensitivity.pdf",
    "fig22_simple_sequential_budget_sensitivity.pdf": "figS19_simple_sequential_budget_sensitivity.pdf",
    "fig23_weighted_sequential_budget_sensitivity.pdf": "figS20_weighted_sequential_budget_sensitivity.pdf",
    "fig24_arima001_sequential_budget_sensitivity.pdf": "figS21_arima001_sequential_budget_sensitivity.pdf",
    "fig26_family_best_cumulative.pdf": "figS22_family_best_cumulative.pdf",
    "fig27_family_uplift.pdf": "figS23_family_uplift.pdf",
    "fig28_top10_weekly_distributions.pdf": "figS24_top10_weekly_distributions.pdf",
    "fig29_external_benchmark.pdf": "figS25_fixed_split_external_benchmark.pdf",
}


def organize_retained_figures(
    staging_dir: Path,
    main_dir: Path,
    supplementary_dir: Path,
) -> None:
    """Move recomputed figures into their final manuscript locations."""
    promoted_main_figures = {
        "fig02_position_correlations.pdf": "fig02_position_correlations.pdf",
        "fig08_arima_variants.pdf": "fig05_arima_variants.pdf",
        "fig25_pareto_frontier.pdf": "fig07_pareto_frontier.pdf",
    }
    for source_name, destination_name in promoted_main_figures.items():
        source = staging_dir / source_name
        if not source.is_file():
            raise RuntimeError(f"Promoted main-paper figure was not generated: {source_name}")
        source.replace(main_dir / destination_name)
    for source_name, destination_name in RETAINED_SUPPLEMENTARY_FIGURES.items():
        source = staging_dir / source_name
        if not source.is_file():
            raise RuntimeError(f"Retained figure was not generated: {source_name}")
        source.replace(supplementary_dir / destination_name)
    for temporary in staging_dir.glob(".*.writing.pdf"):
        temporary.unlink()
    unexpected = sorted(path.name for path in staging_dir.glob("*.pdf"))
    if unexpected:
        raise RuntimeError(f"Unclassified retained figures: {unexpected}")
    staging_dir.rmdir()


def remove_intermediate_pdfs(output_dir: Path) -> None:
    """Remove diagnostic PDFs after their content has been consolidated."""
    experiment_root = output_dir / "analysis_outputs"
    if not experiment_root.exists():
        return
    for path in experiment_root.rglob("*.pdf"):
        path.unlink()


def figure_manifest(output_dir: Path) -> pd.DataFrame:
    """Validate every indexed PDF and record its caption, size, and hash."""
    captions = {
        "main_figures/fig01_framework.pdf": "Author-designed data, forecasting, and optimization workflow",
        "main_figures/fig02_position_correlations.pdf": "Position-specific feature--points correlations",
        "main_figures/fig03_feature_regularization_evidence.pdf": "Feature redundancy, cost-vector stability, and LASSO selection",
        "main_figures/fig04_primary_model_performance.pdf": "Corrected static and fixed-squad sequential performance",
        "main_figures/fig05_arima_variants.pdf": "ARIMA variants",
        "main_figures/fig06_budget_policy_sensitivity.pdf": "Controlled starting-XI cap and bench-budget-rule sensitivity",
        "main_figures/fig07_pareto_frontier.pdf": "Epsilon-constraint starter--bench Pareto frontier",
        "main_figures/fig08_external_benchmark.pdf": "Corrected joint-model comparison with Santoro strategies",
        "main_figures/fig09_availability_sensitivity.pdf": "Stochastic starting-availability sensitivity analysis",
        "supplementary_figures/figS01_feature_correlation_heatmap.pdf": "Pearson correlation heatmap of modeling features",
        "supplementary_figures/figS02_feature_distributions.pdf": "Numeric feature distributions by position",
        "supplementary_figures/figS03a_shap_goalkeepers.pdf": "SHAP summary for goalkeepers",
        "supplementary_figures/figS03b_shap_defenders.pdf": "SHAP summary for defenders",
        "supplementary_figures/figS03c_shap_midfielders.pdf": "SHAP summary for midfielders",
        "supplementary_figures/figS03d_shap_forwards.pdf": "SHAP summary for forwards",
        "supplementary_figures/figS04_method_similarity.pdf": "Hierarchical clustering of strategy scores",
        "supplementary_figures/figS05a_averaging_robust.pdf": "Averaging and robust forecasting families",
        "supplementary_figures/figS05b_simulation.pdf": "Simulation forecasting families",
        "supplementary_figures/figS06_alternative_objectives.pdf": "ICT, robust ICT, and involvement objectives",
        "supplementary_figures/figS07a_hybrid_averaging.pdf": "Hybrid averaging methods",
        "supplementary_figures/figS07b_hybrid_simulation.pdf": "Hybrid simulation methods",
        "supplementary_figures/figS08a_hybrid_arima.pdf": "Hybrid ARIMA methods",
        "supplementary_figures/figS08b_hybrid_ict_linear.pdf": "Hybrid ICT and linear methods",
        "supplementary_figures/figS09_simple_budget_ranks.pdf": "Simple-average budget ranks",
        "supplementary_figures/figS10_weighted_budget_ranks.pdf": "Weighted-average budget ranks",
        "supplementary_figures/figS11_arima100_sequential_budget_ranks.pdf": "ARIMA(1,0,0) static and corrected sequential budget ranks",
        "supplementary_figures/figS12_arima101_sequential_budget_ranks.pdf": "ARIMA(1,0,1) static and corrected sequential budget ranks",
        "supplementary_figures/figS13_arima001_budget_sensitivity.pdf": "ARIMA(0,0,1) static budget sensitivity",
        "supplementary_figures/figS14_arima100_budget_sensitivity.pdf": "ARIMA(1,0,0) static budget sensitivity",
        "supplementary_figures/figS15_arima101_budget_sensitivity.pdf": "ARIMA(1,0,1) static budget sensitivity",
        "supplementary_figures/figS16_ict_budget_sensitivity.pdf": "ICT budget sensitivity",
        "supplementary_figures/figS17_hybrid_ict_budget_sensitivity.pdf": "Hybrid ICT budget sensitivity",
        "supplementary_figures/figS18_monte_carlo_budget_sensitivity.pdf": "Monte Carlo budget sensitivity",
        "supplementary_figures/figS19_simple_sequential_budget_sensitivity.pdf": "Corrected simple-average sequential budget sensitivity",
        "supplementary_figures/figS20_weighted_sequential_budget_sensitivity.pdf": "Corrected weighted-average sequential budget sensitivity",
        "supplementary_figures/figS21_arima001_sequential_budget_sensitivity.pdf": "Corrected ARIMA(0,0,1) sequential budget sensitivity",
        "supplementary_figures/figS22_family_best_cumulative.pdf": "Best-performing methods across forecasting families",
        "supplementary_figures/figS23_family_uplift.pdf": "Weekly method uplift relative to the corrected fixed-squad benchmark",
        "supplementary_figures/figS24_top10_weekly_distributions.pdf": "Weekly score distributions for the leading methods",
        "supplementary_figures/figS25_fixed_split_external_benchmark.pdf": "Fixed-split corrected sequential comparison with Santoro strategies",
    }
    rows = []
    for relative_path, caption in captions.items():
        path = output_dir / relative_path
        exists = path.is_file()
        payload = path.read_bytes() if exists else b""
        valid_pdf = bool(
            exists
            and payload.startswith(b"%PDF-")
            and payload.rstrip().endswith(b"%%EOF")
        )
        rows.append(
            {
                "relative_path": relative_path,
                "caption_short": caption,
                "exists": exists,
                "valid_pdf": valid_pdf,
                "size_bytes": path.stat().st_size if exists else 0,
                "sha256": sha256(path) if exists else "",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "figure_manifest.csv", index=False)
    if not manifest["exists"].all():
        missing = manifest.loc[~manifest["exists"], "relative_path"].tolist()
        raise RuntimeError(f"Missing publication figures: {missing}")
    if not manifest["valid_pdf"].all():
        invalid = manifest.loc[~manifest["valid_pdf"], "relative_path"].tolist()
        raise RuntimeError(f"Invalid or incomplete publication PDFs: {invalid}")
    return manifest


def output_manifest(output: Path) -> None:
    """Write a hash manifest covering every generated file in the output tree."""
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "output_manifest.csv":
            rows.append({"relative_path": str(path.relative_to(output)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(output / "output_manifest.csv", index=False)


def write_readme(output: Path, args: argparse.Namespace) -> None:
    """Describe the generated directories, provenance, and selection safeguards."""
    text = f"""# FPL paper output bundle

Data split: GW1--26 for training/development and GW27--38 for evaluation.

The final publication set contains nine main-paper PDFs in `main_figures/`
and 31 supplementary PDFs in `supplementary_figures/`. Figure 1 is the
author-designed workflow supplied through `--framework-figure`; it is validated
and copied without alteration, not drawn by the pipeline. The other 39 PDFs are
recomputed and plotted from the analysis outputs. Every analytical PDF is
title-free, uses stable manuscript/supplement indexing, and is saved as a
vector PDF with 300-DPI raster metadata and a tight bounding box. Plot legends
are outside the data panels. Portrait figures are designed for the manuscript's
6.70-inch full-text width. The dense method-similarity matrix is designed for a
9.70-inch landscape supplementary page. Global labels are 9--10 pt and no
explicit annotation or legend text is smaller than 8 pt.

All 94 internal weekly score series are recomputed from `{args.data.name}`:
74 static forecasting/objective/budget variants and 20 corrected fixed-squad
sequential variants. The run never reads a historical Results CSV or a saved
roster table. The four Santoro series are the only external numerical input
and are labeled as externally supplied benchmark data. All manuscript roster
tables are regenerated from the fresh optimization outputs. No realized
GW27--38 outcome is used to choose a budget, Pareto point, regularization
parameter, or forecasting specification.

The retained uplift and top-method distribution figures are descriptive
post-evaluation summaries only and are not used for model selection.

Successful completion is indicated by `RUN_COMPLETED_SUCCESSFULLY.txt`.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Execute the end-to-end workflow and fail if any expected artifact is missing."""
    args = parse_args()
    started = time.perf_counter()
    for attr in ["data", "output", "font", "framework_figure", "analysis_core", "roster_design_runner", "full_internal_runner"]:
        setattr(args, attr, getattr(args, attr).expanduser().resolve())
    for attr in ["existing_roster_design_dir", "existing_regularization_dir", "existing_full_internal_dir"]:
        if getattr(args, attr) is not None:
            setattr(args, attr, getattr(args, attr).expanduser().resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    validate_support_files(args)
    framework_payload = read_valid_pdf(args.framework_figure, "Author-designed workflow figure")
    for path, label in [(args.data, "dataset")]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    main_figures = args.output / "main_figures"
    supplementary_figures = args.output / "supplementary_figures"
    figure_staging = args.output / "_figure_staging"
    results_dir = args.output / "results"
    tables_dir = args.output / "tables"
    for directory in [main_figures, supplementary_figures, figure_staging, results_dir, tables_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    for directory in [main_figures, supplementary_figures, figure_staging]:
        for stale_pdf in directory.glob("*.pdf"):
            stale_pdf.unlink()
        for stale_temporary in directory.glob(".*.writing.pdf"):
            stale_temporary.unlink()
    (main_figures / "fig01_framework.pdf").write_bytes(framework_payload)
    font_name = configure_style(args.font)
    data = load_data(args.data)
    log(f"Loaded {len(data):,} player--gameweek rows; no historical Results CSV will be used.")
    if args.skip_computational_experiments:
        if args.existing_roster_design_dir is None or args.existing_full_internal_dir is None:
            raise ValueError(
                "Skipping computation requires --existing-roster-design-dir "
                "and --existing-full-internal-dir."
            )
        roster_design_dir = args.existing_roster_design_dir
        regularization_dir = args.existing_regularization_dir or (args.output / "analysis_outputs" / "feature_regularization")
        full_internal_dir = args.existing_full_internal_dir
    else:
        roster_design_dir, regularization_dir = run_supporting_experiments(args, args.output / "analysis_outputs")
        full_internal_dir = run_full_internal_screen(
            args, roster_design_dir, args.output / "analysis_outputs"
        )

    results = load_recomputed_results(full_internal_dir / "full_internal_score_matrix.csv")
    log(f"Loaded {results.shape[1] - len(SANTORO_SCORES)} freshly recomputed internal series and {len(SANTORO_SCORES)} external benchmark series.")
    retained_matrix = results.rename(columns=display_label).copy()
    retained_matrix.insert(0, "GW", range(TARGET_GW, FINAL_GW + 1))
    retained_matrix.to_csv(results_dir / "raw_recomputed_figure_score_matrix.csv", index=False)
    pd.DataFrame(
        {
            "series": list(SANTORO_SCORES),
            "source_type": "externally supplied benchmark",
            "source_note": "Santoro comparison series supplied by the cited study/first author; not derived from the FPL raw panel.",
        }
    ).to_csv(results_dir / "external_benchmark_provenance.csv", index=False)

    regularization_evidence_figure(
        data,
        regularization_dir,
        main_figures / "fig03_feature_regularization_evidence.pdf",
    )
    corrected_performance_figure(
        roster_design_dir,
        regularization_dir,
        main_figures / "fig04_primary_model_performance.pdf",
    )
    budget_policy_figure(
        roster_design_dir,
        main_figures / "fig06_budget_policy_sensitivity.pdf",
    )
    corrected_external_benchmark(
        roster_design_dir,
        main_figures / "fig08_external_benchmark.pdf",
    )
    availability_figure(
        roster_design_dir,
        main_figures / "fig09_availability_sensitivity.pdf",
    )

    eda_figures(
        data,
        figure_staging,
        min(args.shap_sample, 300 if args.smoke_test else args.shap_sample),
        args.seed,
        args.skip_shap or args.smoke_test,
    )
    generate_score_figures(
        results,
        figure_staging,
        roster_design_dir,
    )
    organize_retained_figures(
        figure_staging,
        main_figures,
        supplementary_figures,
    )
    for directory in [main_figures, supplementary_figures]:
        for temporary in directory.glob(".*.writing.pdf"):
            temporary.unlink()

    export_score_results(results, results_dir, tables_dir, full_internal_dir)
    copy_controlled_tables(roster_design_dir, regularization_dir, tables_dir, results_dir)
    copy_full_internal_results(full_internal_dir, results_dir, tables_dir)
    build_primary_comparison_table(roster_design_dir, regularization_dir, tables_dir)
    remove_intermediate_pdfs(args.output)
    for cache_dir in sorted(args.output.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache_dir, ignore_errors=True)
    manifest = figure_manifest(args.output)
    config = {
        "data": str(args.data), "data_sha256": sha256(args.data), "output": str(args.output),
        "font": font_name, "workers": args.workers, "bootstrap": 4 if args.smoke_test else args.bootstrap,
        "simulation_draws": 20 if args.smoke_test else args.simulation_draws,
        "seed": args.seed, "shap_sample": min(args.shap_sample, 300 if args.smoke_test else args.shap_sample),
        "smoke_test": args.smoke_test, "main_figure_files": 9, "supplementary_figure_files": 31,
        "analytical_figure_files": 39, "author_supplied_figure_files": 1,
        "framework_figure_source": str(args.framework_figure),
        "figure_full_text_width_in": FULL_TEXT_WIDTH_IN,
        "figure_landscape_width_in": LANDSCAPE_TEXT_WIDTH_IN,
        "base_font_size_pt": BASE_FONT_SIZE_PT,
        "minimum_explicit_figure_text_pt": MIN_FIGURE_TEXT_PT,
        "figure_format": "vector PDF",
        "rasterized_content_dpi": 300,
        "raw_recomputed_internal_series": 94, "external_benchmark_series": 4,
        "historical_results_csv_used": False, "saved_roster_snapshot_used": False,
        "python": sys.version.replace("\n", " "), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
    }
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_readme(args.output, args)
    elapsed = time.perf_counter() - started
    (args.output / "RUN_COMPLETED_SUCCESSFULLY.txt").write_text(f"Completed successfully in {elapsed/60:.2f} minutes. Generated {len(manifest)} figure files.\n", encoding="utf-8")
    output_manifest(args.output)
    log(f"EXPERIMENT COMPLETED SUCCESSFULLY: {len(manifest)} indexed figure files; elapsed {elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    main()
