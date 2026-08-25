#!/usr/bin/env python3
"""Run the feature-regularization and model-stability analyses for FPL.

The workflow builds a leakage-safe player--gameweek training panel, tunes Ridge
and LASSO models with rolling-origin validation, and compares full, reduced,
and parsimonious feature sets.  It then measures cost-vector and selected-XI
stability with a position-stratified player-cluster bootstrap.  A fixed-squad
evaluator applies legal formations, automatic substitutions, and invariant
captaincy when computing GW27--38 performance.

The module can also launch the companion roster-design analysis so that a
standalone run produces all controlled optimization and regularization results.
Configuration, software versions, hashes, and a completion marker are exported
for reproducibility.  Generated files are always written beneath a new output
directory; the input dataset and any historical outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from collections import OrderedDict
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fpl_regularization_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    from scipy.stats import spearmanr
except ImportError as exc:  # pragma: no cover - dependency guard for server use
    raise SystemExit(
        "SciPy is required. Install the packages listed in requirements_fpl_pipeline.txt."
    ) from exc

try:
    import sklearn
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required. Install the packages listed in "
        "requirements_fpl_pipeline.txt."
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_GW = 27
FINAL_GW = 38
TOTAL_BUDGET = 100.0
EWM_SPAN = 5
HISTORY_WEIGHT = 1.0 / 3.0
MODEL_WEIGHT = 2.0 / 3.0
MIN_HISTORY = 4
OBJECTIVE_TOL = 1e-6
TIE_EPS = 1e-9

SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_BOUNDS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3
POSITIONS = ["GK", "DEF", "MID", "FWD"]

FULL_FEATURES = [
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
    "ict_index",
    "selected",
    "starts",
]

MODEL_SPECS = OrderedDict(
    [
        (
            "Full Ridge",
            {
                "kind": "ridge",
                "features": FULL_FEATURES,
                "description": "All seven features retained with L2 shrinkage",
            },
        ),
        (
            "Reduced Ridge",
            {
                "kind": "ridge",
                "features": [
                    "expected_goal_involvements",
                    "expected_goals_conceded",
                    "ict_index",
                    "selected",
                    "starts",
                ],
                "description": "Drops xG and xA because xGI is their sum",
            },
        ),
        (
            "Parsimonious Ridge",
            {
                "kind": "ridge",
                "features": [
                    "expected_goal_involvements",
                    "expected_goals_conceded",
                    "selected",
                    "starts",
                ],
                "description": "Keeps xGI as the sole attacking feature",
            },
        ),
        (
            "Full LASSO",
            {
                "kind": "lasso",
                "features": FULL_FEATURES,
                "description": "All seven candidates with L1 feature selection",
            },
        ),
    ]
)

RIDGE_ALPHAS = np.logspace(-3, 3, 13)
LASSO_ALPHAS = np.logspace(-5, 0, 16)
CV_FOLDS = [(14, 15, 18), (18, 19, 22), (22, 23, 26)]

FEATURE_SHORT = {
    "expected_assists": "xA",
    "expected_goal_involvements": "xGI",
    "expected_goals": "xG",
    "expected_goals_conceded": "xGC",
    "ict_index": "ICT",
    "selected": "Selected",
    "starts": "Starts",
}


def parse_args() -> argparse.Namespace:
    """Parse file locations, bootstrap settings, and server-run options."""
    parser = argparse.ArgumentParser(
        description="Run FPL roster-design, regularization, and stability analyses."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("merged_gw_2324.csv"),
        help="Path to merged_gw_2324.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fpl_analysis_outputs_server"),
        help="New directory for all generated outputs.",
    )
    parser.add_argument(
        "--roster-design-script", dest="roster_design_script",
        type=Path,
        default=Path(__file__).with_name("fpl_roster_design_experiments.py"),
        help="Path to the companion roster-design and Pareto runner.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=500,
        help="Position-stratified player-cluster bootstrap replications (default: 500).",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="ARIMA workers passed to the roster-design runner.",
    )
    parser.add_argument(
        "--skip-roster-design", dest="skip_roster_design",
        action="store_true",
        help="Skip the roster-design analysis (for local dependency tests only).",
    )
    parser.add_argument(
        "--skip-legacy-figures",
        action="store_true",
        help="Skip consolidated figures based on the optional legacy Results CSV.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast installation test using four bootstraps and a reduced roster-design configuration.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    """Write an immediately flushed message suitable for ``tail -f``."""
    print(message, flush=True)


def safe_spearman(a, b) -> float:
    """Compute Spearman correlation after filtering invalid or constant inputs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.isclose(np.std(a[mask]), 0) or np.isclose(np.std(b[mask]), 0):
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def jaccard(left, right) -> float:
    """Measure overlap between two selected-player sets."""
    left, right = set(left), set(right)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def save_pdf(fig, path: Path) -> None:
    """Save a title-free vector PDF with a tight bounding box and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload) -> None:
    """Serialize a JSON-compatible object with stable, human-readable formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    """Collect software and platform versions needed to reproduce the run."""
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import statsmodels

        versions["statsmodels"] = statsmodels.__version__
    except Exception:
        versions["statsmodels"] = "not installed (required for ARIMA roster forecasts)"
    return versions


# ---------------------------------------------------------------------------
# Data preparation: leakage-safe lagged feature panel
# ---------------------------------------------------------------------------
def load_player_gameweek(path: Path) -> pd.DataFrame:
    """Load, type-check, and collapse the raw data to one row per player and GW.

    Double-gameweek additive statistics are summed.  Player metadata and price
    use the latest observation, while the selected-by count uses its maximum.
    """
    required = {
        "name",
        "position",
        "team",
        "value",
        "GW",
        "total_points",
        "minutes",
        *FULL_FEATURES,
    }
    raw = pd.read_csv(path)
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    raw = raw.copy()
    raw["GW"] = pd.to_numeric(raw["GW"], errors="raise").astype(int)
    numeric = ["value", "total_points", "minutes", *FULL_FEATURES]
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0.0)
    if raw["value"].median() > 20:
        raw["value"] = raw["value"] / 10.0

    # Aggregate defensive duplicates, including double gameweeks.  Additive
    # match statistics and points are summed; selection counts use their maximum.
    additive = [
        "total_points",
        "minutes",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals",
        "expected_goals_conceded",
        "ict_index",
        "starts",
    ]
    aggregations = {column: "sum" for column in additive}
    aggregations.update(
        {
            "selected": "max",
            "position": "last",
            "team": "last",
            "value": "last",
        }
    )
    panel = (
        raw.sort_values(["name", "GW"])
        .groupby(["name", "GW"], as_index=False)
        .agg(aggregations)
        .sort_values(["name", "GW"])
        .reset_index(drop=True)
    )
    return panel


def add_lagged_summaries(panel: pd.DataFrame) -> pd.DataFrame:
    """Add exponentially weighted histories shifted by one GW to prevent leakage."""
    panel = panel.sort_values(["name", "GW"]).copy()
    history_columns = [*FULL_FEATURES, "total_points"]
    grouped = panel.groupby("name", sort=False)
    panel["history_count"] = grouped.cumcount()
    for column in history_columns:
        panel[f"lag_{column}"] = grouped[column].transform(
            lambda series: series.shift(1).ewm(span=EWM_SPAN, adjust=False).mean()
        )
    return panel


def snapshot_at_cutoff(panel: pd.DataFrame, cutoff_gw: int) -> pd.DataFrame:
    """Construct the latest player metadata and pre-cutoff feature summary."""
    history = panel.loc[panel["GW"] < cutoff_gw].sort_values(["name", "GW"])
    if history.empty:
        raise ValueError(f"No history is available before GW{cutoff_gw}.")

    metadata = (
        history.groupby("name", as_index=False)
        .tail(1)[["name", "team", "position", "value"]]
        .drop_duplicates("name")
        .set_index("name")
    )
    summary_rows = []
    for name, group in history.groupby("name", sort=False):
        row = {"name": name}
        for feature in FULL_FEATURES:
            row[f"lag_{feature}"] = float(
                group[feature].ewm(span=EWM_SPAN, adjust=False).mean().iloc[-1]
            )
        row["lag_total_points"] = float(
            group["total_points"].ewm(span=EWM_SPAN, adjust=False).mean().iloc[-1]
        )
        row["history_gameweeks"] = int(group["GW"].nunique())
        summary_rows.append(row)
    snapshot = pd.DataFrame(summary_rows).set_index("name").join(metadata, how="inner")
    snapshot = snapshot.reset_index()
    snapshot = snapshot.loc[snapshot["position"].isin(POSITIONS)].copy()
    snapshot = snapshot.dropna(subset=["name", "team", "position", "value"])
    return snapshot.sort_values("name", kind="stable").reset_index(drop=True)


def future_player_outcomes(panel: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized GW27--38 outcomes for evaluation only."""
    future = panel.loc[panel["GW"].between(TARGET_GW, FINAL_GW)]
    totals = future.groupby("name")["total_points"].sum()
    appearances = future.loc[future["minutes"] > 0].groupby("name")["GW"].nunique()
    out = candidates[["name", "position"]].copy()
    out["future_total_points"] = out["name"].map(totals).fillna(0.0)
    out["future_mean_points"] = out["future_total_points"] / (FINAL_GW - TARGET_GW + 1)
    out["future_appearances"] = out["name"].map(appearances).fillna(0).astype(int)
    return out


# ---------------------------------------------------------------------------
# Correct Ridge/LASSO pipelines and rolling-origin tuning
# ---------------------------------------------------------------------------
def make_pipeline(kind: str, alpha: float) -> Pipeline:
    """Create a standardized Ridge or LASSO regression pipeline."""
    if kind == "ridge":
        regressor = Ridge(alpha=float(alpha), fit_intercept=True)
    elif kind == "lasso":
        regressor = Lasso(
            alpha=float(alpha),
            fit_intercept=True,
            max_iter=100_000,
            tol=1e-6,
            selection="cyclic",
        )
    else:
        raise ValueError(f"Unknown regularizer: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])


def tune_alpha(
    train_panel: pd.DataFrame,
    position: str,
    spec: dict,
) -> tuple[float, pd.DataFrame]:
    """Choose regularization strength using predeclared rolling-origin folds.

    Ridge minimizes validation RMSE. LASSO uses the one-standard-error rule and
    selects the largest eligible penalty to favor a reproducibly sparse model.
    """
    features = [f"lag_{feature}" for feature in spec["features"]]
    alpha_grid = RIDGE_ALPHAS if spec["kind"] == "ridge" else LASSO_ALPHAS
    rows = []
    position_data = train_panel.loc[train_panel["position"] == position].copy()

    for alpha in alpha_grid:
        fold_rmses = []
        fold_maes = []
        for train_end, valid_start, valid_end in CV_FOLDS:
            fit_rows = position_data.loc[position_data["GW"] <= train_end]
            valid_rows = position_data.loc[position_data["GW"].between(valid_start, valid_end)]
            if len(fit_rows) < 30 or len(valid_rows) < 10:
                continue
            pipeline = make_pipeline(spec["kind"], alpha)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(fit_rows[features], fit_rows["total_points"])
            pred = pipeline.predict(valid_rows[features])
            fold_rmses.append(math.sqrt(mean_squared_error(valid_rows["total_points"], pred)))
            fold_maes.append(mean_absolute_error(valid_rows["total_points"], pred))
        if fold_rmses:
            rows.append(
                {
                    "alpha": float(alpha),
                    "cv_rmse_mean": float(np.mean(fold_rmses)),
                    "cv_rmse_se": float(
                        np.std(fold_rmses, ddof=1) / math.sqrt(len(fold_rmses))
                        if len(fold_rmses) > 1
                        else 0.0
                    ),
                    "cv_mae_mean": float(np.mean(fold_maes)),
                    "folds": len(fold_rmses),
                }
            )
    grid = pd.DataFrame(rows)
    if grid.empty:
        raise RuntimeError(f"No valid CV folds for {position} and {spec['kind']}.")

    best_idx = grid["cv_rmse_mean"].idxmin()
    threshold = grid.loc[best_idx, "cv_rmse_mean"] + grid.loc[best_idx, "cv_rmse_se"]
    eligible = grid.loc[grid["cv_rmse_mean"] <= threshold + 1e-12]
    if spec["kind"] == "lasso":
        # For LASSO, the one-standard-error rule provides a predeclared,
        # validation-only preference for the sparser model.
        selected_alpha = float(eligible["alpha"].max())
        selection_rule = "one-standard-error (largest eligible alpha)"
    else:
        # Ridge does not perform variable selection, so use the direct
        # rolling-origin RMSE minimizer instead of pushing shrinkage to an
        # arbitrary grid boundary.
        selected_alpha = float(grid.loc[best_idx, "alpha"])
        selection_rule = "minimum rolling-origin RMSE"
    grid["one_se_threshold"] = float(threshold)
    grid["selected"] = np.isclose(grid["alpha"], selected_alpha)
    grid["selection_rule"] = selection_rule
    return selected_alpha, grid


def fit_reference_models(train_panel: pd.DataFrame):
    """Tune and fit every model-by-position specification on the training panel."""
    fitted: dict[str, dict[str, Pipeline]] = {}
    selected_alphas: dict[str, dict[str, float]] = {}
    tuning_rows = []
    coefficient_rows = []

    for model_name, spec in MODEL_SPECS.items():
        log(f"Tuning {model_name} ...")
        fitted[model_name] = {}
        selected_alphas[model_name] = {}
        features = [f"lag_{feature}" for feature in spec["features"]]
        for position in POSITIONS:
            alpha, grid = tune_alpha(train_panel, position, spec)
            grid.insert(0, "position", position)
            grid.insert(0, "model", model_name)
            tuning_rows.append(grid)
            selected_alphas[model_name][position] = alpha

            rows = train_panel.loc[train_panel["position"] == position]
            pipeline = make_pipeline(spec["kind"], alpha)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(rows[features], rows["total_points"])
            fitted[model_name][position] = pipeline

            coef = pipeline.named_steps["regressor"].coef_
            for feature, value in zip(spec["features"], coef):
                coefficient_rows.append(
                    {
                        "model": model_name,
                        "position": position,
                        "feature": feature,
                        "feature_label": FEATURE_SHORT[feature],
                        "alpha": alpha,
                        "standardized_coefficient": float(value),
                        "selected_nonzero": bool(abs(value) > 1e-10),
                    }
                )

    return (
        fitted,
        selected_alphas,
        pd.concat(tuning_rows, ignore_index=True),
        pd.DataFrame(coefficient_rows),
    )


def score_snapshot(
    snapshot: pd.DataFrame,
    model_name: str,
    fitted_models: dict[str, dict[str, Pipeline]],
) -> pd.DataFrame:
    """Predict a cutoff snapshot and form the hybrid optimization cost vector."""
    spec = MODEL_SPECS[model_name]
    features = [f"lag_{feature}" for feature in spec["features"]]
    out = snapshot.copy()
    out["model_prediction"] = np.nan
    for position in POSITIONS:
        mask = out["position"] == position
        if not mask.any():
            continue
        out.loc[mask, "model_prediction"] = fitted_models[model_name][position].predict(
            out.loc[mask, features]
        )
    if out["model_prediction"].isna().any():
        raise AssertionError(f"Missing predictions for {model_name}.")
    # Both terms are points per gameweek, so they can be combined directly; no
    # arbitrary min--max rescaling is needed for the predeclared 1:2 hybrid.
    out["cost_vector"] = (
        HISTORY_WEIGHT * out["lag_total_points"]
        + MODEL_WEIGHT * out["model_prediction"]
    )
    out["model"] = model_name
    return out


# ---------------------------------------------------------------------------
# Joint roster optimizer and fixed-squad evaluator
# ---------------------------------------------------------------------------
class MilpRows:
    """Incrementally build sparse linear constraints for SciPy's MILP solver."""
    def __init__(self, n_variables: int):
        self.n_variables = int(n_variables)
        self.row_ids: list[int] = []
        self.col_ids: list[int] = []
        self.values: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, coefficients: dict[int, float], lower=-np.inf, upper=np.inf) -> None:
        row = len(self.lower)
        for column, value in coefficients.items():
            if not np.isclose(value, 0.0):
                self.row_ids.append(row)
                self.col_ids.append(int(column))
                self.values.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def constraint(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.values, (self.row_ids, self.col_ids)),
            shape=(len(self.lower), self.n_variables),
        ).tocsr()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


def solve_binary(objective, rows: MilpRows, label: str) -> np.ndarray:
    """Maximize a linear objective over binary variables and return 0/1 values."""
    objective = np.asarray(objective, dtype=float)
    result = milp(
        c=-objective,
        integrality=np.ones(len(objective), dtype=int),
        bounds=Bounds(np.zeros(len(objective)), np.ones(len(objective))),
        constraints=rows.constraint(),
        options={"time_limit": 120.0, "mip_rel_gap": 0.0, "presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed for {label}: {result.message}")
    return np.rint(result.x).astype(int)


def joint_rows(pool: pd.DataFrame, xi_floor: float | None = None):
    """Build legal 15-player squad, XI, captain, budget, and club constraints."""
    pool = pool.reset_index(drop=True)
    n = len(pool)
    q = np.arange(n)
    u = n + np.arange(n)
    y = 2 * n + np.arange(n)
    rows = MilpRows(3 * n)
    prices = pool["value"].to_numpy(float)
    costs = pool["cost_vector"].to_numpy(float)

    rows.add({int(q[j]): 1 for j in range(n)}, lower=15, upper=15)
    rows.add({int(u[j]): 1 for j in range(n)}, lower=11, upper=11)
    rows.add({int(y[j]): 1 for j in range(n)}, lower=1, upper=1)
    rows.add({int(q[j]): prices[j] for j in range(n)}, upper=TOTAL_BUDGET)
    for j in range(n):
        rows.add({int(u[j]): 1, int(q[j]): -1}, upper=0)
        rows.add({int(y[j]): 1, int(u[j]): -1}, upper=0)

    for position, quota in SQUAD_QUOTAS.items():
        idx = np.flatnonzero(pool["position"].to_numpy() == position)
        rows.add({int(q[j]): 1 for j in idx}, lower=quota, upper=quota)
    for position, (minimum, maximum) in XI_BOUNDS.items():
        idx = np.flatnonzero(pool["position"].to_numpy() == position)
        rows.add({int(u[j]): 1 for j in idx}, lower=minimum, upper=maximum)
    for team in sorted(pool["team"].unique()):
        idx = np.flatnonzero(pool["team"].to_numpy() == team)
        rows.add({int(q[j]): 1 for j in idx}, upper=MAX_PER_CLUB)
    if xi_floor is not None:
        coefficients = {
            **{int(u[j]): costs[j] for j in range(n)},
            **{int(y[j]): costs[j] for j in range(n)},
        }
        rows.add(coefficients, lower=float(xi_floor))
    return rows, q, u, y


def solve_joint_lexicographic(pool: pd.DataFrame, label: str) -> dict:
    """Maximize XI value first, then bench value without sacrificing XI quality."""
    pool = pool.reset_index(drop=True).copy()
    n = len(pool)
    costs = pool["cost_vector"].to_numpy(float)
    stable = np.linspace(1.0, 0.0, n, endpoint=False)

    rows, q, u, y = joint_rows(pool)
    xi_objective = np.zeros(3 * n)
    xi_objective[u] = costs + TIE_EPS * stable
    xi_objective[y] = costs + TIE_EPS * stable
    phase1 = solve_binary(xi_objective, rows, f"{label}, phase 1")
    xi_star = float(costs[np.flatnonzero(phase1[u])].sum() + costs[np.flatnonzero(phase1[y])].sum())

    # Among rosters with the maximum XI objective, select the best bench.  This
    # removes arbitrary bench ties without sacrificing starter quality.
    rows, q, u, y = joint_rows(pool, xi_floor=xi_star - OBJECTIVE_TOL)
    bench_objective = np.zeros(3 * n)
    bench_objective[q] = costs + TIE_EPS * stable
    bench_objective[u] = -costs
    phase2 = solve_binary(bench_objective, rows, f"{label}, phase 2")

    squad_idx = np.flatnonzero(phase2[q])
    starter_idx = np.flatnonzero(phase2[u])
    captain_idx = int(np.flatnonzero(phase2[y])[0])
    bench_idx = np.asarray(sorted(set(squad_idx) - set(starter_idx)), dtype=int)
    captain = str(pool.loc[captain_idx, "name"])
    vice_candidates = pool.loc[starter_idx].loc[lambda frame: frame["name"] != captain]
    vice = str(vice_candidates.sort_values(["cost_vector", "name"], ascending=[False, True]).iloc[0]["name"])

    solution = {
        "squad": pool.loc[squad_idx].reset_index(drop=True),
        "starters": pool.loc[starter_idx].reset_index(drop=True),
        "bench": pool.loc[bench_idx].reset_index(drop=True),
        "captain": captain,
        "vice_captain": vice,
        "xi_score": float(costs[starter_idx].sum() + costs[captain_idx]),
        "bench_score": float(costs[bench_idx].sum()),
        "total_spend": float(pool.loc[squad_idx, "value"].sum()),
    }
    validate_solution(solution)
    return solution


def validate_solution(solution: dict) -> None:
    """Assert squad size, formation, budget, club, and captaincy feasibility."""
    squad, starters, bench = solution["squad"], solution["starters"], solution["bench"]
    assert len(squad) == 15 and squad["name"].nunique() == 15
    assert len(starters) == 11 and starters["name"].nunique() == 11
    assert len(bench) == 4 and bench["name"].nunique() == 4
    assert solution["captain"] in set(starters["name"])
    assert solution["vice_captain"] in set(starters["name"])
    assert squad["position"].value_counts().to_dict() == SQUAD_QUOTAS
    assert squad["team"].value_counts().max() <= MAX_PER_CLUB
    assert solution["total_spend"] <= TOTAL_BUDGET + 1e-6


def solve_fixed_squad_lineup(solution: dict, score_map: pd.Series):
    """Update the legal starting XI inside a fixed squad using new forecasts."""
    squad = solution["squad"][["name", "team", "position", "value"]].copy()
    squad["cost_vector"] = squad["name"].map(score_map).fillna(0.0)
    squad = squad.reset_index(drop=True)
    n = len(squad)
    rows = MilpRows(n)
    rows.add({j: 1 for j in range(n)}, lower=11, upper=11)
    for position, (minimum, maximum) in XI_BOUNDS.items():
        idx = np.flatnonzero(squad["position"].to_numpy() == position)
        rows.add({int(j): 1 for j in idx}, lower=minimum, upper=maximum)
    # Captaincy is intentionally fixed; forcing captain and vice into the XI
    # prevents the sequential policy from becoming dynamic captaincy.
    for fixed_name in [solution["captain"], solution["vice_captain"]]:
        idx = np.flatnonzero(squad["name"].to_numpy() == fixed_name)
        rows.add({int(idx[0]): 1}, lower=1, upper=1)
    objective = squad["cost_vector"].to_numpy(float) + TIE_EPS * np.linspace(1, 0, n)
    z = solve_binary(objective, rows, "fixed-squad weekly XI")
    starters = squad.loc[np.flatnonzero(z)].reset_index(drop=True)
    bench = squad.loc[np.flatnonzero(1 - z)].reset_index(drop=True)
    return starters, bench


def actual_lookup(panel: pd.DataFrame) -> dict:
    """Map each evaluation player--GW pair to minutes and realized points."""
    return {
        (int(row.GW), str(row.name)): (float(row.minutes), float(row.total_points))
        for row in panel.itertuples(index=False)
        if TARGET_GW <= int(row.GW) <= FINAL_GW
    }


def partial_formation_feasible(position_counts: dict, final_size: int) -> bool:
    """Check whether a partially filled XI can still reach a legal formation."""
    empty_slots = 11 - int(final_size)
    for position, (minimum, maximum) in XI_BOUNDS.items():
        count = position_counts.get(position, 0)
        if count > maximum or count + empty_slots < minimum:
            return False
    return True


def apply_autosubs(planned_xi, ordered_bench, gw: int, lookup: dict):
    """Apply goalkeeper and outfield automatic substitutions in bench order."""
    def played(name):
        return lookup.get((int(gw), str(name)), (0.0, 0.0))[0] > 0

    playing = planned_xi.loc[planned_xi["name"].map(played)].copy()
    absent = planned_xi.loc[~planned_xi["name"].map(played)].copy()
    substitutes: list[str] = []

    if not any(playing["position"] == "GK"):
        reserve_gk = ordered_bench.loc[
            (ordered_bench["position"] == "GK") & ordered_bench["name"].map(played)
        ]
        if not reserve_gk.empty:
            chosen = reserve_gk.iloc[[0]]
            playing = pd.concat([playing, chosen], ignore_index=True)
            substitutes.append(str(chosen.iloc[0]["name"]))

    playable_outfield = ordered_bench.loc[
        (ordered_bench["position"] != "GK") & ordered_bench["name"].map(played)
    ].reset_index(drop=True)
    absent_outfield = int((absent["position"] != "GK").sum())
    chosen_combo = ()
    for k in range(min(absent_outfield, len(playable_outfield)), -1, -1):
        feasible = []
        for combo in combinations(range(len(playable_outfield)), k):
            additions = playable_outfield.iloc[list(combo)] if combo else playable_outfield.iloc[[]]
            candidate = pd.concat([playing, additions], ignore_index=True)
            if partial_formation_feasible(candidate["position"].value_counts().to_dict(), len(candidate)):
                feasible.append(combo)
        if feasible:
            chosen_combo = min(feasible)
            break
    if chosen_combo:
        additions = playable_outfield.iloc[list(chosen_combo)]
        playing = pd.concat([playing, additions], ignore_index=True)
        substitutes.extend(additions["name"].astype(str).tolist())
    return playing.drop_duplicates("name"), absent, substitutes


def evaluate_regularization_solution(
    panel: pd.DataFrame,
    solution: dict,
    model_name: str,
    fitted_models: dict,
    policy: str,
) -> pd.DataFrame:
    """Evaluate one model under either a static or fixed-squad sequential policy."""
    lookup = actual_lookup(panel)
    records = []
    for gw in range(TARGET_GW, FINAL_GW + 1):
        if policy == "static":
            planned_xi = solution["starters"].copy()
            bench = solution["bench"].copy()
        elif policy == "sequential":
            weekly_snapshot = snapshot_at_cutoff(panel, gw)
            weekly_scores = score_snapshot(weekly_snapshot, model_name, fitted_models)
            score_map = weekly_scores.set_index("name")["cost_vector"]
            planned_xi, bench = solve_fixed_squad_lineup(solution, score_map)
        else:
            raise ValueError("policy must be static or sequential")

        outfield = bench.loc[bench["position"] != "GK"].sort_values(
            ["cost_vector", "name"], ascending=[False, True]
        )
        reserve_gk = bench.loc[bench["position"] == "GK"].sort_values(
            ["cost_vector", "name"], ascending=[False, True]
        )
        ordered_bench = pd.concat([outfield, reserve_gk], ignore_index=True)
        final_players, absent, substitutes = apply_autosubs(
            planned_xi, ordered_bench, gw, lookup
        )
        point_map = {
            str(name): lookup.get((gw, str(name)), (0.0, 0.0))[1]
            for name in final_players["name"]
        }
        base_points = float(sum(point_map.values()))
        captain_used = None
        captain_bonus = 0.0
        if solution["captain"] in point_map:
            captain_used = solution["captain"]
            captain_bonus = point_map[captain_used]
        elif solution["vice_captain"] in point_map:
            captain_used = solution["vice_captain"]
            captain_bonus = point_map[captain_used]
        records.append(
            {
                "model": model_name,
                "policy": policy,
                "GW": gw,
                "weekly_points": base_points + captain_bonus,
                "base_points": base_points,
                "captain_bonus": captain_bonus,
                "captain_used": captain_used,
                "planned_absences": int(len(absent)),
                "autosubs_used": int(len(substitutes)),
                "substitute_points": float(sum(point_map.get(name, 0.0) for name in substitutes)),
                "squad_members": "|".join(sorted(solution["squad"]["name"].astype(str))),
                "planned_xi_members": "|".join(sorted(planned_xi["name"].astype(str))),
                "bench_order": "|".join(ordered_bench["name"].astype(str)),
                "captain": solution["captain"],
                "vice_captain": solution["vice_captain"],
            }
        )
    out = pd.DataFrame(records)
    out["cumulative_points"] = out["weekly_points"].cumsum()
    return out


# ---------------------------------------------------------------------------
# Position-stratified player-cluster bootstrap
# ---------------------------------------------------------------------------
def bootstrap_indices(train_panel: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Sample players with replacement within position, retaining full histories."""
    index_chunks = []
    for position in POSITIONS:
        part = train_panel.loc[train_panel["position"] == position]
        players = part["name"].unique()
        by_player = {name: part.index[part["name"] == name].to_numpy() for name in players}
        sampled = rng.choice(players, size=len(players), replace=True)
        index_chunks.extend(by_player[name] for name in sampled)
    return np.concatenate(index_chunks)


def fit_with_fixed_alphas(
    boot_panel: pd.DataFrame,
    selected_alphas: dict[str, dict[str, float]],
) -> dict[str, dict[str, Pipeline]]:
    """Fit one bootstrap replicate using training-selected penalties unchanged."""
    fitted: dict[str, dict[str, Pipeline]] = {}
    for model_name, spec in MODEL_SPECS.items():
        fitted[model_name] = {}
        features = [f"lag_{feature}" for feature in spec["features"]]
        for position in POSITIONS:
            rows = boot_panel.loc[boot_panel["position"] == position]
            pipeline = make_pipeline(spec["kind"], selected_alphas[model_name][position])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(rows[features], rows["total_points"])
            fitted[model_name][position] = pipeline
    return fitted


def run_bootstrap(
    train_panel: pd.DataFrame,
    snapshot: pd.DataFrame,
    reference_scores: dict[str, pd.DataFrame],
    reference_solutions: dict[str, dict],
    selected_alphas: dict[str, dict[str, float]],
    n_bootstrap: int,
    seed: int,
):
    """Estimate prediction, coefficient, and selected-XI stability by bootstrap."""
    rng = np.random.default_rng(seed)
    diagnostic_rows = []
    coefficient_rows = []
    accumulators = {
        model: {
            "sum": np.zeros(len(snapshot), dtype=float),
            "sum_sq": np.zeros(len(snapshot), dtype=float),
        }
        for model in MODEL_SPECS
    }

    for replication in range(1, n_bootstrap + 1):
        sampled_indices = bootstrap_indices(train_panel, rng)
        boot_panel = train_panel.loc[sampled_indices].copy()
        boot_models = fit_with_fixed_alphas(boot_panel, selected_alphas)

        for model_name, spec in MODEL_SPECS.items():
            scored = score_snapshot(snapshot, model_name, boot_models)
            values = scored["cost_vector"].to_numpy(float)
            reference = reference_scores[model_name]["cost_vector"].to_numpy(float)
            accumulators[model_name]["sum"] += values
            accumulators[model_name]["sum_sq"] += values**2

            boot_solution = solve_joint_lexicographic(
                scored, f"bootstrap {replication}, {model_name}"
            )
            ref_solution = reference_solutions[model_name]
            top_ref = set(reference_scores[model_name].nlargest(15, "cost_vector")["name"])
            top_boot = set(scored.nlargest(15, "cost_vector")["name"])
            normalized_rmse = float(
                np.sqrt(np.mean((values - reference) ** 2))
                / max(np.std(reference), 1e-12)
            )
            diagnostic_rows.append(
                {
                    "bootstrap": replication,
                    "model": model_name,
                    "cost_spearman": safe_spearman(values, reference),
                    "normalized_cost_rmse": normalized_rmse,
                    "top15_jaccard": jaccard(top_boot, top_ref),
                    "squad_jaccard": jaccard(
                        boot_solution["squad"]["name"], ref_solution["squad"]["name"]
                    ),
                    "xi_jaccard": jaccard(
                        boot_solution["starters"]["name"], ref_solution["starters"]["name"]
                    ),
                    "captain_same": bool(
                        boot_solution["captain"] == ref_solution["captain"]
                    ),
                }
            )

            for position in POSITIONS:
                coefficients = boot_models[model_name][position].named_steps["regressor"].coef_
                for feature, value in zip(spec["features"], coefficients):
                    coefficient_rows.append(
                        {
                            "bootstrap": replication,
                            "model": model_name,
                            "position": position,
                            "feature": feature,
                            "feature_label": FEATURE_SHORT[feature],
                            "standardized_coefficient": float(value),
                            "selected_nonzero": bool(abs(value) > 1e-10),
                        }
                    )
        if replication == 1 or replication % max(1, n_bootstrap // 10) == 0:
            log(f"Bootstrap progress: {replication}/{n_bootstrap}")

    diagnostics = pd.DataFrame(diagnostic_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    player_stability_rows = []
    for model_name in MODEL_SPECS:
        mean = accumulators[model_name]["sum"] / n_bootstrap
        variance = np.maximum(
            accumulators[model_name]["sum_sq"] / n_bootstrap - mean**2, 0.0
        )
        sd = np.sqrt(variance)
        reference = reference_scores[model_name]["cost_vector"].to_numpy(float)
        for index, row in snapshot.iterrows():
            player_stability_rows.append(
                {
                    "model": model_name,
                    "name": row["name"],
                    "position": row["position"],
                    "reference_cost": reference[index],
                    "bootstrap_mean_cost": mean[index],
                    "bootstrap_sd_cost": sd[index],
                    "absolute_cv": sd[index] / max(abs(mean[index]), 1e-8),
                }
            )
    return diagnostics, coefficients, pd.DataFrame(player_stability_rows)


# ---------------------------------------------------------------------------
# Regularization and feature-stability output tables and diagnostic figures
# ---------------------------------------------------------------------------
def regularization_figures(
    output_dir: Path,
    feature_correlations: pd.DataFrame,
    predictive_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    weekly_scores: pd.DataFrame,
) -> None:
    """Create diagnostic figures for redundancy, stability, and LASSO selection."""
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # Corrected feature-redundancy figure: xGI is almost exactly xG+xA, while
    # the other pairwise relationships are reported rather than overstated.
    labels = [FEATURE_SHORT[column] for column in FULL_FEATURES]
    matrix = feature_correlations.loc[FULL_FEATURES, FULL_FEATURES].to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if abs(matrix[i, j]) > 0.55 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, shrink=0.82, label="Pearson correlation")
    fig.tight_layout()
    save_pdf(fig, figures / "reviewer1_comment3_feature_correlations.pdf")

    order = list(MODEL_SPECS)
    metrics = predictive_metrics.set_index("model").reindex(order)
    stability = bootstrap_summary.set_index("model").reindex(order)
    x = np.arange(len(order))
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    bars0 = axes[0].bar(x, metrics["test_rmse"], color=colors)
    axes[0].bar_label(bars0, fmt="%.3f", fontsize=7, padding=2)
    axes[0].set_ylabel("RMSE (future mean points)")
    axes[0].grid(axis="y", alpha=0.25)

    bars1 = axes[1].bar(
        x,
        stability["cost_spearman_mean"],
        yerr=stability["cost_spearman_sd"],
        color=colors,
        capsize=3,
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Spearman correlation")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].bar_label(bars1, fmt="%.3f", fontsize=7, padding=2)

    bars2 = axes[2].bar(
        x,
        stability["xi_jaccard_mean"],
        yerr=stability["xi_jaccard_sd"],
        color=colors,
        capsize=3,
    )
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("Jaccard similarity")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].bar_label(bars2, fmt="%.3f", fontsize=7, padding=2)
    for ax in axes:
        ax.set_xticks(x, labels=order, rotation=28, ha="right")
    fig.tight_layout()
    save_pdf(fig, figures / "reviewer1_comments3_4_performance_stability.pdf")

    lasso = coefficient_summary.loc[coefficient_summary["model"] == "Full LASSO"]
    pivot = lasso.pivot(index="position", columns="feature_label", values="selection_frequency")
    pivot = pivot.reindex(index=POSITIONS, columns=[FEATURE_SHORT[f] for f in FULL_FEATURES])
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(pivot.shape[0]), labels=pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 0.6 else "black",
            )
    fig.colorbar(image, ax=ax, label="Selection frequency", shrink=0.8)
    fig.tight_layout()
    save_pdf(fig, figures / "reviewer1_comment4_lasso_selection.pdf")

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    for (model, policy), group in weekly_scores.groupby(["model", "policy"], sort=False):
        if policy != "sequential":
            continue
        ax.plot(
            group["GW"],
            group["cumulative_points"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=model,
        )
    ax.set_xticks(range(TARGET_GW, FINAL_GW + 1))
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative realized points")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               frameon=True, ncol=2, fontsize=8)
    fig.subplots_adjust(bottom=0.22)
    save_pdf(fig, figures / "reviewer1_comments3_4_cumulative_scores.pdf")


def run_regularization_experiment(
    data_path: Path,
    output_dir: Path,
    bootstrap_replications: int,
    seed: int,
) -> dict:
    """Run the complete leakage-safe regularization and stability experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_player_gameweek(data_path)
    lagged = add_lagged_summaries(panel)
    train_panel = lagged.loc[
        (lagged["GW"] < TARGET_GW) & (lagged["history_count"] >= MIN_HISTORY)
    ].copy()
    lag_columns = [f"lag_{feature}" for feature in [*FULL_FEATURES, "total_points"]]
    train_panel = train_panel.dropna(subset=lag_columns + ["total_points", "position"])
    snapshot = snapshot_at_cutoff(panel, TARGET_GW)
    outcomes = future_player_outcomes(panel, snapshot)

    feature_correlations = panel.loc[panel["GW"] < TARGET_GW, FULL_FEATURES].corr()
    feature_correlations.to_csv(output_dir / "reviewer1_comment3_feature_correlations.csv")
    identity_error = (
        panel.loc[panel["GW"] < TARGET_GW, "expected_goal_involvements"]
        - panel.loc[panel["GW"] < TARGET_GW, "expected_goals"]
        - panel.loc[panel["GW"] < TARGET_GW, "expected_assists"]
    ).abs()
    redundancy_audit = pd.DataFrame(
        [
            {
                "relationship": "xGI = xG + xA",
                "training_rows": int(len(identity_error)),
                "exact_or_rounding_equal_share": float((identity_error <= 0.01 + 1e-12).mean()),
                "exact_equal_share": float((identity_error <= 1e-12).mean()),
                "maximum_absolute_difference": float(identity_error.max()),
            }
        ]
    )
    redundancy_audit.to_csv(output_dir / "reviewer1_comment3_redundancy_audit.csv", index=False)

    fitted, selected_alphas, tuning, reference_coefficients = fit_reference_models(train_panel)
    tuning.to_csv(output_dir / "reviewer1_comments3_4_alpha_tuning.csv", index=False)
    reference_coefficients.to_csv(
        output_dir / "reviewer1_comments3_4_reference_coefficients.csv", index=False
    )

    reference_scores: dict[str, pd.DataFrame] = {}
    reference_solutions: dict[str, dict] = {}
    prediction_rows = []
    metric_rows = []
    roster_rows = []
    weekly_frames = []

    outcome_map = outcomes.set_index("name")
    for model_name in MODEL_SPECS:
        scored = score_snapshot(snapshot, model_name, fitted)
        scored = scored.join(
            outcome_map[["future_total_points", "future_mean_points", "future_appearances"]],
            on="name",
        )
        reference_scores[model_name] = scored
        prediction_rows.append(scored)
        metric_rows.append(
            {
                "model": model_name,
                "n_features": len(MODEL_SPECS[model_name]["features"]),
                "test_mae": mean_absolute_error(
                    scored["future_mean_points"], scored["cost_vector"]
                ),
                "test_rmse": math.sqrt(
                    mean_squared_error(scored["future_mean_points"], scored["cost_vector"])
                ),
                "test_spearman": safe_spearman(
                    scored["future_mean_points"], scored["cost_vector"]
                ),
            }
        )
        solution = solve_joint_lexicographic(scored, model_name)
        reference_solutions[model_name] = solution
        starters = set(solution["starters"]["name"])
        for row in solution["squad"].itertuples(index=False):
            roster_rows.append(
                {
                    "model": model_name,
                    "name": row.name,
                    "team": row.team,
                    "position": row.position,
                    "value": row.value,
                    "cost_vector": row.cost_vector,
                    "role": "XI" if row.name in starters else "Bench",
                    "is_captain": row.name == solution["captain"],
                    "is_vice_captain": row.name == solution["vice_captain"],
                }
            )
        for policy in ["static", "sequential"]:
            weekly_frames.append(
                evaluate_regularization_solution(panel, solution, model_name, fitted, policy)
            )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictive_metrics = pd.DataFrame(metric_rows)
    rosters = pd.DataFrame(roster_rows)
    weekly_scores = pd.concat(weekly_frames, ignore_index=True)
    predictions.to_csv(output_dir / "reviewer1_comments3_4_cost_vectors.csv", index=False)
    predictive_metrics.to_csv(
        output_dir / "reviewer1_comments3_4_predictive_metrics.csv", index=False
    )
    rosters.to_csv(output_dir / "reviewer1_comments3_4_rosters.csv", index=False)
    weekly_scores.to_csv(output_dir / "reviewer1_comments3_4_weekly_scores.csv", index=False)

    log(f"Running {bootstrap_replications} clustered bootstrap replications ...")
    diagnostics, boot_coefficients, player_stability = run_bootstrap(
        train_panel,
        snapshot,
        reference_scores,
        reference_solutions,
        selected_alphas,
        bootstrap_replications,
        seed,
    )
    diagnostics.to_csv(
        output_dir / "reviewer1_comments3_4_bootstrap_diagnostics.csv", index=False
    )
    boot_coefficients.to_csv(
        output_dir / "reviewer1_comments3_4_bootstrap_coefficients.csv", index=False
    )
    player_stability.to_csv(
        output_dir / "reviewer1_comment3_player_cost_stability.csv", index=False
    )

    bootstrap_summary = (
        diagnostics.groupby("model", as_index=False)
        .agg(
            cost_spearman_mean=("cost_spearman", "mean"),
            cost_spearman_sd=("cost_spearman", "std"),
            normalized_cost_rmse_mean=("normalized_cost_rmse", "mean"),
            normalized_cost_rmse_sd=("normalized_cost_rmse", "std"),
            top15_jaccard_mean=("top15_jaccard", "mean"),
            squad_jaccard_mean=("squad_jaccard", "mean"),
            squad_jaccard_sd=("squad_jaccard", "std"),
            xi_jaccard_mean=("xi_jaccard", "mean"),
            xi_jaccard_sd=("xi_jaccard", "std"),
            captain_stability=("captain_same", "mean"),
        )
    )
    bootstrap_summary.to_csv(
        output_dir / "reviewer1_comment3_cost_vector_stability.csv", index=False
    )

    coefficient_summary = (
        boot_coefficients.groupby(
            ["model", "position", "feature", "feature_label"], as_index=False
        )
        .agg(
            coefficient_mean=("standardized_coefficient", "mean"),
            coefficient_sd=("standardized_coefficient", "std"),
            selection_frequency=("selected_nonzero", "mean"),
        )
    )
    coefficient_summary["sign_consistency"] = coefficient_summary.apply(
        lambda row: float(
            (
                np.sign(
                    boot_coefficients.loc[
                        (boot_coefficients["model"] == row["model"])
                        & (boot_coefficients["position"] == row["position"])
                        & (boot_coefficients["feature"] == row["feature"]),
                        "standardized_coefficient",
                    ]
                )
                == np.sign(row["coefficient_mean"])
            ).mean()
        ),
        axis=1,
    )
    coefficient_summary.to_csv(
        output_dir / "reviewer1_comments3_4_coefficient_stability.csv", index=False
    )
    coefficient_summary.loc[coefficient_summary["model"] == "Full LASSO"].to_csv(
        output_dir / "reviewer1_comment4_lasso_feature_selection.csv", index=False
    )

    realized_summary = (
        weekly_scores.groupby(["model", "policy"], as_index=False)
        .agg(
            realized_points=("weekly_points", "sum"),
            mean_weekly_points=("weekly_points", "mean"),
            total_autosubs=("autosubs_used", "sum"),
            substitute_points=("substitute_points", "sum"),
        )
    )
    main_comparison = (
        predictive_metrics.merge(bootstrap_summary, on="model", how="left")
        .merge(
            realized_summary.pivot(index="model", columns="policy", values="realized_points")
            .rename(columns={"static": "static_points", "sequential": "sequential_points"})
            .reset_index(),
            on="model",
            how="left",
        )
    )
    for position in POSITIONS:
        main_comparison[f"alpha_{position}"] = main_comparison["model"].map(
            {model: selected_alphas[model][position] for model in MODEL_SPECS}
        )
    main_comparison.to_csv(
        output_dir / "reviewer1_comments3_4_main_comparison.csv", index=False
    )

    # Audit the regularization experiment's fixed-squad sequential evaluator too.
    audit_rows = []
    for (model, policy), group in weekly_scores.groupby(["model", "policy"]):
        audit_rows.append(
            {
                "model": model,
                "policy": policy,
                "unique_squad_memberships": int(group["squad_members"].nunique()),
                "external_transfers": 0,
                "transfer_hit_points": 0,
                "unique_captains": int(group["captain"].nunique()),
                "unique_vice_captains": int(group["vice_captain"].nunique()),
                "unique_starting_lineups": int(group["planned_xi_members"].nunique()),
                "unique_bench_orders": int(group["bench_order"].nunique()),
                "audit_pass": bool(
                    group["squad_members"].nunique() == 1
                    and group["captain"].nunique() == 1
                    and group["vice_captain"].nunique() == 1
                ),
            }
        )
    policy_audit = pd.DataFrame(audit_rows)
    if not policy_audit["audit_pass"].all():
        raise AssertionError("Regularization fixed-squad audit failed.")
    policy_audit.to_csv(
        output_dir / "reviewer1_comment1_regularization_policy_audit.csv", index=False
    )

    regularization_figures(
        output_dir,
        feature_correlations,
        predictive_metrics,
        bootstrap_summary,
        coefficient_summary,
        weekly_scores,
    )

    return {
        "main_comparison": main_comparison,
        "policy_audit": policy_audit,
        "weekly_scores": weekly_scores,
        "feature_correlations": feature_correlations,
    }


# ---------------------------------------------------------------------------
# Reproducibility packaging and main entry point
# ---------------------------------------------------------------------------
def run_roster_design_analysis(args: argparse.Namespace, output_dir: Path) -> None:
    """Launch the companion roster-design analysis as a subprocess."""
    script = args.roster_design_script.expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Roster-design runner not found: {script}")
    command = [
        sys.executable,
        "-u",
        str(script),
        "--data",
        str(args.data),
        "--output",
        str(output_dir),
        "--workers",
        str(args.workers),
    ]
    if args.smoke_test:
        command.append("--smoke-test")
    log("Running roster-design, Pareto, and availability analyses ...")
    subprocess.run(command, check=True)


def create_manifest(output_dir: Path) -> pd.DataFrame:
    """Hash every generated artifact and write a machine-readable manifest."""
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "output_manifest.csv":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "relative_path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "output_manifest.csv", index=False)
    return manifest


def write_reproduction_readme(output_dir: Path, args: argparse.Namespace) -> None:
    """Write standalone server instructions and data-leakage safeguards."""
    text = f"""# FPL computational workflow

## Full server run

```bash
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
python -u fpl_regularization_experiments.py \\
  --data merged_gw_2324.csv \\
  --output fpl_analysis_outputs_server \\
  --bootstrap {args.bootstrap} \\
  --workers {args.workers} \\
  > fpl_analysis_run.log 2>&1 &
```

Monitor with `tail -f fpl_analysis_run.log` and check the process with
`pgrep -af "[f]pl_regularization_experiments.py"`.  Successful completion creates
`RUN_COMPLETED_SUCCESSFULLY.txt`.

## Data split and leakage controls

* Model development and hyperparameter selection: GW1--26 only.
* Final evaluation: GW27--38 only.
* Ridge/LASSO tuning: rolling-origin validation folds ending no later than GW26.
* Bootstrap: position-stratified resampling of players, preserving each player's
  full training history.
* No budget, regularization parameter, Pareto point, or model is selected using
  realized GW27--38 points.

## Principal outputs

* `roster_design/reviewer1_comment1_policy_audit.csv`
* `roster_design/reviewer1_comment2_main_comparison.csv`
* `roster_design/reviewer1_comment2_pareto_frontier.csv`
* `roster_design/reviewer1_comment5_availability_comparison.csv`
* `regularization/reviewer1_comments3_4_main_comparison.csv`
* `regularization/reviewer1_comment3_cost_vector_stability.csv`
* `regularization/reviewer1_comment4_lasso_feature_selection.csv`
* diagnostic vector PDFs beneath `regularization/figures/`; the final indexed
  publication PDFs are created by `run_fpl_paper_pipeline.py`
* `run_config.json`, `software_environment.txt`, and `output_manifest.csv`
"""
    (output_dir / "README_reproduction.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the analyses, export metadata, and create the success marker."""
    args = parse_args()
    start = time.perf_counter()
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    if args.bootstrap < 2:
        raise ValueError("--bootstrap must be at least 2.")
    bootstrap_replications = 4 if args.smoke_test else args.bootstrap

    args.output.mkdir(parents=True, exist_ok=True)
    roster_design_dir = args.output / "roster_design"
    regularization_dir = args.output / "regularization"

    config = {
        "data": str(args.data),
        "output": str(args.output),
        "target_gw": TARGET_GW,
        "final_gw": FINAL_GW,
        "ewm_span": EWM_SPAN,
        "history_weight": HISTORY_WEIGHT,
        "model_weight": MODEL_WEIGHT,
        "bootstrap_replications": bootstrap_replications,
        "seed": args.seed,
        "workers": args.workers,
        "skip_roster_design": args.skip_roster_design,
        "smoke_test": args.smoke_test,
        "model_specs": MODEL_SPECS,
        "versions": package_versions(),
    }
    write_json(args.output / "run_config.json", config)
    (args.output / "software_environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in package_versions().items()) + "\n",
        encoding="utf-8",
    )

    log("=" * 78)
    log("FPL computational analysis workflow")
    log(f"Data:       {args.data}")
    log(f"Output:     {args.output}")
    log(f"Bootstraps: {bootstrap_replications}")
    log("=" * 78)

    if not args.skip_roster_design:
        run_roster_design_analysis(args, roster_design_dir)
        audit = roster_design_dir / "reviewer1_comment1_policy_audit.csv"
        if not audit.is_file():
            raise RuntimeError(
                "The roster-design runner completed without its fixed-squad policy audit. "
                "Use the bundled roster-design runner."
            )
        shutil.copy2(audit, args.output / "reviewer1_comment1_policy_audit.csv")

    run_regularization_experiment(
        args.data,
        regularization_dir,
        bootstrap_replications,
        args.seed,
    )

    write_reproduction_readme(args.output, args)
    elapsed = time.perf_counter() - start
    (args.output / "RUN_COMPLETED_SUCCESSFULLY.txt").write_text(
        f"Experiment completed successfully in {elapsed / 60:.2f} minutes.\n",
        encoding="utf-8",
    )
    manifest = create_manifest(args.output)
    log("EXPERIMENT COMPLETED SUCCESSFULLY")
    log(f"Generated {len(manifest)} auditable files in {args.output}")
    log(f"Total elapsed time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
