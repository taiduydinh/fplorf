#!/usr/bin/env python3
"""Recompute every internal forecasting-family result from the raw FPL panel.

This runner is the provenance layer for the retained manuscript figures.  It
does not read ``Results.csv``, ``Results(1).csv``, or a saved roster snapshot.
All 94 internal GW27--38 series are rebuilt from ``merged_gw_2324.csv``.

The four Santoro series are intentionally excluded: they are externally
supplied benchmark observations and are added by the publication pipeline with
an explicit external-source flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except ImportError as exc:  # pragma: no cover
    raise SystemExit("SciPy with scipy.optimize.milp is required.") from exc

try:
    import sklearn
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required for the hybrid Ridge models.") from exc


TARGET_GW = 27
FINAL_GW = 38
TOTAL_BUDGET = 100.0
DEFAULT_XI_CAP = 83.5
SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_BOUNDS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3
TIE_EPS = 1e-8
FEATURES = [
    "starts",
    "ict_index",
    "expected_goals_conceded",
    "selected",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
]
ARIMA_ORDERS = [
    (0, 0, 1),
    (1, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
]


def parse_args() -> argparse.Namespace:
    """Parse raw-data, solver, simulation, and output options."""
    parser = argparse.ArgumentParser(
        description="Recompute all internal FPL manuscript results from raw data."
    )
    parser.add_argument("--data", type=Path, default=Path("merged_gw_2324.csv"))
    parser.add_argument(
        "--roster-design-dir",
        type=Path,
        required=True,
        help="Directory containing the controlled roster-design weekly scores.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("full_internal_experiments")
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    parser.add_argument("--simulation-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--mip-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Layout/dependency test only; time-series fits use fast mean fallbacks.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for provenance records."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    """Derive a repeatable sub-seed from a base seed and task identifiers."""
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def arima_label(order: tuple[int, int, int]) -> str:
    """Format an ARIMA order consistently with table and figure labels."""
    return f"ARIMA ({order[0]},{order[1]},{order[2]})"


def load_panel(path: Path) -> pd.DataFrame:
    """Load, validate, type-convert, and order the raw player--GW panel."""
    required = {
        "name",
        "position",
        "team",
        "value",
        "GW",
        "total_points",
        "minutes",
        *FEATURES,
    }
    data = pd.read_csv(path)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data = data.copy()
    data["GW"] = pd.to_numeric(data["GW"], errors="raise").astype(int)
    for column in ["value", "total_points", "minutes", *FEATURES]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    if data["value"].median(skipna=True) > 20:
        data["value"] /= 10.0
    data = data.sort_values(["name", "GW"], kind="stable").reset_index(drop=True)
    return data


def latest_snapshot(train: pd.DataFrame) -> pd.DataFrame:
    """Return one pre-evaluation metadata row for each candidate player."""
    snapshot = (
        train.sort_values(["name", "GW"], kind="stable")
        .groupby("name", as_index=False)
        .tail(1)[["name", "team", "position", "value"]]
        .dropna()
        .drop_duplicates("name")
        .sort_values("name", kind="stable")
        .reset_index(drop=True)
    )
    unknown = set(snapshot["position"]) - set(SQUAD_QUOTAS)
    if unknown:
        raise ValueError(f"Unexpected positions: {sorted(unknown)}")
    return snapshot


def minmax(values: np.ndarray) -> np.ndarray:
    """Scale finite values to [0, 1], handling constant and missing vectors."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    clean = values.copy()
    clean[~finite] = np.nanmedian(clean[finite])
    lo, hi = float(clean.min()), float(clean.max())
    if np.isclose(lo, hi):
        return np.zeros_like(clean)
    return (clean - lo) / (hi - lo)


def weighted_mean(values: np.ndarray) -> float:
    """Compute a linearly recency-weighted historical mean."""
    if len(values) == 0:
        return 0.0
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.average(values, weights=weights))


def forecast_scalar(
    values: np.ndarray,
    kind: str,
    *,
    horizon: int,
    seed: int,
    draws: int,
    arima_order: tuple[int, int, int] | None = None,
    smoke_test: bool = False,
) -> float:
    """Forecast one player's points using the requested univariate method."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    if kind == "simple":
        estimate = float(values.mean())
    elif kind == "weighted":
        estimate = weighted_mean(values)
    elif kind == "bootstrap":
        rng = np.random.default_rng(seed)
        estimate = float(rng.choice(values, size=(draws, max(1, horizon)), replace=True).mean())
    elif kind == "monte_carlo":
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rng = np.random.default_rng(seed)
        estimate = float(rng.normal(mean, std, size=(draws, max(1, horizon))).mean())
    elif kind == "linear":
        if len(values) < 2:
            estimate = float(values.mean())
        else:
            x = np.arange(1, len(values) + 1, dtype=float)
            slope, intercept = np.polyfit(x, values, 1)
            future = np.arange(len(values) + 1, len(values) + horizon + 1, dtype=float)
            estimate = float(np.mean(intercept + slope * future))
    elif kind in {"exponential", "arima"}:
        if smoke_test:
            estimate = float(values.mean())
        else:
            try:
                if kind == "exponential":
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing

                    if len(values) < 2:
                        estimate = float(values.mean())
                    else:
                        fitted = ExponentialSmoothing(
                            values,
                            trend="add",
                            seasonal=None,
                            initialization_method="estimated",
                        ).fit(optimized=True)
                        estimate = float(np.asarray(fitted.forecast(max(1, horizon))).mean())
                else:
                    from statsmodels.tsa.arima.model import ARIMA

                    if len(values) < 2 or np.isclose(values.mean(), 0.0):
                        estimate = float(values.mean())
                    else:
                        fitted = ARIMA(values, order=tuple(arima_order)).fit()
                        estimate = float(np.asarray(fitted.forecast(max(1, horizon))).mean())
            except Exception:
                estimate = float(values.mean())
    else:
        raise ValueError(f"Unknown forecast kind: {kind}")
    return float(estimate) if np.isfinite(estimate) else 0.0


def build_point_forecasts(
    histories: dict[str, pd.DataFrame],
    snapshot: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.Series]:
    """Build every point-forecast table required by the internal method screen."""
    specifications: list[tuple[str, str, tuple[int, int, int] | None]] = [
        ("simple", "simple", None),
        ("weighted", "weighted", None),
        ("bootstrap", "bootstrap", None),
        ("monte_carlo", "monte_carlo", None),
        ("exponential", "exponential", None),
        ("linear", "linear", None),
    ] + [(f"arima{p}{d}{q}", "arima", (p, d, q)) for p, d, q in ARIMA_ORDERS]
    names = snapshot["name"].astype(str).tolist()
    output: dict[str, pd.Series] = {}

    for key, kind, order in specifications:
        def task(name: str) -> float:
            values = histories[name]["total_points"].to_numpy(float)
            return forecast_scalar(
                values,
                kind,
                horizon=FINAL_GW - TARGET_GW + 1,
                seed=stable_seed(args.seed, key, name),
                draws=args.simulation_draws,
                arima_order=order,
                smoke_test=args.smoke_test,
            )

        if kind == "arima" and args.workers > 1 and not args.smoke_test:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                values = list(executor.map(task, names))
        else:
            values = [task(name) for name in names]
        output[key] = pd.Series(values, index=names, dtype=float)
        print(f"Built raw-data forecast coefficients: {key}", flush=True)
    return output


def build_uncertainty(
    histories: dict[str, pd.DataFrame], snapshot: pd.DataFrame, column: str
) -> pd.Series:
    """Estimate player-level forecast dispersion from pre-evaluation history."""
    values = {}
    for name, frame in histories.items():
        arr = frame[column].to_numpy(float)
        values[name] = float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan
    series = pd.Series(values, dtype=float).reindex(snapshot["name"])
    positions = snapshot.set_index("name")["position"]
    for position in SQUAD_QUOTAS:
        idx = positions.index[positions == position]
        local = series.reindex(idx)
        fallback = float(local.median(skipna=True))
        if not np.isfinite(fallback):
            fallback = float(series.median(skipna=True)) if series.notna().any() else 0.0
        series.loc[idx] = local.fillna(fallback)
    return series.fillna(0.0)


def aggregate_feature_table(
    key: str,
    histories: dict[str, pd.DataFrame],
    snapshot: pd.DataFrame,
    point_forecasts: dict[str, pd.Series],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Aggregate expected-statistics features for ICT and hybrid objectives."""
    names = snapshot["name"].astype(str).tolist()
    columns = [*FEATURES, "total_points"]
    rows: list[dict[str, float | str]] = []
    for name in names:
        frame = histories[name]
        row: dict[str, float | str] = {"name": name}
        for column in columns:
            values = frame[column].to_numpy(float)
            if key == "arima001" and column == "total_points":
                value = float(point_forecasts["arima001"].loc[name])
            elif key == "arima001":
                value = float(values.mean()) if len(values) else 0.0
            else:
                value = forecast_scalar(
                    values,
                    key,
                    horizon=FINAL_GW - TARGET_GW + 1,
                    seed=stable_seed(args.seed, "feature", key, name, column),
                    draws=args.simulation_draws,
                    smoke_test=args.smoke_test,
                )
            row[column] = value
        rows.append(row)
    return snapshot.merge(pd.DataFrame(rows), on="name", how="left", validate="one_to_one")


def hybrid_scores(table: pd.DataFrame, history_weight: float, ridge_weight: float) -> pd.Series:
    """Combine recent points and Ridge predictions on a common normalized scale."""
    output = pd.Series(index=table["name"].astype(str), dtype=float)
    for position in SQUAD_QUOTAS:
        subset = table.loc[table["position"] == position].copy()
        if subset.empty:
            continue
        x = subset[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        y = subset["total_points"].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(x_scaled, y)
        prediction = ridge.predict(x_scaled)
        combined = (
            history_weight * minmax(y) + ridge_weight * minmax(prediction)
        ) / (history_weight + ridge_weight)
        scale = max(1.0, float(np.nanmax(np.abs(y))))
        output.loc[subset["name"].astype(str)] = combined * scale
    return output.reindex(table["name"].astype(str)).fillna(0.0)


class MilpRows:
    """Sparse linear-constraint builder for SciPy's mixed-integer solver."""
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
            if value != 0:
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


def stable_weights(pool: pd.DataFrame) -> np.ndarray:
    """Create tiny deterministic tie-break weights ordered by player name."""
    rank = pd.Series(pool["name"]).rank(method="first", ascending=True).to_numpy()
    return (len(pool) + 1.0 - rank) / max(1.0, len(pool))


def solve_binary(
    objective: np.ndarray,
    rows: MilpRows,
    label: str,
    args: argparse.Namespace,
) -> np.ndarray:
    """Solve a binary linear maximization model and validate solver success."""
    result = milp(
        c=-np.asarray(objective, dtype=float),
        integrality=np.ones(len(objective), dtype=int),
        bounds=Bounds(np.zeros(len(objective)), np.ones(len(objective))),
        constraints=rows.constraint(),
        options={"time_limit": args.mip_time_limit, "mip_rel_gap": args.mip_gap},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed for {label}: {result.status}; {result.message}")
    return np.rint(result.x).astype(int)


def validate_solution(solution: dict[str, object]) -> None:
    """Assert roster, XI, formation, club, budget, and captaincy feasibility."""
    squad = solution["squad"]
    starters = solution["starters"]
    bench = solution["bench"]
    assert isinstance(squad, pd.DataFrame)
    assert isinstance(starters, pd.DataFrame)
    assert isinstance(bench, pd.DataFrame)
    assert len(squad) == 15 and squad["name"].nunique() == 15
    assert len(starters) == 11 and starters["name"].nunique() == 11
    assert len(bench) == 4 and bench["name"].nunique() == 4
    assert set(starters["name"]).isdisjoint(set(bench["name"]))
    assert solution["captain"] in set(starters["name"])
    assert solution["vice_captain"] in set(starters["name"])
    assert float(solution["total_spend"]) <= TOTAL_BUDGET + 1e-6
    assert squad["position"].value_counts().to_dict() == SQUAD_QUOTAS
    assert squad["team"].value_counts().max() <= MAX_PER_CLUB
    counts = starters["position"].value_counts().to_dict()
    for position, (minimum, maximum) in XI_BOUNDS.items():
        assert minimum <= counts.get(position, 0) <= maximum


def solve_two_stage(
    pool: pd.DataFrame,
    xi_cap: float,
    label: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Select an XI and captain first, then fill the four legal bench slots."""
    pool = pool.reset_index(drop=True).copy()
    n = len(pool)
    objective_values = pool["objective"].to_numpy(float)
    bench_values = pool["bench_forecast"].to_numpy(float)
    prices = pool["value"].to_numpy(float)
    stable = stable_weights(pool)

    s = np.arange(n)
    y = n + np.arange(n)
    rows = MilpRows(2 * n)
    rows.add({int(s[j]): 1 for j in range(n)}, lower=11, upper=11)
    rows.add({int(s[j]): prices[j] for j in range(n)}, upper=float(xi_cap))
    rows.add({int(y[j]): 1 for j in range(n)}, lower=1, upper=1)
    for j in range(n):
        rows.add({int(y[j]): 1, int(s[j]): -1}, upper=0)
    for position, (minimum, maximum) in XI_BOUNDS.items():
        idx = np.flatnonzero(pool["position"].to_numpy() == position)
        rows.add({int(s[j]): 1 for j in idx}, lower=minimum, upper=maximum)
    for club in sorted(pool["team"].unique()):
        idx = np.flatnonzero(pool["team"].to_numpy() == club)
        rows.add({int(s[j]): 1 for j in idx}, upper=MAX_PER_CLUB)

    objective = np.zeros(2 * n)
    objective[s] = objective_values + TIE_EPS * stable
    objective[y] = objective_values + TIE_EPS * stable
    z = solve_binary(objective, rows, f"{label}: XI", args)
    starter_idx = np.flatnonzero(z[s])
    captain_idx = int(np.flatnonzero(z[y])[0])

    bench_rows = MilpRows(n)
    bench_rows.add({j: 1 for j in range(n)}, lower=4, upper=4)
    bench_rows.add(
        {j: prices[j] for j in range(n)}, upper=TOTAL_BUDGET - float(xi_cap)
    )
    for j in starter_idx:
        bench_rows.add({int(j): 1}, lower=0, upper=0)
    starter_positions = pool.loc[starter_idx, "position"].value_counts().to_dict()
    for position, quota in SQUAD_QUOTAS.items():
        requirement = quota - starter_positions.get(position, 0)
        idx = np.flatnonzero(pool["position"].to_numpy() == position)
        bench_rows.add({int(j): 1 for j in idx}, lower=requirement, upper=requirement)
    starter_clubs = pool.loc[starter_idx, "team"].value_counts().to_dict()
    for club in sorted(pool["team"].unique()):
        idx = np.flatnonzero(pool["team"].to_numpy() == club)
        bench_rows.add(
            {int(j): 1 for j in idx},
            upper=MAX_PER_CLUB - starter_clubs.get(club, 0),
        )
    bz = solve_binary(
        bench_values + TIE_EPS * stable,
        bench_rows,
        f"{label}: bench",
        args,
    )
    bench_idx = np.flatnonzero(bz)
    squad_idx = np.concatenate([starter_idx, bench_idx])
    squad = pool.loc[squad_idx].copy()
    starters = pool.loc[starter_idx].copy()
    bench = pool.loc[bench_idx].copy()
    captain = str(pool.loc[captain_idx, "name"])
    vice = str(
        starters.loc[starters["name"] != captain]
        .sort_values(["objective", "name"], ascending=[False, True])
        .iloc[0]["name"]
    )
    solution: dict[str, object] = {
        "label": label,
        "xi_cap": float(xi_cap),
        "squad": squad.reset_index(drop=True),
        "starters": starters.reset_index(drop=True),
        "bench": bench.reset_index(drop=True),
        "captain": captain,
        "vice_captain": vice,
        "xi_spend": float(prices[starter_idx].sum()),
        "bench_spend": float(prices[bench_idx].sum()),
        "total_spend": float(prices[squad_idx].sum()),
    }
    validate_solution(solution)
    return solution


def actual_lookup(test: pd.DataFrame) -> dict[tuple[int, str], tuple[float, float]]:
    """Map evaluation player--GW pairs to realized minutes and points."""
    grouped = (
        test.groupby(["GW", "name"], as_index=False)
        .agg(minutes=("minutes", "max"), total_points=("total_points", "sum"))
    )
    return {
        (int(row.GW), str(row.name)): (float(row.minutes), float(row.total_points))
        for row in grouped.itertuples(index=False)
    }


def partial_formation_feasible(counts: dict[str, int], final_size: int) -> bool:
    """Check whether a partially filled XI can still satisfy formation bounds."""
    empty = 11 - int(final_size)
    for position, (minimum, maximum) in XI_BOUNDS.items():
        count = counts.get(position, 0)
        if count > maximum or count + empty < minimum:
            return False
    return True


def apply_autosubs(
    planned_xi: pd.DataFrame,
    ordered_bench: pd.DataFrame,
    gw: int,
    lookup: dict[tuple[int, str], tuple[float, float]],
) -> tuple[pd.DataFrame, list[str]]:
    """Apply FPL-style goalkeeper and outfield substitutions in bench order."""
    planned_xi = planned_xi.reset_index(drop=True)
    ordered_bench = ordered_bench.reset_index(drop=True)

    def played(name: str) -> bool:
        return lookup.get((int(gw), str(name)), (0.0, 0.0))[0] > 0

    playing = planned_xi.loc[planned_xi["name"].map(played)].copy()
    absent = planned_xi.loc[~planned_xi["name"].map(played)].copy()
    substitutes: list[str] = []
    if not any(playing["position"] == "GK"):
        reserve_gk = ordered_bench.loc[
            (ordered_bench["position"] == "GK")
            & ordered_bench["name"].map(played)
        ]
        if not reserve_gk.empty:
            replacement = reserve_gk.iloc[[0]]
            playing = pd.concat([playing, replacement], ignore_index=True)
            substitutes.append(str(replacement.iloc[0]["name"]))

    outfield = ordered_bench.loc[
        (ordered_bench["position"] != "GK") & ordered_bench["name"].map(played)
    ].reset_index(drop=True)
    absent_outfield = int((absent["position"] != "GK").sum())
    chosen: tuple[int, ...] = ()
    for k in range(min(absent_outfield, len(outfield)), -1, -1):
        feasible = []
        for combo in combinations(range(len(outfield)), k):
            additions = outfield.iloc[list(combo)] if combo else outfield.iloc[[]]
            candidate = pd.concat([playing, additions], ignore_index=True)
            if partial_formation_feasible(
                candidate["position"].value_counts().to_dict(), len(candidate)
            ):
                feasible.append(combo)
        if feasible:
            chosen = min(feasible)
            break
    if chosen:
        additions = outfield.iloc[list(chosen)]
        playing = pd.concat([playing, additions], ignore_index=True)
        substitutes.extend(additions["name"].astype(str).tolist())
    return playing.drop_duplicates("name"), substitutes


def evaluate_static(
    solution: dict[str, object],
    lookup: dict[tuple[int, str], tuple[float, float]],
) -> pd.DataFrame:
    """Evaluate a fixed GW27 roster and lineup over the full test horizon."""
    starters = solution["starters"]
    bench = solution["bench"]
    assert isinstance(starters, pd.DataFrame)
    assert isinstance(bench, pd.DataFrame)
    outfield = bench.loc[bench["position"] != "GK"].sort_values(
        ["bench_forecast", "name"], ascending=[False, True]
    )
    goalkeeper = bench.loc[bench["position"] == "GK"].sort_values(
        ["bench_forecast", "name"], ascending=[False, True]
    )
    ordered_bench = pd.concat([outfield, goalkeeper], ignore_index=True)
    records = []
    for gw in range(TARGET_GW, FINAL_GW + 1):
        final_players, substitutes = apply_autosubs(starters, ordered_bench, gw, lookup)
        points = {
            name: lookup.get((gw, str(name)), (0.0, 0.0))[1]
            for name in final_players["name"]
        }
        base = float(sum(points.values()))
        captain_used = None
        bonus = 0.0
        if solution["captain"] in points:
            captain_used = solution["captain"]
            bonus = float(points[captain_used])
        elif solution["vice_captain"] in points:
            captain_used = solution["vice_captain"]
            bonus = float(points[captain_used])
        records.append(
            {
                "GW": gw,
                "weekly_points": base + bonus,
                "base_points": base,
                "captain_bonus": bonus,
                "captain_used": captain_used,
                "autosubs_used": len(substitutes),
                "substitute_points": float(sum(points.get(n, 0.0) for n in substitutes)),
            }
        )
    result = pd.DataFrame(records)
    result["cumulative_points"] = result["weekly_points"].cumsum()
    return result


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable description of one forecast/objective/budget experiment."""
    label: str
    objective_key: str
    bench_key: str
    xi_cap: float = DEFAULT_XI_CAP
    family: str = ""


def build_specs() -> list[ExperimentSpec]:
    """Declare the 74 static internal series in stable manuscript order."""
    specs: list[ExperimentSpec] = [
        ExperimentSpec("Simple Avg.", "simple", "simple", family="Average"),
        ExperimentSpec("Weighted Avg.", "weighted", "weighted", family="Average"),
        ExperimentSpec("Simulation (Non Parametric)", "bootstrap", "bootstrap", family="Simulation"),
        ExperimentSpec("Monte Carlo Simulation", "monte_carlo", "monte_carlo", family="Simulation"),
        ExperimentSpec("Exponential Smoothing", "exponential", "exponential", family="Smoothing"),
        ExperimentSpec("Robust Simple Avg.", "robust_simple", "simple", family="Robust"),
        ExperimentSpec("Robust Weighted Avg.", "robust_weighted", "weighted", family="Robust"),
        ExperimentSpec("Robust Exponential Smoothing", "robust_exponential", "exponential", family="Robust"),
    ]
    for order in ARIMA_ORDERS:
        key = f"arima{order[0]}{order[1]}{order[2]}"
        specs.append(ExperimentSpec(arima_label(order), key, key, family="ARIMA"))
    specs.extend(
        [
            ExperimentSpec("ICT Score", "ict", "weighted", family="Alternative objective"),
            ExperimentSpec("Robust ICT Score", "robust_ict", "weighted", family="Alternative objective"),
            ExperimentSpec("Involvement", "involvement", "weighted", family="Alternative objective"),
            ExperimentSpec("Linear Regression", "linear", "linear", family="Linear trend"),
        ]
    )
    hybrid_definitions = [
        ("Hybrid Simple Avg.", "hybrid_simple", "simple"),
        ("Hybrid Weighted", "hybrid_weighted", "weighted"),
        ("Hybrid Exponential Smoothing", "hybrid_exponential", "exponential"),
        ("Hybrid Simulation (Non Parametric)", "hybrid_bootstrap", "bootstrap"),
        ("Hybrid Monte Carlo", "hybrid_monte_carlo", "monte_carlo"),
        ("Hybrid ARIMA", "hybrid_arima001", "arima001"),
        ("Hybrid ICT Score", "hybrid_ict", "weighted"),
        ("Hybrid Linear Regression", "hybrid_linear", "linear"),
    ]
    for prefix, key, bench_key in hybrid_definitions:
        specs.append(
            ExperimentSpec(
                f"{prefix} 1:2 (Higher Total Points)",
                f"{key}_history2",
                bench_key,
                family="Hybrid",
            )
        )
        specs.append(
            ExperimentSpec(
                f"{prefix} 2:1 (Lower Total Points)",
                f"{key}_ridge2",
                bench_key,
                family="Hybrid",
            )
        )

    for cap in [55, 60, 65, 70, 75, 80]:
        specs.append(ExperimentSpec(f"Simple Avg. (Budget = {cap})", "simple", "simple", cap, "Budget"))
    for cap in [60, 65, 70, 75, 80]:
        specs.append(ExperimentSpec(f"Weighted Avg. (Budget = {cap})", "weighted", "weighted", cap, "Budget"))
        specs.append(ExperimentSpec(f"Monte Carlo (Budget = {cap})", "monte_carlo", "monte_carlo", cap, "Budget"))
        specs.append(ExperimentSpec(f"ARIMA (1,0,0, Budget = {cap})", "arima100", "arima100", cap, "Budget"))
        specs.append(ExperimentSpec(f"ARIMA (1,0,1, Budget = {cap})", "arima101", "arima101", cap, "Budget"))
    for cap in [65, 70, 75, 80]:
        specs.append(ExperimentSpec(f"ARIMA (0,0,1, Budget = {cap})", "arima001", "arima001", cap, "Budget"))
        specs.append(ExperimentSpec(f"Hybrid ICT Score (Budget = {cap})", "hybrid_ict_history2", "weighted", cap, "Budget"))
    for cap in [70, 75, 80]:
        specs.append(ExperimentSpec(f"ICT Score (Budget = {cap})", "ict", "weighted", cap, "Budget"))
    if len(specs) != 74:
        raise AssertionError(f"Expected 74 static internal series, got {len(specs)}")
    if len({spec.label for spec in specs}) != len(specs):
        raise AssertionError("Static experiment labels must be unique.")
    return specs


def sequential_mapping() -> dict[str, tuple[str, float]]:
    """Map the 20 sequential series to forecast method and XI cap."""
    return {
        "Simple Avg. Sequential": ("Simple average", 83.5),
        "Simple Avg. Sequential (Budget = 80)": ("Simple average", 80.0),
        "Simple Avg. Sequential (Budget = 70)": ("Simple average", 70.0),
        "Simple Avg. Sequential (Budget = 65)": ("Simple average", 65.0),
        "Weighted Avg. Sequential": ("Weighted average", 83.5),
        "Weighted Avg. Sequential (Budget = 80)": ("Weighted average", 80.0),
        "Weighted Avg. Sequential (Budget = 70)": ("Weighted average", 70.0),
        "Weighted Avg. Sequential (Budget = 60)": ("Weighted average", 60.0),
        "ARIMA (0,0,1) Sequential": ("ARIMA(0,0,1)", 83.5),
        "ARIMA (0,0,1) Sequential (Budget = 80)": ("ARIMA(0,0,1)", 80.0),
        "ARIMA (0,0,1) Sequential (Budget = 70)": ("ARIMA(0,0,1)", 70.0),
        "ARIMA (0,0,1) Sequential (Budget = 60)": ("ARIMA(0,0,1)", 60.0),
        "ARIMA (1,0,0) Sequential": ("ARIMA(1,0,0)", 83.5),
        "ARIMA (1,0,0) Sequential (Budget = 75)": ("ARIMA(1,0,0)", 75.0),
        "ARIMA (1,0,0) Sequential (Budget = 70)": ("ARIMA(1,0,0)", 70.0),
        "ARIMA (1,0,0) Sequential (Budget = 60)": ("ARIMA(1,0,0)", 60.0),
        "ARIMA (1,0,1) Sequential": ("ARIMA(1,0,1)", 83.5),
        "ARIMA (1,0,1) Sequential (Budget = 80)": ("ARIMA(1,0,1)", 80.0),
        "ARIMA (1,0,1) Sequential (Budget = 70)": ("ARIMA(1,0,1)", 70.0),
        "ARIMA (1,0,1) Sequential (Budget = 60)": ("ARIMA(1,0,1)", 60.0),
    }


def add_sequential_scores(
    matrix: pd.DataFrame,
    roster_design_dir: Path,
    provenance_rows: list[dict[str, object]],
) -> None:
    """Append fresh fixed-squad sequential scores and their provenance records."""
    weekly_path = roster_design_dir / "reviewer1_comment2_weekly_scores.csv"
    weekly = pd.read_csv(weekly_path)
    for label, (method, cap) in sequential_mapping().items():
        solution_id = f"{method} | two_stage | fixed_split | xi_cap={cap:g}"
        selected = weekly.loc[
            (weekly["method"] == method)
            & (weekly["policy"] == "sequential")
            & (weekly["solution_id"] == solution_id)
        ].sort_values("GW")
        if len(selected) != FINAL_GW - TARGET_GW + 1:
            raise RuntimeError(
                f"Missing fresh sequential output for {method}, cap={cap:g}: {len(selected)} rows."
            )
        matrix[label] = selected["weekly_points"].to_numpy(float)
        provenance_rows.append(
            {
                "series": label,
                "source_type": "raw-data recomputation",
                "source_file": str(weekly_path),
                "forecast_method": method,
                "policy": "fixed-squad sequential lineup updating",
                "xi_cap": cap,
            }
        )


def main() -> None:
    """Recompute all internal series and export matrices, rosters, and audits."""
    args = parse_args()
    started = time.perf_counter()
    args.data = args.data.expanduser().resolve()
    args.roster_design_dir = args.roster_design_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not args.roster_design_dir.is_dir():
        raise FileNotFoundError(args.roster_design_dir)
    if args.workers < 1 or args.simulation_draws < 1:
        raise ValueError("--workers and --simulation-draws must be positive.")
    if not args.smoke_test:
        try:
            import statsmodels
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("statsmodels is required for the full publication run.") from exc
    else:
        statsmodels = None

    warnings.filterwarnings("ignore")
    data = load_panel(args.data)
    train = data.loc[data["GW"] < TARGET_GW].copy()
    test = data.loc[data["GW"].between(TARGET_GW, FINAL_GW)].copy()
    snapshot = latest_snapshot(train)
    histories = {
        str(name): frame.sort_values("GW", kind="stable").reset_index(drop=True)
        for name, frame in train.groupby("name", sort=False)
    }
    if set(snapshot["name"].astype(str)) - set(histories):
        raise AssertionError("Every candidate must have a pre-GW27 history.")
    print("=" * 78, flush=True)
    print("Full raw-data internal experiment screen", flush=True)
    print(f"Data: {args.data}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Training rows: {len(train):,}; test rows: {len(test):,}; candidates: {len(snapshot):,}", flush=True)
    print("=" * 78, flush=True)

    point_forecasts = build_point_forecasts(histories, snapshot, args)
    point_uncertainty = build_uncertainty(histories, snapshot, "total_points")
    ict_uncertainty = build_uncertainty(histories, snapshot, "ict_index")

    aggregate_keys = ["simple", "weighted", "bootstrap", "monte_carlo", "exponential", "linear", "arima001"]
    aggregates = {
        key: aggregate_feature_table(key, histories, snapshot, point_forecasts, args)
        for key in aggregate_keys
    }
    objectives: dict[str, pd.Series] = {key: value.copy() for key, value in point_forecasts.items()}
    objectives["robust_simple"] = point_forecasts["simple"] - point_uncertainty.to_numpy(float)
    objectives["robust_weighted"] = point_forecasts["weighted"] - point_uncertainty.to_numpy(float)
    objectives["robust_exponential"] = point_forecasts["exponential"] - point_uncertainty.to_numpy(float)

    weighted = aggregates["weighted"].set_index("name")
    objectives["ict"] = weighted["ict_index"].reindex(snapshot["name"])
    objectives["robust_ict"] = objectives["ict"] - ict_uncertainty.to_numpy(float)
    objectives["involvement"] = (
        weighted["expected_goal_involvements"] - weighted["expected_goals_conceded"]
    ).reindex(snapshot["name"])

    hybrid_sources = {
        "hybrid_simple": "simple",
        "hybrid_weighted": "weighted",
        "hybrid_exponential": "exponential",
        "hybrid_bootstrap": "bootstrap",
        "hybrid_monte_carlo": "monte_carlo",
        "hybrid_arima001": "arima001",
        "hybrid_ict": "simple",
        "hybrid_linear": "linear",
    }
    for hybrid_key, aggregate_key in hybrid_sources.items():
        table = aggregates[aggregate_key]
        objectives[f"{hybrid_key}_history2"] = hybrid_scores(table, 2.0, 1.0)
        objectives[f"{hybrid_key}_ridge2"] = hybrid_scores(table, 1.0, 2.0)

    # Normalize Series indexing before model construction.
    candidate_names = snapshot["name"].astype(str).tolist()
    for key, values in list(objectives.items()):
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=candidate_names, dtype=float)
        if list(values.index.astype(str)) != candidate_names:
            values.index = values.index.astype(str)
            values = values.reindex(candidate_names)
        objectives[key] = pd.to_numeric(values, errors="coerce").fillna(0.0)

    specs = build_specs()
    lookup = actual_lookup(test)
    matrix = pd.DataFrame({"GW": range(TARGET_GW, FINAL_GW + 1)})
    weekly_rows: list[pd.DataFrame] = []
    roster_rows: list[dict[str, object]] = []
    coefficient_rows: list[pd.DataFrame] = []
    provenance_rows: list[dict[str, object]] = []
    solution_cache: dict[tuple[str, str, float], tuple[dict[str, object], pd.DataFrame]] = {}

    for number, spec in enumerate(specs, start=1):
        cache_key = (spec.objective_key, spec.bench_key, float(spec.xi_cap))
        if cache_key not in solution_cache:
            pool = snapshot.copy()
            pool["objective"] = objectives[spec.objective_key].to_numpy(float)
            pool["bench_forecast"] = point_forecasts[spec.bench_key].reindex(candidate_names).to_numpy(float)
            solution = solve_two_stage(pool, spec.xi_cap, spec.label, args)
            evaluation = evaluate_static(solution, lookup)
            solution_cache[cache_key] = (solution, evaluation)
        solution, evaluation = solution_cache[cache_key]
        matrix[spec.label] = evaluation["weekly_points"].to_numpy(float)
        long = evaluation.copy()
        long.insert(0, "series", spec.label)
        long.insert(1, "policy", "static")
        weekly_rows.append(long)
        squad = solution["squad"]
        starters = solution["starters"]
        assert isinstance(squad, pd.DataFrame) and isinstance(starters, pd.DataFrame)
        starter_names = set(starters["name"].astype(str))
        for row in squad.itertuples(index=False):
            roster_rows.append(
                {
                    "series": spec.label,
                    "name": row.name,
                    "team": row.team,
                    "position": row.position,
                    "value": row.value,
                    "role": "XI" if row.name in starter_names else "Bench",
                    "is_captain": row.name == solution["captain"],
                    "is_vice_captain": row.name == solution["vice_captain"],
                    "xi_cap": spec.xi_cap,
                    "xi_spend": solution["xi_spend"],
                    "bench_spend": solution["bench_spend"],
                    "total_spend": solution["total_spend"],
                }
            )
        coefficient_rows.append(
            pd.DataFrame(
                {
                    "series": spec.label,
                    "name": candidate_names,
                    "objective_coefficient": objectives[spec.objective_key].to_numpy(float),
                    "bench_forecast": point_forecasts[spec.bench_key].reindex(candidate_names).to_numpy(float),
                    "xi_cap": spec.xi_cap,
                }
            )
        )
        provenance_rows.append(
            {
                "series": spec.label,
                "source_type": "raw-data recomputation",
                "source_file": str(args.data),
                "forecast_method": spec.objective_key,
                "policy": "static",
                "xi_cap": spec.xi_cap,
            }
        )
        print(f"[{number:02d}/{len(specs)}] Solved and evaluated: {spec.label}", flush=True)

    add_sequential_scores(matrix, args.roster_design_dir, provenance_rows)
    if matrix.shape[1] != 95:  # GW + 94 internal result series
        raise AssertionError(f"Expected GW plus 94 internal series, got {matrix.shape[1]} columns.")
    if matrix.columns.duplicated().any():
        raise AssertionError("Internal result series must have unique labels.")
    if matrix.drop(columns="GW").isna().any().any():
        raise AssertionError("Internal score matrix contains missing values.")

    # Cross-check freshly recomputed static families against the separately
    # executed controlled experiment.  This is provenance validation, not reuse.
    controlled_weekly = pd.read_csv(
        args.roster_design_dir / "reviewer1_comment2_weekly_scores.csv"
    )
    checks = {
        "Simple Avg.": ("Simple average", 83.5),
        "Weighted Avg.": ("Weighted average", 83.5),
        "ARIMA (0,0,1)": ("ARIMA(0,0,1)", 83.5),
        "ARIMA (1,0,0)": ("ARIMA(1,0,0)", 83.5),
        "ARIMA (1,0,1)": ("ARIMA(1,0,1)", 83.5),
    }
    audit = []
    for label, (method, cap) in checks.items():
        solution_id = f"{method} | two_stage | fixed_split | xi_cap={cap:g}"
        selected = controlled_weekly.loc[
            (controlled_weekly["method"] == method)
            & (controlled_weekly["policy"] == "static")
            & (controlled_weekly["solution_id"] == solution_id)
        ].sort_values("GW")
        if len(selected) != 12:
            raise RuntimeError(f"Missing controlled static cross-check: {label}")
        left = matrix[label].to_numpy(float)
        right = selected["weekly_points"].to_numpy(float)
        matched = bool(np.allclose(left, right, atol=1e-8, rtol=0.0))
        audit.append(
            {
                "series": label,
                "independent_total": float(left.sum()),
                "controlled_total": float(right.sum()),
                "weekly_values_match": matched,
            }
        )
        if not matched and not args.smoke_test:
            raise AssertionError(f"Independent static recomputation disagrees with controlled run: {label}")

    matrix.to_csv(args.output / "full_internal_score_matrix.csv", index=False)
    pd.concat(weekly_rows, ignore_index=True).to_csv(
        args.output / "full_internal_static_weekly_scores_long.csv", index=False
    )
    pd.DataFrame(roster_rows).to_csv(args.output / "full_internal_rosters.csv", index=False)
    pd.concat(coefficient_rows, ignore_index=True).to_csv(
        args.output / "full_internal_cost_vectors.csv", index=False
    )
    pd.DataFrame(provenance_rows).to_csv(
        args.output / "full_internal_series_provenance.csv", index=False
    )
    pd.DataFrame(audit).to_csv(
        args.output / "independent_controlled_crosscheck.csv", index=False
    )

    elapsed = time.perf_counter() - started
    config = {
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "roster_design_dir": str(args.roster_design_dir),
        "output": str(args.output),
        "training_gameweeks": [1, TARGET_GW - 1],
        "evaluation_gameweeks": [TARGET_GW, FINAL_GW],
        "internal_series": 94,
        "static_series": 74,
        "sequential_series": 20,
        "external_series_included": 0,
        "simulation_draws": args.simulation_draws,
        "workers": args.workers,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "uncertainty_single_observation_fallback": "position median empirical SD",
        "hybrid_ridge_alpha": 1.0,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "statsmodels": None if args.smoke_test else statsmodels.__version__,
        "elapsed_minutes": elapsed / 60.0,
    }
    (args.output / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "RUN_COMPLETED_SUCCESSFULLY.txt").write_text(
        f"Recomputed 94 internal series from raw data in {elapsed/60:.2f} minutes.\n",
        encoding="utf-8",
    )
    print("=" * 78, flush=True)
    print("FULL INTERNAL EXPERIMENT SCREEN COMPLETED SUCCESSFULLY", flush=True)
    print(f"Recomputed 74 static + 20 sequential = 94 internal series.", flush=True)
    print(f"Elapsed: {elapsed/60:.2f} minutes", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
