#!/usr/bin/env python3
"""Compare FPL roster-construction and budget-allocation strategies.

The same player forecasts, candidate pool, FPL constraints, and GW27--38
evaluation protocol are used for every design.  This isolates the effect of
the roster formulation from differences in predictive inputs.  The script
implements:

* a two-stage roster with a fixed starting-XI/bench allocation;
* a two-stage roster that passes unused XI budget to the bench;
* a joint 15-player mixed-integer program with endogenous expenditures;
* an epsilon-constraint Pareto frontier for starter and bench quality;
* static and fixed-squad sequential lineup evaluation; and
* a starting-availability sensitivity analysis.

Pareto compromise points are selected in predicted-objective space before
realized GW27--38 outcomes are evaluated, avoiding test-period tuning.

Examples
--------
Full run (weighted average and ARIMA):

    python -u roster_design_experiments.py \
        --data merged_gw_2324.csv \
        --output roster_design_outputs_server

Short installation/feasibility test:

    python -u roster_design_experiments.py \
        --data merged_gw_2324.csv --smoke-test

The script never reads or overwrites a historical results CSV.  All tables,
figures, and run metadata are written beneath the selected output directory.
"""

# %% [markdown]
# ## Two-stage versus joint 15-player roster optimization
#
# This program is intentionally self-contained. It reloads the 2023/24
# player--gameweek data and compares roster-construction rules while holding the
# forecasts, candidate pool, FPL constraints, and GW27--38 evaluation protocol
# fixed. It does **not** overwrite `Results.csv` or any earlier paper output.
#
# The experiment includes:
#
# 1. the published two-stage baseline with an £83.5m XI cap and £16.5m bench cap;
# 2. a corrected two-stage variant that passes any unspent XI allowance to the bench;
# 3. a joint MILP for the entire 15-player roster, with endogenous XI expenditure;
# 4. an epsilon-constraint Pareto frontier for XI quality versus bench quality; and
# 5. a common, leakage-safe GW27--38 evaluator.
#
# The Pareto compromise is selected only from predicted objective values. Realized
# GW27--38 points are reported afterwards and are never used to tune the budget,
# epsilon, or compromise point.

# %%
# Configuration, imports, and leakage-safe data preparation
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
import os
import platform
import sys
import time
import warnings

# Force a non-interactive backend before importing pyplot on a remote server.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fpl_roster_design_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except ImportError as exc:
    raise SystemExit(
        "SciPy with scipy.optimize.milp is required. Install dependencies with: "
        "python -m pip install numpy pandas scipy statsmodels matplotlib"
    ) from exc

try:
    import statsmodels
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:
    raise SystemExit(
        "statsmodels is required for ARIMA. Install dependencies with: "
        "python -m pip install numpy pandas scipy statsmodels matplotlib"
    ) from exc


def display(value):
    """Readable console replacement for notebook display()."""
    if isinstance(value, pd.DataFrame):
        print(value.to_string())
    else:
        print(value)


def parse_args():
    """Parse data, solver, forecasting, and evaluation options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the two-stage, corrected two-stage, joint-roster, and "
            "epsilon-constraint roster-design experiments."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("merged_gw_2324.csv"),
        help="Path to merged_gw_2324.csv (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("roster_design_outputs_server"),
        help="Directory for all generated results.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["simple", "weighted", "arima001", "arima100", "arima101"],
        default=["simple", "weighted", "arima001", "arima100", "arima101"],
        help=(
            "Forecasting methods to run. The default covers every method used "
            "in the retained fixed-squad sequential figures."
        ),
    )
    parser.add_argument(
        "--xi-caps",
        nargs="+",
        type=float,
        default=[55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 83.5],
        help="Starting-XI caps for the two-stage sensitivity analysis.",
    )
    parser.add_argument(
        "--pareto-deltas",
        nargs="+",
        type=float,
        default=[0.000, 0.005, 0.010, 0.020, 0.050, 0.100],
        help="Permitted proportional XI losses for the epsilon constraints.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["static", "sequential"],
        default=["static", "sequential"],
        help="Out-of-sample lineup policies to evaluate.",
    )
    parser.add_argument(
        "--mip-time-limit",
        type=float,
        default=120.0,
        help="Per-MILP time limit in seconds (default: 120).",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.0,
        help="Relative MILP optimality gap (default: 0 for exact solves).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help=(
            "Threads used for independent ARIMA fits (default: min(4, CPU count)). "
            "Set OMP_NUM_THREADS=1 when using several workers to avoid oversubscription."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a quick weighted-only installation test using cap 83.5, "
            "deltas 0 and 0.05, and the static policy."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
START_TIME = time.perf_counter()

warnings.filterwarnings("ignore", category=Warning, module="statsmodels")

# ---------- Reproducible configuration ----------
DATA_PATH = ARGS.data.expanduser().resolve()
OUTPUT_DIR = ARGS.output.expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

TARGET_GW = 27
FINAL_GW = 38
TOTAL_BUDGET = 100.0
DEFAULT_XI_CAP = 83.5

# Official FPL squad quotas and legal starting-XI bounds.
SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_BOUNDS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3

# Interpretable forecasters used by the controlled roster-design comparison and
# by the retained budget and fixed-squad sequential analyses.
METHOD_CATALOG = {
    "simple": {"label": "Simple average", "kind": "simple"},
    "weighted": {"label": "Weighted average", "kind": "weighted"},
    "arima001": {"label": "ARIMA(0,0,1)", "kind": "arima", "order": (0, 0, 1)},
    "arima100": {"label": "ARIMA(1,0,0)", "kind": "arima", "order": (1, 0, 0)},
    "arima101": {"label": "ARIMA(1,0,1)", "kind": "arima", "order": (1, 0, 1)},
}
METHOD_SPECS = [METHOD_CATALOG[name] for name in ARGS.methods]

# The published default plus the caps used for the diagnostic budget sweep.
TWO_STAGE_XI_CAPS = list(ARGS.xi_caps)

# Maximum permitted proportional loss relative to the best predicted XI value.
PARETO_DELTAS = list(ARGS.pareto_deltas)

EVALUATION_POLICIES = list(ARGS.policies)
FORECAST_WORKERS = int(ARGS.workers)

if FORECAST_WORKERS < 1:
    raise ValueError("--workers must be at least 1.")

if ARGS.smoke_test:
    METHOD_SPECS = [METHOD_CATALOG["weighted"]]
    TWO_STAGE_XI_CAPS = [83.5]
    PARETO_DELTAS = [0.0, 0.05]
    EVALUATION_POLICIES = ["static"]

MILP_TIME_LIMIT = float(ARGS.mip_time_limit)
MILP_GAP = float(ARGS.mip_gap)
OBJECTIVE_TOL = 1e-6
TIE_EPS = 1e-8
AVAILABILITY_GAMMA = 0.5

if any(cap <= 0 or cap >= TOTAL_BUDGET for cap in TWO_STAGE_XI_CAPS):
    raise ValueError("Every XI cap must be strictly between 0 and 100.")
if any(delta < 0 or delta >= 1 for delta in PARETO_DELTAS):
    raise ValueError("Every Pareto delta must satisfy 0 <= delta < 1.")
if DEFAULT_XI_CAP not in TWO_STAGE_XI_CAPS:
    print(
        "WARNING: the requested XI-cap grid omits the published default £83.5m; "
        "the main-comparison CSV may contain only joint designs."
    )

run_config = {
    "data": str(DATA_PATH),
    "output": str(OUTPUT_DIR),
    "methods": [spec["label"] for spec in METHOD_SPECS],
    "xi_caps": TWO_STAGE_XI_CAPS,
    "pareto_deltas": PARETO_DELTAS,
    "policies": EVALUATION_POLICIES,
    "mip_time_limit": MILP_TIME_LIMIT,
    "mip_gap": MILP_GAP,
    "availability_gamma": AVAILABILITY_GAMMA,
    "availability_estimator": "Beta(1,1)-smoothed pre-GW27 starting frequency",
    "forecast_workers": FORECAST_WORKERS,
    "smoke_test": bool(ARGS.smoke_test),
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
    "matplotlib": matplotlib.__version__,
}
(OUTPUT_DIR / "run_config.json").write_text(
    json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("=" * 78)
print("FPL roster-design and budget-allocation experiment")
print(f"Data:    {DATA_PATH}")
print(f"Output:  {OUTPUT_DIR}")
print(f"Methods: {', '.join(run_config['methods'])}")
print(f"Policies: {', '.join(EVALUATION_POLICIES)}")
print(f"ARIMA forecast workers: {FORECAST_WORKERS}")
print("=" * 78)

required_columns = {
    "name", "position", "team", "value", "total_points", "minutes", "starts", "GW"
}
raw = pd.read_csv(DATA_PATH)
missing_columns = required_columns.difference(raw.columns)
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

raw = raw.copy()
raw["GW"] = pd.to_numeric(raw["GW"], errors="raise").astype(int)
raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
raw["total_points"] = pd.to_numeric(raw["total_points"], errors="coerce").fillna(0.0)
raw["minutes"] = pd.to_numeric(raw["minutes"], errors="coerce").fillna(0.0)
raw["starts"] = pd.to_numeric(raw["starts"], errors="coerce").fillna(0.0)

# The supplied data stores price in tenths of £m (e.g., 55 means £5.5m).
if raw["value"].median(skipna=True) > 20:
    raw["value"] = raw["value"] / 10.0

training_data = raw.loc[raw["GW"] < TARGET_GW].copy()
evaluation_data = raw.loc[raw["GW"].between(TARGET_GW, FINAL_GW)].copy()

if training_data.empty or evaluation_data.empty:
    raise ValueError("The requested GW1--26/GW27--38 split is unavailable.")

# Match the original study's candidate definition: every player observed by GW26,
# using the latest pre-GW27 team, position, and price. No GW27 outcome is used.
candidate_snapshot = (
    training_data.sort_values(["name", "GW"])
    .groupby("name", as_index=False)
    .tail(1)[["name", "team", "position", "value"]]
    .dropna()
    .drop_duplicates("name")
    .reset_index(drop=True)
)

unknown_positions = set(candidate_snapshot["position"]) - set(SQUAD_QUOTAS)
if unknown_positions:
    raise ValueError(f"Unexpected position labels: {sorted(unknown_positions)}")

print(
    f"Training rows: {len(training_data):,} (GW < {TARGET_GW}); "
    f"test rows: {len(evaluation_data):,} (GW {TARGET_GW}--{FINAL_GW}); "
    f"candidate players: {len(candidate_snapshot):,}."
)

# %%
# Forecast tables: the same prediction table is supplied to every roster design
def _forecast_one_player(points, method_spec, forecast_steps):
    """Return one scalar expected-points coefficient from pre-target history."""
    values = pd.Series(points, dtype=float).dropna().reset_index(drop=True)
    if values.empty:
        return 0.0

    kind = method_spec["kind"]
    if kind == "simple":
        estimate = values.mean()
    elif kind == "weighted":
        weights = np.arange(1, len(values) + 1, dtype=float)
        estimate = np.average(values.to_numpy(), weights=weights)
    elif kind == "arima":
        if len(values) < 2 or np.isclose(values.mean(), 0.0):
            estimate = values.mean()
        else:
            try:
                fit = ARIMA(values, order=tuple(method_spec["order"])).fit()
                estimate = float(np.asarray(fit.forecast(steps=max(1, forecast_steps))).mean())
            except Exception:
                estimate = values.mean()
    else:
        raise ValueError(f"Unsupported forecasting kind: {kind}")

    return float(estimate) if np.isfinite(estimate) else 0.0


FORECAST_EXECUTOR = (
    ThreadPoolExecutor(max_workers=FORECAST_WORKERS)
    if FORECAST_WORKERS > 1
    else None
)


def _forecast_task(task):
    """Worker wrapper that makes independent player forecasts thread-safe."""
    points, method_spec, forecast_steps = task
    return _forecast_one_player(points, method_spec, forecast_steps)


def _forecast_many(tasks, method_spec):
    """Run independent ARIMA fits concurrently; keep cheap methods sequential."""
    if FORECAST_EXECUTOR is not None and method_spec["kind"] == "arima":
        return list(FORECAST_EXECUTOR.map(_forecast_task, tasks))
    return [_forecast_task(task) for task in tasks]


def make_forecast_table(history, snapshot, method_spec, forecast_steps):
    """Forecast every candidate using only rows contained in `history`."""
    ordered = history.sort_values(["name", "GW"])
    histories = ordered.groupby("name", sort=False)["total_points"].apply(list).to_dict()

    out = snapshot.copy()
    tasks = [
        (histories.get(name, []), method_spec, forecast_steps)
        for name in out["name"]
    ]
    out["forecast"] = _forecast_many(tasks, method_spec)
    out = out.dropna(subset=["name", "team", "position", "value", "forecast"])
    out = out.sort_values("name", kind="stable").reset_index(drop=True)
    if out["name"].duplicated().any():
        raise AssertionError("Candidate names must be unique for the MILP index.")
    return out


forecast_tables = {}
for spec in METHOD_SPECS:
    # 12 future gameweeks: GW27 through GW38 inclusive.
    forecast_tables[spec["label"]] = make_forecast_table(
        training_data,
        candidate_snapshot,
        spec,
        forecast_steps=FINAL_GW - TARGET_GW + 1,
    )
    print(f"Built forecast table: {spec['label']}")

display(
    pd.DataFrame(
        {
            label: table["forecast"].describe()[["count", "mean", "std", "min", "max"]]
            for label, table in forecast_tables.items()
        }
    ).round(3)
)

# %%
# Sparse MILP helpers and common solution checks
class _MilpRows:
    """Build sparse linear constraints for the roster MILP formulations."""
    def __init__(self, n_variables):
        self.n_variables = int(n_variables)
        self.row_ids = []
        self.col_ids = []
        self.values = []
        self.lower = []
        self.upper = []

    def add(self, coefficients, lower=-np.inf, upper=np.inf):
        row = len(self.lower)
        for col, value in coefficients.items():
            value = float(value)
            if not np.isclose(value, 0.0):
                self.row_ids.append(row)
                self.col_ids.append(int(col))
                self.values.append(value)
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def constraint(self):
        matrix = coo_matrix(
            (self.values, (self.row_ids, self.col_ids)),
            shape=(len(self.lower), self.n_variables),
        ).tocsr()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


def _solve_binary_milp(maximize_coefficients, rows, label):
    """Maximize a binary linear objective and return rounded decision values."""
    maximize_coefficients = np.asarray(maximize_coefficients, dtype=float)
    result = milp(
        c=-maximize_coefficients,
        integrality=np.ones(len(maximize_coefficients), dtype=int),
        bounds=Bounds(np.zeros(len(maximize_coefficients)), np.ones(len(maximize_coefficients))),
        constraints=rows.constraint(),
        options={"time_limit": MILP_TIME_LIMIT, "mip_rel_gap": MILP_GAP},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed for {label}: status={result.status}; {result.message}")
    return np.rint(result.x).astype(int)


def _stable_weights(pool):
    """Generate deterministic name-based weights used only to break exact ties."""
    # Tiny deterministic tie-break only; it is excluded from all reported values.
    rank = pd.Series(pool["name"]).rank(method="first", ascending=True).to_numpy()
    return (len(pool) + 1.0 - rank) / max(1.0, len(pool))


def _formation_string(starters):
    """Format the DEF--MID--FWD counts of a legal starting XI."""
    counts = starters["position"].value_counts().to_dict()
    return f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}"


def _package_solution(pool, squad_indices, starter_indices, captain_index, metadata):
    """Convert MILP indices into auditable squad, XI, bench, and captain tables."""
    squad_indices = np.asarray(sorted(set(map(int, squad_indices))), dtype=int)
    starter_indices = np.asarray(sorted(set(map(int, starter_indices))), dtype=int)
    bench_indices = np.asarray(sorted(set(squad_indices) - set(starter_indices)), dtype=int)

    squad = pool.loc[squad_indices].copy()
    starters = pool.loc[starter_indices].copy()
    bench = pool.loc[bench_indices].copy()
    captain_name = str(pool.loc[int(captain_index), "name"])

    starters = starters.sort_values(["position", "forecast", "name"], ascending=[True, False, True])
    bench = bench.sort_values(["position", "forecast", "name"], ascending=[True, False, True])

    vice_candidates = starters.loc[starters["name"] != captain_name]
    vice_name = str(
        vice_candidates.sort_values(["forecast", "name"], ascending=[False, True]).iloc[0]["name"]
    )

    scores = pool["forecast"].to_numpy(float)
    values = pool["value"].to_numpy(float)
    xi_score = float(scores[starter_indices].sum() + scores[int(captain_index)])
    bench_score = float(scores[bench_indices].sum())
    xi_spend = float(values[starter_indices].sum())
    bench_spend = float(values[bench_indices].sum())

    solution = {
        **metadata,
        "squad": squad.reset_index(drop=True),
        "starters": starters.reset_index(drop=True),
        "bench": bench.reset_index(drop=True),
        "captain": captain_name,
        "vice_captain": vice_name,
        "xi_score": xi_score,
        "bench_score": bench_score,
        "xi_spend": xi_spend,
        "bench_spend": bench_spend,
        "total_spend": xi_spend + bench_spend,
        "formation": _formation_string(starters),
    }
    _validate_solution(solution)
    return solution


def _validate_solution(solution):
    """Assert all roster, formation, budget, club, and captaincy constraints."""
    squad = solution["squad"]
    starters = solution["starters"]
    bench = solution["bench"]

    assert len(squad) == 15 and squad["name"].nunique() == 15
    assert len(starters) == 11 and starters["name"].nunique() == 11
    assert len(bench) == 4 and bench["name"].nunique() == 4
    assert set(starters["name"]).isdisjoint(set(bench["name"]))
    assert solution["captain"] in set(starters["name"])
    assert solution["vice_captain"] in set(starters["name"])
    assert solution["total_spend"] <= TOTAL_BUDGET + 1e-5
    assert squad["position"].value_counts().to_dict() == SQUAD_QUOTAS
    assert squad["team"].value_counts().max() <= MAX_PER_CLUB

    counts = starters["position"].value_counts().to_dict()
    for position, (minimum, maximum) in XI_BOUNDS.items():
        assert minimum <= counts.get(position, 0) <= maximum

# %%
# Model A: published and corrected two-stage formulations
def solve_two_stage(pool, xi_cap=DEFAULT_XI_CAP, bench_budget_rule="fixed_split", label=None):
    """
    Select XI+captain first, then select the positional bench complement.

    bench_budget_rule="fixed_split" reproduces 100-xi_cap.
    bench_budget_rule="actual_remainder" uses 100-actual_XI_spend.
    """
    pool = pool.reset_index(drop=True).copy()
    n = len(pool)
    scores = pool["forecast"].to_numpy(float)
    prices = pool["value"].to_numpy(float)
    stable = _stable_weights(pool)

    # Stage 1 variables: s_j (starter), y_j (captain).
    nvar = 2 * n
    s = np.arange(n)
    y = n + np.arange(n)
    rows = _MilpRows(nvar)
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

    objective = np.zeros(nvar)
    objective[s] = scores + TIE_EPS * stable
    objective[y] = scores + TIE_EPS * stable
    z = _solve_binary_milp(objective, rows, f"two-stage XI ({xi_cap})")
    starter_idx = np.flatnonzero(z[s])
    captain_idx = int(np.flatnonzero(z[y])[0])
    actual_xi_spend = float(prices[starter_idx].sum())

    if bench_budget_rule == "fixed_split":
        bench_budget = TOTAL_BUDGET - float(xi_cap)
    elif bench_budget_rule == "actual_remainder":
        bench_budget = TOTAL_BUDGET - actual_xi_spend
    else:
        raise ValueError("bench_budget_rule must be 'fixed_split' or 'actual_remainder'.")

    # Stage 2 variables: b_j (bench).
    brows = _MilpRows(n)
    brows.add({j: 1 for j in range(n)}, lower=4, upper=4)
    brows.add({j: prices[j] for j in range(n)}, upper=bench_budget)

    starter_set = set(starter_idx)
    for j in starter_set:
        brows.add({int(j): 1}, lower=0, upper=0)

    starter_positions = pool.loc[starter_idx, "position"].value_counts().to_dict()
    for position, squad_total in SQUAD_QUOTAS.items():
        requirement = squad_total - starter_positions.get(position, 0)
        idx = np.flatnonzero(pool["position"].to_numpy() == position)
        brows.add({int(j): 1 for j in idx}, lower=requirement, upper=requirement)

    starter_clubs = pool.loc[starter_idx, "team"].value_counts().to_dict()
    for club in sorted(pool["team"].unique()):
        idx = np.flatnonzero(pool["team"].to_numpy() == club)
        brows.add(
            {int(j): 1 for j in idx},
            upper=MAX_PER_CLUB - starter_clubs.get(club, 0),
        )

    bobjective = scores + TIE_EPS * stable
    bz = _solve_binary_milp(
        bobjective,
        brows,
        f"two-stage bench ({xi_cap}, {bench_budget_rule})",
    )
    bench_idx = np.flatnonzero(bz)
    squad_idx = np.concatenate([starter_idx, bench_idx])

    return _package_solution(
        pool,
        squad_idx,
        starter_idx,
        captain_idx,
        metadata={
            "design": label or f"Two-stage {bench_budget_rule}",
            "design_family": "Two-stage",
            "xi_cap": float(xi_cap),
            "bench_budget_rule": bench_budget_rule,
            "bench_budget_available": float(bench_budget),
            "delta": np.nan,
        },
    )

# %%
# Model B: joint 15-player formulation and epsilon-constraint Pareto frontier
def solve_joint(
    pool,
    xi_objective_weight=0.0,
    bench_objective_weight=0.0,
    xi_minimum=None,
    bench_minimum=None,
    label="Joint",
):
    """Jointly select the 15-player squad, XI, and captain."""
    pool = pool.reset_index(drop=True).copy()
    n = len(pool)
    scores = pool["forecast"].to_numpy(float)
    prices = pool["value"].to_numpy(float)
    stable = _stable_weights(pool)

    # x_j: squad, s_j: starting XI, y_j: captain.
    nvar = 3 * n
    x = np.arange(n)
    s = n + np.arange(n)
    y = 2 * n + np.arange(n)
    rows = _MilpRows(nvar)

    rows.add({int(x[j]): 1 for j in range(n)}, lower=15, upper=15)
    rows.add({int(s[j]): 1 for j in range(n)}, lower=11, upper=11)
    rows.add({int(y[j]): 1 for j in range(n)}, lower=1, upper=1)
    rows.add({int(x[j]): prices[j] for j in range(n)}, upper=TOTAL_BUDGET)

    for j in range(n):
        rows.add({int(s[j]): 1, int(x[j]): -1}, upper=0)
        rows.add({int(y[j]): 1, int(s[j]): -1}, upper=0)

    for position, quota in SQUAD_QUOTAS.items():
        idx = np.flatnonzero(pool["position"].to_numpy() == position)
        rows.add({int(x[j]): 1 for j in idx}, lower=quota, upper=quota)

    for position, (minimum, maximum) in XI_BOUNDS.items():
        idx = np.flatnonzero(pool["position"].to_numpy() == position)
        rows.add({int(s[j]): 1 for j in idx}, lower=minimum, upper=maximum)

    for club in sorted(pool["team"].unique()):
        idx = np.flatnonzero(pool["team"].to_numpy() == club)
        rows.add({int(x[j]): 1 for j in idx}, upper=MAX_PER_CLUB)

    # F_XI = sum c_j s_j + sum c_j y_j.
    xi_terms = {int(s[j]): scores[j] for j in range(n)}
    xi_terms.update({int(y[j]): scores[j] for j in range(n)})

    # F_B = sum c_j (x_j - s_j).
    bench_terms = {int(x[j]): scores[j] for j in range(n)}
    for j in range(n):
        bench_terms[int(s[j])] = bench_terms.get(int(s[j]), 0.0) - scores[j]

    if xi_minimum is not None:
        rows.add(xi_terms, lower=float(xi_minimum))
    if bench_minimum is not None:
        rows.add(bench_terms, lower=float(bench_minimum))

    objective = np.zeros(nvar)
    objective[x] += bench_objective_weight * scores
    objective[s] += (xi_objective_weight - bench_objective_weight) * scores
    objective[y] += xi_objective_weight * scores

    # Stable tie-break; too small to change the substantive objective.
    objective[x] += TIE_EPS * stable
    objective[s] += TIE_EPS * stable
    objective[y] += TIE_EPS * stable

    z = _solve_binary_milp(objective, rows, label)
    squad_idx = np.flatnonzero(z[x])
    starter_idx = np.flatnonzero(z[s])
    captain_idx = int(np.flatnonzero(z[y])[0])

    return _package_solution(
        pool,
        squad_idx,
        starter_idx,
        captain_idx,
        metadata={
            "design": label,
            "design_family": "Joint Pareto",
            "xi_cap": np.nan,
            "bench_budget_rule": "endogenous",
            "bench_budget_available": np.nan,
            "delta": np.nan,
        },
    )


def build_pareto_frontier(pool, deltas=PARETO_DELTAS):
    """Use epsilon constraints, followed by lexicographic refinement."""
    xi_best = solve_joint(
        pool,
        xi_objective_weight=1.0,
        bench_objective_weight=0.0,
        label="Joint XI optimum",
    )
    xi_star = xi_best["xi_score"]

    solutions = []
    for delta in deltas:
        xi_floor = (1.0 - float(delta)) * xi_star
        # Allow only a numerical tolerance at delta=0.
        xi_floor -= OBJECTIVE_TOL

        phase_1 = solve_joint(
            pool,
            xi_objective_weight=0.0,
            bench_objective_weight=1.0,
            xi_minimum=xi_floor,
            label=f"Joint Pareto delta={delta:.3f}, bench phase",
        )
        bench_floor = phase_1["bench_score"] - OBJECTIVE_TOL

        # Lexicographic refinement: among maximum-bench solutions, maximize XI.
        refined = solve_joint(
            pool,
            xi_objective_weight=1.0,
            bench_objective_weight=0.0,
            xi_minimum=xi_floor,
            bench_minimum=bench_floor,
            label=f"Joint Pareto delta={delta:.3f}",
        )
        refined["delta"] = float(delta)
        refined["xi_floor"] = float(xi_floor)
        refined["xi_loss_pct"] = 100.0 * (xi_star - refined["xi_score"]) / abs(xi_star)
        refined["design"] = f"Joint Pareto δ={100*delta:.1f}%"
        solutions.append(refined)

    # Mark nondominated points among the computed epsilon solutions.
    for candidate in solutions:
        dominated = False
        for other in solutions:
            weakly_better = (
                other["xi_score"] >= candidate["xi_score"] - OBJECTIVE_TOL
                and other["bench_score"] >= candidate["bench_score"] - OBJECTIVE_TOL
            )
            strictly_better = (
                other["xi_score"] > candidate["xi_score"] + OBJECTIVE_TOL
                or other["bench_score"] > candidate["bench_score"] + OBJECTIVE_TOL
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        candidate["is_nondominated"] = not dominated

    nondominated = [s for s in solutions if s["is_nondominated"]]
    xi_values = np.array([s["xi_score"] for s in nondominated], dtype=float)
    bench_values = np.array([s["bench_score"] for s in nondominated], dtype=float)

    def normalize(values):
        span = values.max() - values.min()
        return np.ones_like(values) if np.isclose(span, 0.0) else (values - values.min()) / span

    xi_norm = normalize(xi_values)
    bench_norm = normalize(bench_values)
    distances = np.sqrt((1.0 - xi_norm) ** 2 + (1.0 - bench_norm) ** 2)
    compromise_index = int(np.argmin(distances))

    for s in solutions:
        s["is_compromise"] = False
    nondominated[compromise_index]["is_compromise"] = True

    return solutions, xi_star

# %%
# Common out-of-sample evaluator: no transfers and no test-outcome tuning
def _actual_lookup(test_data):
    """Map each evaluation player--GW pair to realized minutes and points."""
    # A defensive aggregation in case a player--GW key is duplicated.
    grouped = (
        test_data.groupby(["GW", "name"], as_index=False)
        .agg(minutes=("minutes", "max"), total_points=("total_points", "sum"))
    )
    return {
        (int(row.GW), row.name): (float(row.minutes), float(row.total_points))
        for row in grouped.itertuples(index=False)
    }


evaluation_lookup = _actual_lookup(evaluation_data)
weekly_forecast_cache = {}


def _weekly_forecasts_for_squad(solution, method_spec, gw):
    """Cache pre-GW forecasts by method, gameweek, and player."""
    method_label = method_spec["label"]
    players = list(solution["squad"].itertuples(index=False))
    missing_keys = []
    missing_tasks = []

    for player in players:
        key = (method_label, int(gw), player.name)
        if key not in weekly_forecast_cache:
            player_history = raw.loc[
                (raw["name"] == player.name) & (raw["GW"] < gw),
                ["GW", "total_points"],
            ].sort_values("GW")
            missing_keys.append(key)
            missing_tasks.append(
                (
                    player_history["total_points"].tolist(),
                    method_spec,
                    FINAL_GW - gw + 1,
                )
            )

    if missing_tasks:
        values = _forecast_many(missing_tasks, method_spec)
        weekly_forecast_cache.update(zip(missing_keys, values))

    rows = []
    for player in players:
        key = (method_label, int(gw), player.name)
        rows.append(
            {
                "name": player.name,
                "team": player.team,
                "position": player.position,
                "value": player.value,
                "forecast": weekly_forecast_cache[key],
            }
        )
    return pd.DataFrame(rows)


def _is_partial_formation_feasible(position_counts, final_size):
    """Check whether empty slots can be reserved for unmet formation minima."""
    empty_slots = 11 - int(final_size)
    for position, (minimum, maximum) in XI_BOUNDS.items():
        count = position_counts.get(position, 0)
        if count > maximum:
            return False
        if count + empty_slots < minimum:
            return False
    return True


def _apply_automatic_substitutions(planned_xi, ordered_bench, gw):
    """Apply GK replacement and formation-valid outfield autosubs."""
    planned_xi = planned_xi.copy().reset_index(drop=True)
    ordered_bench = ordered_bench.copy().reset_index(drop=True)

    def played(name):
        return evaluation_lookup.get((int(gw), name), (0.0, 0.0))[0] > 0

    playing_starters = planned_xi.loc[planned_xi["name"].map(played)].copy()
    absent_starters = planned_xi.loc[~planned_xi["name"].map(played)].copy()
    substitutes = []

    # Goalkeeper substitution is position-specific.
    starting_gk_played = any(playing_starters["position"] == "GK")
    if not starting_gk_played:
        bench_gk = ordered_bench.loc[
            (ordered_bench["position"] == "GK") & ordered_bench["name"].map(played)
        ]
        if not bench_gk.empty:
            replacement = bench_gk.iloc[[0]]
            playing_starters = pd.concat([playing_starters, replacement], ignore_index=True)
            substitutes.append(str(replacement.iloc[0]["name"]))

    # Preserve the declared order among the three outfield bench players.
    playable_outfield = ordered_bench.loc[
        (ordered_bench["position"] != "GK") & ordered_bench["name"].map(played)
    ].reset_index(drop=True)
    absent_outfield_count = int((absent_starters["position"] != "GK").sum())

    # Select the maximum feasible number of substitutes; ties use bench order.
    chosen = ()
    max_k = min(absent_outfield_count, len(playable_outfield))
    for k in range(max_k, -1, -1):
        feasible_combinations = []
        for combo in combinations(range(len(playable_outfield)), k):
            additions = playable_outfield.iloc[list(combo)] if combo else playable_outfield.iloc[[]]
            candidate = pd.concat([playing_starters, additions], ignore_index=True)
            counts = candidate["position"].value_counts().to_dict()
            if _is_partial_formation_feasible(counts, len(candidate)):
                feasible_combinations.append(combo)
        if feasible_combinations:
            chosen = min(feasible_combinations)
            break

    if chosen:
        additions = playable_outfield.iloc[list(chosen)]
        playing_starters = pd.concat([playing_starters, additions], ignore_index=True)
        substitutes.extend(additions["name"].astype(str).tolist())

    return playing_starters.drop_duplicates("name"), absent_starters, substitutes


def _fixed_squad_weekly_lineup(solution, weekly_forecasts):
    """Update only XI membership and bench order; keep captain/vice fixed."""
    squad = solution["squad"][["name", "team", "position", "value"]].copy()
    score_map = weekly_forecasts.set_index("name")["forecast"]
    squad["forecast"] = squad["name"].map(score_map).fillna(0.0)
    squad = squad.reset_index(drop=True)
    n = len(squad)

    rows = _MilpRows(n)
    rows.add({j: 1 for j in range(n)}, lower=11, upper=11)
    for position, (minimum, maximum) in XI_BOUNDS.items():
        idx = np.flatnonzero(squad["position"].to_numpy() == position)
        rows.add({int(j): 1 for j in idx}, lower=minimum, upper=maximum)

    # These fixed assignments avoid introducing dynamic captaincy into this test.
    for fixed_name in [solution["captain"], solution["vice_captain"]]:
        idx = np.flatnonzero(squad["name"].to_numpy() == fixed_name)
        rows.add({int(idx[0]): 1}, lower=1, upper=1)

    objective = squad["forecast"].to_numpy(float) + TIE_EPS * _stable_weights(squad)
    z = _solve_binary_milp(objective, rows, "fixed-squad weekly XI")
    starters = squad.loc[np.flatnonzero(z)].copy()
    bench = squad.loc[np.flatnonzero(1 - z)].copy()
    return starters.reset_index(drop=True), bench.reset_index(drop=True)


def evaluate_solution(solution, method_spec, policy="static"):
    """
    Score GW27--38 with official-style autosubs.

    policy="static" keeps the optimized GW27 XI and bench order.
    policy="sequential" updates only XI membership and bench order from GW<t data;
    the 15-player squad, captain, and vice-captain remain fixed.
    """
    if policy not in {"static", "sequential"}:
        raise ValueError("policy must be 'static' or 'sequential'.")

    records = []
    for gw in range(TARGET_GW, FINAL_GW + 1):
        if policy == "static":
            planned_xi = solution["starters"].copy()
            bench = solution["bench"].copy()
        else:
            weekly_forecasts = _weekly_forecasts_for_squad(
                solution, method_spec, gw
            )
            planned_xi, bench = _fixed_squad_weekly_lineup(solution, weekly_forecasts)

        # Outfield bench order is predicted score descending; reserve GK is separate.
        outfield_bench = bench.loc[bench["position"] != "GK"].sort_values(
            ["forecast", "name"], ascending=[False, True]
        )
        goalkeeper_bench = bench.loc[bench["position"] == "GK"].sort_values(
            ["forecast", "name"], ascending=[False, True]
        )
        ordered_bench = pd.concat([outfield_bench, goalkeeper_bench], ignore_index=True)

        final_players, absentees, substitutes = _apply_automatic_substitutions(
            planned_xi, ordered_bench, gw
        )
        point_map = {
            name: evaluation_lookup.get((gw, name), (0.0, 0.0))[1]
            for name in final_players["name"]
        }
        base_points = float(sum(point_map.values()))

        captain_used = None
        captain_bonus = 0.0
        if solution["captain"] in point_map:
            captain_used = solution["captain"]
            captain_bonus = float(point_map[captain_used])
        elif solution["vice_captain"] in point_map:
            captain_used = solution["vice_captain"]
            captain_bonus = float(point_map[captain_used])

        substitute_points = float(sum(point_map.get(name, 0.0) for name in substitutes))
        records.append(
            {
                "GW": gw,
                "policy": policy,
                "weekly_points": base_points + captain_bonus,
                "base_points": base_points,
                "captain_bonus": captain_bonus,
                "captain_used": captain_used,
                "planned_absences": len(absentees),
                "autosubs_used": len(substitutes),
                "substitute_points": substitute_points,
                "final_players": len(final_players),
                # Store complete policy state so the fixed-squad assumptions can
                # be checked mechanically. Squad membership and captaincy remain
                # invariant; only the planned XI and bench order may change.
                "squad_members": "|".join(sorted(solution["squad"]["name"].astype(str))),
                "planned_xi_members": "|".join(sorted(planned_xi["name"].astype(str))),
                "bench_order": "|".join(ordered_bench["name"].astype(str)),
                "captain": str(solution["captain"]),
                "vice_captain": str(solution["vice_captain"]),
            }
        )

    result = pd.DataFrame(records)
    result["cumulative_points"] = result["weekly_points"].cumsum()
    return result

# %%
# Run all controlled designs
all_solutions = []
pareto_rows = []

for method_spec in METHOD_SPECS:
    method_label = method_spec["label"]
    pool = forecast_tables[method_label]
    print(f"\nOptimizing designs for {method_label} ...")

    # Budget sweep for both interpretations of the second-stage budget.
    for xi_cap in TWO_STAGE_XI_CAPS:
        for rule, rule_label in [
            ("fixed_split", "fixed allocation"),
            ("actual_remainder", "actual remainder"),
        ]:
            solution = solve_two_stage(
                pool,
                xi_cap=xi_cap,
                bench_budget_rule=rule,
                label=f"Two-stage {rule_label}, XI cap £{xi_cap:g}m",
            )
            solution["method"] = method_label
            solution["method_spec"] = method_spec
            solution["solution_id"] = (
                f"{method_label} | two_stage | {rule} | xi_cap={xi_cap:g}"
            )
            all_solutions.append(solution)

    frontier, xi_star = build_pareto_frontier(pool)
    for solution in frontier:
        solution["method"] = method_label
        solution["method_spec"] = method_spec
        solution["xi_star"] = xi_star
        solution["solution_id"] = (
            f"{method_label} | joint_pareto | delta={solution['delta']:.3f}"
        )
        all_solutions.append(solution)
        pareto_rows.append(
            {
                "method": method_label,
                "solution_id": solution["solution_id"],
                "delta": solution["delta"],
                "xi_score": solution["xi_score"],
                "bench_score": solution["bench_score"],
                "xi_loss_pct": solution["xi_loss_pct"],
                "xi_spend": solution["xi_spend"],
                "bench_spend": solution["bench_spend"],
                "total_spend": solution["total_spend"],
                "formation": solution["formation"],
                "is_nondominated": solution["is_nondominated"],
                "is_compromise": solution["is_compromise"],
            }
        )

pareto_frontier = pd.DataFrame(pareto_rows)
print(f"Created {len(all_solutions)} roster designs.")
display(pareto_frontier.round(3))

# %%
# Evaluate GW27--38, summarize, plot, and export without overwriting old results
design_rows = []
roster_rows = []
weekly_frames = []

for solution in all_solutions:
    design_rows.append(
        {
            "method": solution["method"],
            "solution_id": solution["solution_id"],
            "design": solution["design"],
            "design_family": solution["design_family"],
            "xi_cap": solution["xi_cap"],
            "bench_budget_rule": solution["bench_budget_rule"],
            "delta": solution.get("delta", np.nan),
            "is_nondominated": solution.get("is_nondominated", np.nan),
            "is_compromise": solution.get("is_compromise", False),
            "xi_score": solution["xi_score"],
            "bench_score": solution["bench_score"],
            "xi_spend": solution["xi_spend"],
            "bench_spend": solution["bench_spend"],
            "total_spend": solution["total_spend"],
            "formation": solution["formation"],
            "captain": solution["captain"],
            "vice_captain": solution["vice_captain"],
        }
    )

    starter_names = set(solution["starters"]["name"])
    for row in solution["squad"].itertuples(index=False):
        roster_rows.append(
            {
                "method": solution["method"],
                "solution_id": solution["solution_id"],
                "name": row.name,
                "team": row.team,
                "position": row.position,
                "value": row.value,
                "forecast": row.forecast,
                "role": "XI" if row.name in starter_names else "Bench",
                "is_captain": row.name == solution["captain"],
                "is_vice_captain": row.name == solution["vice_captain"],
            }
        )

    for policy in EVALUATION_POLICIES:
        weekly = evaluate_solution(solution, solution["method_spec"], policy=policy)
        weekly.insert(0, "solution_id", solution["solution_id"])
        weekly.insert(0, "method", solution["method"])
        weekly_frames.append(weekly)

design_results = pd.DataFrame(design_rows)
roster_membership = pd.DataFrame(roster_rows)
weekly_results = pd.concat(weekly_frames, ignore_index=True)

design_summary = (
    weekly_results.groupby(["method", "solution_id", "policy"], as_index=False)
    .agg(
        realized_points=("weekly_points", "sum"),
        mean_weekly_points=("weekly_points", "mean"),
        total_autosubs=("autosubs_used", "sum"),
        substitute_points=("substitute_points", "sum"),
        total_planned_absences=("planned_absences", "sum"),
    )
    .merge(design_results, on=["method", "solution_id"], how="left")
)

# Main comparison only: the published default, corrected default, joint delta=0,
# and the predicted-space Pareto compromise. No method is selected by test score.
main_mask = (
    (
        (design_summary["design_family"] == "Two-stage")
        & np.isclose(design_summary["xi_cap"], DEFAULT_XI_CAP, equal_nan=False)
    )
    | (
        (design_summary["design_family"] == "Joint Pareto")
        & (
            np.isclose(design_summary["delta"], 0.0, equal_nan=False)
            | design_summary["is_compromise"].fillna(False).astype(bool)
        )
    )
)
main_comparison = design_summary.loc[main_mask].copy()


def _external_transfer_count(group):
    """Count changes in 15-player membership across adjacent gameweeks."""
    ordered = group.sort_values("GW")
    memberships = [set(str(value).split("|")) for value in ordered["squad_members"]]
    return int(
        sum(len(current.difference(previous)) for previous, current in zip(memberships, memberships[1:]))
    )


# Machine-checkable policy audit. A passing row verifies fixed squad membership,
# fixed captain/vice-captain assignments, and zero external transfers. The last
# two fields document whether the permitted XI and bench-order updates occurred.
policy_audit_rows = []
for (method, solution_id, policy), group in weekly_results.groupby(
    ["method", "solution_id", "policy"], sort=False
):
    external_transfers = _external_transfer_count(group)
    unique_squads = int(group["squad_members"].nunique())
    unique_captains = int(group["captain"].nunique())
    unique_vice_captains = int(group["vice_captain"].nunique())
    policy_audit_rows.append(
        {
            "method": method,
            "solution_id": solution_id,
            "policy": policy,
            "gameweeks": int(group["GW"].nunique()),
            "unique_squad_memberships": unique_squads,
            "external_transfers": external_transfers,
            "transfer_hit_points": 0,
            "unique_captains": unique_captains,
            "unique_vice_captains": unique_vice_captains,
            "unique_starting_lineups": int(group["planned_xi_members"].nunique()),
            "unique_bench_orders": int(group["bench_order"].nunique()),
            "audit_pass": bool(
                unique_squads == 1
                and external_transfers == 0
                and unique_captains == 1
                and unique_vice_captains == 1
            ),
        }
    )
policy_audit = pd.DataFrame(policy_audit_rows)
if not policy_audit["audit_pass"].all():
    failed = policy_audit.loc[~policy_audit["audit_pass"]]
    raise AssertionError(
        "Fixed-squad policy audit failed:\n" + failed.to_string(index=False)
    )

# Starting-availability sensitivity analysis. The pre-GW27 probability that a
# player starts is estimated with Beta(1,1)
# smoothing. Weeks missing from the player panel contribute zero starts because
# the denominator covers every training gameweek. The availability-aware
# coefficient is p_j c_j; the robust coefficient also subtracts
# Gamma |c_j| sqrt(p_j(1-p_j)), which penalizes uncertain selection.
start_counts = (
    training_data.groupby(["name", "GW"], as_index=False)["starts"].max()
    .assign(started=lambda frame: (frame["starts"] > 0).astype(int))
    .groupby("name", as_index=False)
    .agg(training_starts=("started", "sum"))
)
start_counts["starting_probability"] = (
    start_counts["training_starts"] + 1.0
) / ((TARGET_GW - 1) + 2.0)
start_counts["starting_probability"] = start_counts[
    "starting_probability"
].clip(1.0 / (TARGET_GW + 1.0), (TARGET_GW - 0.0) / (TARGET_GW + 1.0))
starting_probability_map = start_counts.set_index("name")["starting_probability"]

availability_rows = []
availability_roster_rows = []
for method_spec in METHOD_SPECS:
    method_label = method_spec["label"]
    raw_pool = forecast_tables[method_label].copy()
    raw_pool["starting_probability"] = (
        raw_pool["name"].map(starting_probability_map).fillna(1.0 / (TARGET_GW + 1.0))
    )
    raw_pool["raw_forecast"] = raw_pool["forecast"]

    baseline = next(
        solution for solution in all_solutions
        if solution["method"] == method_label
        and solution["design_family"] == "Joint Pareto"
        and np.isclose(solution.get("delta", np.nan), 0.0, equal_nan=False)
    )
    candidates = [("Unadjusted joint $\\delta=0$", baseline, raw_pool)]

    for availability_design in ["Expected availability", "Availability robust"]:
        adjusted_pool = raw_pool.copy()
        probability = adjusted_pool["starting_probability"].to_numpy(float)
        coefficient = adjusted_pool["raw_forecast"].to_numpy(float)
        adjusted = probability * coefficient
        if availability_design == "Availability robust":
            adjusted -= (
                AVAILABILITY_GAMMA
                * np.abs(coefficient)
                * np.sqrt(probability * (1.0 - probability))
            )
        adjusted_pool["forecast"] = adjusted
        frontier, _ = build_pareto_frontier(adjusted_pool, deltas=[0.0])
        solution = frontier[0]
        solution["method"] = method_label
        solution["method_spec"] = method_spec
        solution["solution_id"] = (
            f"{method_label} | availability | "
            f"{availability_design.lower().replace(' ', '_')}"
        )
        candidates.append((availability_design, solution, adjusted_pool))

    for availability_design, solution, source_pool in candidates:
        weekly = evaluate_solution(solution, method_spec, policy="static")
        squad_names = solution["squad"]["name"].astype(str)
        squad_probabilities = squad_names.map(starting_probability_map).fillna(
            1.0 / (TARGET_GW + 1.0)
        )
        availability_rows.append(
            {
                "method": method_label,
                "availability_design": availability_design,
                "gamma": 0.0 if availability_design != "Availability robust" else AVAILABILITY_GAMMA,
                "realized_points": float(weekly["weekly_points"].sum()),
                "mean_squad_start_probability": float(squad_probabilities.mean()),
                "minimum_squad_start_probability": float(squad_probabilities.min()),
                "players_below_70pct_start_probability": int((squad_probabilities < 0.70).sum()),
                "total_autosubs": int(weekly["autosubs_used"].sum()),
                "substitute_points": float(weekly["substitute_points"].sum()),
                "formation": solution["formation"],
                "captain": solution["captain"],
                "vice_captain": solution["vice_captain"],
            }
        )
        source_index = source_pool.set_index("name")
        starter_names = set(solution["starters"]["name"])
        for row in solution["squad"].itertuples(index=False):
            availability_roster_rows.append(
                {
                    "method": method_label,
                    "availability_design": availability_design,
                    "name": row.name,
                    "team": row.team,
                    "position": row.position,
                    "value": row.value,
                    "raw_forecast": float(source_index.loc[row.name, "raw_forecast"]),
                    "starting_probability": float(source_index.loc[row.name, "starting_probability"]),
                    "adjusted_cost": float(row.forecast),
                    "role": "XI" if row.name in starter_names else "Bench",
                    "is_captain": row.name == solution["captain"],
                    "is_vice_captain": row.name == solution["vice_captain"],
                }
            )

availability_comparison = pd.DataFrame(availability_rows)
availability_rosters = pd.DataFrame(availability_roster_rows)

# Export detailed, machine-readable tables without touching historical outputs.
pareto_frontier.to_csv(OUTPUT_DIR / "reviewer1_comment2_pareto_frontier.csv", index=False)
design_results.to_csv(OUTPUT_DIR / "reviewer1_comment2_design_results.csv", index=False)
roster_membership.to_csv(OUTPUT_DIR / "reviewer1_comment2_rosters.csv", index=False)
weekly_results.to_csv(OUTPUT_DIR / "reviewer1_comment2_weekly_scores.csv", index=False)
design_summary.to_csv(OUTPUT_DIR / "reviewer1_comment2_summary.csv", index=False)
main_comparison.to_csv(OUTPUT_DIR / "reviewer1_comment2_main_comparison.csv", index=False)
policy_audit.to_csv(OUTPUT_DIR / "reviewer1_comment1_policy_audit.csv", index=False)
availability_comparison.to_csv(
    OUTPUT_DIR / "reviewer1_comment5_availability_comparison.csv", index=False
)
availability_rosters.to_csv(
    OUTPUT_DIR / "reviewer1_comment5_availability_rosters.csv", index=False
)

# Pareto frontier in predicted objective space.
fig, axes = plt.subplots(1, len(METHOD_SPECS), figsize=(6.4 * len(METHOD_SPECS), 4.8), squeeze=False)
for ax, method_spec in zip(axes.ravel(), METHOD_SPECS):
    method_label = method_spec["label"]
    part = pareto_frontier.loc[pareto_frontier["method"] == method_label].sort_values("xi_score")
    ax.plot(part["xi_score"], part["bench_score"], color="#2A6F97", linewidth=1.8, zorder=1)
    scatter = ax.scatter(
        part["xi_score"],
        part["bench_score"],
        c=100 * part["delta"],
        cmap="viridis",
        s=65,
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
    compromise = part.loc[part["is_compromise"]]
    ax.scatter(
        compromise["xi_score"],
        compromise["bench_score"],
        marker="*",
        s=240,
        color="#D1495B",
        edgecolor="black",
        label="Predicted-space compromise",
        zorder=3,
    )
    for row in part.itertuples(index=False):
        ax.annotate(f"{100*row.delta:g}%", (row.xi_score, row.bench_score), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.text(0.03, 0.96, method_label, transform=ax.transAxes,
            ha="left", va="top", weight="bold")
    ax.set_xlabel(r"Predicted XI + captain value ($F_{XI}$)")
    ax.set_ylabel(r"Predicted bench value ($F_B$)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27),
              frameon=True, fontsize=8)
fig.colorbar(scatter, ax=axes.ravel().tolist(), label=r"Permitted XI loss, $\delta$ (%)", shrink=0.85)
fig.subplots_adjust(wspace=0.25, right=0.90)
fig.savefig(
    OUTPUT_DIR / "reviewer1_comment2_pareto_frontier.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Realized total-points comparison for the predeclared main designs.
plot_data = main_comparison.copy()
plot_data["short_design"] = np.select(
    [
        (plot_data["design_family"] == "Two-stage") & (plot_data["bench_budget_rule"] == "fixed_split"),
        (plot_data["design_family"] == "Two-stage") & (plot_data["bench_budget_rule"] == "actual_remainder"),
        (plot_data["design_family"] == "Joint Pareto") & np.isclose(plot_data["delta"], 0.0, equal_nan=False),
        (plot_data["design_family"] == "Joint Pareto") & plot_data["is_compromise"].fillna(False).astype(bool),
    ],
    ["Two-stage fixed split", "Two-stage actual remainder", "Joint δ=0", "Joint compromise"],
    default=plot_data["design"],
)

fig, axes = plt.subplots(len(METHOD_SPECS), 1, figsize=(10, 4.5 * len(METHOD_SPECS)), squeeze=False)
colors = {"static": "#4C78A8", "sequential": "#F58518"}
for ax, method_spec in zip(axes.ravel(), METHOD_SPECS):
    part = plot_data.loc[plot_data["method"] == method_spec["label"]].copy()
    pivot = part.pivot_table(index="short_design", columns="policy", values="realized_points", aggfunc="first")
    desired_order = ["Two-stage fixed split", "Two-stage actual remainder", "Joint δ=0", "Joint compromise"]
    pivot = pivot.reindex([name for name in desired_order if name in pivot.index])
    pivot.plot(kind="bar", ax=ax, color=[colors.get(c, "gray") for c in pivot.columns], width=0.75)
    ax.text(0.02, 0.96, method_spec["label"], transform=ax.transAxes,
            ha="left", va="top", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Realized GW27--38 points")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Lineup policy", loc="upper left",
              bbox_to_anchor=(1.01, 1.0), frameon=True)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "reviewer1_comment2_realized_comparison.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

columns_to_show = [
    "method", "policy", "design", "xi_score", "bench_score", "xi_spend",
    "bench_spend", "total_spend", "realized_points", "total_autosubs",
    "substitute_points", "is_compromise",
]
display(main_comparison[columns_to_show].sort_values(["method", "policy", "design"]).round(3))
print(f"Roster-design outputs saved in: {OUTPUT_DIR.resolve()}")
if FORECAST_EXECUTOR is not None:
    FORECAST_EXECUTOR.shutdown(wait=True)
elapsed_seconds = time.perf_counter() - START_TIME
print(f"Total elapsed time: {elapsed_seconds / 60:.2f} minutes")

# %% [markdown]
# ### Generated-output guide
#
# Use the generated files as follows:
#
# - a predicted-space Pareto frontier for XI and bench quality;
# - a main table comparing the fixed split, actual-remainder rule, joint optimum,
#   and predicted-space compromise;
# - the full budget sensitivity grid and all Pareto solutions;
# - auditable player-level roster membership and role assignments;
# - gameweek-level realized scores and automatic-substitution diagnostics; and
# - a policy audit verifying invariant squad and captaincy assignments, zero
#   external transfers, and zero transfer-hit points.
#
# Do not choose the XI cap or Pareto point by taking the highest realized GW27--38
# score. That would tune on the test period. The `is_compromise` flag is computed
# entirely in forecast-objective space before realized scores are examined.
