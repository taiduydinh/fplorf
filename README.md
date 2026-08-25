# FPL forecasting and roster-optimization pipeline

This repository reproduces the computational results for the study
*A data-driven framework for team selection in Fantasy Premier League*.
Starting from the raw player--gameweek panel, the pipeline rebuilds forecasting
cost vectors, solves the roster and lineup optimization models, evaluates the
resulting policies, and exports all tables and figures.

## Quick start

Install the required packages in the intended Python environment:

```bash
python -m pip install -r requirements_fpl_pipeline.txt
```

Keep the main script, the three `fpl_*_experiments.py` modules, the font, and
the dataset in the same directory. Then run:

```bash
python -u run_fpl_paper_pipeline.py \
  --data merged_gw_2324.csv \
  --output fpl_paper_outputs \
  --workers 8 \
  --bootstrap 500 \
  --simulation-draws 1000
```

For a detached server run and monitoring commands, see
`SERVER_RUN_INSTRUCTIONS.md`.

## What the pipeline computes

1. Validates and standardizes the player--gameweek data.
2. Uses GW1--26 for forecast construction and model development.
3. Builds cost vectors for averaging, simulation, smoothing, ARIMA, robust,
   alternative-objective, linear, and Ridge-hybrid specifications.
4. Solves legal two-stage and joint 15-player roster models under FPL budget,
   positional, formation, captaincy, and club constraints.
5. Evaluates static and fixed-squad sequential lineup policies over GW27--38,
   including automatic substitutions.
6. Runs starter--bench Pareto, Ridge/LASSO stability, and starting-availability
   sensitivity analyses.
7. Writes indexed PDF figures, CSV tables, player-level rosters, configurations,
   software versions, provenance records, checksums, and completion markers.

No historical score matrix or saved roster snapshot is used. All 94 internal
weekly series are recomputed from `merged_gw_2324.csv`. The four Santoro series
are the only externally supplied numerical benchmarks and are labeled as such
in the exported provenance.

## Output layout

- `main_figures/`: seven indexed main-paper PDFs.
- `supplementary_figures/`: 33 indexed supplementary PDFs.
- `tables/`: compact reporting tables and policy audits.
- `results/`: weekly scores, cost vectors, rosters, and provenance records.
- `analysis_outputs/roster_design/`: detailed roster-design and Pareto results.
- `analysis_outputs/feature_regularization/`: Ridge/LASSO and stability results.
- `analysis_outputs/full_internal_raw_recomputation/`: complete 94-series run.
- `figure_manifest.csv` and `output_manifest.csv`: file existence, size, and
  SHA-256 checksums.
- `RUN_COMPLETED_SUCCESSFULLY.txt`: created only after all required outputs pass
  validation.

All figures are title-free vector PDFs, specify 300 DPI for rasterized content,
use tight bounding boxes, and place legends away from the plotted data. They
are generated at their intended printed width: 6.70 inches for ordinary
full-width figures and 9.70 inches for the landscape method-similarity matrix.
Global labels are 9--10 pt, and no explicit annotation or legend text is below
8 pt.

For LaTeX placement, use each portrait PDF at full text width. Do not combine
the former Figures 6--15 into half-width subfigures. Place
`figS05_method_similarity.pdf` alone on a landscape supplementary page; it was
specifically designed for that orientation.

## Reproducibility safeguards

- Forecasts and hyperparameters use only information available before GW27.
- Ridge and LASSO penalties are selected with rolling-origin validation.
- Bootstrap samples retain each selected player's complete training history.
- Budget caps, regularization parameters, and Pareto compromises are not chosen
  using realized GW27--38 scores.
- Fixed-squad sequential policies keep squad membership and captaincy invariant;
  only the planned XI and bench order can change.
- Randomized components use deterministic seeds recorded in the run metadata.

Use `--help` to view all command-line options. The `--smoke-test --skip-shap`
combination provides a quick installation and output-layout check; it is not a
substitute for the full reported run.

## Source-code organization

- `run_fpl_paper_pipeline.py`: command-line interface, figure generation,
  output validation, and manifests.
- `fpl_roster_design_experiments.py`: forecasts, two-stage and joint roster
  models, Pareto analysis, policy evaluation, and availability sensitivity.
- `fpl_regularization_experiments.py`: Ridge/LASSO tuning, player-cluster
  bootstrap stability, and associated optimization/evaluation routines.
- `fpl_internal_experiments.py`: full 94-series raw-data recomputation and
  independent consistency checks.
- `run_fpl_paper_pipeline_single.py`: a small compatibility launcher for older
  commands; it contains no embedded or encoded source code.
