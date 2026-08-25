# FPL forecasting and roster-optimization pipeline

This repository reproduces the computational results for the study
*A data-driven framework for team selection in Fantasy Premier League*.
Starting from the raw player--gameweek panel, the pipeline rebuilds forecasting
cost vectors, solves the roster and lineup optimization models, evaluates the
resulting policies, and exports all manuscript tables and figures.

The workflow can be run on a Windows, macOS, or Linux laptop, or as a detached
job on a Linux server. A server is helpful for speed but is not required.

## Repository contents

Keep the following files together in the repository root:

- `run_fpl_paper_pipeline.py`: command-line entry point, figure generation,
  output validation, and manifests.
- `fpl_roster_design_experiments.py`: forecasting, two-stage and joint roster
  models, Pareto analysis, policy evaluation, and availability sensitivity.
- `fpl_regularization_experiments.py`: Ridge/LASSO tuning, player-cluster
  bootstrap stability, and associated optimization and evaluation routines.
- `fpl_internal_experiments.py`: complete 94-series raw-data recomputation and
  independent consistency checks.
- `merged_gw_2324.csv`: player--gameweek input data used by the study.
- `lmroman10-regular.otf`: font used to make figure typography consistent.
- `requirements_fpl_pipeline.txt`: minimum supported package versions.

The committed `fpl_paper_outputs_final/` directory contains the validated
reference outputs from the reported run. When reproducing the analysis, use a
new output directory so the reference package remains unchanged.

## Software requirements

Python 3.11 is recommended because the reported run used Python 3.11.15. A
current Python 3.10 or newer environment with the package versions listed in
`requirements_fpl_pipeline.txt` should also work.

Clone the repository and enter its directory:

```bash
git clone https://github.com/taiduydinh/fplorf.git
cd fplorf
```

### Option A: standard virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_fpl_pipeline.txt
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_fpl_pipeline.txt
```

On Windows Command Prompt, activate the environment with:

```bat
.venv\Scripts\activate.bat
```

### Option B: Conda

The environment name is arbitrary. For example:

```bash
conda create -n fplorf python=3.11 -y
conda activate fplorf
python -m pip install --upgrade pip
python -m pip install -r requirements_fpl_pipeline.txt
```

## Quick smoke test

Run this short test after installation. It verifies dependencies, input loading,
optimization feasibility, and the output layout, but it does not reproduce the
publication results:

```bash
python -u run_fpl_paper_pipeline.py --data merged_gw_2324.csv --output fpl_paper_smoke_test --workers 2 --bootstrap 10 --simulation-draws 20 --smoke-test --skip-shap
```

A successful smoke test creates:

```text
fpl_paper_smoke_test/RUN_COMPLETED_SUCCESSFULLY.txt
```

The `--skip-shap` option is intended only for the smoke test. Do not use it for
the full publication run.

## Full run on a local laptop or desktop

Run the command from the repository root. The following one-line form works in
Windows PowerShell, Windows Command Prompt, macOS, and Linux:

```bash
python -u run_fpl_paper_pipeline.py --data merged_gw_2324.csv --output fpl_paper_outputs_local --workers 4 --bootstrap 500 --simulation-draws 1000 --seed 20260824
```

The number of workers affects runtime rather than the intended analysis. A
typical laptop can use `--workers 2` or `--workers 4`. A machine with more CPU
cores can use `--workers 8`. Keep `--bootstrap 500`, `--simulation-draws 1000`,
and `--seed 20260824` unchanged when reproducing the reported results.

The full run is computationally intensive because it fits many ARIMA models,
performs 500 player-cluster bootstrap replications, solves many mixed-integer
programs, and regenerates every figure. The reported Linux server run completed
in approximately 35.62 minutes with eight forecast workers; a laptop may take
considerably longer. Keep the terminal open until the local run completes.

## Full detached run on a Linux server

Clone the repository, enter its directory, and activate either the virtual
environment or Conda environment created above. If the work is performed
inside Docker, first enter the container and activate the Python environment;
the remaining commands are unchanged.

Start the full run with `nohup` so that it continues after the terminal or SSH
session is closed:

```bash
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u run_fpl_paper_pipeline.py \
  --data merged_gw_2324.csv \
  --output fpl_paper_outputs_server \
  --workers 8 \
  --bootstrap 500 \
  --simulation-draws 1000 \
  --seed 20260824 \
  > fpl_paper_pipeline.log 2>&1 &
```

The shell prints a process identifier. It can also be displayed immediately
after launch with:

```bash
echo $!
```

Monitor the log:

```bash
tail -f fpl_paper_pipeline.log
```

Press `Ctrl+C` to stop following the log; this does not stop the experiment.

Check whether the process is still running:

```bash
if pgrep -af "[r]un_fpl_paper_pipeline.py" >/dev/null; then
    echo "PIPELINE IS STILL RUNNING"
    pgrep -af "[r]un_fpl_paper_pipeline.py"
else
    echo "NO PIPELINE PROCESS IS RUNNING"
fi
```

Inspect the most recent log messages:

```bash
tail -n 50 fpl_paper_pipeline.log
```

Successful completion is confirmed by both the log message
`EXPERIMENT COMPLETED SUCCESSFULLY` and this file:

```text
fpl_paper_outputs_server/RUN_COMPLETED_SUCCESSFULLY.txt
```

You can check the completion marker with:

```bash
test -f fpl_paper_outputs_server/RUN_COMPLETED_SUCCESSFULLY.txt && echo "OUTPUT VALIDATION PASSED"
```

## What the pipeline computes

1. Validates and standardizes the player--gameweek data.
2. Uses GW1--26 for forecast construction and model development.
3. Builds cost vectors for averaging, simulation, smoothing, ARIMA, robust,
   alternative-objective, linear, and Ridge-hybrid specifications.
4. Solves legal two-stage and joint 15-player roster models under FPL budget,
   positional, formation, captaincy, and club constraints.
5. Evaluates static and fixed-squad sequential lineup policies over GW27--38,
   including formation-valid automatic substitutions.
6. Runs starter--bench Pareto, Ridge/LASSO stability, and starting-availability
   sensitivity analyses.
7. Writes indexed PDF figures, CSV tables, player-level rosters, configurations,
   software versions, provenance records, checksums, and completion markers.

No historical score matrix or saved roster snapshot is used. All 94 internal
weekly series are recomputed from `merged_gw_2324.csv`. The four Santoro series
are the only externally supplied numerical benchmarks and are labeled as such
in the exported provenance.

## Output layout

Each full run creates the following structure under the selected output path:

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

For LaTeX placement, use each portrait PDF at full text width. Place
`figS05_method_similarity.pdf` alone on a landscape supplementary page. Do not
combine detailed supplementary figures into small half-width panels, because
that would reduce their effective label size.

## Reproducibility safeguards

- Forecasts and hyperparameters use only information available before GW27.
- Ridge and LASSO penalties are selected with rolling-origin validation.
- Bootstrap samples retain each selected player's complete training history.
- Budget caps, regularization parameters, and Pareto compromises are not chosen
  using realized GW27--38 scores.
- Fixed-squad sequential policies keep squad membership, captaincy, and
  vice-captaincy invariant; only the planned XI and bench order can change.
- Randomized components use deterministic seeds recorded in the run metadata.
- Output manifests record file sizes and SHA-256 checksums.

Use `python run_fpl_paper_pipeline.py --help` to view all command-line options.
The options that reuse existing output directories are intended only for figure
layout diagnostics and must not be used to claim a fresh publication run.

## Troubleshooting

### A required package is missing

Activate the intended environment and reinstall the requirements:

```bash
python -m pip install -r requirements_fpl_pipeline.txt
```

### The full run is too slow on a laptop

First run the smoke test. For the full analysis, reduce `--workers` to `2`.
Do not reduce the bootstrap replications or simulation draws if exact
publication-level reproduction is required.

### The plotting font is not found

Keep `lmroman10-regular.otf` in the repository root. Alternatively, supply a
font path explicitly:

```bash
python -u run_fpl_paper_pipeline.py --font /path/to/font.otf --data merged_gw_2324.csv --output fpl_paper_outputs_local --workers 4 --bootstrap 500 --simulation-draws 1000 --seed 20260824
```

### A previous output directory already exists

Use a new output-directory name so that the validated reference outputs are
not mixed with a new run. The pipeline never needs to read the committed
`fpl_paper_outputs_final/` directory for a fresh reproduction.

## Data source and citation

The included player--gameweek panel was constructed from the public 2023/24
Fantasy Premier League data maintained in the
[Fantasy-Premier-League repository](https://github.com/vaastav/Fantasy-Premier-League).

This repository accompanies the following paper, which has been submitted to
*Operations Research Forum* (Springer Nature) and is also available as an
arXiv preprint:

```bibtex
@article{ramezani2025data,
  title   = {A Data-Driven Framework for Team Selection in Fantasy Premier League},
  author  = {Ramezani, Danial and Dinh, Tai},
  journal = {arXiv preprint arXiv:2505.02170},
  year    = {2025},
  note    = {Submitted to Operations Research Forum (Springer Nature)}
}
```

Please cite the paper and acknowledge the upstream data source when reusing
the code or data.
