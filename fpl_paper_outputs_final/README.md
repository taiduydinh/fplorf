# FPL paper output bundle

Data split: GW1--26 for training/development and GW27--38 for evaluation.

The final publication set contains seven main-paper PDFs in `main_figures/`
and 33 supplementary PDFs in `supplementary_figures/`. The 35 files from the
original Figures 1--29 and their subpanels are all retained: the framework and
Pareto frontier remain in the main paper, while the other 33 retained files
are placed in the supplement. Five additional main figures provide the
regularization, controlled-performance, budget-allocation, external-benchmark,
and starting-availability evidence. Every PDF is
title-free, uses stable manuscript/supplement indexing, and is saved as a
vector PDF with 300-DPI raster metadata and a tight bounding box. Plot legends
are outside the data panels. Portrait figures are designed for the manuscript's
6.70-inch full-text width. The dense method-similarity matrix is designed for a
9.70-inch landscape supplementary page. Global labels are 9--10 pt and no
explicit annotation or legend text is smaller than 8 pt.

All 94 internal weekly score series are recomputed from `merged_gw_2324.csv`:
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
