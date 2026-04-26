# OHIE QJE 2012 --- Replication and Reanalysis

Replication and reanalysis of Finkelstein et al. (QJE 2012) using the public OHIE files and estimator diagnostics.

## How to run

```bash
# (1) basic replication + simple reanalysis (baseline adjustment, 2-period DiD)
python code/analysis.py

# (2) wider survey-table replication
python code/replicate_tables.py

# (3) Table I auxiliary output + OOP quantile figure
python code/extra_replications.py

# (4) reanalysis (ex-post weights + BBMT decomposition)
python code/reanalysis.py
```

To rebuild the PDF:

```bash
latexmk -pdf -interaction=nonstopmode ohie_reanalysis_full.tex
```

## Code files (`code/`)

| File | Purpose |
| --- | --- |
| `analysis.py` | Loads the OHIE public-use Stata files, builds the analysis frame, and runs the basic replication (first-stage ever-Medicaid, ITT + LATE for outpatient visits, OOP spending, self-rated health, no-depression) plus the reanalysis comparison (baseline-adjusted ANCOVA and 0m--12m DiD ITT on the balanced panel). Writes to `output/` and makes `figures/replication_plot.*` / `reanalysis_plot.*`. |
| `reanalysis.py` | Reanalysis built on top of `analysis.py`. Computes OLS ex-post weights $C_i$, the BBMT'22 2SLS decomposition into complier and always-taker terms under three control specifications, the difference between unconditional-complier and covariate-adjusted 2SLS weights, and ex-post weights for the two-period DiD on the balanced panel. Writes `output/reanalysis_decomposition.csv`, `output/reanalysis_weights.json`, `output/late_weighting_diagnostics.csv`, and the `figures/weights_*` plots. |
| `replicate_tables.py` | Reproduces the public-data portions of the published tables. Writes `output/replication_all_tables.{csv,json}` and the LaTeX fragment `output/replication_results_tables.tex`. |
| `extra_replications.py` | Optional: extra figure (OOP quantile regression, `figures/figure1_oop_quantile.*`) and secondary robustness cuts. |

## Output files (`output/`)

| File | Contents |
| --- | --- |
| `replication_results.csv` | Selected ITT and LATE estimates, SEs, control means, and sample sizes, with the corresponding published LATE in the `paper_late` column. Produced by `analysis.py`. |
| `reanalysis_results.csv` | Per-outcome comparison of (original ITT/IV, baseline-adjusted ITT/IV, 2-period DiD ITT) on the balanced 0m--12m panel. Produced by `analysis.py`. |
| `replication_all_tables.csv` / `.json` | Public-data table replications: Table III public columns, Tables V, VI, VIII, IX, X, and Table XI standardized survey effects. From `replicate_tables.py`. |
| `replication_results_tables.tex` | LaTeX tables for the public-data replication results. From `replicate_tables.py`, using Python estimates rather than the authors' `.do` files. |
| `table1.json`, `table1_replication.tex`, `figure1.json` | Table I summary output and quantile-figure data from `extra_replications.py`. |
| `summary.json` | Small top-level summary with sample sizes and selected replication/reanalysis estimates. |
| `reanalysis_decomposition.csv` | BBMT'22 2SLS decomposition table: per-outcome $\hat\beta_{2SLS}$, $\max\lvert \mathbb{E}[Z\mid W]-\mathbb{L}[Z\mid W]\rvert$, implied always-taker share, and numerator contributions by $(Z,D)$ cell, across three covariate specifications. Produced by `reanalysis.py`. |
| `reanalysis_weights.json` | Diagnostics for each regression: number of negative ex-post weights, max $n\cdot C_i$, effective sample size $n_\text{eff}=1/\sum C_i^2$, and BBMT sub-blocks. Produced by `reanalysis.py`. |
| `late_weighting_diagnostics.csv` | Stratum-level comparison of unconditional-complier weights $P(W) \pi(W)$ and covariate-adjusted 2SLS weights $P(W)\operatorname{Var}(Z\mid W)\pi(W)$. Produced by `reanalysis.py`. |

## Figures (`figures/`)

| File | Produced by | Content |
| --- | --- | --- |
| `replication_plot.pdf`/`.png` | `analysis.py` | Replicated LATEs vs published LATEs (doc visits + OOP). |
| `reanalysis_plot.pdf`/`.png` | `analysis.py` | Original IV vs baseline-adjusted IV vs DiD ITT. |
| `figure1_oop_quantile.pdf`/`.png` | `extra_replications.py` | OOP-spending quantile treatment effect. |
| `weights_ols_first_stage.pdf`/`.png` | `reanalysis.py` | Histogram of $n\cdot C_i$ for Medicaid-on-lottery. |
| `weights_ols_itt_docvisits.pdf`/`.png` | `reanalysis.py` | Histogram of $n\cdot C_i$ for ITT on outpatient visits. |
| `weights_iv_docvisits.pdf`/`.png` | `reanalysis.py` | Per-observation 2SLS weight-on-$Y$ histogram (doc visits, saturated design). |

## Write-ups

| File | Contents |
| --- | --- |
| `ohie_reanalysis_full.tex` / `.pdf` | **Main reanalysis note.** Method and results for OLS ex-post weights, the BBMT decomposition, and the two-period DiD check.|

## Reanalysis design

The selected targets from the paper:

- Table III, row 1 --- ever-on-Medicaid first stage
- Table V --- outpatient visits last six months
- Table VIII --- any out-of-pocket medical expenses, last six months
- Table IX, Panel B --- self-reported health good/very good/excellent
- Table IX, Panel B --- did not screen positive for depression

Reanalysis Methods:

- FWL + covariate-adjusted OLS, ex-post weights under the heterogeneous-effect
  model, and the omitted-variable-bias term under saturated strata.
- Imbens--Angrist IV/LATE and the BBMT 2SLS decomposition into complier and
  always-taker terms; the always-taker term mechanically vanishes under
  saturated strata because $\mathbb{E}[Z\mid W] = \mathbb{L}[Z\mid W]$.
- Two-period 0m--12m DiD/TWFE ITT on the balanced panel; with $T=2$ and no staggering,
  Goodman-Bacon reduces to a single 2$\times$2 DiD and ex-post weights
  are all non-negative.

Qualitative findings:

- Utilization effects remain large.
- Financial-protection effects remain large.
- Self-reported health effects attenuate in the balanced-panel reanalysis
  (sample-composition effect, not a weighting artifact).
- No always-taker contamination under the paper's saturated stratum
  controls --- but substituting linear covariate adjustment shifts
  $\hat\beta_{2SLS}$ by $\sim 5\%$, illustrating the BBMT warning.
- The IV estimates are covariate- and survey-weight-adjusted averages
  of conditional complier effects, not the raw unconditional complier
  LATE. In the survey specification, the total-variation distance
  between these two stratum-weight vectors is about 0.023.
