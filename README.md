# Frugal AI feasibility experiment (ICSIC 2026, paper 8566)

Code and results for "A Frugal AI Framework for Resource-Constrained SMEs in Palestine: Readiness-Gated Adoption and a Feasibility Evaluation" (S. A. Jumhour).

## Data
Online Retail II, UCI Machine Learning Repository, doi:10.24432/C5CG6D (CC BY 4.0).
Download `online_retail_II.xlsx` and run `python prepare_data.py online_retail_II.xlsx`.

## Scripts (run in this order; single CPU core, no GPU)
| Script | Purpose |
|---|---|
| `run_experiment.py raw.parquet out_orig` | Primary pre-specified pipeline (cleaning, H1, H2; seed 42) |
| `h1_corrected.py` | H1 with corrected moving averages, Croston/SBA/TSB baselines, data scarcity |
| `h1_ext.py` | H1 equal tuning budget (20 configs/model) and ten seeds |
| `h1_final.py` | H1 metrics, model sizes and training times for Table IV |
| `h2_ext.py` | H2 RFM baselines, equal tuning budget and ten seeds |
| `h2_more.py` | H2 customer-clustered bootstrap CIs, tuned model sizes and times |
| `h2_scar.py` | H2 data scarcity (Fig. 2b) |

`results/` holds the JSON outputs reported in the paper.

## Correction note
The first version of the pipeline computed the 4- and 8-week moving averages with a pandas index misalignment, so the values were attached to the wrong item-week rows. This affected only the H1 moving-average baseline and two H1 model features. It was found while preparing this release and corrected in the revised paper; `FIX_MA=0 python run_experiment.py raw.parquet out_v1` reproduces the original numbers. H2 (churn) does not use these features and is unaffected.
