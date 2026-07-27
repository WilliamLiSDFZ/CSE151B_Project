# Data Analysis Plan — from `results/` to paper-ready tables & figures

Goal: one script, `src/analyze.py`, that reads everything under `results/` and
mechanically produces every number, table, and figure the paper needs, so each
claim in the writeup traces to a generated artifact. Run it locally (no GPU
needed); re-run any time new results appear.

## 1. Input discovery (robust to folder layout)

Scan `results/**/*.json` and classify by content, not by path:
- **baseline files** — contain a `random` key (4 found: easy/challenge × val/test)
- **run files** — contain `val_accs` + `config` (metadata like model/subset/lr/seed
  read from `config` inside the JSON, so your folder renames never break anything)
- **sweep summaries** — contain `runs` (used only as cross-checks; runs are the truth)
- **eval files** (`eval_*_*.json`, from evaluate.py) — none yet; auto-included when
  they appear (test evals, per-subset decompositions, prediction records)

Current inventory the script will report: 6 sweeps (BERT/DeBERTa × easy/combined/
challenge), 63 completed runs (note: BERT-easy sweep has 3/12 — the quota incident;
coverage is reported honestly per sweep), 4 baseline files, mixed seeds (42/77/...)
grouped by config.

## 2. Outputs (written to `results/analysis/`)

**T1 — Main results table** (the paper's Table 1).
Rows: Random, Majority-position, BERT-base (best config), DeBERTa-v3-base (best
config). Columns: Easy / Challenge / Combined validation accuracy, each with a
95% binomial confidence interval (n = 570 / 299 / 869). Test columns appear
automatically once `eval_*_test.json` files exist. Emitted as `.md`, `.tex`
(booktabs), and `.csv`.

**F1 — Hyperparameter heatmaps** (2 models × 3 subsets grid).
lr × epochs → best_val_acc per cell, champion cell marked. This is the figure that
shows the paper's cleanest secondary finding: BERT's optimum sits at high lr
(5–7e-5) while DeBERTa's sits at 1.5–3e-5 — optimal fine-tuning hyperparameters do
not transfer across pretrained models. Missing cells (failed runs) rendered gray.

**F2 — Training dynamics** for the 6 champions: per-epoch val accuracy (left) and
training-loss curve (right). Documents the overfitting pattern (train loss → ~0
while val plateaus) that motivates best-epoch checkpointing; feeds the Discussion.

**T2 — Consolidated sweep leaderboard** (appendix material): every completed run
with model, subset, lr, epochs, seed, best_val_acc, epochs-to-best, wall time.

**T3 — Efficiency table**: mean s/epoch and total GPU time per model family
(BERT ~18 s/epoch vs DeBERTa ~25 s/epoch from `epoch_times`), parameter counts.
One paragraph of the Experiments section, fully sourced.

**Derived analysis in the report:**
- *Combined vs separate training*: expected combined-val accuracy of the two
  separate champions (weighted 570:299) vs the actual combined champion — e.g.
  DeBERTa separate models imply ~(0.761·570 + 0.522·299)/869 ≈ 67.9% vs actual
  70.0% → combined training helps (+2.1 pts). Computed for both model families.
- *Significance checks*: two-proportion tests / CI overlap for the headline gaps
  (DeBERTa vs BERT per subset; champion vs runner-up within each sweep — the
  latter is expected to be non-significant and the report will say so).
- *Caveats auto-flagged*: single-seed sweeps (until final 3-seed runs land),
  BERT-easy partial coverage, combined-champion per-subset decomposition marked
  as TODO until you run `evaluate.py --subset easy/challenge` on the combined
  champion (2 × ~30 s on DataHub; exact commands printed by the script).

**analysis.md — auto-generated report** stitching all of the above together with
the observations written out in plain sentences (champion configs, gaps with CIs,
caveats), so paper writing becomes copy-editing rather than number-hunting.

## 3. What this plan deliberately does NOT need

No new training. Test-set evaluation stays untouched until final models are
chosen (the script leaves clearly-labeled empty slots for those numbers). Error
analysis (confidence distributions, accuracy vs question length, hand-labeled
error categories) activates automatically once you produce prediction records
via `evaluate.py --save_predictions` — the hooks are in the script from day one.

## 4. Implementation notes

pandas + matplotlib only (already in requirements). ~250 lines, one file
(`src/analyze.py`), same offline-testable style as the rest of the repo (I'll add
a synthetic-results test to `tests/`). CLI: `python src/analyze.py`
(defaults: `--results_dir results --out results/analysis`).
