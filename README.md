# CSE 151B Project — Fine-tuning BERT on ARC

Multiple-choice science QA on the [AI2 Reasoning Challenge (ARC)](https://arxiv.org/abs/1803.05457)
dataset, using `BertForMultipleChoice` (BERT encoder + linear scoring head).
See `IMPLEMENTATION_PLAN.md` (EN) / `IMPLEMENTATION_PLAN_ZH.md` (中文) for the full plan.

## Layout

```
src/data.py          data loading, label normalization, variable-choice collator
src/train.py         fine-tuning loop (CUDA / MPS / CPU, fp16 on CUDA, --resume)
src/evaluate.py      accuracy on any subset/split, per-example records for error analysis
src/baselines.py     random-guess + majority-position baselines
tests/offline_smoke_test.py   verifies all custom logic without network access
notebooks/eda.ipynb  dataset statistics + figures for the paper
results/             one JSON per run/eval (committed; every paper number traces here)
checkpoints/         model weights (gitignored)
```

## Setup

**Mac (local, MPS):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**UCSD DataHub (JupyterHub, CUDA):** open a terminal, then
```bash
pip install --user -r requirements.txt
nvidia-smi   # confirm the GPU
```

## Verify the pipeline

```bash
# 1. offline logic test (no downloads needed) — should print ALL OFFLINE SMOKE TESTS PASSED
python tests/offline_smoke_test.py

# 2. online smoke test (downloads ARC + a tiny model; should overfit to ~1.0 val acc)
python src/train.py --model_name prajjwal1/bert-tiny --subset easy \
    --max_train_samples 100 --max_val_samples 100 --epochs 5 --lr 1e-4 \
    --output_dir checkpoints/online_smoke
```

## Baselines

```bash
python src/baselines.py --subset easy --split test
python src/baselines.py --subset challenge --split test
```

## Training

```bash
# default full run (bert-base-uncased, combined Easy+Challenge train set)
python src/train.py --subset combined

# resume after a culled DataHub session (same args + --resume)
python src/train.py --subset combined --resume

# hyperparameter sweep from a config file (plan section 4.4)
python main.py --config configs/sweep.json

# final runs: after the sweep, edit configs/final.json (set lr/epochs to the
# champion's values from results/sweep_summary.json), then
python main.py --config configs/final.json
```

`main.py` expands the `grid` in the config (cartesian product; any train.py
argument can be an axis), runs one training subprocess per combination, and —
to save server storage — **keeps only the best run's checkpoints**, deleting
each run's `checkpoints/<run_name>_<YYYYmmdd_HHMMSS>/` (folder names carry the
run's start timestamp) as soon as it is beaten. Every run's
metrics JSON is always kept in `results/` — train.py rewrites it after EVERY
epoch (`status: running` → `completed`), so even a culled session leaves its
metrics on disk for comparison. The ranked leaderboard lands in
`results/<sweep_name>_summary.json`. An interrupted sweep resumes for free:
completed combos are skipped, half-finished ones are resumed from `last.pt`
(`--rerun` forces a full redo). Use `--dry_run` to preview
the commands. Config knobs: `keep_champion_last_pt: false` also drops the
champion's `last.pt` (~1.3GB, only needed for `--resume`);
`keep_all_checkpoints: true` disables deletion entirely — that is why it is on
in `configs/final.json`, where all 3 seed checkpoints are needed for test
evaluation.

Each run writes `results/<run_name>.json` (config, loss curve, per-epoch val
accuracy, wall time) and saves the best-val checkpoint to
`checkpoints/<run_name>_<YYYYmmdd_HHMMSS>/best` (the exact path is recorded in
the summary's `champion.checkpoint` field).

## Evaluation (test sets: run ONCE, at the very end)

```bash
python src/evaluate.py --checkpoint checkpoints/<run_name>/best \
    --subset easy --split test --save_predictions
python src/evaluate.py --checkpoint checkpoints/<run_name>/best \
    --subset challenge --split test --save_predictions
```

`--save_predictions` stores per-example records (id, pred, label, confidence)
in the JSON for the paper's error analysis.
