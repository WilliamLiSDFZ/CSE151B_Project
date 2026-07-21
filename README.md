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

# hyperparameter sweep (plan section 4.4)
for lr in 1e-5 2e-5 3e-5 5e-5; do
  for ep in 2 3 4; do
    python src/train.py --subset combined --lr $lr --epochs $ep
  done
done

# final runs: best config, 3 seeds
for seed in 42 43 44; do
  python src/train.py --subset combined --lr <BEST_LR> --epochs <BEST_EP> --seed $seed
done
```

Each run writes `results/<run_name>.json` (config, loss curve, per-epoch val
accuracy, wall time) and saves the best-val checkpoint to
`checkpoints/<run_name>/best`.

## Evaluation (test sets: run ONCE, at the very end)

```bash
python src/evaluate.py --checkpoint checkpoints/<run_name>/best \
    --subset easy --split test --save_predictions
python src/evaluate.py --checkpoint checkpoints/<run_name>/best \
    --subset challenge --split test --save_predictions
```

`--save_predictions` stores per-example records (id, pred, label, confidence)
in the JSON for the paper's error analysis.
