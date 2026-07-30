# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CSE 151B course project (model-building track): fine-tune pretrained encoders on the
[AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457) multiple-choice science QA
benchmark, with no retrieval, no external corpus, and no intermediate-task training. Two
model families are compared (BERT-base, DeBERTa-v3-base) across three training subsets
(easy / challenge / combined). GPU work runs on UCSD DataHub; analysis runs locally.

Deliverables are a paper (Related Work + Methods + Results) and a lightning talk, so
**every number in the writeup must trace back to a generated artifact under
`results/analysis/`** — that constraint drives the design below.

## Commands

```bash
# setup (Mac local)
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# the only test suite — fully offline, no downloads, ~1 min
python tests/offline_smoke_test.py        # must print ALL OFFLINE SMOKE TESTS PASSED

# online smoke test (covers the HF download path the offline test can't)
python src/train.py --model_name prajjwal1/bert-tiny --subset easy \
    --max_train_samples 100 --max_val_samples 100 --epochs 5 --lr 1e-4 \
    --output_dir checkpoints/online_smoke     # should overfit to ~1.0 val acc

python src/baselines.py --subset easy --split test    # random + majority-position
python src/train.py --subset combined                 # one run
python main.py --config configs/sweep.json            # grid sweep (--dry_run to preview)
python src/evaluate.py --checkpoint <dir>/best --subset easy --split test --save_predictions
python src/analyze.py                                 # results/ -> results/analysis/
```

There is no pytest, linter, or CI. `tests/offline_smoke_test.py` is a single script that
builds a local WordPiece vocab and a random-init tiny BERT so it needs no network; it
covers label normalization, collator shapes/NaN-freedom on mixed-choice batches, an
end-to-end overfit check, checkpoint save→reload, and `--resume`. Run it after touching
anything in `src/data.py`, `src/train.py`, or `src/evaluate.py`.

`src/analyze.py` needs `tabulate` (for `DataFrame.to_markdown`) and reads the HF dataset
cache for part of the error analysis — see below.

## Architecture

### The `results/` contract

This is the most important thing to understand. `src/analyze.py` discovers everything by
`rglob("*.json")` and **classifies files by their content, not their path** — directory
names under `results/` are purely human organization and carry no meaning:

| file kind | identified by | produced by |
|---|---|---|
| baseline | has `random` + `majority_position` | `src/baselines.py` |
| run | has `val_accs` + `config` | `src/train.py` |
| sweep summary | has `runs` | `main.py` |
| eval | has `accuracy` + `checkpoint` | `src/evaluate.py` |

Consequences that have caused real problems:

- **`train.py` rewrites `results/<run_name>.json` after every epoch**, with
  `status: "running"` → `"completed"`, so a culled DataHub session still leaves its
  metrics. `analyze.py` counts only `status == "completed"`.
- **Duplicate configs are not deduplicated.** If the same
  `(model, subset, lr, epochs, seed)` exists in two directories, both are counted and the
  reported champion becomes a max over them — i.e. silent cherry-picking. When re-running a
  sweep, delete or move the superseded directory.
- **Runs land in `results/` root**, not in a subdirectory. The `bert_easy_*`-style folders
  were moved there by hand afterwards.
- Eval JSONs record only a `checkpoint` path, so `resolve_eval_model()` maps them back to a
  model by finding the longest `run_name` that prefixes the checkpoint directory name
  (`train.py` saves to `checkpoints/<run_name>_<timestamp>/best`).

### Model / data path

`AutoModelForMultipleChoice` scores each `(question, option_i)` pair in a **separate**
forward pass and only compares them via a softmax over the resulting logits. Three
invariants live across `src/data.py`, `train.py` and `evaluate.py`:

1. ARC mixes label schemes (`A`–`E` and `1`–`5`); `normalize_example` converts both to an
   integer index into `choices`.
2. Choice counts vary (3–5). `ArcCollator` pads the choice dimension by **duplicating the
   example's first real choice** — never an all-padding sequence, which makes the attention
   softmax produce NaNs — and marks real slots in `choice_mask`.
3. `masked_choice_logits` must be applied to the logits before any softmax or
   cross-entropy, in both the training loop and `evaluate_model`. Skipping it lets dummy
   choice slots win.

Tokenization uses `truncation="only_first"`: the question is truncated, never the answer
option. `example_id` rides along as a plain Python list and is popped before the forward
pass.

`pick_device()` (in `evaluate.py`, reused by `train.py`) resolves CUDA → MPS → CPU; fp16 is
enabled only on CUDA.

### Sweep runner (`main.py`)

Expands `grid` as a cartesian product and runs `src/train.py` once per combination as a
**subprocess**, so GPU memory is fully released between runs. Any `train.py` argument can be
a grid axis.

- `run_name = sweep_name + "_" + "lr{v}_epochs{v}"`, so **two sweeps must not share a
  `sweep_name`** or their result files collide.
- Resume is free: a combo whose `results/<run_name>.json` is `completed` is skipped; an
  incomplete one resumes from `last.pt`. `--rerun` forces.
- Storage is tight on DataHub, so non-champion checkpoints are **deleted as soon as they are
  beaten**. `keep_all_checkpoints: true` disables this (needed when several checkpoints must
  survive for test evaluation); `keep_champion_last_pt: false` also drops the champion's
  ~1.3 GB `last.pt`.
- A failed subprocess is recorded as `{"status": "failed"}` in the summary, but **the exit
  code and stderr are not persisted** — diagnosing a failure means re-running it.

### Analysis (`src/analyze.py`)

One script produces every paper artifact into `results/analysis/`: T1 main results
(validation + test, Wilson CIs), T2 run leaderboard, T3 efficiency, T4 calibration,
T5 error categories, hyperparameter heatmaps, training dynamics, error-analysis figures, and
an auto-generated `analysis.md` whose findings, caveats and "Still outstanding" section are
**derived from the data on disk**, not hardcoded. Re-run it after any new result lands.

Figure conventions — do not cross these channels:

- **colour encodes the model family** (`MODEL_COLOR`); a single-model figure uses one colour
- **line style / marker encodes the subset** (`SUBSET_STYLE`, `SUBSET_MARKER`)

Error analysis is two-tiered so the script never hard-fails: confidence calibration and
answer-position bias need only the prediction records, while the per-question-category
breakdowns additionally join question text via `load_arc` and are skipped with a printed
notice if the dataset is unreachable.

The question categories (`CUE_PATTERNS`) are a coarse **lexical proxy** matched
first-to-last, not the hand-labelled knowledge/reasoning taxonomy of Clark et al. (2018),
and the report says so. ~27–33% of questions match more than one pattern, so match order
silently decides their class; `which-question` and `other` are effectively residual buckets.

## Working notes

- `docs/` is gitignored (`IMPLEMENTATION_PLAN.md`, its Chinese translation, and
  `related_work.md` with verified ARC benchmark numbers from the primary papers).
  `ANALYSIS_PLAN.md` specifies what `analyze.py` must produce.
- `checkpoints/` is gitignored and routinely empty — sweeps delete all but the champion.
- Test-set evaluation is meant to run **once, at the end**. Published ARC numbers are
  test-set numbers, so validation figures are not comparable to the literature.
- When a claim rests on hyperparameter positions, check `analyze.py`'s grid-boundary
  caveats first: the two model families were swept over different learning-rate ranges
  (BERT `{2,3,5,7}e-5`, DeBERTa `{1,1.5,2,3}e-5`, overlapping only at 2e-5 and 3e-5), and
  several champions sit on a grid edge.
