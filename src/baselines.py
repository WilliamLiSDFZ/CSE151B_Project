"""Baselines for ARC (plan section 3.2).

1. Random guess  -- analytic E[acc] = mean(1/n_choices) + empirical check
2. Majority position -- always predict the most frequent answer position seen in train

Usage:
    python src/baselines.py --subset challenge --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_arc  # noqa: E402


def random_baseline(labels: list[int], n_choices: list[int],
                    trials: int = 1000, seed: int = 42) -> dict:
    analytic = float(np.mean([1.0 / n for n in n_choices]))
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(trials):
        guesses = [rng.integers(0, n) for n in n_choices]
        accs.append(np.mean([g == y for g, y in zip(guesses, labels)]))
    return {
        "analytic_accuracy": analytic,
        "empirical_accuracy_mean": float(np.mean(accs)),
        "empirical_accuracy_std": float(np.std(accs)),
        "trials": trials,
    }


def majority_position_baseline(train_labels: list[int], labels: list[int],
                               n_choices: list[int]) -> dict:
    majority_pos, count = Counter(train_labels).most_common(1)[0]
    correct = [y == majority_pos and majority_pos < n
               for y, n in zip(labels, n_choices)]
    return {
        "majority_position": int(majority_pos),
        "train_frequency": count / len(train_labels),
        "accuracy": float(np.mean(correct)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", default="combined", choices=["easy", "challenge", "combined"])
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    eval_ds = load_arc(args.subset, args.split)
    train_ds = load_arc(args.subset, "train")
    labels = [ex["label"] for ex in eval_ds]
    n_choices = [len(ex["choices"]) for ex in eval_ds]
    train_labels = [ex["label"] for ex in train_ds]

    out = {
        "subset": args.subset,
        "split": args.split,
        "n_examples": len(labels),
        "choice_count_distribution": dict(Counter(n_choices)),
        "random": random_baseline(labels, n_choices, args.trials, args.seed),
        "majority_position": majority_position_baseline(train_labels, labels, n_choices),
    }
    print(json.dumps(out, indent=2))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"baselines_{args.subset}_{args.split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[baselines] wrote {out_path}")


if __name__ == "__main__":
    main()
