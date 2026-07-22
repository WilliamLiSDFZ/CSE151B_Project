"""Evaluate a fine-tuned checkpoint on any ARC subset/split.

Writes results/eval_{subset}_{split}.json with overall accuracy and (optionally)
per-example records for error analysis (plan section 6).

Usage:
    python src/evaluate.py --checkpoint checkpoints/bert_base_combined/best \
        --subset challenge --split test --save_predictions
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ArcCollator, load_arc, masked_choice_logits  # noqa: E402


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_model(model, loader, device) -> tuple[float, list[dict]]:
    """Return (accuracy, per-example records)."""
    model.eval()
    records: list[dict] = []
    for batch in loader:
        ids = batch.pop("example_id", None)
        labels = batch.pop("labels")
        mask = batch.pop("choice_mask").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits            # (B, C)
        logits = masked_choice_logits(logits, mask)
        probs = logits.float().softmax(dim=-1)
        conf, preds = probs.max(dim=-1)
        for j in range(labels.shape[0]):
            records.append({
                "id": ids[j] if ids is not None else None,
                "label": int(labels[j]),
                "pred": int(preds[j]),
                "correct": bool(preds[j] == labels[j]),
                "confidence": float(conf[j]),
            })
    accuracy = sum(r["correct"] for r in records) / max(len(records), 1)
    return accuracy, records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="dir saved by train.py (e.g. .../best)")
    p.add_argument("--subset", default="combined", choices=["easy", "challenge", "combined"])
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=None, help="debug: evaluate a subset only")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--save_predictions", action="store_true",
                   help="also store per-example records in the JSON (for error analysis)")
    args = p.parse_args()

    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    device = pick_device()
    print(f"[evaluate] device = {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForMultipleChoice.from_pretrained(args.checkpoint).float().to(device)

    ds = load_arc(args.subset, args.split, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ArcCollator(tokenizer, args.max_length),
    )
    t0 = time.time()
    accuracy, records = evaluate_model(model, loader, device)
    print(f"[evaluate] {args.subset}/{args.split}: n={len(records)} "
          f"accuracy={accuracy:.4f} ({time.time() - t0:.1f}s)")

    out = {
        "checkpoint": args.checkpoint,
        "subset": args.subset,
        "split": args.split,
        "n_examples": len(records),
        "accuracy": accuracy,
    }
    if args.save_predictions:
        out["predictions"] = records
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"eval_{args.subset}_{args.split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[evaluate] wrote {out_path}")


if __name__ == "__main__":
    main()
