"""Offline smoke test: verifies every piece of custom logic WITHOUT network access.

Builds a tiny random-init BERT + local WordPiece vocab, generates synthetic
ARC-schema examples (mixed 3/4/5 choices, mixed A-E / 1-5 label schemes), then:

  1. tests normalize_example on every label-scheme variant
  2. tests ArcCollator shapes, choice_mask, and NaN-freedom on mixed batches
  3. runs train.run_training end to end and checks the model OVERFITS (sanity
     that gradients flow through the masked softmax)
  4. tests best-checkpoint save -> AutoModel reload -> identical accuracy
  5. tests --resume from last.pt

Run:  python tests/offline_smoke_test.py
The real-data pipeline (load_arc / bert-base-uncased) uses standard HF calls;
run the online smoke test from README on Mac/DataHub to cover the download path.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from data import ArcCollator, masked_choice_logits, normalize_example  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from train import run_training, set_seed  # noqa: E402

WORDS = [
    "which", "animal", "can", "fly", "swim", "bark", "climb", "what", "color",
    "is", "the", "sky", "grass", "sun", "snow", "a", "bird", "fish", "dog",
    "cat", "rock", "tree", "blue", "green", "yellow", "white", "red", "black",
    "best", "worst", "answer", "option", "number", "one", "two", "three",
]


def build_tokenizer(tmp: Path):
    """Build a real WordPiece tokenizer fully offline via the `tokenizers` lib."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import Lowercase
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS)}
    t = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    t.normalizer = Lowercase()
    t.pre_tokenizer = Whitespace()
    t.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )


def build_model(vocab_size: int):
    from transformers import BertConfig, BertForMultipleChoice
    config = BertConfig(
        vocab_size=vocab_size, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=128, max_position_embeddings=64,
    )
    return BertForMultipleChoice(config)


def make_synthetic_raw() -> list[dict]:
    """Synthetic examples in the exact raw HF ARC schema, covering all quirks."""
    base = [
        ("which animal can fly", ["bird", "fish", "dog", "rock"], 0),
        ("which animal can swim", ["rock", "fish", "cat", "tree"], 1),
        ("which animal can bark", ["bird", "tree", "dog", "sun"], 2),
        ("which animal can climb", ["fish", "sun", "snow", "cat"], 3),
        ("what color is the sky", ["blue", "green", "red", "black"], 0),
        ("what color is the grass", ["white", "green", "blue", "red"], 1),
        ("what color is the sun", ["black", "blue", "yellow", "white"], 2),
        ("what color is the snow", ["red", "green", "black", "white"], 3),
    ]
    raw = []
    for i, (q, choices, ans) in enumerate(base):
        # variant A: standard A-D labels
        letters = ["A", "B", "C", "D"]
        raw.append({"id": f"syn-{i}-letters", "question": q,
                    "choices": {"text": choices, "label": letters},
                    "answerKey": letters[ans]})
        # variant B: numeric 1-4 labels
        nums = ["1", "2", "3", "4"]
        raw.append({"id": f"syn-{i}-nums", "question": q,
                    "choices": {"text": choices, "label": nums},
                    "answerKey": nums[ans]})
        # variant C: 3 choices (drop one wrong option)
        keep = [j for j in range(4) if j != (ans + 1) % 4]
        c3 = [choices[j] for j in keep]
        raw.append({"id": f"syn-{i}-three", "question": q,
                    "choices": {"text": c3, "label": ["A", "B", "C"]},
                    "answerKey": ["A", "B", "C"][keep.index(ans)]})
        # variant D: 5 choices, numeric labels
        c5 = choices + ["option"]
        raw.append({"id": f"syn-{i}-five", "question": q,
                    "choices": {"text": c5, "label": ["1", "2", "3", "4", "5"]},
                    "answerKey": str(ans + 1)})
    return raw


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="arc_smoke_"))
    try:
        set_seed(0)
        tokenizer = build_tokenizer(tmp)
        model = build_model(tokenizer.vocab_size)
        device = torch.device("cpu")
        model.to(device)

        # --- 1. normalization ------------------------------------------------
        raw = make_synthetic_raw()
        ds = [normalize_example(ex) for ex in raw]
        assert all(0 <= ex["label"] < len(ex["choices"]) for ex in ds)
        assert ds[0]["label"] == 0 and ds[1]["label"] == 0          # A-D vs 1-4 agree
        assert len(ds[2]["choices"]) == 3 and len(ds[3]["choices"]) == 5
        counts = {len(ex["choices"]) for ex in ds}
        assert counts == {3, 4, 5}, counts
        print(f"[1/5] normalize_example OK on {len(ds)} examples "
              f"(choice counts {sorted(counts)}, A-E and 1-5 schemes)")

        # --- 2. collator on a mixed batch -----------------------------------
        collator = ArcCollator(tokenizer, max_length=32)
        batch = collator(ds[:6])                                    # mixes 3/4/5 choices
        B, C, L = batch["input_ids"].shape
        assert C == 5, f"expected batch padded to 5 choices, got {C}"
        assert batch["choice_mask"].shape == (B, C)
        assert batch["choice_mask"].sum().item() == sum(len(ex["choices"]) for ex in ds[:6])
        with torch.no_grad():
            logits = model(**{k: v for k, v in batch.items()
                              if k not in ("labels", "choice_mask", "example_id")}).logits
        assert logits.shape == (B, C)
        assert not torch.isnan(logits).any(), "NaN logits — dummy-choice handling broken"
        masked = masked_choice_logits(logits, batch["choice_mask"])
        probs = masked.softmax(-1)
        assert (probs[~batch["choice_mask"]] < 1e-6).all(), "dummy choices got probability"
        print(f"[2/5] ArcCollator OK: shapes ({B},{C},{L}), mask correct, no NaN, "
              "dummy choices excluded from softmax")

        # --- 3. training loop overfits synthetic data -----------------------
        args = argparse.Namespace(
            model_name="tiny-random-bert", subset="synthetic", lr=1e-3, epochs=70,
            batch_size=8, grad_accum=1, max_length=32, weight_decay=0.01,
            warmup_ratio=0.1, max_grad_norm=1.0, seed=0, num_workers=0,
            log_every=1000, output_dir=str(tmp / "ckpt"), run_name="smoke",
            results_dir=str(tmp / "results"), resume=False, no_fp16=True,
        )
        results = run_training(model, tokenizer, ds, ds, args, device)
        assert results["best_val_acc"] >= 0.9, (
            f"tiny model failed to overfit (acc={results['best_val_acc']:.3f}) — "
            "gradients or masking are broken")
        assert results["loss_curve"][-1] < results["loss_curve"][0]
        rec = json.loads((tmp / "results" / "smoke.json").read_text())
        assert rec["status"] == "completed" and rec["epochs_done"] == args.epochs
        print(f"[3/5] run_training OK: overfit acc={results['best_val_acc']:.3f}, "
              f"loss {results['loss_curve'][0]:.3f} -> {results['loss_curve'][-1]:.3f}, "
              "per-epoch results JSON persisted")

        # --- 4. checkpoint save/reload roundtrip ----------------------------
        from transformers import AutoModelForMultipleChoice, AutoTokenizer
        from torch.utils.data import DataLoader
        reloaded = AutoModelForMultipleChoice.from_pretrained(tmp / "ckpt" / "best").to(device)
        tok2 = AutoTokenizer.from_pretrained(tmp / "ckpt" / "best")
        loader = DataLoader(ds, batch_size=8, collate_fn=ArcCollator(tok2, 32))
        acc, records = evaluate_model(reloaded, loader, device)
        assert acc >= 0.9, f"reloaded checkpoint accuracy dropped: {acc:.3f}"
        assert len(records) == len(ds) and all(r["id"] for r in records)
        print(f"[4/5] save/reload OK: reloaded best checkpoint acc={acc:.3f}, "
              f"{len(records)} per-example records with ids")

        # --- 5. --resume ------------------------------------------------------
        set_seed(0)
        model2 = build_model(tokenizer.vocab_size).to(device)
        args2 = argparse.Namespace(**{**vars(args), "epochs": 2,
                                      "output_dir": str(tmp / "ckpt2")})
        r1 = run_training(model2, tokenizer, ds, ds, args2, device)
        args3 = argparse.Namespace(**{**vars(args2), "epochs": 4, "resume": True})
        r2 = run_training(model2, tokenizer, ds, ds, args3, device)
        assert len(r1["val_accs"]) == 2 and len(r2["val_accs"]) == 4, (
            len(r1["val_accs"]), len(r2["val_accs"]))
        rec = json.loads((tmp / "results" / "smoke.json").read_text())
        assert rec["status"] == "completed" and rec["epochs_done"] == 4
        print("[5/5] --resume OK: continued from epoch 2 to 4, results JSON updated")

        print("\nALL OFFLINE SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
