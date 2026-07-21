"""Data loading & preprocessing for ARC multiple-choice QA (CSE 151B project).

Implements IMPLEMENTATION_PLAN.md section 2.2:
  * label-scheme normalization ("1"-"5" -> "A"-"E", answerKey -> integer index)
  * variable choice counts (3-5) handled by ArcCollator via choice padding + mask
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

HF_DATASET = "allenai/ai2_arc"
SUBSET_TO_CONFIGS = {
    "easy": ("ARC-Easy",),
    "challenge": ("ARC-Challenge",),
    "combined": ("ARC-Easy", "ARC-Challenge"),
}
_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def normalize_example(example: dict) -> dict:
    """Map one raw HF example to {id, question, choices: list[str], label: int}.

    Handles the mixed label schemes in ARC: most questions label choices A-E,
    a few use 1-5. After this function every example has an integer `label`
    that indexes into `choices`.
    """
    raw_labels = [str(lab).strip() for lab in example["choices"]["label"]]
    labels = [_NUM_TO_LETTER.get(lab, lab.upper()) for lab in raw_labels]
    answer = str(example["answerKey"]).strip()
    answer = _NUM_TO_LETTER.get(answer, answer.upper())
    if answer not in labels:
        raise ValueError(
            f"answerKey {example['answerKey']!r} not found in choice labels "
            f"{raw_labels!r} (id={example.get('id')!r})"
        )
    label_idx = labels.index(answer)
    choices = [str(t) for t in example["choices"]["text"]]
    assert 0 <= label_idx < len(choices)
    return {
        "id": str(example.get("id", "")),
        "question": example["question"],
        "choices": choices,
        "label": label_idx,
    }


def load_arc(subset: str = "combined", split: str = "train", max_samples: int | None = None):
    """Load and normalize ARC from Hugging Face.

    subset: "easy" | "challenge" | "combined"
    split:  "train" | "validation" | "test"
    """
    # Imported here (not module level) so offline tests don't need network access.
    from datasets import concatenate_datasets, load_dataset

    if subset not in SUBSET_TO_CONFIGS:
        raise ValueError(f"unknown subset {subset!r}, expected one of {list(SUBSET_TO_CONFIGS)}")
    parts = [load_dataset(HF_DATASET, cfg, split=split) for cfg in SUBSET_TO_CONFIGS[subset]]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    ds = ds.map(normalize_example, remove_columns=ds.column_names,
                desc=f"normalize {subset}/{split}")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    # Cheap full-pass sanity check; the datasets are small (plan section 2.2).
    for ex in ds:
        assert 0 <= ex["label"] < len(ex["choices"]), f"bad label for id={ex['id']}"
    return ds


@dataclass
class ArcCollator:
    """Collate a list of normalized examples into (B, C, L) tensors.

    C is the max number of choices in the batch. Dummy choice slots are filled
    with a copy of the example's *first real choice* -- never an all-padding
    sequence, which would make BERT's attention softmax produce NaNs -- and are
    excluded from the answer softmax via `choice_mask` (True = real choice).
    """

    tokenizer: object
    max_length: int = 128

    def __call__(self, batch: list[dict]) -> dict:
        n_choices = [len(ex["choices"]) for ex in batch]
        max_c = max(n_choices)
        firsts, seconds = [], []
        for ex in batch:
            for i in range(max_c):
                choice = ex["choices"][i] if i < len(ex["choices"]) else ex["choices"][0]
                firsts.append(ex["question"])
                seconds.append(choice)
        enc = self.tokenizer(
            firsts,
            seconds,
            truncation="only_first",  # truncate the question, never the option (plan 4.2)
            max_length=self.max_length,
            padding="longest",  # dynamic padding to the batch max length
            return_tensors="pt",
        )
        out = {k: v.view(len(batch), max_c, -1) for k, v in enc.items()}
        out["choice_mask"] = torch.tensor(
            [[True] * n + [False] * (max_c - n) for n in n_choices], dtype=torch.bool
        )
        if "label" in batch[0]:
            out["labels"] = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        if "id" in batch[0]:
            out["example_id"] = [ex["id"] for ex in batch]  # plain list, popped before forward
        return out


def masked_choice_logits(logits: torch.Tensor, choice_mask: torch.Tensor) -> torch.Tensor:
    """Give dummy choice slots ~zero softmax probability."""
    return logits.masked_fill(~choice_mask, torch.finfo(logits.dtype).min)
