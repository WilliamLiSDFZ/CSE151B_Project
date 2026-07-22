"""Fine-tune a pretrained transformer on ARC multiple-choice QA.

Device-agnostic (CUDA on DataHub / MPS on Mac / CPU fallback), fp16 on CUDA,
epoch-level --resume for culled DataHub sessions, JSON result logging
(plan sections 4-5).

Typical runs:
    # smoke test (tiny model, tiny sample -- should overfit)
    python src/train.py --model_name prajjwal1/bert-tiny --subset easy \
        --max_train_samples 100 --max_val_samples 100 --epochs 5 --lr 1e-4

    # full default run
    python src/train.py --subset combined --output_dir checkpoints/bert_base_combined
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ArcCollator, load_arc, masked_choice_logits  # noqa: E402
from evaluate import evaluate_model, pick_device  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, lr: float, weight_decay: float) -> torch.optim.AdamW:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def linear_warmup_schedule(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


def run_training(model, tokenizer, train_ds, val_ds, args, device) -> dict:
    """Core training loop; returns the results record that is also written to JSON."""
    collator = ArcCollator(tokenizer, args.max_length)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=gen, collate_fn=collator,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            collate_fn=collator, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = linear_warmup_schedule(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    fp16 = device.type == "cuda" and not args.no_fp16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if fp16 else nullcontext

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"

    results_path = None
    if getattr(args, "results_dir", None):
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"{args.run_name}.json"

    start_epoch, best_val_acc = 0, -1.0
    loss_curve: list[float] = []
    val_accs: list[float] = []
    epoch_times: list[dict] = []
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_val_acc = state["best_val_acc"]
        loss_curve = state["loss_curve"]
        val_accs = state["val_accs"]
        epoch_times = state.get("epoch_times", [])
        print(f"[train] resumed from {last_path} at epoch {start_epoch}")

    t0 = time.time()

    def write_results(status: str) -> dict:
        """Persist current metrics to results/<run_name>.json. Called after EVERY
        epoch (status="running") so a killed/culled session never loses metrics,
        and once at the end (status="completed")."""
        record = {
            "run_name": args.run_name,
            "status": status,
            "epochs_done": len(val_accs),
            "config": {
                "model_name": args.model_name, "subset": args.subset, "lr": args.lr,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch_size": args.batch_size * args.grad_accum,
                "max_length": args.max_length, "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio, "seed": args.seed, "fp16": fp16,
                "device": device.type,
            },
            "n_train": len(train_ds), "n_val": len(val_ds),
            "val_accs": val_accs, "best_val_acc": best_val_acc,
            "loss_curve": loss_curve,
            "epoch_times": epoch_times,
            "wall_time_s": round(time.time() - t0, 1),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if results_path is not None:
            results_path.write_text(json.dumps(record, indent=2))
        return record

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        model.train()
        running, n_running = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch.pop("example_id", None)
            labels = batch.pop("labels").to(device)
            mask = batch.pop("choice_mask").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                logits = model(**batch).logits
                loss = F.cross_entropy(masked_choice_logits(logits, mask), labels)
            scaler.scale(loss / args.grad_accum).backward()
            running += loss.item()
            n_running += 1
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if n_running % (args.log_every * args.grad_accum) == 0:
                avg = running / n_running
                loss_curve.append(round(avg, 5))
                print(f"[train] epoch {epoch} step {step + 1}/{len(train_loader)} "
                      f"loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                running, n_running = 0.0, 0
        if n_running:
            loss_curve.append(round(running / n_running, 5))

        t_train = time.time() - t_epoch
        val_acc, _ = evaluate_model(model, val_loader, device)
        t_eval = time.time() - t_epoch - t_train
        epoch_times.append({"epoch": epoch, "train_s": round(t_train, 1),
                            "eval_s": round(t_eval, 1),
                            "total_s": round(t_train + t_eval, 1)})
        val_accs.append(round(val_acc, 5))
        print(f"[train] epoch {epoch} done: val_acc={val_acc:.4f} (best={best_val_acc:.4f}) | "
              f"train {t_train:.1f}s + eval {t_eval:.1f}s = {t_train + t_eval:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"[train] new best -> saved to {best_dir}")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_acc": best_val_acc,
            "loss_curve": loss_curve,
            "val_accs": val_accs,
            "epoch_times": epoch_times,
        }, last_path)
        write_results("running")

    return write_results("completed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--subset", default="combined", choices=["easy", "challenge", "combined"])
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8, help="per-device batch size")
    p.add_argument("--grad_accum", type=int, default=2,
                   help="effective batch = batch_size * grad_accum (plan: 16)")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=20, help="in optimizer steps")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--output_dir", default=None,
                   help="default: checkpoints/{model}_{subset}_lr{lr}_s{seed}")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume", action="store_true", help="resume from output_dir/last.pt")
    p.add_argument("--no_fp16", action="store_true", help="disable fp16 even on CUDA")
    args = p.parse_args()

    model_slug = args.model_name.split("/")[-1]
    if args.run_name is None:
        args.run_name = f"{model_slug}_{args.subset}_lr{args.lr:g}_e{args.epochs}_s{args.seed}"
    if args.output_dir is None:
        args.output_dir = f"checkpoints/{args.run_name}"

    from transformers import AutoModelForMultipleChoice, AutoTokenizer

    set_seed(args.seed)
    device = pick_device()
    print(f"[train] device = {device}, run = {args.run_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # .float(): some checkpoints (e.g. deberta-v3) are stored in fp16 and load as
    # fp16 params; GradScaler requires fp32 master weights (fp16 is for compute only).
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name).float().to(device)

    train_ds = load_arc(args.subset, "train", max_samples=args.max_train_samples)
    val_ds = load_arc(args.subset, "validation", max_samples=args.max_val_samples)
    print(f"[train] n_train={len(train_ds)} n_val={len(val_ds)}")

    results = run_training(model, tokenizer, train_ds, val_ds, args, device)
    print(f"[train] best_val_acc={results['best_val_acc']:.4f}; "
          f"wrote {Path(args.results_dir) / (args.run_name + '.json')}")


if __name__ == "__main__":
    main()
