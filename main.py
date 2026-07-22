"""Config-driven hyperparameter sweep runner.

Reads a JSON config listing hyperparameter values to test, expands the grid,
runs src/train.py once per combination (as a subprocess, so GPU memory is fully
released between runs), and — because server storage is limited — keeps only
the checkpoints of the BEST run (by best_val_acc), deleting the rest as soon
as each run is beaten.

Usage:
    python main.py --config configs/sweep.json
    python main.py --config configs/sweep.json --dry_run   # print commands only

Config format (JSON):
    {
      "sweep_name": "sweep",              # prefix for run names + summary file
      "model_name": "bert-base-uncased",
      "subset": "combined",
      "seed": 42,
      "grid":  {"lr": [1e-5, 3e-5], "epochs": [2, 3]},   # cartesian product;
                                          # any train.py argument may be a grid axis
      "fixed": {"batch_size": 8, "grad_accum": 2},        # applied to every run
      "keep_champion_last_pt": true,      # false: champion keeps only best/ (saves ~1.3GB)
      "keep_all_checkpoints": false       # true: delete nothing (use for final multi-seed runs)
    }

Interrupted sweeps resume for free: combinations whose results/<run_name>.json
already exists are skipped (use --rerun to force).
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent


def expand_grid(grid: dict) -> list[dict]:
    """{"lr": [a, b], "epochs": [2]} -> [{"lr": a, "epochs": 2}, {"lr": b, "epochs": 2}]"""
    keys = list(grid)
    return [dict(zip(keys, values))
            for values in itertools.product(*(grid[k] for k in keys))]


def fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def find_ckpt_dir(ckpt_root: Path, run_name: str) -> Path | None:
    """Newest existing checkpoint dir for run_name: timestamped
    <run_name>_YYYYmmdd_HHMMSS preferred (max = newest), else legacy
    exact <run_name>, else None."""
    stamped = sorted(p for p in ckpt_root.glob(run_name + "_*")
                     if p.is_dir()
                     and re.fullmatch(re.escape(run_name) + r"_\d{8}_\d{6}", p.name))
    if stamped:
        return stamped[-1]
    legacy = ckpt_root / run_name
    return legacy if legacy.is_dir() else None


def build_run(combo: dict, cfg: dict) -> tuple[str, list[str]]:
    """Return (run_name, command) for one grid combination."""
    merged = {k: cfg[k] for k in ("model_name", "subset", "seed") if k in cfg}
    merged.update(cfg.get("fixed", {}))
    merged.update(combo)
    run_name = cfg["sweep_name"] + "_" + "_".join(f"{k}{fmt(v)}" for k, v in combo.items())
    cmd = [sys.executable, str(REPO / cfg.get("train_script", "src/train.py")),
           "--run_name", run_name,
           "--results_dir", str(REPO / "results")]
    for k, v in merged.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return run_name, cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/sweep.json")
    ap.add_argument("--dry_run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--rerun", action="store_true",
                    help="rerun combinations even if their results JSON exists")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    keep_all = cfg.get("keep_all_checkpoints", False)
    combos = expand_grid(cfg["grid"])
    results_dir = REPO / "results"
    ckpt_root = REPO / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] {cfg['sweep_name']}: {len(combos)} runs, config={args.config}")

    champion: tuple[float, str, Path] | None = None   # (best_val_acc, run_name, ckpt_dir)
    summary: list[dict] = []
    t0 = time.time()
    for i, combo in enumerate(combos):
        run_name, cmd = build_run(combo, cfg)
        # reuse an existing folder (resume/overwrite semantics, avoids orphans),
        # otherwise mint a fresh timestamped one
        run_dir = (find_ckpt_dir(ckpt_root, run_name)
                   or ckpt_root / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}")
        cmd += ["--output_dir", str(run_dir)]
        res_path = results_dir / f"{run_name}.json"
        if args.dry_run:
            print("[dry]", " ".join(cmd))
            continue
        need_run, resume = True, False
        if res_path.exists() and not args.rerun:
            prev = json.loads(res_path.read_text())
            if prev.get("status", "completed") == "completed":
                need_run = False
                print(f"[sweep] ({i + 1}/{len(combos)}) {run_name}: "
                      f"completed results exist, skipping run")
            else:
                resume = (run_dir / "last.pt").exists()
                print(f"[sweep] ({i + 1}/{len(combos)}) {run_name}: incomplete results "
                      f"({prev.get('epochs_done', 0)} epochs done), "
                      f"{'resuming' if resume else 'restarting'}")
        if need_run:
            if resume:
                cmd.append("--resume")
            if not res_path.exists():
                print(f"[sweep] ({i + 1}/{len(combos)}) {run_name}: starting")
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"[sweep] {run_name}: FAILED (exit {proc.returncode}), "
                      f"checkpoints removed, continuing with next combo")
                shutil.rmtree(run_dir, ignore_errors=True)
                summary.append({"run_name": run_name, **combo, "status": "failed"})
                continue
        acc = json.loads(res_path.read_text())["best_val_acc"]
        summary.append({"run_name": run_name, **combo,
                        "best_val_acc": acc, "status": "ok"})

        if keep_all:
            print(f"[sweep] {run_name}: best_val_acc={acc:.4f} (keep_all_checkpoints on)")
            if champion is None or acc > champion[0]:
                champion = (acc, run_name, run_dir)
            continue

        # keep only the champion's checkpoints
        if champion is None or acc > champion[0]:
            if champion is not None:
                shutil.rmtree(champion[2], ignore_errors=True)
                print(f"[sweep] {run_name}: new champion ({acc:.4f}), "
                      f"deleted checkpoints of {champion[1]}")
            else:
                print(f"[sweep] {run_name}: first champion ({acc:.4f})")
            champion = (acc, run_name, run_dir)
            if not cfg.get("keep_champion_last_pt", True):
                (run_dir / "last.pt").unlink(missing_ok=True)
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
            print(f"[sweep] {run_name}: {acc:.4f} <= champion {champion[0]:.4f}, "
                  f"checkpoints deleted")

    if args.dry_run:
        return

    summary.sort(key=lambda r: r.get("best_val_acc", -1.0), reverse=True)
    out = {
        "sweep_name": cfg["sweep_name"],
        "config": cfg,
        "n_runs": len(combos),
        "champion": ({"run_name": champion[1], "best_val_acc": champion[0],
                      "checkpoint": str(champion[2] / "best")}
                     if champion else None),
        "runs": summary,
        "wall_time_s": round(time.time() - t0, 1),
    }
    summary_path = results_dir / f"{cfg['sweep_name']}_summary.json"
    summary_path.write_text(json.dumps(out, indent=2))

    print(f"\n[sweep] done in {out['wall_time_s']}s — leaderboard:")
    for r in summary:
        acc = f"{r['best_val_acc']:.4f}" if "best_val_acc" in r else "FAILED"
        print(f"    {acc}  {r['run_name']}")
    if champion:
        champ_best = champion[2] / "best"
        print(f"[sweep] champion: {champion[1]} (best_val_acc={champion[0]:.4f})")
        print(f"[sweep] checkpoint kept at: {champ_best}")
        if not champ_best.exists():
            print("[sweep] WARNING: champion checkpoint missing on disk (its run was "
                  "probably skipped after an earlier sweep deleted it). To restore: "
                  f"delete results/{champion[1]}.json and rerun this sweep.")
    print(f"[sweep] summary written to {summary_path}")


if __name__ == "__main__":
    main()
