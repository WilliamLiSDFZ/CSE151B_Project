"""Aggregate all experiment results under results/ into paper-ready tables & figures.

Reads every *.json under --results_dir (recursively), classifies files by content
(runs / baselines / eval files — folder names never matter), and writes tables
(md + tex + csv), figures (png), and an auto-generated analysis.md report to
--out. See ANALYSIS_PLAN.md for the full specification.

Usage:
    python src/analyze.py                     # results/ -> results/analysis/
    python src/analyze.py --results_dir results --out results/analysis
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

# ---- palette (validated reference palette; see dataviz notes) ----------------
C_BERT = "#2a78d6"      # categorical slot 1 (blue)  — BERT family
C_DEBERTA = "#008300"   # categorical slot 2 (green) — DeBERTa family
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, NEUTRAL = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

MODEL_INFO = {  # model_name -> (short label, params)
    "bert-base-uncased": ("BERT-base", "110M"),
    "microsoft/deberta-v3-base": ("DeBERTa-v3-base", "184M"),
}
SUBSETS = ["easy", "challenge", "combined"]
MODEL_COLOR = {"BERT-base": C_BERT, "DeBERTa-v3-base": C_DEBERTA}
SUBSET_STYLE = {"easy": "-", "challenge": ":", "combined": "--"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 200,
})


# ---- discovery ---------------------------------------------------------------

def short_model(name: str) -> str:
    return MODEL_INFO.get(name, (name.split("/")[-1], "?"))[0]


def discover(results_dir: Path) -> dict:
    runs, baselines, evals = [], {}, []
    for p in sorted(results_dir.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "random" in d and "majority_position" in d:
            baselines[(d["subset"], d["split"])] = d
        elif "val_accs" in d and "config" in d:
            cfg = d["config"]
            runs.append({
                "path": str(p), "run_name": d.get("run_name", p.stem),
                "status": d.get("status", "completed"),
                "model_name": cfg["model_name"], "model": short_model(cfg["model_name"]),
                "subset": cfg["subset"], "lr": cfg["lr"], "epochs": cfg["epochs"],
                "seed": cfg.get("seed"), "epochs_done": d.get("epochs_done", len(d["val_accs"])),
                "best_val_acc": d["best_val_acc"], "val_accs": d["val_accs"],
                "loss_curve": d.get("loss_curve", []),
                "mean_epoch_s": (np.mean([t["total_s"] for t in d["epoch_times"]])
                                 if d.get("epoch_times") else np.nan),
                "wall_time_s": d.get("wall_time_s", np.nan),
            })
        elif "accuracy" in d and "checkpoint" in d:
            evals.append(d)
    return {"runs": pd.DataFrame(runs), "baselines": baselines, "evals": evals}


# ---- statistics --------------------------------------------------------------

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def two_prop_z(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-sided p-value for H0: p1 == p2."""
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


# ---- tables ------------------------------------------------------------------

def fmt_pct(p: float, n: int | None = None) -> str:
    if pd.isna(p):
        return "—"
    if n:
        lo, hi = wilson_ci(p, n)
        return f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{100 * p:.1f}"


def write_table(df: pd.DataFrame, out: Path, stem: str, index: bool = False) -> None:
    df.to_csv(out / f"{stem}.csv", index=index)
    (out / f"{stem}.md").write_text(df.to_markdown(index=index) + "\n")
    (out / f"{stem}.tex").write_text(df.to_latex(index=index) + "\n")


def build_t1(champs: pd.DataFrame, baselines: dict, n_val: dict) -> pd.DataFrame:
    rows = []

    def baseline_row(label, key):
        cells = {}
        for s in SUBSETS:
            if s == "combined":
                be, bc = baselines.get(("easy", "validation")), baselines.get(("challenge", "validation"))
                if be and bc:
                    ne, nc = be["n_examples"], bc["n_examples"]
                    acc = (key(be) * ne + key(bc) * nc) / (ne + nc)
                    cells[s] = fmt_pct(acc, ne + nc)
                else:
                    cells[s] = "—"
            else:
                b = baselines.get((s, "validation"))
                cells[s] = fmt_pct(key(b), b["n_examples"]) if b else "—"
        return cells

    rows.append({"Model": "Random guess",
                 **baseline_row("random", lambda b: b["random"]["analytic_accuracy"])})
    rows.append({"Model": "Majority position",
                 **baseline_row("majority", lambda b: b["majority_position"]["accuracy"])})
    for model in sorted(champs["model"].unique()):
        cells = {"Model": f"{model} (best config)"}
        for s in SUBSETS:
            row = champs[(champs["model"] == model) & (champs["subset"] == s)]
            cells[s] = fmt_pct(row.iloc[0]["best_val_acc"], n_val[s]) if len(row) else "—"
        rows.append(cells)
    t1 = pd.DataFrame(rows)
    t1.columns = ["Model"] + [f"{s.capitalize()} val acc % [95% CI]" for s in SUBSETS]
    return t1


# ---- figures -----------------------------------------------------------------

def fig_heatmaps(runs: pd.DataFrame, out: Path) -> None:
    models = sorted(runs["model"].unique())
    fig, axes = plt.subplots(len(models), len(SUBSETS),
                             figsize=(3.4 * len(SUBSETS), 2.6 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, subset in enumerate(SUBSETS):
            ax = axes[i][j]
            ax.grid(False)
            sub = runs[(runs["model"] == model) & (runs["subset"] == subset)]
            lrs = sorted(sub["lr"].unique())
            eps = sorted(sub["epochs"].unique())
            if not len(sub):
                ax.set_axis_off()
                continue
            mat = np.full((len(lrs), len(eps)), np.nan)
            for _, r in sub.iterrows():
                mat[lrs.index(r["lr"]), eps.index(r["epochs"])] = r["best_val_acc"]
            vmin, vmax = np.nanmin(mat), np.nanmax(mat)
            masked = np.ma.masked_invalid(mat)
            cmap = SEQ_BLUE.copy()
            cmap.set_bad(NEUTRAL)
            ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            best = np.unravel_index(np.nanargmax(mat), mat.shape)
            for r_i in range(len(lrs)):
                for c_i in range(len(eps)):
                    v = mat[r_i, c_i]
                    if np.isnan(v):
                        ax.text(c_i, r_i, "—", ha="center", va="center", color=MUTED)
                        continue
                    rel = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    color = "#ffffff" if rel > 0.6 else INK
                    weight = "bold" if (r_i, c_i) == best else "normal"
                    ax.text(c_i, r_i, f"{100 * v:.1f}", ha="center", va="center",
                            color=color, fontsize=8, fontweight=weight)
            ax.set_xticks(range(len(eps)), [str(e) for e in eps])
            ax.set_yticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
            if i == len(models) - 1:
                ax.set_xlabel("epochs")
            if j == 0:
                ax.set_ylabel(f"{model}\nlearning rate")
            if i == 0:
                ax.set_title(subset.capitalize(), color=INK2)
    fig.suptitle("Validation accuracy (%) across the hyperparameter grid "
                 "(bold = sweep champion, — = run not completed)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_hparam_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


def fig_dynamics(champs: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for _, r in champs.sort_values(["model", "subset"]).iterrows():
        color, ls = MODEL_COLOR.get(r["model"], MUTED), SUBSET_STYLE[r["subset"]]
        label = f"{r['model']} · {r['subset']}"
        accs = [100 * a for a in r["val_accs"]]
        ax1.plot(range(1, len(accs) + 1), accs, ls, color=color, lw=2,
                 marker="o", markersize=4, label=label)
        if r["loss_curve"]:
            x = np.linspace(0, r["epochs_done"], len(r["loss_curve"]))
            ax2.plot(x, r["loss_curve"], ls, color=color, lw=2, label=label)
    ax1.axhline(25, color=MUTED, lw=1, ls=(0, (2, 2)))
    ax1.text(1.02, 25, "random", transform=ax1.get_yaxis_transform(),
             color=MUTED, va="center", fontsize=8)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation accuracy (%)")
    ax1.set_title("Champions: val accuracy per epoch", color=INK2)
    ax1.set_ylim(bottom=20)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("training loss")
    ax2.set_title("Champions: training loss", color=INK2)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_training_dynamics.png", bbox_inches="tight")
    plt.close(fig)


# ---- report ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()
    results_dir, out = Path(args.results_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = discover(results_dir)
    runs, baselines, evals = data["runs"], data["baselines"], data["evals"]
    if runs.empty:
        raise SystemExit(f"no run JSONs found under {results_dir}")
    completed = runs[runs["status"] == "completed"].copy()

    n_val = {"easy": 570, "challenge": 299}
    for s in ("easy", "challenge"):
        if (s, "validation") in baselines:
            n_val[s] = baselines[(s, "validation")]["n_examples"]
    n_val["combined"] = n_val["easy"] + n_val["challenge"]

    champs = (completed.sort_values("best_val_acc", ascending=False)
              .groupby(["model", "subset"], as_index=False).first())

    # T1 main results
    t1 = build_t1(champs, baselines, n_val)
    write_table(t1, out, "t1_main_results")

    # T2 leaderboard (all completed runs)
    t2 = completed[["model", "subset", "lr", "epochs", "seed", "best_val_acc",
                    "epochs_done", "mean_epoch_s", "wall_time_s"]].copy()
    t2 = t2.sort_values(["model", "subset", "best_val_acc"],
                        ascending=[True, True, False])
    t2["best_val_acc"] = (100 * t2["best_val_acc"]).round(2)
    write_table(t2, out, "t2_all_runs")

    # T3 efficiency
    t3_rows = []
    for model, g in completed.groupby("model"):
        name = [k for k, v in MODEL_INFO.items() if v[0] == model]
        params = MODEL_INFO[name[0]][1] if name else "?"
        t3_rows.append({
            "Model": model, "Params": params,
            "Mean s/epoch": round(float(g["mean_epoch_s"].mean()), 1),
            "Completed runs": len(g),
            "Total GPU time (min)": round(float(g["wall_time_s"].sum()) / 60, 1),
        })
    t3 = pd.DataFrame(t3_rows)
    write_table(t3, out, "t3_efficiency")

    # figures
    fig_heatmaps(completed, out)
    fig_dynamics(champs, out)

    # derived analyses
    lines = []
    for model in sorted(champs["model"].unique()):
        c = {s: champs[(champs["model"] == model) & (champs["subset"] == s)]
             for s in SUBSETS}
        if all(len(c[s]) for s in SUBSETS):
            ae, ac = c["easy"].iloc[0]["best_val_acc"], c["challenge"].iloc[0]["best_val_acc"]
            expected = (ae * n_val["easy"] + ac * n_val["challenge"]) / n_val["combined"]
            actual = c["combined"].iloc[0]["best_val_acc"]
            lines.append(
                f"- **{model} — combined vs separate:** separate champions imply an "
                f"expected combined-val accuracy of {100 * expected:.1f}%; the combined-"
                f"trained champion reaches {100 * actual:.1f}% "
                f"({100 * (actual - expected):+.1f} pts → combined training "
                f"{'helps' if actual > expected else 'does not help'}). Note: the exact "
                f"per-subset split of the combined champion still needs the two "
                f"decomposition evals (commands below).")
    models = sorted(champs["model"].unique())
    if len(models) == 2:
        for s in SUBSETS:
            a = champs[(champs["model"] == models[0]) & (champs["subset"] == s)]
            b = champs[(champs["model"] == models[1]) & (champs["subset"] == s)]
            if len(a) and len(b):
                p = two_prop_z(a.iloc[0]["best_val_acc"], n_val[s],
                               b.iloc[0]["best_val_acc"], n_val[s])
                gap = 100 * (b.iloc[0]["best_val_acc"] - a.iloc[0]["best_val_acc"])
                lines.append(f"- **{models[1]} vs {models[0]} on {s}:** "
                             f"{gap:+.1f} pts, two-proportion p = {p:.4f} "
                             f"({'significant' if p < 0.05 else 'NOT significant'} at 0.05).")

    caveats = []
    for (model, subset), g in completed.groupby(["model", "subset"]):
        seeds = sorted(set(g["seed"].dropna().astype(int)))
        note = f"  - {model} / {subset}: {len(g)} completed runs, seeds {seeds}"
        if len(g) < 12:
            note += "  ← partial sweep coverage"
        caveats.append(note)

    eval_lines = []
    for e in evals:
        eval_lines.append(f"  - `{e['checkpoint']}` on {e['subset']}/{e['split']}: "
                          f"{100 * e['accuracy']:.1f}% (n={e['n_examples']})")

    report = ["# Results Analysis (auto-generated by src/analyze.py)", "",
              f"Inputs: {len(completed)} completed runs / {len(runs)} run files, "
              f"{len(baselines)} baseline files, {len(evals)} eval files.", "",
              "## Table 1 — main results (validation)", "", t1.to_markdown(index=False), "",
              "## Findings", "", *lines, "",
              "## Figures", "",
              "- `fig_hparam_heatmaps.png` — lr × epochs sensitivity, per model × subset",
              "- `fig_training_dynamics.png` — champions' val-accuracy and loss curves", "",
              "## Efficiency", "", t3.to_markdown(index=False), "",
              "## Coverage & caveats", "",
              "- Sweep coverage (12 = full grid):", *caveats,
              "- All numbers are validation accuracy; test sets remain untouched.",
              "- Single-seed sweeps: differences of ~±2 pts within a sweep are noise; "
              "run the 3-seed final config before claiming a specific champion.", "",
              "## TODO evals still needed (run on DataHub)", "",
              "```",
              "# decompose the combined champions per subset:",
              "python src/evaluate.py --checkpoint <combined_champion>/best --subset easy --split validation",
              "python src/evaluate.py --checkpoint <combined_champion>/best --subset challenge --split validation",
              "# final (once, at the very end): test evals with --save_predictions",
              "```", ""]
    if eval_lines:
        report.insert(-6, "## Eval files found\n")
        report.insert(-6, "\n".join(eval_lines) + "\n")
    (out / "analysis.md").write_text("\n".join(report))

    print(f"[analyze] {len(completed)} completed runs -> {out}")
    print(t1.to_markdown(index=False))
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
