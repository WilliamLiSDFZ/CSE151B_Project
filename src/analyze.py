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
import re
import sys
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
SPLITS = ["validation", "test"]          # column pair reported per subset in T1
SPLIT_LABEL = {"validation": "val", "test": "test"}
MODEL_COLOR = {"BERT-base": C_BERT, "DeBERTa-v3-base": C_DEBERTA}
SUBSET_STYLE = {"easy": "-", "challenge": ":", "combined": "--"}
SUBSET_MARKER = {"easy": "o", "challenge": "s", "combined": "D"}

# Error-analysis bins. The cue categories are a coarse *lexical* proxy, matched
# first-to-last — not the hand-labeled knowledge/reasoning taxonomy of Clark et al.
CONF_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
CUE_PATTERNS = [
    ("negation / except", r"\b(not|except|least|never)\b"),
    ("superlative", r"\b(most|best|greatest|largest|highest)\b"),
    ("comparative", r"\b(more|less|increase|decrease|greater|faster|slower)\b"),
    ("which-question", r"\bwhich\b"),
]

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
    runs_df = pd.DataFrame(runs)
    eval_index = index_evals(evals, runs_df)
    return {"runs": runs_df, "baselines": baselines,
            "evals": evals, "eval_index": eval_index}


def resolve_eval_model(checkpoint: str, runs: pd.DataFrame) -> str | None:
    """Map an eval's checkpoint path back to the run — hence model — that produced it.

    train.py saves to `checkpoints/<run_name>_<timestamp>/best`, so the checkpoint
    directory starts with the run_name. Longest matching run_name wins, so
    `..._epochs5` never steals a match from `..._epochs5_seed77`.
    """
    if runs.empty:
        return None
    parts = [p for p in Path(checkpoint).parts if p not in ("best", "last.pt")]
    if not parts:
        return None
    stem, best = parts[-1], None
    for _, r in runs.iterrows():
        name = str(r["run_name"])
        if stem.startswith(name) and (best is None or len(name) > len(best[0])):
            best = (name, r["model"])
    return best[1] if best else None


def index_evals(evals: list[dict], runs: pd.DataFrame) -> dict:
    """Key eval files by (model, subset, split); annotate each with its model."""
    index = {}
    for e in evals:
        e["model"] = resolve_eval_model(e["checkpoint"], runs)
        if e["model"]:
            index[(e["model"], e["subset"], e["split"])] = e
    return index


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


def baseline_acc(baselines: dict, subset: str, split: str, key) -> tuple:
    """Measured baseline for (subset, split).

    Reads the baseline file for that subset directly. Pooling easy+challenge is only
    a fallback for older results/ trees that predate baselines_combined_*.json — it
    assumes both subsets' training sets agree on the majority answer position, which
    the measured file does not have to assume.
    """
    b = baselines.get((subset, split))
    if b is not None:
        return key(b), b["n_examples"]
    if subset == "combined":
        be, bc = baselines.get(("easy", split)), baselines.get(("challenge", split))
        if be and bc:
            ne, nc = be["n_examples"], bc["n_examples"]
            return (key(be) * ne + key(bc) * nc) / (ne + nc), ne + nc
    return None, None


def build_t1(champs: pd.DataFrame, baselines: dict, eval_index: dict,
             n_val: dict) -> pd.DataFrame:
    """Table 1: baselines + per-model champions, validation and test, per subset."""
    rows = []

    def baseline_row(label, key):
        cells = {"Model": label}
        for s in SUBSETS:
            for sp in SPLITS:
                acc, n = baseline_acc(baselines, s, sp, key)
                cells[(s, sp)] = fmt_pct(acc, n) if acc is not None else "—"
        return cells

    rows.append(baseline_row("Random guess", lambda b: b["random"]["analytic_accuracy"]))
    rows.append(baseline_row("Majority position", lambda b: b["majority_position"]["accuracy"]))
    for model in sorted(champs["model"].unique()):
        cells = {"Model": f"{model} (best config)"}
        for s in SUBSETS:
            row = champs[(champs["model"] == model) & (champs["subset"] == s)]
            cells[(s, "validation")] = (fmt_pct(row.iloc[0]["best_val_acc"], n_val[s])
                                        if len(row) else "—")
            e = eval_index.get((model, s, "test"))
            cells[(s, "test")] = fmt_pct(e["accuracy"], e["n_examples"]) if e else "—"
        rows.append(cells)
    t1 = pd.DataFrame(rows)
    t1.columns = ["Model"] + [f"{s.capitalize()} {SPLIT_LABEL[sp]} % [95% CI]"
                              for s in SUBSETS for sp in SPLITS]
    return t1


# ---- error analysis ----------------------------------------------------------

def cue_category(question: str) -> str:
    q = question.lower()
    for name, pattern in CUE_PATTERNS:
        if re.search(pattern, q):
            return name
    return "other"


def option_len_band(choices: list[str]) -> str:
    m = float(np.mean([len(c.split()) for c in choices]))
    return ("1. <=2 words" if m <= 2 else "2. 3-5 words" if m <= 5
            else "3. 6-10 words" if m <= 10 else "4. >10 words")


def question_len_band(question: str) -> str:
    n = len(question.split())
    return ("1. <=10 words" if n <= 10 else "2. 11-20 words" if n <= 20
            else "3. 21-30 words" if n <= 30 else "4. >30 words")


def load_questions(subset: str, split: str) -> dict | None:
    """Question text keyed by example id, or None if the dataset is unreachable.

    Reuses load_arc from src/data.py. Content breakdowns are best-effort so the
    script still runs (confidence + position analysis only) without the HF cache.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from data import load_arc
        return {ex["id"]: ex for ex in load_arc(subset, split)}
    except Exception as exc:  # noqa: BLE001 — any failure degrades to tier 1
        print(f"[analyze] question text for {subset}/{split} unavailable "
              f"({type(exc).__name__}); skipping content breakdowns")
        return None


def error_analysis(evals: list[dict], out: Path) -> tuple:
    """Confidence, position, and (where question text is reachable) content breakdowns.

    Only easy/challenge evals are broken down — `combined` is their union, so
    including it would count every example twice.
    """
    scored = [e for e in evals
              if e.get("predictions") and e["subset"] in ("easy", "challenge")]
    calib_rows, cat_rows, lines = [], [], []

    for e in sorted(scored, key=lambda d: (d["split"], d["subset"])):
        subset, split, preds = e["subset"], e["split"], e["predictions"]
        n = len(preds)

        for lo, hi in CONF_BINS:
            binned = [p for p in preds if lo <= p["confidence"] < hi]
            if not binned:
                continue
            acc = float(np.mean([p["correct"] for p in binned]))
            conf = float(np.mean([p["confidence"] for p in binned]))
            calib_rows.append({
                "Subset": subset, "Split": split,
                "Confidence bin": f"[{lo:.1f}, {min(hi, 1.0):.1f})", "n": len(binned),
                "Accuracy %": round(100 * acc, 1),
                "Mean confidence %": round(100 * conf, 1),
                "Gap (conf - acc)": round(100 * (conf - acc), 1)})

        errors = [p for p in preds if not p["correct"]]
        overconf = [p for p in errors if p["confidence"] > 0.9]
        lines.append(
            f"- **{subset}/{split} — confidently wrong:** {len(overconf)} predictions "
            f"({100 * len(overconf) / n:.1f}% of the set, "
            f"{100 * len(overconf) / max(len(errors), 1):.1f}% of all errors) are "
            f"incorrect at confidence > 0.9.")

        n_pos = max(max(p["pred"] for p in preds), max(p["label"] for p in preds)) + 1
        pred_d = [sum(p["pred"] == i for p in preds) / n for i in range(n_pos)]
        gold_d = [sum(p["label"] == i for p in preds) / n for i in range(n_pos)]
        worst = max(abs(a - b) for a, b in zip(pred_d, gold_d))
        lines.append(
            f"- **{subset}/{split} — position bias:** predicted position distribution "
            f"{[f'{100 * x:.1f}%' for x in pred_d]} vs gold "
            f"{[f'{100 * x:.1f}%' for x in gold_d]}; largest deviation "
            f"{100 * worst:.1f} pts → no meaningful answer-position prior.")

        questions = load_questions(subset, split)
        if questions is None:
            continue
        groups: dict = {}
        for p in preds:
            ex = questions.get(p["id"])
            if ex is None:
                continue
            for dim, cat in (("lexical cue", cue_category(ex["question"])),
                             ("answer-option length", option_len_band(ex["choices"])),
                             ("question length", question_len_band(ex["question"]))):
                groups.setdefault((dim, cat), []).append(p["correct"])
        for dim in ("lexical cue", "answer-option length", "question length"):
            cats = {c: v for (d, c), v in groups.items() if d == dim}
            if not cats:
                continue
            overall = float(np.mean([x for v in cats.values() for x in v]))
            for cat, hits in sorted(cats.items()):
                rest = [x for c, v in cats.items() if c != cat for x in v]
                acc = float(np.mean(hits))
                pval = (two_prop_z(acc, len(hits), float(np.mean(rest)), len(rest))
                        if rest else 1.0)
                cat_rows.append({
                    "Subset": subset, "Split": split, "Dimension": dim, "Category": cat,
                    "n": len(hits), "Accuracy %": round(100 * acc, 1),
                    "Delta vs subset overall": round(100 * (acc - overall), 1),
                    "p (vs rest)": round(pval, 4)})

    calib = pd.DataFrame(calib_rows)
    cats_df = pd.DataFrame(cat_rows)
    if not calib.empty:
        write_table(calib, out, "t4_calibration")
    if not cats_df.empty:
        write_table(cats_df, out, "t5_error_categories")
        for subset, g in cats_df[cats_df["Dimension"] == "lexical cue"].groupby("Subset"):
            worst = g.sort_values("Delta vs subset overall").iloc[0]
            lines.append(
                f"- **{subset} — weakest question type:** *{worst['Category']}* scores "
                f"{worst['Accuracy %']}% (n={worst['n']}), "
                f"{worst['Delta vs subset overall']:+.1f} pts vs the subset overall, "
                f"two-proportion p = {worst['p (vs rest)']:.4f} "
                f"({'significant' if worst['p (vs rest)'] < 0.05 else 'NOT significant'} "
                f"at 0.05).")
    return calib, cats_df, lines


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


def _preferred_split(df: pd.DataFrame) -> str:
    return "test" if "test" in set(df["Split"]) else df["Split"].iloc[0]


def ordered_subsets(available) -> list[str]:
    """SUBSETS order, not alphabetical — keeps legends and n-labels in step."""
    return [s for s in SUBSETS if s in set(available)]


def fig_errors(calib: pd.DataFrame, cats: pd.DataFrame, out: Path) -> None:
    """Calibration + weakest question types for the evaluated champion.

    Colour stays C_DEBERTA throughout: in this repo colour encodes the *model*
    family, and every panel here is one model. Subsets are separated by the same
    line styles as fig_training_dynamics, plus marker shape in the dot plot.
    """
    if calib.empty:
        return
    calib = calib[calib["Split"] == _preferred_split(calib)]
    has_cats = not cats.empty
    if has_cats:
        cats = cats[(cats["Split"] == _preferred_split(cats))
                    & (cats["Dimension"] == "lexical cue")]
        has_cats = not cats.empty

    fig, axes = plt.subplots(1, 2 if has_cats else 1,
                             figsize=(9 if has_cats else 4.8, 3.4), squeeze=False)
    ax1 = axes[0][0]

    # equal aspect so the 45° reference really reads as 45° on screen
    ax1.set_aspect("equal", adjustable="box")
    ax1.plot([0, 100], [0, 100], color=AXIS, lw=1, zorder=1)
    # sits on the clear lower-left stretch of the diagonal; the curves start at ~28%
    ax1.text(4, 6, "perfect calibration", color=MUTED, fontsize=7,
             ha="left", va="bottom", rotation=45, rotation_mode="anchor")
    for subset in ordered_subsets(calib["Subset"]):
        g = calib[calib["Subset"] == subset].sort_values("Mean confidence %")
        ax1.plot(g["Mean confidence %"], g["Accuracy %"], SUBSET_STYLE.get(subset, "-"),
                 color=C_DEBERTA, lw=2, marker=SUBSET_MARKER.get(subset, "o"),
                 markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                 label=subset, zorder=3)
    ax1.set_xlabel("mean confidence in bin (%)")
    ax1.set_ylabel("accuracy in bin (%)")
    ax1.set_title("Calibration — below the line = overconfident", color=INK2)
    ax1.set_xlim(0, 105)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=7, frameon=False, loc="upper left")

    if has_cats:
        ax2 = axes[0][1]
        delta = cats.pivot(index="Category", columns="Subset",
                           values="Delta vs subset overall")
        counts = cats.pivot(index="Category", columns="Subset", values="n")
        shown = ordered_subsets(delta.columns)
        order = list(delta.sort_values(shown[0]).index)
        ypos = np.arange(len(order))
        ax2.axvline(0, color=AXIS, lw=1, zorder=1)
        for subset in shown:
            ax2.scatter(delta.loc[order, subset], ypos, s=70,
                        marker=SUBSET_MARKER[subset], color=C_DEBERTA,
                        edgecolor=SURFACE, linewidth=2, label=subset, zorder=3)
        # n's are listed in the same order as the legend, so they stay readable
        ticks = []
        for cat in order:
            ns = "/".join(str(int(counts.loc[cat, s])) for s in shown
                          if pd.notna(counts.loc[cat, s]))
            ticks.append(f"{cat}\n(n={ns})")
        ax2.set_yticks(ypos, ticks, fontsize=7)
        ax2.set_ylim(-0.6, len(order) - 0.4)
        ax2.set_xlabel("accuracy − subset overall (pts)")
        ax2.set_title("Weakest question types", color=INK2)
        ax2.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(out / "fig_error_analysis.png", bbox_inches="tight")
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
    eval_index = data["eval_index"]
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
    t1 = build_t1(champs, baselines, eval_index, n_val)
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

    # error analysis (activates as soon as evals carry --save_predictions records)
    calib, cats, error_lines = error_analysis(evals, out)

    # figures
    fig_heatmaps(completed, out)
    fig_dynamics(champs, out)
    fig_errors(calib, cats, out)

    # derived analyses
    lines = []
    for model in sorted(champs["model"].unique()):
        c = {s: champs[(champs["model"] == model) & (champs["subset"] == s)]
             for s in SUBSETS}
        if all(len(c[s]) for s in SUBSETS):
            ae, ac = c["easy"].iloc[0]["best_val_acc"], c["challenge"].iloc[0]["best_val_acc"]
            expected = (ae * n_val["easy"] + ac * n_val["challenge"]) / n_val["combined"]
            actual = c["combined"].iloc[0]["best_val_acc"]
            note = ""
            ee = eval_index.get((model, "easy", "test"))
            ec = eval_index.get((model, "challenge", "test"))
            if ee and ec:
                note = (f" On test, that same combined champion decomposes to "
                        f"{100 * ee['accuracy']:.1f}% easy (n={ee['n_examples']}) / "
                        f"{100 * ec['accuracy']:.1f}% challenge (n={ec['n_examples']}).")
            lines.append(
                f"- **{model} — combined vs separate:** separate champions imply an "
                f"expected combined-val accuracy of {100 * expected:.1f}%; the combined-"
                f"trained champion reaches {100 * actual:.1f}% "
                f"({100 * (actual - expected):+.1f} pts → combined training "
                f"{'helps' if actual > expected else 'does not help'}).{note}")
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

    # provenance: which checkpoint produced each test cell in T1
    provenance = [f"- `{e['checkpoint']}` → {e['model'] or 'unmatched model'} on "
                  f"{e['subset']}/{e['split']}: {100 * e['accuracy']:.1f}% "
                  f"(n={e['n_examples']})"
                  for e in sorted(evals, key=lambda d: (d["split"], d["subset"]))]

    # what is genuinely still missing, derived from state rather than hardcoded
    todo = []
    tested_models = {m for (m, _, sp) in eval_index if sp == "test"}
    untested = sorted(set(champs["model"]) - tested_models)
    for model in untested:
        todo.append(f"- **No test numbers for {model}.** Its champion checkpoint has to "
                    f"be retrained before a cross-model *test* comparison (or any "
                    f"combined-vs-separate claim on test) is possible.")
    if tested_models and len({e["checkpoint"] for e in evals if e["split"] == "test"}) == 1:
        todo.append("- All test cells come from a **single** checkpoint (the combined "
                    "champion), while the validation cells come from three different "
                    "per-subset champions — do not read a row as one model's profile.")
    # multi-seed means repeated seeds *within* a sweep, not merely different seeds
    # across sweeps — the latter says nothing about run-to-run variance.
    multi_seed = {m for (m, _), g in completed.groupby(["model", "subset"])
                  if len(set(g["seed"].dropna())) > 1}
    for model in sorted(set(champs["model"]) - multi_seed):
        todo.append(f"- **{model} sweeps are single-seed**: gaps of ~±2 pts within a "
                    f"sweep are noise. Run the 3-seed final config before naming a "
                    f"champion in the paper.")

    report = ["# Results Analysis (auto-generated by src/analyze.py)", "",
              f"Inputs: {len(completed)} completed runs / {len(runs)} run files, "
              f"{len(baselines)} baseline files, {len(evals)} eval files.", "",
              "## Table 1 — main results", "",
              "Cells are accuracy % with a 95% Wilson confidence interval. Validation "
              "cells come from each subset's sweep champion; test cells come from the "
              "checkpoints listed under Provenance below.", "",
              t1.to_markdown(index=False), "",
              "## Findings", "", *lines, ""]
    if error_lines:
        report += ["## Error analysis", "",
                   "Question categories are an automatic **lexical proxy** (regex over "
                   "the question text), not the hand-labeled knowledge/reasoning "
                   "taxonomy of Clark et al. (2018).", "",
                   *error_lines, ""]
    report += ["## Figures", "",
               "- `fig_hparam_heatmaps.png` — lr × epochs sensitivity, per model × subset",
               "- `fig_training_dynamics.png` — champions' val-accuracy and loss curves"]
    if not calib.empty:
        report += ["- `fig_error_analysis.png` — calibration and weakest question types"]
    report += ["", "## Efficiency", "", t3.to_markdown(index=False), ""]
    if provenance:
        report += ["## Provenance — evaluated checkpoints", "", *provenance, ""]
    report += ["## Coverage & caveats", "",
               "- Sweep coverage (12 = full grid):", *caveats, ""]
    if todo:
        report += ["## Still outstanding", "", *todo, ""]
    (out / "analysis.md").write_text("\n".join(report))

    print(f"[analyze] {len(completed)} completed runs, {len(evals)} evals -> {out}")
    print(t1.to_markdown(index=False))
    for line in lines + error_lines:
        print(line)


if __name__ == "__main__":
    main()
