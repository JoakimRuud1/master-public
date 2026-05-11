from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.judge_schema import (
        GATE_SCORE_KEY,
        QUALITY_SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
        VOICE_CASES,
    )
except Exception:
    from judge_schema import (  # type: ignore
        GATE_SCORE_KEY,
        QUALITY_SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
        VOICE_CASES,
    )


DIMENSION_LABELS = {
    "citation": "Citation",
    "accurate": "Accurate",
    "thorough": "Thorough",
    "useful": "Useful",
    "organized": "Organized",
    "comprehensible": "Comprehensible",
    "succinct": "Succinct",
    "synthesized": "Synthesized",
    "abstraction": "Abstraction (gate)",
}

VOICE_CASE_LABELS = {
    "no_stigmatizing_language_detected": "Ingen stigma",
    "neutralized_stigmatizing_language": "Nøytralisert stigma",
    "propagated_stigmatizing_language": "Videreført stigma",
    "introduced_stigmatizing_language": "Introdusert stigma",
}

VOICE_CASE_COLORS = {
    "no_stigmatizing_language_detected": "#cfd8dc",
    "neutralized_stigmatizing_language": "#7ac87a",
    "propagated_stigmatizing_language": "#e08c3e",
    "introduced_stigmatizing_language": "#d65a5a",
}

VOICE_CASE_ORDER = [
    "no_stigmatizing_language_detected",
    "neutralized_stigmatizing_language",
    "propagated_stigmatizing_language",
    "introduced_stigmatizing_language",
]

HEATMAP_CMAP = "YlGnBu"


def load_strategy_names(strategies_path: Path) -> Dict[str, str]:
    if not strategies_path.exists():
        return {}
    with strategies_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["id"]: s.get("name", s["id"]) for s in data}


def strategy_label(strategy_id: str, names: Dict[str, str]) -> str:
    parts = strategy_id.split("_", 1)
    tag = parts[0] if len(parts) == 2 and parts[0].isdigit() else ""
    name = names.get(strategy_id, strategy_id.replace("_", " "))
    return f"{tag}. {name}" if tag else name


def wrap_label(label: str, max_chars: int = 22) -> str:
    words = label.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def build_strategy_palette(strategy_ids: List[str]) -> Dict[str, tuple]:
    palette = sns.color_palette("tab10", n_colors=max(len(strategy_ids), 10))
    return {sid: palette[i] for i, sid in enumerate(sorted(strategy_ids))}


def apply_global_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#111111",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "text.color": "#222222",
        "font.size": 10,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight", facecolor="white")


def plot_strategy_ranking(
    per_summary: pd.DataFrame,
    labels: Dict[str, str],
    colors: Dict[str, tuple],
    out_dir: Path,
) -> None:
    stats = (
        per_summary.groupby("strategy_id")["primary_score"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )
    n_per = int(stats["count"].iloc[0]) if len(stats) else 0
    same_n = stats["count"].nunique() == 1

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ypos = np.arange(len(stats))
    bar_colors = [colors[s] for s in stats["strategy_id"]]
    ax.barh(
        ypos,
        stats["mean"],
        xerr=stats["sem"].fillna(0),
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[s] for s in stats["strategy_id"]])
    ax.set_xlabel("Gjennomsnittlig primary score (1–5)")
    ax.set_xlim(1, 5.25)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    overall_mean = per_summary["primary_score"].mean()
    ax.axvline(overall_mean, color="#666666", linestyle="--", linewidth=0.9, label=f"Samlet snitt = {overall_mean:.2f}")

    for y, (mean, sem) in zip(ypos, zip(stats["mean"], stats["sem"].fillna(0))):
        offset = max(sem, 0) + 0.05
        ax.text(mean + offset, y, f"{mean:.2f}", va="center", fontsize=9, color="#222222")

    title_suffix = f" (n = {n_per} per strategi)" if same_n else ""
    ax.set_title(f"Strategi-ranking etter primary score{title_suffix}")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_dir / "01_strategy_ranking")
    plt.close(fig)


def plot_dimension_heatmap(
    per_summary: pd.DataFrame,
    labels: Dict[str, str],
    out_dir: Path,
) -> None:
    strat_order = sorted(per_summary["strategy_id"].unique())
    dim_cols = [*QUALITY_SCORE_KEYS, SYNTHESIS_SCORE_KEY]
    matrix = per_summary.groupby("strategy_id")[dim_cols].mean().reindex(strat_order)

    col_labels = [DIMENSION_LABELS[c] for c in dim_cols]
    row_labels = [labels[s] for s in strat_order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        matrix.values,
        annot=True,
        fmt=".2f",
        cmap=HEATMAP_CMAP,
        vmin=1,
        vmax=5,
        cbar_kws={"label": "Gjennomsnittlig score (1–5)"},
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.6,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 9},
    )
    ax.set_title("Gjennomsnittsscore per strategi × rubrikk-dimensjon")
    ax.set_xlabel("Dimensjon")
    ax.set_ylabel("Strategi")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    save_fig(fig, out_dir / "02_dimension_heatmap")
    plt.close(fig)


def plot_score_distribution(
    per_summary: pd.DataFrame,
    labels: Dict[str, str],
    colors: Dict[str, tuple],
    out_dir: Path,
) -> None:
    strat_order = sorted(per_summary["strategy_id"].unique())
    palette = [colors[s] for s in strat_order]

    fig, ax = plt.subplots(figsize=(11.5, 6))
    sns.boxplot(
        data=per_summary,
        x="strategy_id",
        y="primary_score",
        order=strat_order,
        hue="strategy_id",
        hue_order=strat_order,
        palette=palette,
        width=0.55,
        fliersize=0,
        linewidth=1.1,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=per_summary,
        x="strategy_id",
        y="primary_score",
        order=strat_order,
        color="#222222",
        alpha=0.55,
        size=4,
        jitter=0.18,
        ax=ax,
    )
    ax.set_xticks(np.arange(len(strat_order)))
    ax.set_xticklabels([wrap_label(labels[s]) for s in strat_order], rotation=0, fontsize=9)
    ax.set_xlabel("Strategi")
    ax.set_ylabel("Primary score")
    ax.set_ylim(1, 5.2)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Fordeling av primary score per strategi")
    fig.tight_layout()
    save_fig(fig, out_dir / "03_score_distribution")
    plt.close(fig)


def plot_conversation_heatmap(
    per_summary: pd.DataFrame,
    labels: Dict[str, str],
    out_dir: Path,
) -> None:
    strat_order = sorted(per_summary["strategy_id"].unique())
    matrix = per_summary.pivot_table(
        index="strategy_id",
        columns="conversation_id",
        values="primary_score",
        aggfunc="mean",
    ).reindex(strat_order)
    matrix = matrix[sorted(matrix.columns)]

    conv_labels = [c.split(":")[-1] for c in matrix.columns]
    n_cols = len(matrix.columns)
    width = max(9.0, min(22.0, 2.5 + 0.55 * n_cols))

    fig, ax = plt.subplots(figsize=(width, 5.5))
    sns.heatmap(
        matrix.values,
        annot=True,
        fmt=".2f",
        cmap=HEATMAP_CMAP,
        vmin=1,
        vmax=5,
        cbar_kws={"label": "Primary score (1–5)"},
        xticklabels=conv_labels,
        yticklabels=[labels[s] for s in matrix.index],
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Primary score per strategi × samtale")
    ax.set_xlabel("Samtale (source_id)")
    ax.set_ylabel("Strategi")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    save_fig(fig, out_dir / "04_conversation_heatmap")
    plt.close(fig)


def plot_abstraction_synthesis(
    per_summary: pd.DataFrame,
    labels: Dict[str, str],
    colors: Dict[str, tuple],
    out_dir: Path,
) -> None:
    strat_order = sorted(per_summary["strategy_id"].unique())
    abstraction_rate = (
        per_summary.groupby("strategy_id")[GATE_SCORE_KEY].mean().reindex(strat_order)
    )
    synth_df = per_summary[per_summary[GATE_SCORE_KEY] == 1]
    synth_mean = synth_df.groupby("strategy_id")[SYNTHESIS_SCORE_KEY].mean().reindex(strat_order)
    synth_n = synth_df.groupby("strategy_id").size().reindex(strat_order).fillna(0).astype(int)

    xs = np.arange(len(strat_order))
    bar_colors = [colors[s] for s in strat_order]
    tick_labels = [labels[s] for s in strat_order]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6.2))

    ax_left.bar(xs, abstraction_rate.values, color=bar_colors, edgecolor="#333333", linewidth=0.6)
    ax_left.set_xticks(xs)
    ax_left.set_xticklabels(tick_labels, fontsize=9, rotation=35, ha="right")
    ax_left.set_ylim(0, 1.08)
    ax_left.set_ylabel("Andel sammendrag med abstraksjon")
    ax_left.set_title("Abstraksjonsrate (gate)")
    for x, v in zip(xs, abstraction_rate.values):
        if pd.notna(v):
            ax_left.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax_left.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax_left.set_axisbelow(True)

    synth_values = synth_mean.values.astype(float)
    plot_values = np.where(np.isnan(synth_values), 0.0, synth_values)
    ax_right.bar(xs, plot_values, color=bar_colors, edgecolor="#333333", linewidth=0.6)
    ax_right.set_xticks(xs)
    ax_right.set_xticklabels(tick_labels, fontsize=9, rotation=35, ha="right")
    ax_right.set_ylim(0, 5.4)
    ax_right.set_yticks([1, 2, 3, 4, 5])
    ax_right.set_ylabel("Gjennomsnittlig synthesized score")
    ax_right.set_title("Synthesized (bare hvor abstraksjon = 1)")
    for x, v, n in zip(xs, synth_values, synth_n.values):
        if np.isnan(v):
            ax_right.text(x, 0.1, f"n/a\n(n={n})", ha="center", fontsize=8, color="#666666")
        else:
            ax_right.text(x, v + 0.12, f"{v:.2f}\n(n={n})", ha="center", fontsize=8)
    ax_right.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax_right.set_axisbelow(True)

    fig.suptitle("Abstraksjon og synthesized per strategi", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "05_abstraction_synthesis")
    plt.close(fig)


def plot_voice_analysis(
    per_summary: pd.DataFrame,
    labels: Dict[str, str],
    out_dir: Path,
) -> None:
    strat_order = sorted(per_summary["strategy_id"].unique())
    counts = (
        per_summary.groupby(["strategy_id", "voice_case"]).size().unstack(fill_value=0)
    )
    for case in VOICE_CASES:
        if case not in counts.columns:
            counts[case] = 0
    counts = counts[VOICE_CASE_ORDER].reindex(strat_order).fillna(0)
    totals = counts.sum(axis=1).replace(0, np.nan)
    proportions = counts.div(totals, axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    xs = np.arange(len(strat_order))
    bottom = np.zeros(len(strat_order))
    for case in VOICE_CASE_ORDER:
        vals = proportions[case].values
        ax.bar(
            xs,
            vals,
            bottom=bottom,
            label=VOICE_CASE_LABELS[case],
            color=VOICE_CASE_COLORS[case],
            edgecolor="white",
            linewidth=0.6,
        )
        bottom += vals

    ax.set_xticks(xs)
    ax.set_xticklabels([wrap_label(labels[s]) for s in strat_order], fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Andel sammendrag")
    ax.set_title("Voice/stigma-analyse per strategi")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, title="Voice-kategori")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, out_dir / "06_voice_analysis")
    plt.close(fig)


def write_summary_table(
    per_summary: pd.DataFrame,
    labels: Dict[str, str],
    out_dir: Path,
) -> None:
    strat_order = sorted(per_summary["strategy_id"].unique())
    rows = []
    for sid in strat_order:
        sub = per_summary[per_summary["strategy_id"] == sid]
        synth_sub = sub[sub[GATE_SCORE_KEY] == 1][SYNTHESIS_SCORE_KEY].dropna()
        rows.append({
            "Strategi": labels[sid],
            "n": len(sub),
            "Mean": sub["primary_score"].mean(),
            "SD": sub["primary_score"].std(ddof=1) if len(sub) > 1 else float("nan"),
            "Min": sub["primary_score"].min(),
            "Median": sub["primary_score"].median(),
            "Max": sub["primary_score"].max(),
            "Abstraksjonsrate": sub[GATE_SCORE_KEY].mean(),
            "Synthesized (mean, n=abstracted)": (
                f"{synth_sub.mean():.2f} (n={len(synth_sub)})"
                if len(synth_sub)
                else f"n/a (n=0)"
            ),
        })

    header = list(rows[0].keys())
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        formatted: List[str] = []
        for key in header:
            val = row[key]
            if isinstance(val, float):
                formatted.append("n/a" if np.isnan(val) else f"{val:.3f}")
            else:
                formatted.append(str(val))
        lines.append("| " + " | ".join(formatted) + " |")

    (out_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_per_summary_csv(run_dir: Path, regenerate: bool) -> Path:
    per_summary_path = run_dir / "per_summary_results.csv"
    if regenerate or not per_summary_path.exists():
        script = Path(__file__).resolve().parent / "report_results.py"
        print(f"Bygger CSV-tabeller via {script}...")
        subprocess.run(
            [sys.executable, str(script), "--run-dir", str(run_dir)],
            check=True,
        )
    return per_summary_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generer figurer for en judgement-kjøring.")
    ap.add_argument("--run-dir", required=True, help="Path to runs/<run_id>")
    ap.add_argument(
        "--strategies",
        default="configs/strategies.json",
        help="Path til strategies-config (for pene navn på aksene)",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Mappen figurene skrives til (default: <run-dir>/plots)",
    )
    ap.add_argument(
        "--regenerate-csv",
        action="store_true",
        help="Kjør report_results.py på nytt selv om CSV-ene finnes",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing {run_dir}")

    per_summary_path = ensure_per_summary_csv(run_dir, args.regenerate_csv)
    per_summary = pd.read_csv(per_summary_path)
    if per_summary.empty:
        raise RuntimeError(f"{per_summary_path} er tom — fant ingen judgement-rader å plotte.")

    names = load_strategy_names(Path(args.strategies))
    strategy_ids = sorted(per_summary["strategy_id"].unique())
    labels = {sid: strategy_label(sid, names) for sid in strategy_ids}
    colors = build_strategy_palette(strategy_ids)

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_global_style()

    plot_strategy_ranking(per_summary, labels, colors, out_dir)
    plot_dimension_heatmap(per_summary, labels, out_dir)
    plot_score_distribution(per_summary, labels, colors, out_dir)
    plot_conversation_heatmap(per_summary, labels, out_dir)
    plot_abstraction_synthesis(per_summary, labels, colors, out_dir)
    plot_voice_analysis(per_summary, labels, out_dir)
    write_summary_table(per_summary, labels, out_dir)

    png_count = len(list(out_dir.glob("*.png")))
    pdf_count = len(list(out_dir.glob("*.pdf")))
    print(f"Skrev {png_count} PNG + {pdf_count} PDF til {out_dir}")
    print(f"Samlet tabell: {out_dir / 'summary_table.md'}")


if __name__ == "__main__":
    main()
