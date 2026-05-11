from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from src.judge_schema import (
        GATE_SCORE_KEY,
        QUALITY_SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
        VOICE_CASES,
        VOICE_SCORE_KEYS,
    )
except Exception:
    from judge_schema import (  # type: ignore
        GATE_SCORE_KEY,
        QUALITY_SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
        VOICE_CASES,
        VOICE_SCORE_KEYS,
    )


EXPECTED_STRATEGY_IDS = [
    "01_zero_shot_minimal_baseline",
    "02_zero_shot_structured_instruction",
    "03_one_shot",
    "04_two_shot",
    "05_two_shot_decomposition",
    "06_two_shot_self_criticism",
    "07_two_shot_decomposition_self_criticism",
    "08_two_shot_decomposition_self_criticism_ensemble",
]

BASELINE_STRATEGY_ID = "01_zero_shot_minimal_baseline"
DIMENSION_COLUMNS = [
    *QUALITY_SCORE_KEYS,
    "synthesized_when_applicable",
]
VOICE_SUMMARY_COLUMNS = [
    "strategy_id",
    "n",
    "voice_note_rate",
    "voice_summ_rate",
    "introduced_stigmatizing_language_rate",
    "propagated_stigmatizing_language_rate",
    "neutralized_stigmatizing_language_rate",
    "no_stigmatizing_language_detected_rate",
]

CONDITIONAL_SUMMARY_COLUMNS = [
    "condition",
    "strategy_id",
    "n_conversations",
    "mean_primary_score",
    "median_primary_score",
    "std_primary_score",
    "min_primary_score",
    "max_primary_score",
    "mean_delta_vs_baseline",
    "win_rate_vs_baseline",
    "tie_rate_vs_baseline",
    "loss_rate_vs_baseline",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object on line {line_no} in {path}")
            rows.append(record)
    return rows


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def voice_case(voice_note: int, voice_summ: int) -> str:
    if voice_note == 0 and voice_summ == 1:
        return "introduced_stigmatizing_language"
    if voice_note == 1 and voice_summ == 1:
        return "propagated_stigmatizing_language"
    if voice_note == 1 and voice_summ == 0:
        return "neutralized_stigmatizing_language"
    return "no_stigmatizing_language_detected"


def require_score(scores: Dict[str, Any], key: str) -> Any:
    if key not in scores:
        raise ValueError(f"Stored judge record missing scores.{key}")
    return scores[key]


def coerce_numeric(value: Any, key: str, record_label: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Missing numeric value for {key} in {record_label}")
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid numeric value for {key} in {record_label}: {value!r}") from e


def build_analysis_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    required_record_keys = ["conversation_id", "strategy_id", "split", "scores", "primary_score"]

    for idx, record in enumerate(records, start=1):
        for key in required_record_keys:
            if key not in record:
                raise ValueError(f"Judgement record {idx} missing required key: {key}")

        conversation_id = str(record["conversation_id"]).strip()
        strategy_id = str(record["strategy_id"]).strip()
        split = str(record["split"]).strip()
        record_label = f"{conversation_id} / {strategy_id}"
        scores = record["scores"]
        if not isinstance(scores, dict):
            raise ValueError(f"Judgement record {record_label} must contain a scores object")

        row: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "strategy_id": strategy_id,
            "split": split,
            "primary_score": coerce_numeric(record["primary_score"], "primary_score", record_label),
        }

        for key in QUALITY_SCORE_KEYS:
            row[key] = coerce_numeric(require_score(scores, key), f"scores.{key}", record_label)

        abstraction = int(coerce_numeric(require_score(scores, GATE_SCORE_KEY), f"scores.{GATE_SCORE_KEY}", record_label))
        if abstraction not in {0, 1}:
            raise ValueError(f"scores.{GATE_SCORE_KEY} must be 0 or 1 in {record_label}")
        row[GATE_SCORE_KEY] = abstraction

        synthesized = require_score(scores, SYNTHESIS_SCORE_KEY)
        if abstraction == 1:
            row["synthesized_when_applicable"] = coerce_numeric(
                synthesized,
                f"scores.{SYNTHESIS_SCORE_KEY}",
                record_label,
            )
        else:
            row["synthesized_when_applicable"] = np.nan

        for key in VOICE_SCORE_KEYS:
            value = int(coerce_numeric(require_score(scores, key), f"scores.{key}", record_label))
            if value not in {0, 1}:
                raise ValueError(f"scores.{key} must be 0 or 1 in {record_label}")
            row[key] = value

        row["voice_case"] = voice_case(int(row["voice_note"]), int(row["voice_summ"]))
        rows.append(row)

    return pd.DataFrame(rows)


def validate_design(df: pd.DataFrame, warnings: List[str]) -> None:
    if df.empty:
        raise ValueError("No judgement records found.")

    duplicate_mask = df.duplicated(["conversation_id", "strategy_id"], keep=False)
    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, ["conversation_id", "strategy_id"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Expected at most one row per conversation_id + strategy_id, "
            f"but found duplicates. Examples: {examples}"
        )

    expected = set(EXPECTED_STRATEGY_IDS)
    missing_baseline_conversations: List[str] = []
    invalid_conversations: List[str] = []
    for conversation_id, group in df.groupby("conversation_id", sort=True):
        present = set(group["strategy_id"])
        if BASELINE_STRATEGY_ID not in present:
            missing_baseline_conversations.append(conversation_id)
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing or extra:
            invalid_conversations.append(
                f"{conversation_id}: missing={missing or 'none'}, extra={extra or 'none'}"
            )

        splits = sorted(group["split"].dropna().unique())
        if len(splits) != 1:
            invalid_conversations.append(f"{conversation_id}: expected one split, found {splits}")

    if missing_baseline_conversations:
        message = (
            "Baseline strategy missing for conversations: "
            + ", ".join(missing_baseline_conversations[:10])
            + (" ..." if len(missing_baseline_conversations) > 10 else "")
        )
        warnings.append(message)
        warn(message)

    if invalid_conversations:
        raise ValueError(
            "Every conversation must have exactly the 8 expected strategies while preserving split. "
            "Problems:\n" + "\n".join(invalid_conversations[:30])
        )


def bootstrap_mean_ci(
    values: Iterable[float],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    clean = np.asarray(list(values), dtype=float)
    clean = clean[~np.isnan(clean)]
    if len(clean) == 0:
        return (np.nan, np.nan)
    if len(clean) == 1:
        return (float(clean[0]), float(clean[0]))

    sample_idx = rng.integers(0, len(clean), size=(n_bootstrap, len(clean)))
    means = clean[sample_idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return (float(low), float(high))


def strategy_summary(df: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy_id in EXPECTED_STRATEGY_IDS:
        group = df[df["strategy_id"] == strategy_id]
        ci_low, ci_high = bootstrap_mean_ci(group["primary_score"], rng, n_bootstrap)
        rows.append({
            "strategy_id": strategy_id,
            "n": int(len(group)),
            "mean_primary_score": group["primary_score"].mean(),
            "median_primary_score": group["primary_score"].median(),
            "std_primary_score": group["primary_score"].std(ddof=1),
            "min_primary_score": group["primary_score"].min(),
            "max_primary_score": group["primary_score"].max(),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        })
    return pd.DataFrame(rows)


def dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy_id in EXPECTED_STRATEGY_IDS:
        group = df[df["strategy_id"] == strategy_id]
        row: Dict[str, Any] = {"strategy_id": strategy_id}
        for key in QUALITY_SCORE_KEYS:
            row[key] = group[key].mean()
        row["synthesized_when_applicable"] = group["synthesized_when_applicable"].mean()
        row["abstraction_rate"] = group[GATE_SCORE_KEY].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def split_strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (split, strategy_id), group in df.groupby(["split", "strategy_id"], sort=True):
        row: Dict[str, Any] = {
            "split": split,
            "strategy_id": strategy_id,
            "n": int(len(group)),
            "mean_primary_score": group["primary_score"].mean(),
            "median_primary_score": group["primary_score"].median(),
            "std_primary_score": group["primary_score"].std(ddof=1),
        }
        for key in QUALITY_SCORE_KEYS:
            row[key] = group[key].mean()
        row["synthesized_when_applicable"] = group["synthesized_when_applicable"].mean()
        row["abstraction_rate"] = group[GATE_SCORE_KEY].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_scores(df: pd.DataFrame, strategy_a: str, strategy_b: str) -> pd.DataFrame:
    a = df[df["strategy_id"] == strategy_a][["conversation_id", "primary_score"]].rename(
        columns={"primary_score": "score_a"}
    )
    b = df[df["strategy_id"] == strategy_b][["conversation_id", "primary_score"]].rename(
        columns={"primary_score": "score_b"}
    )
    return a.merge(b, on="conversation_id", how="inner")


def baseline_delta(df: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy_id in EXPECTED_STRATEGY_IDS:
        if strategy_id == BASELINE_STRATEGY_ID:
            continue
        pairs = paired_scores(df, strategy_id, BASELINE_STRATEGY_ID)
        deltas = pairs["score_a"] - pairs["score_b"]
        ci_low, ci_high = bootstrap_mean_ci(deltas, rng, n_bootstrap)
        rows.append({
            "strategy_id": strategy_id,
            "n_pairs": int(len(pairs)),
            "mean_delta_vs_baseline": deltas.mean(),
            "median_delta_vs_baseline": deltas.median(),
            "std_delta_vs_baseline": deltas.std(ddof=1),
            "ci95_low_delta": ci_low,
            "ci95_high_delta": ci_high,
            "win_rate_vs_baseline": (deltas > 0).mean(),
            "tie_rate_vs_baseline": (deltas == 0).mean(),
            "loss_rate_vs_baseline": (deltas < 0).mean(),
        })
    return pd.DataFrame(rows)


def pairwise_strategy_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy_a, strategy_b in itertools.combinations(EXPECTED_STRATEGY_IDS, 2):
        pairs = paired_scores(df, strategy_a, strategy_b)
        deltas = pairs["score_a"] - pairs["score_b"]
        rows.append({
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "n_pairs": int(len(pairs)),
            "mean_delta_a_minus_b": deltas.mean(),
            "median_delta_a_minus_b": deltas.median(),
            "win_rate_a": (deltas > 0).mean(),
            "tie_rate": (deltas == 0).mean(),
            "win_rate_b": (deltas < 0).mean(),
        })
    return pd.DataFrame(rows)


def case_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for conversation_id, group in df.groupby("conversation_id", sort=True):
        scores = group["primary_score"]
        max_score = scores.max()
        min_score = scores.min()
        best = sorted(group.loc[group["primary_score"] == max_score, "strategy_id"])
        worst = sorted(group.loc[group["primary_score"] == min_score, "strategy_id"])
        rows.append({
            "conversation_id": conversation_id,
            "split": group["split"].iloc[0],
            "mean_primary_score_across_strategies": scores.mean(),
            "std_primary_score_across_strategies": scores.std(ddof=1),
            "min_primary_score": min_score,
            "max_primary_score": max_score,
            "score_range": max_score - min_score,
            "best_strategy_ids": ";".join(best),
            "worst_strategy_ids": ";".join(worst),
        })
    return pd.DataFrame(rows)


def voice_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for strategy_id in EXPECTED_STRATEGY_IDS:
        group = df[df["strategy_id"] == strategy_id]
        row: Dict[str, Any] = {
            "strategy_id": strategy_id,
            "n": int(len(group)),
            "voice_note_rate": group["voice_note"].mean(),
            "voice_summ_rate": group["voice_summ"].mean(),
        }
        for case in VOICE_CASES:
            row[f"{case}_rate"] = (group["voice_case"] == case).mean()
        rows.append(row)
    return pd.DataFrame(rows, columns=VOICE_SUMMARY_COLUMNS)


def build_condition_sets(case_stats: pd.DataFrame) -> tuple[Dict[str, Set[str]], Dict[str, Dict[str, Any]]]:
    mean_q25 = float(case_stats["mean_primary_score_across_strategies"].quantile(0.25))
    mean_q75 = float(case_stats["mean_primary_score_across_strategies"].quantile(0.75))
    range_q25 = float(case_stats["score_range"].quantile(0.25))
    range_q75 = float(case_stats["score_range"].quantile(0.75))

    conditions: Dict[str, Set[str]] = {
        "difficult_cases": set(
            case_stats.loc[
                case_stats["mean_primary_score_across_strategies"] <= mean_q25,
                "conversation_id",
            ]
        ),
        "easy_cases": set(
            case_stats.loc[
                case_stats["mean_primary_score_across_strategies"] >= mean_q75,
                "conversation_id",
            ]
        ),
        "high_strategy_sensitivity": set(
            case_stats.loc[case_stats["score_range"] >= range_q75, "conversation_id"]
        ),
        "low_strategy_sensitivity": set(
            case_stats.loc[case_stats["score_range"] <= range_q25, "conversation_id"]
        ),
        "test1": set(case_stats.loc[case_stats["split"] == "test1", "conversation_id"]),
        "test2": set(case_stats.loc[case_stats["split"] == "test2", "conversation_id"]),
        "test3": set(case_stats.loc[case_stats["split"] == "test3", "conversation_id"]),
    }

    definitions: Dict[str, Dict[str, Any]] = {
        "difficult_cases": {
            "description": "Conversations in the bottom 25% by mean_primary_score_across_strategies.",
            "metric": "mean_primary_score_across_strategies",
            "percentile": 25,
            "threshold": mean_q25,
            "rule": "<= threshold",
        },
        "easy_cases": {
            "description": "Conversations in the top 25% by mean_primary_score_across_strategies.",
            "metric": "mean_primary_score_across_strategies",
            "percentile": 75,
            "threshold": mean_q75,
            "rule": ">= threshold",
        },
        "high_strategy_sensitivity": {
            "description": "Conversations in the top 25% by score_range.",
            "metric": "score_range",
            "percentile": 75,
            "threshold": range_q75,
            "rule": ">= threshold",
        },
        "low_strategy_sensitivity": {
            "description": "Conversations in the bottom 25% by score_range.",
            "metric": "score_range",
            "percentile": 25,
            "threshold": range_q25,
            "rule": "<= threshold",
        },
        "test1": {
            "description": "Conversations where split = test1.",
            "metric": "split",
            "rule": "split == test1",
        },
        "test2": {
            "description": "Conversations where split = test2.",
            "metric": "split",
            "rule": "split == test2",
        },
        "test3": {
            "description": "Conversations where split = test3.",
            "metric": "split",
            "rule": "split == test3",
        },
    }

    for condition, conversation_ids in conditions.items():
        definitions[condition]["n_conversations"] = len(conversation_ids)

    return conditions, definitions


def conditional_strategy_summary(
    df: pd.DataFrame,
    case_stats: pd.DataFrame,
    warnings: List[str],
) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    condition_sets, condition_definitions = build_condition_sets(case_stats)
    rows: List[Dict[str, Any]] = []

    baseline_conversations = set(
        df.loc[df["strategy_id"] == BASELINE_STRATEGY_ID, "conversation_id"]
    )

    for condition, conversation_ids in condition_sets.items():
        missing_baseline = sorted(conversation_ids - baseline_conversations)
        if missing_baseline:
            message = (
                f"Baseline strategy missing for {len(missing_baseline)} conversations "
                f"in condition {condition}: "
                + ", ".join(missing_baseline[:10])
                + (" ..." if len(missing_baseline) > 10 else "")
            )
            warnings.append(message)
            warn(message)

        condition_df = df[df["conversation_id"].isin(conversation_ids)].copy()
        for strategy_id in EXPECTED_STRATEGY_IDS:
            group = condition_df[condition_df["strategy_id"] == strategy_id]
            pairs = paired_scores(condition_df, strategy_id, BASELINE_STRATEGY_ID)
            deltas = pairs["score_a"] - pairs["score_b"]
            rows.append({
                "condition": condition,
                "strategy_id": strategy_id,
                "n_conversations": int(group["conversation_id"].nunique()),
                "mean_primary_score": group["primary_score"].mean(),
                "median_primary_score": group["primary_score"].median(),
                "std_primary_score": group["primary_score"].std(ddof=1),
                "min_primary_score": group["primary_score"].min(),
                "max_primary_score": group["primary_score"].max(),
                "mean_delta_vs_baseline": deltas.mean(),
                "win_rate_vs_baseline": (deltas > 0).mean(),
                "tie_rate_vs_baseline": (deltas == 0).mean(),
                "loss_rate_vs_baseline": (deltas < 0).mean(),
            })

    return pd.DataFrame(rows, columns=CONDITIONAL_SUMMARY_COLUMNS), condition_definitions


def write_csvs(tables: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(out_dir / filename, index=False, encoding="utf-8")


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def short_strategy_label(strategy_id: str) -> str:
    return strategy_id.split("_", 1)[0]


def plot_mean_primary(strategy_stats: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = strategy_stats.copy()
    plot_df["label"] = plot_df["strategy_id"].map(short_strategy_label)
    yerr = np.vstack([
        plot_df["mean_primary_score"] - plot_df["ci95_low"],
        plot_df["ci95_high"] - plot_df["mean_primary_score"],
    ])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(
        data=plot_df,
        x="label",
        y="mean_primary_score",
        color="#4c78a8",
        edgecolor="#333333",
        linewidth=0.6,
        errorbar=None,
        ax=ax,
    )
    ax.errorbar(
        plot_df["label"],
        plot_df["mean_primary_score"],
        yerr=yerr,
        fmt="none",
        ecolor="#222222",
        elinewidth=1,
        capsize=4,
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Mean primary score")
    ax.set_ylim(1, 5.1)
    ax.set_title("Mean primary score by strategy")
    fig.tight_layout()
    fig.savefig(figures_dir / "mean_primary_score_by_strategy.png", dpi=200)
    plt.close(fig)


def plot_dimension_heatmap(dim_stats: pd.DataFrame, figures_dir: Path) -> None:
    matrix = dim_stats.set_index("strategy_id")[DIMENSION_COLUMNS]
    matrix.index = [short_strategy_label(s) for s in matrix.index]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=1,
        vmax=5,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean score"},
        annot_kws={"fontsize": 8},
        ax=ax,
    )
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Strategy")
    ax.set_title("Dimension means by strategy")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / "dimension_heatmap_by_strategy.png", dpi=200)
    plt.close(fig)


def plot_split_strategy(df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["strategy_label"] = plot_df["strategy_id"].map(short_strategy_label)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    sns.pointplot(
        data=plot_df,
        x="strategy_label",
        y="primary_score",
        hue="split",
        order=[short_strategy_label(s) for s in EXPECTED_STRATEGY_IDS],
        errorbar=None,
        dodge=0.25,
        markers="o",
        linestyles="-",
        ax=ax,
    )
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Primary score")
    ax.set_ylim(1, 5.1)
    ax.set_title("Primary score by strategy and split")
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(figures_dir / "split_strategy_primary_score.png", dpi=200)
    plt.close(fig)


def plot_baseline_delta(delta_stats: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = delta_stats.copy()
    plot_df["label"] = plot_df["strategy_id"].map(short_strategy_label)
    yerr = np.vstack([
        plot_df["mean_delta_vs_baseline"] - plot_df["ci95_low_delta"],
        plot_df["ci95_high_delta"] - plot_df["mean_delta_vs_baseline"],
    ])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(
        data=plot_df,
        x="label",
        y="mean_delta_vs_baseline",
        color="#59a14f",
        edgecolor="#333333",
        linewidth=0.6,
        errorbar=None,
        ax=ax,
    )
    ax.errorbar(
        plot_df["label"],
        plot_df["mean_delta_vs_baseline"],
        yerr=yerr,
        fmt="none",
        ecolor="#222222",
        elinewidth=1,
        capsize=4,
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Mean delta vs baseline")
    ax.set_title("Paired improvement vs baseline")
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_delta_by_strategy.png", dpi=200)
    plt.close(fig)


def plot_pairwise_winrate_matrix(df: pd.DataFrame, figures_dir: Path) -> None:
    matrix = pd.DataFrame(
        np.nan,
        index=EXPECTED_STRATEGY_IDS,
        columns=EXPECTED_STRATEGY_IDS,
        dtype=float,
    )
    for strategy_a in EXPECTED_STRATEGY_IDS:
        for strategy_b in EXPECTED_STRATEGY_IDS:
            if strategy_a == strategy_b:
                continue
            pairs = paired_scores(df, strategy_a, strategy_b)
            deltas = pairs["score_a"] - pairs["score_b"]
            matrix.loc[strategy_a, strategy_b] = (deltas > 0).mean()
    matrix.index = [short_strategy_label(s) for s in matrix.index]
    matrix.columns = [short_strategy_label(s) for s in matrix.columns]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Win rate for row strategy"},
        annot_kws={"fontsize": 8},
        ax=ax,
    )
    ax.set_xlabel("Compared against")
    ax.set_ylabel("Strategy")
    ax.set_title("Pairwise win-rate matrix")
    fig.tight_layout()
    fig.savefig(figures_dir / "pairwise_winrate_matrix.png", dpi=200)
    plt.close(fig)


def plot_case_difficulty(case_stats: pd.DataFrame, figures_dir: Path) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    sns.histplot(
        data=case_stats,
        x="mean_primary_score_across_strategies",
        bins=12,
        color="#4c78a8",
        edgecolor="white",
        ax=ax_left,
    )
    ax_left.set_xlabel("Mean primary score across strategies")
    ax_left.set_ylabel("Conversation count")
    ax_left.set_title("Case mean score")

    sns.histplot(
        data=case_stats,
        x="score_range",
        bins=12,
        color="#f28e2b",
        edgecolor="white",
        ax=ax_right,
    )
    ax_right.set_xlabel("Score range across strategies")
    ax_right.set_ylabel("Conversation count")
    ax_right.set_title("Case score range")

    fig.tight_layout()
    fig.savefig(figures_dir / "case_difficulty_distribution.png", dpi=200)
    plt.close(fig)


def plot_conditional_strategy_summary(conditional_stats: pd.DataFrame, figures_dir: Path) -> None:
    condition_order = [
        "difficult_cases",
        "easy_cases",
        "high_strategy_sensitivity",
        "low_strategy_sensitivity",
        "test1",
        "test2",
        "test3",
    ]
    plot_df = conditional_stats.copy()
    plot_df["strategy_label"] = plot_df["strategy_id"].map(short_strategy_label)
    plot_df["condition"] = pd.Categorical(
        plot_df["condition"],
        categories=condition_order,
        ordered=True,
    )

    grid = sns.catplot(
        data=plot_df,
        kind="bar",
        x="strategy_label",
        y="mean_primary_score",
        col="condition",
        col_wrap=2,
        order=[short_strategy_label(s) for s in EXPECTED_STRATEGY_IDS],
        color="#4c78a8",
        edgecolor="#333333",
        linewidth=0.5,
        errorbar=None,
        height=3.0,
        aspect=1.35,
        sharey=True,
    )
    grid.set_axis_labels("Strategy", "Mean primary score")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flatten():
        ax.set_ylim(1, 5.1)
        ax.tick_params(axis="x", rotation=0)
    grid.figure.suptitle("Conditional mean primary score by strategy", y=1.02)
    grid.figure.tight_layout()
    grid.figure.savefig(figures_dir / "conditional_strategy_summary.png", dpi=200)
    plt.close(grid.figure)


def write_figures(
    df: pd.DataFrame,
    strategy_stats: pd.DataFrame,
    dim_stats: pd.DataFrame,
    delta_stats: pd.DataFrame,
    case_stats: pd.DataFrame,
    conditional_stats: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    plot_mean_primary(strategy_stats, figures_dir)
    plot_dimension_heatmap(dim_stats, figures_dir)
    plot_split_strategy(df, figures_dir)
    plot_baseline_delta(delta_stats, figures_dir)
    plot_pairwise_winrate_matrix(df, figures_dir)
    plot_case_difficulty(case_stats, figures_dir)
    plot_conditional_strategy_summary(conditional_stats, figures_dir)


def write_manifest(
    run_dir: Path,
    judgements_path: Path,
    out_dir: Path,
    figures_dir: Path,
    tables: Dict[str, pd.DataFrame],
    warnings: List[str],
    condition_definitions: Dict[str, Dict[str, Any]],
    bootstrap_iterations: int,
    seed: int,
) -> Path:
    figure_files = sorted(path.name for path in figures_dir.glob("*.png"))
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "judgements_path": str(judgements_path),
        "analysis_type": "offline_exploratory_analysis",
        "primary_score_source": "stored judge record primary_score",
        "baseline_strategy_id": BASELINE_STRATEGY_ID,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": seed,
        "outputs": {
            "csv": sorted(tables.keys()),
            "figures": figure_files,
        },
        "warnings": warnings,
        "condition_definitions": condition_definitions,
        "notes": [
            "No API calls, generation, judging, cost analysis, raw data edits, prompt edits, rubric edits, or score-logic edits are performed.",
            "Rows are keyed by conversation_id + strategy_id; source_id alone is not used as a key.",
            "Split is preserved as an analysis variable.",
        ],
    }
    manifest_path = out_dir / "exploratory_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline exploratory analysis for pooled judgement results.")
    ap.add_argument("--run-dir", required=True, help="Path to runs/<run_id>")
    ap.add_argument(
        "--judgements",
        default="",
        help="Path to judgements.jsonl (default: <run-dir>/judgements.jsonl)",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: <run-dir>/exploratory_analysis)",
    )
    ap.add_argument("--bootstrap-iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260429)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    judgements_path = Path(args.judgements) if args.judgements else (run_dir / "judgements.jsonl")
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "exploratory_analysis")
    figures_dir = out_dir / "figures"

    if not judgements_path.exists():
        raise FileNotFoundError(f"Missing judgement file: {judgements_path}")
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations should be at least 100")

    warnings: List[str] = []
    records = read_jsonl(judgements_path)
    df = build_analysis_frame(records)
    validate_design(df, warnings)

    rng = np.random.default_rng(args.seed)
    strategy_stats = strategy_summary(df, rng, args.bootstrap_iterations)
    dim_stats = dimension_summary(df)
    split_stats = split_strategy_summary(df)
    delta_stats = baseline_delta(df, rng, args.bootstrap_iterations)
    pairwise_stats = pairwise_strategy_comparisons(df)
    case_stats = case_difficulty(df)
    voice_stats = voice_summary(df)
    conditional_stats, condition_definitions = conditional_strategy_summary(df, case_stats, warnings)

    tables = {
        "strategy_summary.csv": strategy_stats,
        "dimension_summary.csv": dim_stats,
        "split_strategy_summary.csv": split_stats,
        "baseline_delta.csv": delta_stats,
        "pairwise_strategy_comparisons.csv": pairwise_stats,
        "case_difficulty.csv": case_stats,
        "voice_summary.csv": voice_stats,
        "conditional_strategy_summary.csv": conditional_stats,
    }
    write_csvs(tables, out_dir)
    write_figures(df, strategy_stats, dim_stats, delta_stats, case_stats, conditional_stats, figures_dir)
    manifest_path = write_manifest(
        run_dir=run_dir,
        judgements_path=judgements_path,
        out_dir=out_dir,
        figures_dir=figures_dir,
        tables=tables,
        warnings=warnings,
        condition_definitions=condition_definitions,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    print(f"Read {len(df)} judgement rows from {judgements_path}")
    print(f"Conversations: {df['conversation_id'].nunique()}")
    print(f"Strategies: {df['strategy_id'].nunique()}")
    print(f"Splits: {', '.join(sorted(df['split'].unique()))}")
    print(f"Wrote exploratory CSVs to {out_dir}")
    print(f"Wrote diagnostic PNG figures to {figures_dir}")
    print(f"Wrote exploratory manifest to {manifest_path}")


if __name__ == "__main__":
    main()
