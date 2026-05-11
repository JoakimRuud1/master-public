from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from src.data_loader import EXCLUDED_EVALUATION_CONVERSATION_IDS
    from src.judge_schema import (
        GATE_SCORE_KEY,
        QUALITY_SCORE_KEYS,
        SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
        VOICE_CASES,
        VOICE_SCORE_KEYS,
        calculate_primary_score,
        coerce_int_score,
        derive_voice_analysis,
    )
except Exception:
    from data_loader import EXCLUDED_EVALUATION_CONVERSATION_IDS  # type: ignore
    from judge_schema import (  # type: ignore
        GATE_SCORE_KEY,
        QUALITY_SCORE_KEYS,
        SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
        VOICE_CASES,
        VOICE_SCORE_KEYS,
        calculate_primary_score,
        coerce_int_score,
        derive_voice_analysis,
    )


PER_SUMMARY_COLUMNS = [
    "run_id",
    "conversation_id",
    "strategy_id",
    "split",
    "transcript_variant",
    "primary_score",
    *QUALITY_SCORE_KEYS,
    GATE_SCORE_KEY,
    SYNTHESIS_SCORE_KEY,
    *VOICE_SCORE_KEYS,
    "included_dimensions",
    "voice_case",
]

VOICE_COLUMNS = [
    "run_id",
    "conversation_id",
    "strategy_id",
    "split",
    "voice_note",
    "voice_summ",
    "voice_case",
]

REPORT_FILENAMES = {
    "per_summary": "per_summary_results.csv",
    "strategy_aggregates": "strategy_aggregates.csv",
    "split_strategy_aggregates": "split_strategy_aggregates.csv",
    "voice_analysis": "voice_analysis.csv",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_no} in {path}")
            rows.append(obj)
    return rows


def is_null_equivalent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "na", "n/a", "null", "none"}:
        return True
    return False


def normalize_scores(record: Dict[str, Any]) -> Dict[str, int | None]:
    scores = record.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Stored judge record must contain scores object.")

    fixed_scores: Dict[str, int | None] = {}
    for key in QUALITY_SCORE_KEYS:
        if key not in scores:
            raise ValueError(f"Stored judge record missing scores.{key}")
        fixed_scores[key] = coerce_int_score(scores[key], key, (1, 2, 3, 4, 5))

    if GATE_SCORE_KEY not in scores:
        raise ValueError(f"Stored judge record missing scores.{GATE_SCORE_KEY}")
    fixed_scores[GATE_SCORE_KEY] = coerce_int_score(scores[GATE_SCORE_KEY], GATE_SCORE_KEY, (0, 1))

    if SYNTHESIS_SCORE_KEY not in scores:
        raise ValueError(f"Stored judge record missing scores.{SYNTHESIS_SCORE_KEY}")
    if fixed_scores[GATE_SCORE_KEY] == 1:
        fixed_scores[SYNTHESIS_SCORE_KEY] = coerce_int_score(
            scores[SYNTHESIS_SCORE_KEY],
            SYNTHESIS_SCORE_KEY,
            (1, 2, 3, 4, 5),
        )
    else:
        if not is_null_equivalent(scores[SYNTHESIS_SCORE_KEY]):
            raise ValueError("Stored scores.synthesized must be null/NA when abstraction = 0")
        fixed_scores[SYNTHESIS_SCORE_KEY] = None

    for key in VOICE_SCORE_KEYS:
        if key not in scores:
            raise ValueError(f"Stored judge record missing scores.{key}")
        fixed_scores[key] = coerce_int_score(scores[key], key, (0, 1))

    return fixed_scores


def validate_stored_record(record: Dict[str, Any], scores: Dict[str, int | None]) -> tuple[float, List[str], str]:
    required = [
        "run_id",
        "conversation_id",
        "strategy_id",
        "split",
        "transcript_variant",
        "scores",
        "primary_score",
        "included_dimensions",
        "voice_analysis",
        "rationale",
    ]
    for key in required:
        if key not in record:
            raise ValueError(f"Stored judge record missing required key: {key}")

    primary_score, included_dimensions = calculate_primary_score(scores)
    stored_primary_score = float(record["primary_score"])
    if abs(stored_primary_score - primary_score) > 1e-6:
        raise ValueError(
            f"Stored primary_score mismatch for {record['conversation_id']} / {record['strategy_id']}: "
            f"{stored_primary_score} != {primary_score}"
        )

    stored_included = record["included_dimensions"]
    if not isinstance(stored_included, list) or stored_included != included_dimensions:
        raise ValueError(
            f"Stored included_dimensions mismatch for {record['conversation_id']} / {record['strategy_id']}"
        )

    voice_analysis = record["voice_analysis"]
    if not isinstance(voice_analysis, dict):
        raise ValueError("Stored judge record must contain voice_analysis object.")
    voice_case = voice_analysis.get("case")
    derived_voice_case = derive_voice_analysis(scores)["case"]
    if voice_case != derived_voice_case:
        raise ValueError(
            f"Stored voice_analysis.case mismatch for {record['conversation_id']} / {record['strategy_id']}: "
            f"{voice_case} != {derived_voice_case}"
        )

    return primary_score, included_dimensions, derived_voice_case


def build_per_summary_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        conversation_id = str(record.get("conversation_id", "")).strip()
        if conversation_id in EXCLUDED_EVALUATION_CONVERSATION_IDS:
            continue

        scores = normalize_scores(record)
        primary_score, included_dimensions, voice_case = validate_stored_record(record, scores)

        row: Dict[str, Any] = {
            "run_id": str(record["run_id"]).strip(),
            "conversation_id": conversation_id,
            "strategy_id": str(record["strategy_id"]).strip(),
            "split": str(record["split"]).strip(),
            "transcript_variant": str(record["transcript_variant"]).strip(),
            "primary_score": primary_score,
            "included_dimensions": ",".join(included_dimensions),
            "voice_case": voice_case,
        }
        for key in SCORE_KEYS:
            row[key] = scores[key]
        rows.append(row)
    return rows


def build_per_summary_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(build_per_summary_rows(records), columns=PER_SUMMARY_COLUMNS)


def aggregate_scores(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    aggregate_columns = [
        *group_cols,
        "n_summaries",
        "mean_primary_score",
        *[f"mean_{key}" for key in QUALITY_SCORE_KEYS],
        "mean_synthesized_when_applicable",
        "abstraction_rate",
        "voice_summ_rate",
        "voice_note_rate",
        "introduced_stigmatizing_language_rate",
        "propagated_stigmatizing_language_rate",
        "neutralized_stigmatizing_language_rate",
    ]
    if df.empty:
        return pd.DataFrame(columns=aggregate_columns)

    rows: List[Dict[str, Any]] = []
    group_key: str | List[str] = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, group in df.groupby(group_key, dropna=False):
        if len(group_cols) == 1:
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["n_summaries"] = int(len(group))
        row["mean_primary_score"] = group["primary_score"].mean()
        for key in QUALITY_SCORE_KEYS:
            row[f"mean_{key}"] = group[key].mean()
        row["mean_synthesized_when_applicable"] = group.loc[
            group[GATE_SCORE_KEY] == 1,
            SYNTHESIS_SCORE_KEY,
        ].mean()
        row["abstraction_rate"] = group[GATE_SCORE_KEY].mean()
        row["voice_summ_rate"] = group["voice_summ"].mean()
        row["voice_note_rate"] = group["voice_note"].mean()
        for case in VOICE_CASES:
            if case == "no_stigmatizing_language_detected":
                continue
            row[f"{case}_rate"] = (group["voice_case"] == case).mean()
        rows.append(row)

    return pd.DataFrame(rows, columns=aggregate_columns)


def build_report_tables(records: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    per_summary = build_per_summary_table(records)
    return {
        "per_summary": per_summary,
        "strategy_aggregates": aggregate_scores(per_summary, ["run_id", "strategy_id"]),
        "split_strategy_aggregates": aggregate_scores(per_summary, ["run_id", "split", "strategy_id"]),
        "voice_analysis": per_summary[VOICE_COLUMNS].copy(),
    }


def write_report_tables(tables: Dict[str, pd.DataFrame], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for key, filename in REPORT_FILENAMES.items():
        out_path = out_dir / filename
        tables[key].to_csv(out_path, index=False, encoding="utf-8")
        written[key] = out_path
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to runs/<run_id> directory")
    ap.add_argument("--judgements", default="", help="Path to judgements.jsonl (default: runs/<run_id>/judgements.jsonl)")
    ap.add_argument("--out-dir", default="", help="Output directory (default: runs/<run_id>)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    judgements_path = Path(args.judgements) if args.judgements else (run_dir / "judgements.jsonl")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir

    if not judgements_path.exists():
        raise FileNotFoundError(f"Missing {judgements_path}")

    records = read_jsonl(judgements_path)
    tables = build_report_tables(records)
    written = write_report_tables(tables, out_dir)

    print(f"Rows reported: {len(tables['per_summary'])}")
    for key, path in written.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
