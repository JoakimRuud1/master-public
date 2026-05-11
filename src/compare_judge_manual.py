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
        VOICE_SCORE_KEYS,
        calculate_primary_score,
        coerce_int_score,
    )
except Exception:
    from data_loader import EXCLUDED_EVALUATION_CONVERSATION_IDS  # type: ignore
    from judge_schema import (  # type: ignore
        GATE_SCORE_KEY,
        QUALITY_SCORE_KEYS,
        SCORE_KEYS,
        SYNTHESIS_SCORE_KEY,
        VOICE_SCORE_KEYS,
        calculate_primary_score,
        coerce_int_score,
    )


KEY_COLS = ["run_id", "conversation_id", "strategy_id"]
CONTEXT_COLS = ["split", "transcript_variant"]
MANUAL_REQUIRED_COLS = [*KEY_COLS, *SCORE_KEYS]
MAIN_COMPARISON_KEYS = [*QUALITY_SCORE_KEYS, GATE_SCORE_KEY, SYNTHESIS_SCORE_KEY]


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


def normalize_key_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [col for col in KEY_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required join column(s): {missing}")
    df = df.copy()
    for col in KEY_COLS:
        df[col] = df[col].astype(str).str.strip()
    return df


def normalize_scores(values: Dict[str, Any]) -> Dict[str, int | None]:
    scores: Dict[str, int | None] = {}
    for key in QUALITY_SCORE_KEYS:
        scores[key] = coerce_int_score(values.get(key), key, (1, 2, 3, 4, 5))

    scores[GATE_SCORE_KEY] = coerce_int_score(values.get(GATE_SCORE_KEY), GATE_SCORE_KEY, (0, 1))

    if scores[GATE_SCORE_KEY] == 1:
        scores[SYNTHESIS_SCORE_KEY] = coerce_int_score(
            values.get(SYNTHESIS_SCORE_KEY),
            SYNTHESIS_SCORE_KEY,
            (1, 2, 3, 4, 5),
        )
    else:
        scores[SYNTHESIS_SCORE_KEY] = None

    for key in VOICE_SCORE_KEYS:
        scores[key] = coerce_int_score(values.get(key), key, (0, 1))

    return scores


def build_manual_table(manual_df: pd.DataFrame, manual_path: Path) -> pd.DataFrame:
    missing = [col for col in MANUAL_REQUIRED_COLS if col not in manual_df.columns]
    if missing:
        raise ValueError(
            f"{manual_path} missing required named rubric column(s): {missing}. "
            "Manual scoring must use explicit rubric columns, not score_1..score_5."
        )

    manual_df = normalize_key_columns(manual_df, manual_path)
    manual_df = manual_df[~manual_df["conversation_id"].isin(EXCLUDED_EVALUATION_CONVERSATION_IDS)].copy()

    rows: List[Dict[str, Any]] = []
    for _, source in manual_df.iterrows():
        scores = normalize_scores(source.to_dict())
        primary_score, included_dimensions = calculate_primary_score(scores)

        row: Dict[str, Any] = {col: source[col] for col in KEY_COLS}
        for col in CONTEXT_COLS:
            if col in manual_df.columns:
                row[f"{col}_manual_context"] = source[col]
        row["manual_primary_score"] = primary_score
        row["manual_included_dimensions"] = ",".join(included_dimensions)
        for key in SCORE_KEYS:
            row[f"{key}_manual"] = scores[key]
        rows.append(row)

    columns = [
        *KEY_COLS,
        *[f"{col}_manual_context" for col in CONTEXT_COLS if col in manual_df.columns],
        "manual_primary_score",
        "manual_included_dimensions",
        *[f"{key}_manual" for key in SCORE_KEYS],
    ]
    return pd.DataFrame(rows, columns=columns)


def build_judge_table(judge_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in judge_rows:
        for key in [*KEY_COLS, *CONTEXT_COLS, "scores", "primary_score"]:
            if key not in record:
                raise ValueError(f"Stored judge record missing required key: {key}")

        conversation_id = str(record["conversation_id"]).strip()
        if conversation_id in EXCLUDED_EVALUATION_CONVERSATION_IDS:
            continue

        scores_obj = record["scores"]
        if not isinstance(scores_obj, dict):
            raise ValueError("Stored judge record scores must be an object.")

        scores = normalize_scores(scores_obj)
        primary_score, included_dimensions = calculate_primary_score(scores)
        stored_primary_score = float(record["primary_score"])
        if abs(stored_primary_score - primary_score) > 1e-6:
            raise ValueError(
                f"Stored judge primary_score mismatch for {conversation_id} / {record['strategy_id']}: "
                f"{stored_primary_score} != {primary_score}"
            )

        row: Dict[str, Any] = {
            "run_id": str(record["run_id"]).strip(),
            "conversation_id": conversation_id,
            "strategy_id": str(record["strategy_id"]).strip(),
            "split": str(record["split"]).strip(),
            "transcript_variant": str(record["transcript_variant"]).strip(),
            "judge_primary_score": primary_score,
            "judge_included_dimensions": ",".join(included_dimensions),
        }
        for key in SCORE_KEYS:
            row[f"{key}_judge"] = scores[key]
        rows.append(row)

    columns = [
        *KEY_COLS,
        *CONTEXT_COLS,
        "judge_primary_score",
        "judge_included_dimensions",
        *[f"{key}_judge" for key in SCORE_KEYS],
    ]
    return pd.DataFrame(rows, columns=columns)


def validate_optional_context(merged: pd.DataFrame) -> None:
    for col in CONTEXT_COLS:
        manual_col = f"{col}_manual_context"
        if manual_col not in merged.columns:
            continue
        manual_values = merged[manual_col]
        mismatch = (~manual_values.isna()) & (manual_values.astype(str).str.strip() != merged[col].astype(str).str.strip())
        if mismatch.any():
            bad = merged.loc[mismatch, KEY_COLS + [col, manual_col]].head(5).to_dict("records")
            raise ValueError(f"Manual {col} values do not match judge records: {bad}")


def add_comparison_columns(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged[
        [
            *KEY_COLS,
            "split",
            "transcript_variant",
            "judge_primary_score",
            "manual_primary_score",
        ]
    ].copy()

    out["primary_score_difference"] = out["judge_primary_score"] - out["manual_primary_score"]
    out["primary_score_abs_difference"] = out["primary_score_difference"].abs()

    for key in SCORE_KEYS:
        out[f"{key}_judge"] = merged[f"{key}_judge"]
        out[f"{key}_manual"] = merged[f"{key}_manual"]

    for key in [*QUALITY_SCORE_KEYS, GATE_SCORE_KEY]:
        out[f"{key}_difference"] = out[f"{key}_judge"] - out[f"{key}_manual"]
        out[f"{key}_exact_match"] = out[f"{key}_judge"] == out[f"{key}_manual"]

    synthesized_applicable = (out[f"{GATE_SCORE_KEY}_judge"] == 1) & (out[f"{GATE_SCORE_KEY}_manual"] == 1)
    out["synthesized_comparison_applicable"] = synthesized_applicable
    out["synthesized_difference"] = pd.NA
    out["synthesized_exact_match"] = pd.NA
    out.loc[synthesized_applicable, "synthesized_difference"] = (
        out.loc[synthesized_applicable, "synthesized_judge"]
        - out.loc[synthesized_applicable, "synthesized_manual"]
    )
    out.loc[synthesized_applicable, "synthesized_exact_match"] = (
        out.loc[synthesized_applicable, "synthesized_judge"]
        == out.loc[synthesized_applicable, "synthesized_manual"]
    )

    for key in VOICE_SCORE_KEYS:
        out[f"{key}_difference"] = out[f"{key}_judge"] - out[f"{key}_manual"]
        out[f"{key}_exact_match"] = out[f"{key}_judge"] == out[f"{key}_manual"]

    return out


def build_comparison_table(manual_df: pd.DataFrame, judge_rows: List[Dict[str, Any]], manual_path: Path) -> pd.DataFrame:
    manual = build_manual_table(manual_df, manual_path)
    judge = build_judge_table(judge_rows)

    merged = judge.merge(manual, on=KEY_COLS, how="inner")
    validate_optional_context(merged)
    return add_comparison_columns(merged)


def print_summary(comparison: pd.DataFrame, manual: pd.DataFrame, judge: pd.DataFrame, out_path: Path) -> None:
    key_cols = KEY_COLS
    manual_keys = set(tuple(x) for x in manual[key_cols].values.tolist())
    judge_keys = set(tuple(x) for x in judge[key_cols].values.tolist())
    missing_in_judge = sorted(list(manual_keys - judge_keys))
    missing_in_manual = sorted(list(judge_keys - manual_keys))

    print(f"Rows manual: {len(manual)}")
    print(f"Rows judge:  {len(judge)}")
    print(f"Rows merged: {len(comparison)}")
    print(f"Output: {out_path}")

    if missing_in_judge:
        print(f"Missing in judgements (from manual): {len(missing_in_judge)}")
    if missing_in_manual:
        print(f"Missing in manual (from judgements): {len(missing_in_manual)}")

    if len(comparison):
        exact_primary = (comparison["primary_score_difference"] == 0).mean()
        mae_primary = comparison["primary_score_abs_difference"].mean()
    else:
        exact_primary = float("nan")
        mae_primary = float("nan")
    print(f"\nPrimary score: exact={exact_primary:.2f}, MAE={mae_primary:.2f}")

    print("\nMain rubric dimensions (exact match rate, MAE):")
    for key in MAIN_COMPARISON_KEYS:
        exact_col = f"{key}_exact_match"
        diff_col = f"{key}_difference"
        if exact_col not in comparison.columns:
            continue
        valid_exact = comparison[exact_col].dropna()
        exact = valid_exact.astype(bool).mean() if len(valid_exact) else float("nan")
        mae = pd.to_numeric(comparison[diff_col], errors="coerce").abs().mean() if diff_col in comparison.columns else float("nan")
        print(f"- {key}: exact={exact:.2f}, MAE={mae:.2f}")

    print("\nVoice dimensions (exact match rate):")
    for key in VOICE_SCORE_KEYS:
        exact = comparison[f"{key}_exact_match"].mean() if len(comparison) else float("nan")
        print(f"- {key}: exact={exact:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to runs/<run_id>")
    ap.add_argument("--manual", required=True, help="Path to manual scoring CSV with named rubric columns")
    ap.add_argument("--out", default="", help="Output CSV path (default: runs/<run_id>/judge_vs_manual.csv)")
    ap.add_argument("--manual-sep", default=";", help="Manual CSV separator (default: ';')")
    ap.add_argument("--manual-encoding", default="cp1252", help="Manual CSV encoding (default: cp1252)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manual_path = Path(args.manual)
    judgements_path = run_dir / "judgements.jsonl"
    out_path = Path(args.out) if args.out else (run_dir / "judge_vs_manual.csv")

    if not judgements_path.exists():
        raise FileNotFoundError(f"Missing {judgements_path}")
    if not manual_path.exists():
        raise FileNotFoundError(f"Missing {manual_path}")

    manual_raw = pd.read_csv(manual_path, sep=args.manual_sep, encoding=args.manual_encoding)
    judge_rows = read_jsonl(judgements_path)
    manual = build_manual_table(manual_raw, manual_path)
    judge = build_judge_table(judge_rows)
    merged = judge.merge(manual, on=KEY_COLS, how="inner")
    validate_optional_context(merged)
    comparison = add_comparison_columns(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_path, index=False, encoding="utf-8")
    print_summary(comparison, manual, judge, out_path)


if __name__ == "__main__":
    main()
