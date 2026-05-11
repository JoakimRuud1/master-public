from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_POOLED_RUN_ID = "test_all_20260428"
DEFAULT_SOURCE_RUN_DIRS = [
    "runs/test1_20260426",
    "runs/test2_20260428",
    "runs/test3_20260428",
]
EXPECTED_SPLITS = {"test1", "test2", "test3"}
EXPECTED_ROWS = 528
KEY_FIELDS = ("conversation_id", "strategy_id")


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def key_for(row: Dict[str, Any]) -> Tuple[str, str]:
    missing = [field for field in KEY_FIELDS if field not in row or row[field] in (None, "")]
    if missing:
        raise ValueError(f"Missing key field(s) {', '.join(missing)} in row: {row}")
    return (str(row["conversation_id"]), str(row["strategy_id"]))


def duplicate_keys(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    counts = Counter(key_for(row) for row in rows)
    return sorted(key for key, count in counts.items() if count > 1)


def pool_rows(
    source_run_dirs: Sequence[Path],
    pooled_run_id: str,
    filename: str,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    pooled_rows: List[Dict[str, Any]] = []
    for source_dir in source_run_dirs:
        source_path = source_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing required input file: {source_path}")

        source_rows = read_jsonl(source_path)
        source_duplicates = duplicate_keys(source_rows)
        if source_duplicates:
            sample = ", ".join(f"{cid} / {sid}" for cid, sid in source_duplicates[:5])
            raise ValueError(f"Duplicate {filename} keys inside {source_dir}: {sample}")

        for row in source_rows:
            old_run_id = row.get("run_id")
            if old_run_id is None:
                warnings.append(f"{source_path} contains a row without run_id; source_run_id set to null.")
            pooled_row = dict(row)
            pooled_row["source_run_id"] = old_run_id
            pooled_row["run_id"] = pooled_run_id
            pooled_rows.append(pooled_row)

    pooled_duplicates = duplicate_keys(pooled_rows)
    if pooled_duplicates:
        sample = ", ".join(f"{cid} / {sid}" for cid, sid in pooled_duplicates[:10])
        raise ValueError(f"Duplicate {filename} keys after pooling: {sample}")

    return pooled_rows


def split_values(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({str(row.get("split", "")).strip() for row in rows if str(row.get("split", "")).strip()})


def validate_expected_output(
    summaries: Sequence[Dict[str, Any]],
    judgements: Sequence[Dict[str, Any]],
    warnings: List[str],
) -> None:
    summary_splits = set(split_values(summaries))
    judgement_splits = set(split_values(judgements))
    for label, actual in (("summaries.jsonl", summary_splits), ("judgements.jsonl", judgement_splits)):
        missing = sorted(EXPECTED_SPLITS - actual)
        if missing:
            raise ValueError(f"{label} is missing expected split(s): {', '.join(missing)}")

    if len(summaries) != EXPECTED_ROWS:
        warnings.append(f"summaries.jsonl has {len(summaries)} rows; expected {EXPECTED_ROWS}.")
    if len(judgements) != EXPECTED_ROWS:
        warnings.append(f"judgements.jsonl has {len(judgements)} rows; expected {EXPECTED_ROWS}.")

    summary_keys = {key_for(row) for row in summaries}
    judgement_keys = {key_for(row) for row in judgements}
    missing_judgements = sorted(summary_keys - judgement_keys)
    missing_summaries = sorted(judgement_keys - summary_keys)
    if missing_judgements:
        sample = ", ".join(f"{cid} / {sid}" for cid, sid in missing_judgements[:10])
        raise ValueError(f"Judgements missing for summary key(s): {sample}")
    if missing_summaries:
        sample = ", ".join(f"{cid} / {sid}" for cid, sid in missing_summaries[:10])
        raise ValueError(f"Summaries missing for judgement key(s): {sample}")


def build_manifest(
    pooled_run_id: str,
    source_run_dirs: Sequence[Path],
    summaries: Sequence[Dict[str, Any]],
    judgements: Sequence[Dict[str, Any]],
    warnings: Sequence[str],
    errors: Sequence[str],
) -> Dict[str, Any]:
    return {
        "pooled_run_id": pooled_run_id,
        "source_run_dirs": [str(path) for path in source_run_dirs],
        "number_of_summary_rows": len(summaries),
        "number_of_judgement_rows": len(judgements),
        "number_of_unique_conversations": len({str(row["conversation_id"]) for row in summaries}),
        "number_of_strategies": len({str(row["strategy_id"]) for row in summaries}),
        "splits_included": split_values(summaries),
        "warnings": list(warnings),
        "errors": list(errors),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def pool_test_runs(source_run_dirs: Sequence[Path], output_dir: Path, pooled_run_id: str) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []

    summaries = pool_rows(source_run_dirs, pooled_run_id, "summaries.jsonl", warnings)
    judgements = pool_rows(source_run_dirs, pooled_run_id, "judgements.jsonl", warnings)
    validate_expected_output(summaries, judgements, warnings)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "summaries.jsonl", summaries)
    write_jsonl(output_dir / "judgements.jsonl", judgements)

    manifest = build_manifest(pooled_run_id, source_run_dirs, summaries, judgements, warnings, errors)
    with (output_dir / "pooled_manifest.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Pool existing test run JSONL files without generation, judging, or score changes."
    )
    ap.add_argument("--pooled-run-id", default=DEFAULT_POOLED_RUN_ID)
    ap.add_argument("--output-dir", default=f"runs/{DEFAULT_POOLED_RUN_ID}")
    ap.add_argument("--source-run-dir", action="append", dest="source_run_dirs", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source_run_dirs = [Path(path) for path in (args.source_run_dirs or DEFAULT_SOURCE_RUN_DIRS)]
    output_dir = Path(args.output_dir)
    manifest = pool_test_runs(source_run_dirs, output_dir, args.pooled_run_id)

    print(f"Pooled run written to: {output_dir}")
    print(f"summary rows: {manifest['number_of_summary_rows']}")
    print(f"judgement rows: {manifest['number_of_judgement_rows']}")
    print(f"unique conversations: {manifest['number_of_unique_conversations']}")
    print(f"strategies: {manifest['number_of_strategies']}")
    print(f"splits: {', '.join(manifest['splits_included'])}")
    if manifest["warnings"]:
        print("warnings:")
        for warning in manifest["warnings"]:
            print(f"- {warning}")
    if manifest["errors"]:
        print("errors:")
        for error in manifest["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
