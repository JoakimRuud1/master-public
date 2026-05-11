# src/run_judge.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Make it work both when running:
# - python src/run_judge.py ...
# - python -m src.run_judge ...
try:
    from src.data_loader import (
        ACI_BENCH_DATA_DIR,
        ACI_BENCH_MAIN_ANALYSIS_SPLITS,
        ACI_BENCH_TRANSCRIPT_VARIANT,
        load_aci_bench_evaluation_dataset,
    )
    from src.endpoint_config import DEFAULT_ENDPOINT_CONFIG_PATH, get_endpoint_settings, load_endpoint_config
except Exception:
    from data_loader import (  # type: ignore
        ACI_BENCH_DATA_DIR,
        ACI_BENCH_MAIN_ANALYSIS_SPLITS,
        ACI_BENCH_TRANSCRIPT_VARIANT,
        load_aci_bench_evaluation_dataset,
    )
    from endpoint_config import DEFAULT_ENDPOINT_CONFIG_PATH, get_endpoint_settings, load_endpoint_config  # type: ignore


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return rows


def write_jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_existing_judgement_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()

    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        conversation_id = row.get("conversation_id")
        strategy_id = row.get("strategy_id")
        if isinstance(conversation_id, str) and isinstance(strategy_id, str):
            keys.add((conversation_id, strategy_id))
    return keys


def parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in value.split(",") if part.strip())
    if not splits:
        raise ValueError("At least one split must be provided.")
    return splits


def infer_splits_from_summaries(summaries: Sequence[Dict[str, Any]]) -> tuple[str, ...]:
    splits = sorted({row.get("split") for row in summaries if isinstance(row.get("split"), str)})
    if splits:
        return tuple(splits)
    return ACI_BENCH_MAIN_ANALYSIS_SPLITS


def build_transcript_lookup(conversations: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for r in conversations:
        cid = r.get("conversation_id")
        txt = r.get("transcript_text")
        if isinstance(cid, str) and isinstance(txt, str):
            m[cid] = r
    if not m:
        raise ValueError("No usable ACI-Bench rows found. Expected fields: conversation_id, transcript_text.")
    return m


def judge_summary_with_client(**kwargs: Any) -> Dict[str, Any]:
    try:
        from src.judge_client import judge_summary
    except Exception:
        from judge_client import judge_summary  # type: ignore
    return judge_summary(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to runs/<run_id> directory")
    ap.add_argument("--data-dir", default=str(ACI_BENCH_DATA_DIR), help="Path to ACI-Bench src_experiment_data_json directory")
    ap.add_argument("--splits", default="", help="Comma-separated ACI-Bench splits. Defaults to split metadata from summaries.jsonl, or test1.")
    ap.add_argument("--endpoint-config", default=str(DEFAULT_ENDPOINT_CONFIG_PATH), help="Path to endpoint config JSON")
    ap.add_argument("--api-endpoint", default="", help="Override judge API endpoint from endpoint config")
    ap.add_argument("--model", default="", help="Override judge model from endpoint config")
    ap.add_argument("--judge-temperature", type=float, default=None, help="Override judge temperature from endpoint config")
    ap.add_argument("--max-output-tokens", type=int, default=0, help="Override judge max output tokens from endpoint config")
    ap.add_argument("--reasoning-effort", default="", help="Override judge reasoning effort from endpoint config")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only judge first N summaries")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep seconds between calls")
    ap.add_argument("--resume", action="store_true", help="Append missing judgements only; keep existing judgements.jsonl and skip completed pairs.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    endpoint_config_path = Path(args.endpoint_config)
    endpoint_settings = get_endpoint_settings(load_endpoint_config(endpoint_config_path), "judge")
    api_endpoint = args.api_endpoint or str(endpoint_settings["api_endpoint"])
    model = args.model or str(endpoint_settings["model"])
    temperature = args.judge_temperature if args.judge_temperature is not None else endpoint_settings.get("temperature")
    max_output_tokens = args.max_output_tokens or int(endpoint_settings["max_output_tokens"])
    reasoning_effort = args.reasoning_effort or endpoint_settings.get("reasoning_effort")
    summaries_path = run_dir / "summaries.jsonl"
    out_path = run_dir / "judgements.jsonl"
    err_path = run_dir / "judgements_errors.jsonl"

    if not summaries_path.exists():
        raise FileNotFoundError(f"Missing {summaries_path}")

    completed_keys: set[tuple[str, str]] = set()
    if args.resume:
        completed_keys = read_existing_judgement_keys(out_path)
    else:
        # Fresh outputs each run (avoid mixing)
        if out_path.exists():
            out_path.unlink()
        if err_path.exists():
            err_path.unlink()

    summaries = read_jsonl(summaries_path)
    splits = parse_splits(args.splits) if args.splits else infer_splits_from_summaries(summaries)
    conversations = load_aci_bench_evaluation_dataset(
        splits=splits,
        data_dir=Path(args.data_dir),
        transcript_variant=ACI_BENCH_TRANSCRIPT_VARIANT,
    )
    transcript_lookup = build_transcript_lookup(conversations)

    n = 0
    skipped = 0
    if completed_keys:
        print(f"Resume: {len(completed_keys)}/{len(summaries)} (conversation_id, strategy_id) pairs already present in {out_path}.")

    for i, s in enumerate(summaries, start=1):
        if args.limit and n >= args.limit:
            break

        run_id = s.get("run_id", run_dir.name)
        conversation_id = s.get("conversation_id")
        strategy_id = s.get("strategy_id")
        summary_text = s.get("summary")

        if isinstance(conversation_id, str) and isinstance(strategy_id, str):
            if (conversation_id, strategy_id) in completed_keys:
                skipped += 1
                continue

        if not isinstance(conversation_id, str) or not isinstance(strategy_id, str) or not isinstance(summary_text, str):
            write_jsonl_append(err_path, {
                "index": i,
                "error": "Missing/invalid fields in summary row (need conversation_id, strategy_id, summary).",
                "row": s,
            })
            continue

        transcript_row = transcript_lookup.get(conversation_id)
        if not transcript_row:
            write_jsonl_append(err_path, {
                "index": i,
                "error": f"No transcript found for conversation_id={conversation_id}",
                "row": s,
            })
            continue
        transcript_text = transcript_row["transcript_text"]

        try:
            judgement = judge_summary_with_client(
                run_id=str(run_id),
                conversation_id=conversation_id,
                strategy_id=str(strategy_id),
                transcript_text=transcript_text,
                summary_text=summary_text,
                api_endpoint=api_endpoint,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )

            # Keep traceability back to generation settings if present
            judgement["judge_model"] = model
            judgement["judge_api_endpoint"] = api_endpoint
            judgement["judge_temperature"] = temperature
            judgement["judge_max_output_tokens"] = max_output_tokens
            judgement["judge_reasoning_effort"] = reasoning_effort
            judgement["endpoint_config_path"] = str(endpoint_config_path)
            judgement["gen_model"] = s.get("model")
            judgement["gen_temperature"] = s.get("temperature")
            judgement["prompt_file"] = s.get("prompt_file")
            judgement["split"] = s.get("split") or transcript_row.get("split")
            judgement["transcript_variant"] = s.get("transcript_variant") or transcript_row.get("transcript_variant")
            judgement["source_id"] = s.get("source_id") or transcript_row.get("source_id")

            write_jsonl_append(out_path, judgement)
            n += 1

        except Exception as e:
            error_row = {
                "index": i,
                "conversation_id": conversation_id,
                "strategy_id": strategy_id,
                "status": "judge_failure",
                "error": str(e),
            }
            raw_response = getattr(e, "raw_response", None)
            if raw_response is not None:
                error_row["raw_response"] = raw_response
            write_jsonl_append(err_path, error_row)

        if args.sleep and args.sleep > 0:
            time.sleep(args.sleep)

    if args.resume:
        print(f"OK: wrote {n} new judgements to {out_path} ({len(completed_keys) + n}/{len(summaries)} total present).")
    else:
        print(f"OK: wrote {n} judgements to {out_path}")
    if err_path.exists():
        print(f"Note: errors logged to {err_path}")


if __name__ == "__main__":
    main()
