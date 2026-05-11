"""
Extract candidate hallucination cases from a judge run for manual review.

For a given run directory, this script reads judgements.jsonl + summaries.jsonl,
joins them with the source transcripts via load_aci_bench_evaluation_dataset,
and writes out the cases where the judge gave a low `accurate` score.

Two thresholds are produced by default:
  - accurate <= 3 : all cases where the judge flagged at least one clear
                    misrepresentation (timing, negation, attribution,
                    fabrication, etc.).
  - accurate <= 2 : the strict subset (one or more major factual errors,
                    fabrications or falsifications).

Output for each threshold:
  - <out_dir>/hallucination_candidates_acc_le_<N>.jsonl
        Machine-readable, one record per row, with everything you need to
        re-judge or auto-categorize later.
  - <out_dir>/hallucination_candidates_acc_le_<N>.md
        Human-readable: one section per case, with the judge rationale,
        the summary, and the transcript side-by-side. This is the file
        intended for manual review.

Example:
    python -m src.extract_hallucination_candidates \\
        --run-dir runs/test_all_20260428 \\
        --out-dir runs/test_all_20260428/hallucination_review
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from src.data_loader import (
        ACI_BENCH_DATA_DIR,
        ACI_BENCH_MAIN_ANALYSIS_SPLITS,
        ACI_BENCH_TRANSCRIPT_VARIANT,
        load_aci_bench_evaluation_dataset,
    )
except Exception:
    from data_loader import (  # type: ignore
        ACI_BENCH_DATA_DIR,
        ACI_BENCH_MAIN_ANALYSIS_SPLITS,
        ACI_BENCH_TRANSCRIPT_VARIANT,
        load_aci_bench_evaluation_dataset,
    )


# Thresholds we generate by default. Edit here if you want different cuts.
DEFAULT_THRESHOLDS: tuple[int, ...] = (3, 2)


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


def build_summary_lookup(summaries: Sequence[Dict[str, Any]]) -> Dict[tuple[str, str], Dict[str, Any]]:
    """Map (conversation_id, strategy_id) -> summary row."""
    lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in summaries:
        cid = row.get("conversation_id")
        sid = row.get("strategy_id")
        if isinstance(cid, str) and isinstance(sid, str):
            lookup[(cid, sid)] = row
    return lookup


def build_transcript_lookup(conversations: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in conversations:
        cid = row.get("conversation_id")
        if isinstance(cid, str):
            lookup[cid] = row
    return lookup


def infer_splits_from_summaries(summaries: Sequence[Dict[str, Any]]) -> tuple[str, ...]:
    splits = sorted({row.get("split") for row in summaries if isinstance(row.get("split"), str)})
    if splits:
        return tuple(splits)
    return ACI_BENCH_MAIN_ANALYSIS_SPLITS


def select_candidates(
    judgements: Iterable[Dict[str, Any]],
    *,
    accurate_max: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for j in judgements:
        scores = j.get("scores") or {}
        acc = scores.get("accurate")
        if isinstance(acc, int) and acc <= accurate_max:
            out.append(j)
    # Sort: lowest accurate first, then lowest citation, then conversation/strategy
    out.sort(
        key=lambda j: (
            j["scores"].get("accurate", 99),
            j["scores"].get("citation", 99),
            j.get("conversation_id", ""),
            j.get("strategy_id", ""),
        )
    )
    return out


def build_record(
    judgement: Dict[str, Any],
    summary_row: Optional[Dict[str, Any]],
    transcript_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    scores = judgement.get("scores") or {}
    return {
        "conversation_id": judgement.get("conversation_id"),
        "strategy_id": judgement.get("strategy_id"),
        "run_id": judgement.get("run_id"),
        "split": judgement.get("split"),
        "transcript_variant": judgement.get("transcript_variant"),
        "source_id": judgement.get("source_id"),
        "scores": {
            "accurate": scores.get("accurate"),
            "citation": scores.get("citation"),
            "thorough": scores.get("thorough"),
            "useful": scores.get("useful"),
            "abstraction": scores.get("abstraction"),
            "synthesized": scores.get("synthesized"),
        },
        "primary_score": judgement.get("primary_score"),
        "rationale": judgement.get("rationale", ""),
        "summary_text": (summary_row or {}).get("summary", ""),
        "transcript_text": (transcript_row or {}).get("transcript_text", ""),
        "reference_note_text": (transcript_row or {}).get("reference_note_text", ""),
        "gen_model": judgement.get("gen_model"),
        "judge_model": judgement.get("judge_model"),
        "prompt_file": (summary_row or {}).get("prompt_file") or judgement.get("prompt_file"),
    }


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_markdown(path: Path, records: Sequence[Dict[str, Any]], *, accurate_max: int, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Hallusinasjons-kandidater (accurate <= {accurate_max})")
    lines.append("")
    lines.append(f"Kjøring: `{run_id}`  ")
    lines.append(f"Antall kandidater: **{len(records)}**")
    lines.append("")
    lines.append("Kolonnen `accurate` i rubrikken straffer fabrikasjon, forfalskning, motsigelse, ")
    lines.append("feil attribusjon, feil timing, feil negasjon og overdrivelse av sikkerhet. ")
    lines.append("`citation` straffer påstander som ikke kan spores tilbake til transkriptet.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, r in enumerate(records, start=1):
        scores = r["scores"]
        lines.append(
            f"## {i}. {r['conversation_id']}  —  strategy `{r['strategy_id']}`"
        )
        lines.append("")
        lines.append(
            f"**Scores:** accurate={scores.get('accurate')}, citation={scores.get('citation')}, "
            f"thorough={scores.get('thorough')}, useful={scores.get('useful')}, "
            f"abstraction={scores.get('abstraction')}, synthesized={scores.get('synthesized')}  "
        )
        lines.append(f"**Primary score:** {r.get('primary_score')}  ")
        lines.append(f"**Gen model:** {r.get('gen_model')}  |  **Judge model:** {r.get('judge_model')}  ")
        lines.append(f"**Prompt:** `{r.get('prompt_file')}`")
        lines.append("")
        lines.append("**Dommer-begrunnelse:**")
        lines.append("")
        lines.append("> " + (r.get("rationale") or "(tom)").replace("\n", "\n> "))
        lines.append("")
        lines.append("**Generert sammendrag:**")
        lines.append("")
        lines.append("```")
        lines.append(r.get("summary_text") or "(tom)")
        lines.append("```")
        lines.append("")
        lines.append("**Kilde-transkript:**")
        lines.append("")
        lines.append("```")
        lines.append(r.get("transcript_text") or "(tom)")
        lines.append("```")
        lines.append("")
        ref = r.get("reference_note_text")
        if ref:
            lines.append("<details><summary>Referansenotat (gold standard)</summary>")
            lines.append("")
            lines.append("```")
            lines.append(ref)
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="Path to runs/<run_id> directory")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Defaults to <run-dir>/hallucination_review.",
    )
    ap.add_argument(
        "--data-dir",
        default=str(ACI_BENCH_DATA_DIR),
        help="Path to ACI-Bench src_experiment_data_json directory",
    )
    ap.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="Comma-separated `accurate <= N` cutoffs to produce. Default: 3,2",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "hallucination_review"

    judgements_path = run_dir / "judgements.jsonl"
    summaries_path = run_dir / "summaries.jsonl"
    if not judgements_path.exists():
        raise FileNotFoundError(f"Missing {judgements_path}")
    if not summaries_path.exists():
        raise FileNotFoundError(f"Missing {summaries_path}")

    thresholds = tuple(int(t.strip()) for t in args.thresholds.split(",") if t.strip())
    if not thresholds:
        raise ValueError("At least one threshold required.")

    print(f"Reading {judgements_path} ...")
    judgements = read_jsonl(judgements_path)
    print(f"  {len(judgements)} judgement rows")

    print(f"Reading {summaries_path} ...")
    summaries = read_jsonl(summaries_path)
    print(f"  {len(summaries)} summary rows")
    summary_lookup = build_summary_lookup(summaries)

    splits = infer_splits_from_summaries(summaries)
    print(f"Loading transcripts for splits {splits} ...")
    conversations = load_aci_bench_evaluation_dataset(
        splits=splits,
        data_dir=Path(args.data_dir),
        transcript_variant=ACI_BENCH_TRANSCRIPT_VARIANT,
    )
    transcript_lookup = build_transcript_lookup(conversations)
    print(f"  {len(transcript_lookup)} unique transcripts")

    run_id = run_dir.name

    for accurate_max in thresholds:
        candidates = select_candidates(judgements, accurate_max=accurate_max)
        records: List[Dict[str, Any]] = []
        missing_summary = 0
        missing_transcript = 0
        for j in candidates:
            cid = j.get("conversation_id")
            sid = j.get("strategy_id")
            summary_row = summary_lookup.get((cid, sid)) if isinstance(cid, str) and isinstance(sid, str) else None
            transcript_row = transcript_lookup.get(cid) if isinstance(cid, str) else None
            if summary_row is None:
                missing_summary += 1
            if transcript_row is None:
                missing_transcript += 1
            records.append(build_record(j, summary_row, transcript_row))

        jsonl_path = out_dir / f"hallucination_candidates_acc_le_{accurate_max}.jsonl"
        md_path = out_dir / f"hallucination_candidates_acc_le_{accurate_max}.md"
        write_jsonl(jsonl_path, records)
        write_markdown(md_path, records, accurate_max=accurate_max, run_id=run_id)

        print(
            f"accurate <= {accurate_max}: {len(records)} cases  "
            f"-> {jsonl_path.relative_to(run_dir.parent)}, {md_path.relative_to(run_dir.parent)}"
        )
        if missing_summary or missing_transcript:
            print(
                f"  WARN: missing_summary={missing_summary}, missing_transcript={missing_transcript}"
            )


if __name__ == "__main__":
    main()
