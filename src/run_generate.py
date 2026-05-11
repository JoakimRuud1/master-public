import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None

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


GENERATION_SYSTEM_PROMPT = """Du er en klinisk assistent som skriver korte journalsammendrag fra legevaktssamtaler. 
Følg instruksjonene i oppgaveprompten.
Bruk kun informasjon som er eksplisitt oppgitt eller tydelig støttet av transkriptet.
Ikke legg til, anta eller overtolk opplysninger.
Hvis informasjon er uklar eller mangler i transkriptet, skal du ikke gjette.
Prioriter korrekthet og klinisk relevans fremfor å få med mest mulig."""

ENSEMBLE_STRATEGY_ID = "08_two_shot_decomposition_self_criticism_ensemble"
ENSEMBLE_N = 3
DEFAULT_ENSEMBLE_CANDIDATE_PROMPT_FILE = "prompts/07_two_shot_decomposition_self_criticism.txt"
DEFAULT_ENSEMBLE_SELECTION_PROMPT_FILE = "prompts/08_two_shot_decomposition_self_criticism_ensemble.txt"
ENSEMBLE_SELECTION_SYSTEM_PROMPT = (
    "Du er en streng klinisk vurderer. Velg nøyaktig én eksisterende kandidat. "
    "Ikke skriv, slå sammen eller forbedre journalsammendrag."
)

SummaryFn = Callable[[str, Dict[str, Any], str], str]
SelectionFn = Callable[[str, Sequence[str], Dict[str, Any], str], Dict[str, Any]]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_existing_keys(path: Path) -> set[tuple[str, str]]:
    """
    Returns the set of (conversation_id, strategy_id) pairs already present in path.
    Used for resume: skip pairs we have already produced.
    """
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Treat malformed lines as "not done" so they are re-generated.
                continue
            cid = obj.get("conversation_id")
            sid = obj.get("strategy_id")
            if isinstance(cid, str) and isinstance(sid, str):
                keys.add((cid, sid))
    return keys


def load_strategies(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in value.split(",") if part.strip())
    if not splits:
        raise ValueError("At least one split must be provided.")
    return splits


def summarize_llm(
    transcript: str,
    strategy: Dict[str, Any],
    *,
    api_endpoint: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None = None,
) -> str:
    try:
        from src.llm_client import generate_text
    except Exception:
        from llm_client import generate_text  # type: ignore

    prompt_template = read_text_file(Path(strategy["prompt_file"]))
    prompt = prompt_template.replace("{TRANSCRIPT}", transcript)

    temperature = strategy.get("temperature", 0.0)

    return generate_text(
        user_prompt=prompt,
        system_prompt=GENERATION_SYSTEM_PROMPT,
        api_endpoint=api_endpoint,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )


def is_ensemble_strategy(strategy: Dict[str, Any]) -> bool:
    ensemble_config = strategy.get("ensemble")
    if isinstance(ensemble_config, dict) and bool(ensemble_config.get("enabled")):
        return True
    return strategy.get("id") == ENSEMBLE_STRATEGY_ID


def get_ensemble_n(strategy: Dict[str, Any]) -> int:
    ensemble_config = strategy.get("ensemble")
    n = ENSEMBLE_N
    if isinstance(ensemble_config, dict) and "n" in ensemble_config:
        n = int(ensemble_config["n"])
    if n != ENSEMBLE_N:
        raise ValueError(f"Ensemble strategy must use n={ENSEMBLE_N}, got n={n}.")
    return n


def get_ensemble_candidate_prompt_file(strategy: Dict[str, Any]) -> str:
    ensemble_config = strategy.get("ensemble")
    if isinstance(ensemble_config, dict) and ensemble_config.get("candidate_prompt_file"):
        return str(ensemble_config["candidate_prompt_file"])
    if strategy.get("id") == ENSEMBLE_STRATEGY_ID:
        return DEFAULT_ENSEMBLE_CANDIDATE_PROMPT_FILE
    return str(strategy.get("prompt_file", DEFAULT_ENSEMBLE_CANDIDATE_PROMPT_FILE))


def get_ensemble_selection_prompt_file(strategy: Dict[str, Any]) -> str:
    ensemble_config = strategy.get("ensemble")
    if isinstance(ensemble_config, dict) and ensemble_config.get("selection_prompt_file"):
        return str(ensemble_config["selection_prompt_file"])
    if strategy.get("id") == ENSEMBLE_STRATEGY_ID:
        return DEFAULT_ENSEMBLE_SELECTION_PROMPT_FILE
    return str(strategy["prompt_file"])


def build_selection_prompt(
    *,
    transcript: str,
    candidates: Sequence[str],
    selection_prompt_file: str,
) -> str:
    if len(candidates) != ENSEMBLE_N:
        raise ValueError(f"Selection prompt requires exactly {ENSEMBLE_N} candidates.")

    prompt = read_text_file(Path(selection_prompt_file))
    replacements = {
        "{TRANSCRIPT}": transcript,
        "{CANDIDATE_1}": candidates[0],
        "{CANDIDATE_2}": candidates[1],
        "{CANDIDATE_3}": candidates[2],
    }
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def parse_selected_candidate(text: str) -> Optional[int]:
    match = re.fullmatch(r"\s*VALGT_KANDIDAT\s*:\s*([123])\s*", text)
    if not match:
        return None
    return int(match.group(1))


def select_ensemble_candidate_llm(
    transcript: str,
    candidates: Sequence[str],
    strategy: Dict[str, Any],
    model: str,
    *,
    api_endpoint: str,
    max_output_tokens: int,
    reasoning_effort: str | None = None,
) -> Dict[str, Any]:
    try:
        from src.llm_client import generate_text
    except Exception:
        from llm_client import generate_text  # type: ignore

    selection_prompt_file = get_ensemble_selection_prompt_file(strategy)
    prompt = build_selection_prompt(
        transcript=transcript,
        candidates=candidates,
        selection_prompt_file=selection_prompt_file,
    )
    temperature = strategy.get("temperature", 0.0)

    for attempt in range(1, 3):
        response = generate_text(
            user_prompt=prompt,
            system_prompt=ENSEMBLE_SELECTION_SYSTEM_PROMPT,
            api_endpoint=api_endpoint,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        selected_candidate = parse_selected_candidate(response)
        if selected_candidate is not None:
            return {
                "selected_candidate": selected_candidate,
                "selection_status": "ok",
                "selection_attempts": attempt,
            }

    return {
        "selected_candidate": None,
        "selection_status": "failed",
        "selection_attempts": 2,
    }


def build_ensemble_summary_row(
    *,
    run_id: str,
    conversation: Dict[str, Any],
    strategy: Dict[str, Any],
    model: str,
    summarize_fn: SummaryFn,
    selection_fn: SelectionFn,
) -> Dict[str, Any]:
    transcript = conversation["transcript_text"]
    n = get_ensemble_n(strategy)
    candidate_prompt_file = get_ensemble_candidate_prompt_file(strategy)
    selection_prompt_file = get_ensemble_selection_prompt_file(strategy)
    candidate_strategy = {**strategy, "prompt_file": candidate_prompt_file}

    candidate_summaries = [
        summarize_fn(transcript, candidate_strategy, model)
        for _ in range(n)
    ]
    selection = selection_fn(transcript, candidate_summaries, strategy, model)
    selected_candidate = selection.get("selected_candidate")
    summary = (
        candidate_summaries[selected_candidate - 1]
        if isinstance(selected_candidate, int) and 1 <= selected_candidate <= n
        else None
    )

    return {
        "run_id": run_id,
        "conversation_id": conversation.get("conversation_id"),
        "strategy_id": strategy["id"],
        "split": conversation.get("split"),
        "transcript_variant": conversation.get("transcript_variant"),
        "source_id": conversation.get("source_id"),
        "model": model,
        "temperature": strategy.get("temperature", 0.0),
        "prompt_file": candidate_prompt_file,
        "selection_prompt_file": selection_prompt_file,
        "summary": summary,
        "ensemble": {
            "enabled": True,
            "n": n,
            "candidates": [
                {"candidate_id": i + 1, "summary": candidate}
                for i, candidate in enumerate(candidate_summaries)
            ],
            "selected_candidate": selected_candidate,
            "selection_status": selection.get("selection_status", "failed"),
            "selection_attempts": selection.get("selection_attempts", 0),
        },
    }


def build_summary_rows(
    *,
    run_id: str,
    conversations: Sequence[Dict[str, Any]],
    strategies: Sequence[Dict[str, Any]],
    model: str,
    summarize_fn: SummaryFn,
    selection_fn: SelectionFn | None = None,
    on_row: Callable[[Dict[str, Any]], None] | None = None,
    completed_keys: set[tuple[str, str]] | None = None,
    error_handler: Callable[[Dict[str, Any], Exception], None] | None = None,
) -> List[Dict[str, Any]]:
    """
    Generates summary rows for (conversation, strategy) pairs.

    Arguments:
      on_row: optional callback called with each row immediately after it is produced.
              Use this to persist the row to disk incrementally so a crash mid-run
              does not throw away earlier work.
      completed_keys: optional set of (conversation_id, strategy_id) pairs that have
              already been produced and should be skipped. Used for resume.
      error_handler: optional callback called when generation of a single row raises.
              If provided, the exception is swallowed for that pair and the loop
              continues. If None, the exception is re-raised (legacy behaviour).
    """
    outputs: List[Dict[str, Any]] = []
    completed = completed_keys if completed_keys is not None else set()
    for conv in conversations:
        cid = conv.get("conversation_id")
        text = conv.get("transcript_text")
        if not cid or not isinstance(text, str):
            raise ValueError("Each conversation must contain fields: conversation_id, transcript_text")

        for strat in strategies:
            pair = (cid, strat["id"])
            if pair in completed:
                continue

            try:
                if is_ensemble_strategy(strat):
                    if selection_fn is None:
                        raise ValueError("Ensemble strategy requires a selection_fn.")
                    row = build_ensemble_summary_row(
                        run_id=run_id,
                        conversation=conv,
                        strategy=strat,
                        model=model,
                        summarize_fn=summarize_fn,
                        selection_fn=selection_fn,
                    )
                else:
                    summary = summarize_fn(text, strat, model)
                    row = {
                        "run_id": run_id,
                        "conversation_id": cid,
                        "strategy_id": strat["id"],
                        "split": conv.get("split"),
                        "transcript_variant": conv.get("transcript_variant"),
                        "source_id": conv.get("source_id"),
                        "model": model,
                        "temperature": strat.get("temperature", 0.0),
                        "prompt_file": strat["prompt_file"],
                        "summary": summary,
                    }
            except Exception as e:
                if error_handler is None:
                    raise
                error_handler({
                    "run_id": run_id,
                    "conversation_id": cid,
                    "strategy_id": strat["id"],
                    "split": conv.get("split"),
                }, e)
                continue

            outputs.append(row)
            if on_row is not None:
                on_row(row)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ACI_BENCH_DATA_DIR), help="Path to ACI-Bench src_experiment_data_json directory")
    ap.add_argument("--splits", default=",".join(ACI_BENCH_MAIN_ANALYSIS_SPLITS), help="Comma-separated ACI-Bench splits to generate, e.g. test1 or valid,test1")
    ap.add_argument("--strategies", required=True, help="Path to strategies.json")
    ap.add_argument("--endpoint-config", default=str(DEFAULT_ENDPOINT_CONFIG_PATH), help="Path to endpoint config JSON")
    ap.add_argument("--out", default="runs", help="Output directory for runs")
    ap.add_argument("--api-endpoint", default="", help="Override generator API endpoint from endpoint config")
    ap.add_argument("--model", default="", help="Override generator model from endpoint config")
    ap.add_argument("--max-output-tokens", type=int, default=0, help="Override generator max output tokens from endpoint config")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of conversations to process (0 = all)")
    ap.add_argument("--resume", default="", help="Resume generation into an existing runs/<run_id> directory. Skips already-produced (conversation_id, strategy_id) pairs.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = parse_splits(args.splits)
    strat_path = Path(args.strategies)
    endpoint_config_path = Path(args.endpoint_config)
    out_root = Path(args.out)
    endpoint_settings = get_endpoint_settings(load_endpoint_config(endpoint_config_path), "generator")
    api_endpoint = args.api_endpoint or str(endpoint_settings["api_endpoint"])
    model = args.model or str(endpoint_settings["model"])
    max_output_tokens = args.max_output_tokens or int(endpoint_settings["max_output_tokens"])
    reasoning_effort = endpoint_settings.get("reasoning_effort")

    conversations = load_aci_bench_evaluation_dataset(
        splits=splits,
        data_dir=data_dir,
        transcript_variant=ACI_BENCH_TRANSCRIPT_VARIANT,
    )
    if args.limit > 0:
        conversations = conversations[:args.limit]
    strategies = load_strategies(strat_path)

    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            raise FileNotFoundError(f"--resume target does not exist: {run_dir}")
        run_id = run_dir.name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    summaries_path = run_dir / "summaries.jsonl"
    errors_path = run_dir / "summaries_errors.jsonl"
    completed_keys = read_existing_keys(summaries_path)

    # Manifest for reproduserbarhet. On resume, keep the original manifest if present.
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "run_id": run_id,
            "data_dir": str(data_dir),
            "splits": list(splits),
            "transcript_variant": ACI_BENCH_TRANSCRIPT_VARIANT,
            "strategies_path": str(strat_path),
            "endpoint_config_path": str(endpoint_config_path),
            "api_endpoint": api_endpoint,
            "model": model,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
            "num_conversations": len(conversations),
            "num_strategies": len(strategies),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total_pairs = len(conversations) * len(strategies)
    if completed_keys:
        print(f"Resume: {len(completed_keys)}/{total_pairs} (conversation_id, strategy_id) pairs already present in {summaries_path}.")

    def on_row(row: Dict[str, Any]) -> None:
        append_jsonl(summaries_path, row)

    def on_error(context: Dict[str, Any], exc: Exception) -> None:
        err = {
            **context,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        append_jsonl(errors_path, err)
        print(f"WARN: generation failed for {context.get('conversation_id')} / {context.get('strategy_id')}: {exc}")

    outputs = build_summary_rows(
        run_id=run_id,
        conversations=conversations,
        strategies=strategies,
        model=model,
        summarize_fn=lambda transcript, strategy, model: summarize_llm(
            transcript,
            strategy,
            api_endpoint=api_endpoint,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        ),
        selection_fn=lambda transcript, candidates, strategy, model: select_ensemble_candidate_llm(
            transcript,
            candidates,
            strategy,
            model,
            api_endpoint=api_endpoint,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        ),
        on_row=on_row,
        completed_keys=completed_keys,
        error_handler=on_error,
    )
    total_in_file = len(completed_keys) + len(outputs)
    print(f"OK: wrote {len(outputs)} new summaries to {summaries_path} "
          f"({total_in_file}/{total_pairs} total).")
    if errors_path.exists():
        print(f"Note: errors logged to {errors_path}. Re-run with --resume {run_dir} to retry them.")


if __name__ == "__main__":
    load_dotenv()
    main()
