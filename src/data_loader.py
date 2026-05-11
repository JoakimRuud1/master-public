from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


ACI_BENCH_DATA_DIR = Path("data/aci_bench/src_experiment_data_json")
ACI_BENCH_TRANSCRIPT_VARIANT = "aci_asrcorr"
ACI_BENCH_SPLITS = ("train", "valid", "test1", "test2", "test3", "legevakt_norsk")
ACI_BENCH_DEVELOPMENT_SPLITS = ("train",)
ACI_BENCH_PILOT_SPLITS = ("valid",)
ACI_BENCH_MAIN_ANALYSIS_SPLITS = ("test1",)
ACI_BENCH_ROBUSTNESS_SPLITS = ("test2", "test3")

ACI_BENCH_SPLIT_USAGE = {
    "train": "development, debugging, and prompt examples",
    "valid": "sanity checks, debugging, and smaller pilot runs",
    "test1": "main analysis",
    "test2": "optional robustness analysis",
    "test3": "optional robustness analysis",
    "legevakt_norsk": "qualitative norwegian out-of-distribution probe",
}

PROMPT_EXAMPLE_CONVERSATION_IDS = frozenset(
    {
        "train:aci_asrcorr:0-aci",
        "train:aci_asrcorr:1-aci",
    }
)
EXCLUDED_EVALUATION_CONVERSATION_IDS = PROMPT_EXAMPLE_CONVERSATION_IDS


def make_conversation_id(*, split: str, transcript_variant: str, source_id: str) -> str:
    return f"{split}:{transcript_variant}:{source_id}"


def validate_aci_bench_split(split: str) -> None:
    if split not in ACI_BENCH_SPLITS:
        allowed = ", ".join(ACI_BENCH_SPLITS)
        raise ValueError(f"Unknown ACI-Bench split '{split}'. Expected one of: {allowed}.")


def is_excluded_from_evaluation(conversation_id: str) -> bool:
    return conversation_id in EXCLUDED_EVALUATION_CONVERSATION_IDS


def aci_bench_path(
    split: str,
    *,
    data_dir: Path = ACI_BENCH_DATA_DIR,
    transcript_variant: str = ACI_BENCH_TRANSCRIPT_VARIANT,
) -> Path:
    validate_aci_bench_split(split)
    return data_dir / f"{split}_{transcript_variant}.json"


def load_aci_bench_split(
    split: str,
    *,
    data_dir: Path = ACI_BENCH_DATA_DIR,
    transcript_variant: str = ACI_BENCH_TRANSCRIPT_VARIANT,
) -> List[Dict[str, Any]]:
    path = aci_bench_path(split, data_dir=data_dir, transcript_variant=transcript_variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing ACI-Bench source file: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or not isinstance(raw.get("data"), list):
        raise ValueError(f"Expected {path} to contain a JSON object with a 'data' list.")

    conversations: List[Dict[str, Any]] = []
    for row_index, row in enumerate(raw["data"], start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid row {row_index} in {path}: expected object.")

        source_id = row.get("file")
        transcript_text = row.get("src")
        reference_note_text = row.get("tgt")

        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"Invalid or missing 'file' in row {row_index} of {path}.")
        if not isinstance(transcript_text, str) or not transcript_text.strip():
            raise ValueError(f"Invalid or missing 'src' in row {row_index} of {path}.")

        conversation_id = make_conversation_id(
            split=split,
            transcript_variant=transcript_variant,
            source_id=source_id,
        )
        excluded_from_evaluation = is_excluded_from_evaluation(conversation_id)

        mapped: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "transcript_text": transcript_text,
            "split": split,
            "transcript_variant": transcript_variant,
            "source_id": source_id,
            "reserved_for_prompt_examples": excluded_from_evaluation,
            "exclude_from_evaluation": excluded_from_evaluation,
        }

        if isinstance(reference_note_text, str):
            mapped["reference_note_text"] = reference_note_text

        conversations.append(mapped)

    return conversations


def filter_evaluation_conversations(conversations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in conversations if not row.get("exclude_from_evaluation", False)]


def load_aci_bench_dataset(
    *,
    splits: Sequence[str] = ACI_BENCH_SPLITS,
    data_dir: Path = ACI_BENCH_DATA_DIR,
    transcript_variant: str = ACI_BENCH_TRANSCRIPT_VARIANT,
) -> List[Dict[str, Any]]:
    conversations: List[Dict[str, Any]] = []
    for split in splits:
        conversations.extend(
            load_aci_bench_split(
                split,
                data_dir=data_dir,
                transcript_variant=transcript_variant,
            )
        )
    return conversations


def load_aci_bench_evaluation_dataset(
    *,
    splits: Sequence[str] = ACI_BENCH_MAIN_ANALYSIS_SPLITS,
    data_dir: Path = ACI_BENCH_DATA_DIR,
    transcript_variant: str = ACI_BENCH_TRANSCRIPT_VARIANT,
    exclude_prompt_examples: bool = True,
) -> List[Dict[str, Any]]:
    conversations = load_aci_bench_dataset(
        splits=splits,
        data_dir=data_dir,
        transcript_variant=transcript_variant,
    )
    if exclude_prompt_examples:
        conversations = filter_evaluation_conversations(conversations)
    return conversations


def load_aci_bench_development_dataset(
    *,
    data_dir: Path = ACI_BENCH_DATA_DIR,
    transcript_variant: str = ACI_BENCH_TRANSCRIPT_VARIANT,
) -> List[Dict[str, Any]]:
    return load_aci_bench_dataset(
        splits=ACI_BENCH_DEVELOPMENT_SPLITS,
        data_dir=data_dir,
        transcript_variant=transcript_variant,
    )


def load_aci_bench_pilot_dataset(
    *,
    data_dir: Path = ACI_BENCH_DATA_DIR,
    transcript_variant: str = ACI_BENCH_TRANSCRIPT_VARIANT,
) -> List[Dict[str, Any]]:
    return load_aci_bench_dataset(
        splits=ACI_BENCH_PILOT_SPLITS,
        data_dir=data_dir,
        transcript_variant=transcript_variant,
    )


def load_aci_bench_main_analysis_dataset(
    *,
    data_dir: Path = ACI_BENCH_DATA_DIR,
    transcript_variant: str = ACI_BENCH_TRANSCRIPT_VARIANT,
) -> List[Dict[str, Any]]:
    return load_aci_bench_evaluation_dataset(
        splits=ACI_BENCH_MAIN_ANALYSIS_SPLITS,
        data_dir=data_dir,
        transcript_variant=transcript_variant,
    )


def load_aci_bench_robustness_dataset(
    *,
    data_dir: Path = ACI_BENCH_DATA_DIR,
    transcript_variant: str = ACI_BENCH_TRANSCRIPT_VARIANT,
) -> List[Dict[str, Any]]:
    return load_aci_bench_evaluation_dataset(
        splits=ACI_BENCH_ROBUSTNESS_SPLITS,
        data_dir=data_dir,
        transcript_variant=transcript_variant,
    )
