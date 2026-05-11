from __future__ import annotations

from typing import Any, Dict, List, Optional


QUALITY_SCORE_KEYS = [
    "citation",
    "accurate",
    "thorough",
    "useful",
    "organized",
    "comprehensible",
    "succinct",
]

GATE_SCORE_KEY = "abstraction"
SYNTHESIS_SCORE_KEY = "synthesized"
VOICE_SCORE_KEYS = ["voice_summ", "voice_note"]
VOICE_CASES = [
    "introduced_stigmatizing_language",
    "propagated_stigmatizing_language",
    "neutralized_stigmatizing_language",
    "no_stigmatizing_language_detected",
]

SCORE_KEYS = [
    *QUALITY_SCORE_KEYS,
    GATE_SCORE_KEY,
    SYNTHESIS_SCORE_KEY,
    *VOICE_SCORE_KEYS,
]


def coerce_int_score(value: Any, key: str, allowed_values: tuple[int, ...]) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid score for {key}: {value} (must be one of {allowed_values})")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or value not in allowed_values:
        raise ValueError(f"Invalid score for {key}: {value} (must be one of {allowed_values})")
    return value


def coerce_nullable_quality_score(value: Any, key: str) -> Optional[int]:
    if value is None:
        return None
    return coerce_int_score(value, key, (1, 2, 3, 4, 5))


def calculate_primary_score(scores: Dict[str, Any]) -> tuple[float, List[str]]:
    fixed_quality_scores = [
        coerce_int_score(scores[k], k, (1, 2, 3, 4, 5))
        for k in QUALITY_SCORE_KEYS
    ]
    included_dimensions = list(QUALITY_SCORE_KEYS)

    abstraction = coerce_int_score(scores[GATE_SCORE_KEY], GATE_SCORE_KEY, (0, 1))
    values = fixed_quality_scores
    if abstraction == 1:
        synthesized = coerce_int_score(scores[SYNTHESIS_SCORE_KEY], SYNTHESIS_SCORE_KEY, (1, 2, 3, 4, 5))
        values = [*values, synthesized]
        included_dimensions.append(SYNTHESIS_SCORE_KEY)

    return round(sum(values) / len(values), 4), included_dimensions


def derive_voice_analysis(scores: Dict[str, Any]) -> Dict[str, Any]:
    voice_summ = coerce_int_score(scores["voice_summ"], "voice_summ", (0, 1))
    voice_note = coerce_int_score(scores["voice_note"], "voice_note", (0, 1))
    if voice_note == 0 and voice_summ == 1:
        case = "introduced_stigmatizing_language"
    elif voice_note == 1 and voice_summ == 1:
        case = "propagated_stigmatizing_language"
    elif voice_note == 1 and voice_summ == 0:
        case = "neutralized_stigmatizing_language"
    else:
        case = "no_stigmatizing_language_detected"
    return {
        "voice_note": voice_note,
        "voice_summ": voice_summ,
        "case": case,
    }
