import unittest

from src.judge_client import _build_stored_judgement, _validate_raw_judge_response
from src.judge_schema import QUALITY_SCORE_KEYS, calculate_primary_score


def build_stored(scores, rationale="ok"):
    fixed_scores, fixed_rationale = _validate_raw_judge_response({**scores, "rationale": rationale})
    return _build_stored_judgement(
        run_id="run-1",
        conversation_id="test1:aci_asrcorr:0-aci",
        strategy_id="01_zero_shot_minimal_baseline",
        scores=fixed_scores,
        rationale=fixed_rationale,
    )


class JudgeSchemaTests(unittest.TestCase):
    def test_primary_score_excludes_synthesized_when_abstraction_is_zero(self) -> None:
        scores = {
            "citation": 3,
            "accurate": 3,
            "thorough": 3,
            "useful": 3,
            "organized": 3,
            "comprehensible": 3,
            "succinct": 3,
            "abstraction": 0,
            "synthesized": None,
            "voice_summ": 1,
            "voice_note": 0,
        }

        fixed = build_stored(scores)

        self.assertIsNone(fixed["scores"]["synthesized"])
        self.assertEqual(fixed["primary_score"], 3.0)
        self.assertEqual(fixed["included_dimensions"], QUALITY_SCORE_KEYS)
        self.assertEqual(fixed["voice_analysis"]["case"], "introduced_stigmatizing_language")

    def test_primary_score_includes_synthesized_when_abstraction_is_one(self) -> None:
        scores = {
            "citation": 4,
            "accurate": 4,
            "thorough": 4,
            "useful": 4,
            "organized": 4,
            "comprehensible": 4,
            "succinct": 4,
            "abstraction": 1,
            "synthesized": 5,
            "voice_summ": 0,
            "voice_note": 1,
        }

        primary_score, included_dimensions = calculate_primary_score(scores)
        fixed = build_stored(scores)

        self.assertEqual(primary_score, 4.125)
        self.assertEqual(fixed["primary_score"], 4.125)
        self.assertEqual(included_dimensions[-1], "synthesized")
        self.assertEqual(fixed["voice_analysis"]["case"], "neutralized_stigmatizing_language")

    def test_raw_response_rejects_metadata_and_derived_fields(self) -> None:
        scores = {
            "citation": 4,
            "accurate": 4,
            "thorough": 4,
            "useful": 4,
            "organized": 4,
            "comprehensible": 4,
            "succinct": 4,
            "abstraction": 0,
            "synthesized": None,
            "voice_summ": 0,
            "voice_note": 0,
            "primary_score": 4.0,
        }

        with self.assertRaises(ValueError):
            _validate_raw_judge_response(scores)

    def test_raw_response_accepts_na_synthesized_when_abstraction_is_zero(self) -> None:
        scores = {
            "citation": 5,
            "accurate": 5,
            "thorough": 5,
            "useful": 5,
            "organized": 5,
            "comprehensible": 5,
            "succinct": 5,
            "abstraction": 0,
            "synthesized": "NA",
            "voice_summ": 0,
            "voice_note": 0,
        }

        fixed_scores, _ = _validate_raw_judge_response(scores)

        self.assertIsNone(fixed_scores["synthesized"])


if __name__ == "__main__":
    unittest.main()
