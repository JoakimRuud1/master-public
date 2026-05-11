import unittest
from pathlib import Path

import pandas as pd

from src.compare_judge_manual import build_comparison_table, build_manual_table
from src.judge_schema import calculate_primary_score, derive_voice_analysis


def stored_judgement(conversation_id, split, scores):
    primary_score, included_dimensions = calculate_primary_score(scores)
    return {
        "run_id": "run-1",
        "conversation_id": conversation_id,
        "strategy_id": "01_zero_shot_minimal_baseline",
        "split": split,
        "transcript_variant": "aci_asrcorr",
        "scores": scores,
        "primary_score": primary_score,
        "included_dimensions": included_dimensions,
        "voice_analysis": derive_voice_analysis(scores),
        "rationale": "ok",
    }


class CompareJudgeManualTests(unittest.TestCase):
    def test_named_rubric_columns_compare_side_by_side_and_filter_excluded_ids(self) -> None:
        judge_rows = [
            stored_judgement(
                "test1:aci_asrcorr:0-aci",
                "test1",
                {
                    "citation": 4,
                    "accurate": 4,
                    "thorough": 4,
                    "useful": 4,
                    "organized": 4,
                    "comprehensible": 4,
                    "succinct": 4,
                    "abstraction": 0,
                    "synthesized": None,
                    "voice_summ": 1,
                    "voice_note": 0,
                },
            ),
            stored_judgement(
                "train:aci_asrcorr:0-aci",
                "train",
                {
                    "citation": 1,
                    "accurate": 1,
                    "thorough": 1,
                    "useful": 1,
                    "organized": 1,
                    "comprehensible": 1,
                    "succinct": 1,
                    "abstraction": 0,
                    "synthesized": None,
                    "voice_summ": 0,
                    "voice_note": 0,
                },
            ),
        ]
        manual = pd.DataFrame(
            [
                {
                    "run_id": "run-1",
                    "conversation_id": "test1:aci_asrcorr:0-aci",
                    "strategy_id": "01_zero_shot_minimal_baseline",
                    "split": "test1",
                    "transcript_variant": "aci_asrcorr",
                    "citation": 5,
                    "accurate": 4,
                    "thorough": 4,
                    "useful": 4,
                    "organized": 4,
                    "comprehensible": 4,
                    "succinct": 4,
                    "abstraction": 0,
                    "synthesized": "NA",
                    "voice_summ": 0,
                    "voice_note": 0,
                },
                {
                    "run_id": "run-1",
                    "conversation_id": "train:aci_asrcorr:0-aci",
                    "strategy_id": "01_zero_shot_minimal_baseline",
                    "split": "train",
                    "transcript_variant": "aci_asrcorr",
                    "citation": 1,
                    "accurate": 1,
                    "thorough": 1,
                    "useful": 1,
                    "organized": 1,
                    "comprehensible": 1,
                    "succinct": 1,
                    "abstraction": 0,
                    "synthesized": "NA",
                    "voice_summ": 0,
                    "voice_note": 0,
                },
            ]
        )

        comparison = build_comparison_table(manual, judge_rows, Path("manual.csv"))

        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison.iloc[0]["conversation_id"], "test1:aci_asrcorr:0-aci")
        self.assertEqual(comparison.iloc[0]["citation_judge"], 4)
        self.assertEqual(comparison.iloc[0]["citation_manual"], 5)
        self.assertEqual(comparison.iloc[0]["citation_difference"], -1)
        self.assertEqual(comparison.iloc[0]["judge_primary_score"], 4.0)
        self.assertAlmostEqual(comparison.iloc[0]["manual_primary_score"], 4.1429)
        self.assertFalse(comparison.iloc[0]["synthesized_comparison_applicable"])
        self.assertTrue(pd.isna(comparison.iloc[0]["synthesized_difference"]))
        self.assertEqual(comparison.iloc[0]["voice_summ_judge"], 1)
        self.assertEqual(comparison.iloc[0]["voice_summ_manual"], 0)
        self.assertFalse(comparison.iloc[0]["voice_summ_exact_match"])

    def test_manual_scoring_requires_named_rubric_columns(self) -> None:
        old_manual = pd.DataFrame(
            [
                {
                    "run_id": "run-1",
                    "conversation_id": "test1:aci_asrcorr:0-aci",
                    "strategy_id": "01_zero_shot_minimal_baseline",
                    "score_1": 1,
                    "score_2": 1,
                    "score_3": 1,
                    "score_4": 1,
                    "score_5": 1,
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "explicit rubric columns"):
            build_manual_table(old_manual, Path("manual.csv"))


if __name__ == "__main__":
    unittest.main()
