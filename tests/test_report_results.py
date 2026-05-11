import unittest

from src.judge_schema import calculate_primary_score, derive_voice_analysis
from src.report_results import build_report_tables


def judgement(conversation_id, split, strategy_id, scores):
    primary_score, included_dimensions = calculate_primary_score(scores)
    return {
        "run_id": "run-1",
        "conversation_id": conversation_id,
        "strategy_id": strategy_id,
        "split": split,
        "transcript_variant": "aci_asrcorr",
        "scores": scores,
        "primary_score": primary_score,
        "included_dimensions": included_dimensions,
        "voice_analysis": derive_voice_analysis(scores),
        "rationale": "ok",
    }


class ReportResultsTests(unittest.TestCase):
    def test_report_tables_use_new_judge_schema_and_filter_excluded_ids(self) -> None:
        strategy_id = "01_zero_shot_minimal_baseline"
        records = [
            judgement(
                "test1:aci_asrcorr:0-aci",
                "test1",
                strategy_id,
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
            judgement(
                "test2:aci_asrcorr:0-aci",
                "test2",
                strategy_id,
                {
                    "citation": 5,
                    "accurate": 5,
                    "thorough": 5,
                    "useful": 5,
                    "organized": 5,
                    "comprehensible": 5,
                    "succinct": 5,
                    "abstraction": 1,
                    "synthesized": 4,
                    "voice_summ": 0,
                    "voice_note": 1,
                },
            ),
            judgement(
                "train:aci_asrcorr:0-aci",
                "train",
                strategy_id,
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

        tables = build_report_tables(records)
        per_summary = tables["per_summary"]
        strategy = tables["strategy_aggregates"]
        split_strategy = tables["split_strategy_aggregates"]
        voice = tables["voice_analysis"]

        self.assertEqual(len(per_summary), 2)
        self.assertNotIn("train:aci_asrcorr:0-aci", set(per_summary["conversation_id"]))
        self.assertNotIn("total", per_summary.columns)
        self.assertIn("primary_score", per_summary.columns)
        self.assertIn("citation", per_summary.columns)
        self.assertIn("voice_case", per_summary.columns)

        self.assertEqual(len(strategy), 1)
        self.assertEqual(strategy.iloc[0]["n_summaries"], 2)
        self.assertEqual(strategy.iloc[0]["mean_primary_score"], 4.4375)
        self.assertEqual(strategy.iloc[0]["mean_synthesized_when_applicable"], 4.0)
        self.assertEqual(strategy.iloc[0]["introduced_stigmatizing_language_rate"], 0.5)
        self.assertEqual(strategy.iloc[0]["neutralized_stigmatizing_language_rate"], 0.5)

        self.assertEqual(set(split_strategy["split"]), {"test1", "test2"})
        self.assertEqual(list(voice.columns), ["run_id", "conversation_id", "strategy_id", "split", "voice_note", "voice_summ", "voice_case"])


if __name__ == "__main__":
    unittest.main()
