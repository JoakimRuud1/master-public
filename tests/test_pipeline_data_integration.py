import json
import tempfile
import unittest
from pathlib import Path

from src.run_generate import (
    GENERATION_SYSTEM_PROMPT,
    build_selection_prompt,
    build_summary_rows,
    parse_selected_candidate,
)
from src.run_judge import build_transcript_lookup, infer_splits_from_summaries, read_existing_judgement_keys
from src.endpoint_config import get_endpoint_settings, load_endpoint_config


class PipelineDataIntegrationTests(unittest.TestCase):
    def test_endpoint_config_sets_generator_and_judge_defaults(self) -> None:
        config = load_endpoint_config()

        generator = get_endpoint_settings(config, "generator")
        judge = get_endpoint_settings(config, "judge")

        self.assertEqual(generator["api_endpoint"], "responses")
        self.assertEqual(generator["model"], "gpt-5.4")
        self.assertEqual(generator["max_output_tokens"], 1500)
        self.assertIsNone(generator["reasoning_effort"])

        self.assertEqual(judge["api_endpoint"], "responses")
        self.assertEqual(judge["model"], "gpt-5.4")
        self.assertEqual(judge["max_output_tokens"], 16000)
        self.assertEqual(judge["reasoning_effort"], "high")

    def test_generation_system_prompt_matches_project_prompt(self) -> None:
        expected = """Du er en klinisk assistent som skriver korte journalsammendrag fra legevaktssamtaler. 
Følg instruksjonene i oppgaveprompten.
Bruk kun informasjon som er eksplisitt oppgitt eller tydelig støttet av transkriptet.
Ikke legg til, anta eller overtolk opplysninger.
Hvis informasjon er uklar eller mangler i transkriptet, skal du ikke gjette.
Prioriter korrekthet og klinisk relevans fremfor å få med mest mulig."""

        self.assertEqual(GENERATION_SYSTEM_PROMPT, expected)

    def test_generation_rows_keep_aci_bench_metadata(self) -> None:
        conversations = [
            {
                "conversation_id": "test1:aci_asrcorr:0-aci",
                "transcript_text": "[doctor] hello",
                "split": "test1",
                "transcript_variant": "aci_asrcorr",
                "source_id": "0-aci",
            }
        ]
        strategies = [
            {
                "id": "01_zero_shot_minimal_baseline",
                "prompt_file": "prompts/01_zero_shot_minimal_baseline.txt",
                "temperature": 0.0,
            }
        ]

        rows = build_summary_rows(
            run_id="run-1",
            conversations=conversations,
            strategies=strategies,
            model="test-model",
            summarize_fn=lambda transcript, strategy, model: f"{model}:{strategy['id']}:{transcript}",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["conversation_id"], "test1:aci_asrcorr:0-aci")
        self.assertEqual(rows[0]["split"], "test1")
        self.assertEqual(rows[0]["transcript_variant"], "aci_asrcorr")
        self.assertEqual(rows[0]["source_id"], "0-aci")
        self.assertEqual(rows[0]["summary"], "test-model:01_zero_shot_minimal_baseline:[doctor] hello")

    def test_ensemble_strategy_generates_three_candidates_and_keeps_selected_summary(self) -> None:
        conversations = [
            {
                "conversation_id": "test1:aci_asrcorr:0-aci",
                "transcript_text": "[doctor] hello",
                "split": "test1",
                "transcript_variant": "aci_asrcorr",
                "source_id": "0-aci",
            }
        ]
        strategies = [
            {
                "id": "08_two_shot_decomposition_self_criticism_ensemble",
                "prompt_file": "prompts/07_two_shot_decomposition_self_criticism.txt",
                "temperature": 0.0,
                "ensemble": {
                    "enabled": True,
                    "n": 3,
                    "candidate_prompt_file": "prompts/07_two_shot_decomposition_self_criticism.txt",
                    "selection_prompt_file": "prompts/08_two_shot_decomposition_self_criticism_ensemble.txt",
                },
            }
        ]
        calls: list[str] = []

        def summarize_stub(transcript, strategy, model):
            calls.append(strategy["prompt_file"])
            return f"candidate-{len(calls)}"

        rows = build_summary_rows(
            run_id="run-1",
            conversations=conversations,
            strategies=strategies,
            model="test-model",
            summarize_fn=summarize_stub,
            selection_fn=lambda transcript, candidates, strategy, model: {
                "selected_candidate": 2,
                "selection_status": "ok",
                "selection_attempts": 1,
            },
        )

        self.assertEqual(calls, ["prompts/07_two_shot_decomposition_self_criticism.txt"] * 3)
        self.assertEqual(rows[0]["summary"], "candidate-2")
        self.assertEqual(rows[0]["selection_prompt_file"], "prompts/08_two_shot_decomposition_self_criticism_ensemble.txt")
        self.assertEqual(rows[0]["ensemble"]["enabled"], True)
        self.assertEqual(rows[0]["ensemble"]["n"], 3)
        self.assertEqual(rows[0]["ensemble"]["selected_candidate"], 2)
        self.assertEqual(rows[0]["ensemble"]["selection_status"], "ok")
        self.assertEqual(rows[0]["ensemble"]["selection_attempts"], 1)
        self.assertEqual(
            rows[0]["ensemble"]["candidates"],
            [
                {"candidate_id": 1, "summary": "candidate-1"},
                {"candidate_id": 2, "summary": "candidate-2"},
                {"candidate_id": 3, "summary": "candidate-3"},
            ],
        )

    def test_ensemble_selection_failure_does_not_pick_arbitrary_candidate(self) -> None:
        conversations = [
            {
                "conversation_id": "test1:aci_asrcorr:0-aci",
                "transcript_text": "[doctor] hello",
            }
        ]
        strategies = [
            {
                "id": "08_two_shot_decomposition_self_criticism_ensemble",
                "prompt_file": "prompts/07_two_shot_decomposition_self_criticism.txt",
                "ensemble": {"enabled": True, "n": 3},
            }
        ]

        rows = build_summary_rows(
            run_id="run-1",
            conversations=conversations,
            strategies=strategies,
            model="test-model",
            summarize_fn=lambda transcript, strategy, model: "candidate",
            selection_fn=lambda transcript, candidates, strategy, model: {
                "selected_candidate": None,
                "selection_status": "failed",
                "selection_attempts": 2,
            },
        )

        self.assertIsNone(rows[0]["summary"])
        self.assertIsNone(rows[0]["ensemble"]["selected_candidate"])
        self.assertEqual(rows[0]["ensemble"]["selection_status"], "failed")
        self.assertEqual(rows[0]["ensemble"]["selection_attempts"], 2)

    def test_selection_response_parser_requires_fixed_format(self) -> None:
        self.assertEqual(parse_selected_candidate("VALGT_KANDIDAT: 3"), 3)
        self.assertEqual(parse_selected_candidate("  VALGT_KANDIDAT: 1  "), 1)
        self.assertIsNone(parse_selected_candidate("3"))
        self.assertIsNone(parse_selected_candidate("VALGT_KANDIDAT: 4"))
        self.assertIsNone(parse_selected_candidate("VALGT_KANDIDAT: 2\nbegrunnelse"))

    def test_selection_prompt_injects_transcript_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "selection.txt"
            prompt_path.write_text(
                json.dumps({
                    "t": "{TRANSCRIPT}",
                    "c1": "{CANDIDATE_1}",
                    "c2": "{CANDIDATE_2}",
                    "c3": "{CANDIDATE_3}",
                }),
                encoding="utf-8",
            )

            prompt = build_selection_prompt(
                transcript="[doctor] hello",
                candidates=["one", "two", "three"],
                selection_prompt_file=str(prompt_path),
            )

        self.assertIn("[doctor] hello", prompt)
        self.assertIn("one", prompt)
        self.assertIn("two", prompt)
        self.assertIn("three", prompt)

    def test_judge_infers_splits_from_summary_metadata(self) -> None:
        summaries = [
            {"conversation_id": "valid:aci_asrcorr:0-aci", "split": "valid"},
            {"conversation_id": "test1:aci_asrcorr:0-aci", "split": "test1"},
            {"conversation_id": "test1:aci_asrcorr:1-aci", "split": "test1"},
        ]

        self.assertEqual(infer_splits_from_summaries(summaries), ("test1", "valid"))

    def test_transcript_lookup_uses_same_conversation_id_model(self) -> None:
        conversations = [
            {
                "conversation_id": "test1:aci_asrcorr:0-aci",
                "transcript_text": "[doctor] hello",
                "split": "test1",
            }
        ]

        lookup = build_transcript_lookup(conversations)

        self.assertEqual(lookup["test1:aci_asrcorr:0-aci"]["transcript_text"], "[doctor] hello")

    def test_judge_resume_reads_existing_judgement_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judgements.jsonl"
            rows = [
                {
                    "conversation_id": "test3:aci_asrcorr:0-aci",
                    "strategy_id": "01_zero_shot_minimal_baseline",
                },
                {
                    "conversation_id": "test3:aci_asrcorr:0-aci",
                    "strategy_id": "02_zero_shot_structured_instruction",
                },
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            keys = read_existing_judgement_keys(path)

        self.assertEqual(
            keys,
            {
                ("test3:aci_asrcorr:0-aci", "01_zero_shot_minimal_baseline"),
                ("test3:aci_asrcorr:0-aci", "02_zero_shot_structured_instruction"),
            },
        )


if __name__ == "__main__":
    unittest.main()
