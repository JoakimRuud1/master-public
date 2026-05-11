import json
import tempfile
import unittest
from pathlib import Path

from src.data_loader import (
    ACI_BENCH_TRANSCRIPT_VARIANT,
    EXCLUDED_EVALUATION_CONVERSATION_IDS,
    load_aci_bench_evaluation_dataset,
    load_aci_bench_main_analysis_dataset,
    load_aci_bench_split,
    make_conversation_id,
)


class AciBenchDataLoaderTests(unittest.TestCase):
    def write_split_file(self, data_dir: Path, split: str, files: list[str]) -> None:
        rows = [
            {
                "file": file_id,
                "src": f"[doctor] transcript for {file_id}",
                "tgt": f"Reference note for {file_id}",
            }
            for file_id in files
        ]
        (data_dir / f"{split}_aci_asrcorr.json").write_text(
            json.dumps({"data": rows}),
            encoding="utf-8",
        )

    def test_make_conversation_id_includes_split_variant_and_source_id(self) -> None:
        self.assertEqual(
            make_conversation_id(
                split="test1",
                transcript_variant=ACI_BENCH_TRANSCRIPT_VARIANT,
                source_id="0-aci",
            ),
            "test1:aci_asrcorr:0-aci",
        )

    def test_load_aci_bench_split_maps_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            path = data_dir / "valid_aci_asrcorr.json"
            path.write_text(
                json.dumps(
                    {
                        "data": [
                            {
                                "file": "0-aci",
                                "src": "[doctor] hello",
                                "tgt": "CHIEF COMPLAINT\nExample",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = load_aci_bench_split("valid", data_dir=data_dir)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["conversation_id"], "valid:aci_asrcorr:0-aci")
        self.assertEqual(rows[0]["transcript_text"], "[doctor] hello")
        self.assertEqual(rows[0]["split"], "valid")
        self.assertEqual(rows[0]["transcript_variant"], "aci_asrcorr")
        self.assertEqual(rows[0]["source_id"], "0-aci")
        self.assertFalse(rows[0]["reserved_for_prompt_examples"])
        self.assertFalse(rows[0]["exclude_from_evaluation"])
        self.assertEqual(rows[0]["reference_note_text"], "CHIEF COMPLAINT\nExample")

    def test_train_prompt_examples_are_marked_but_not_removed_from_raw_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            self.write_split_file(data_dir, "train", ["0-aci", "1-aci", "2-aci"])

            rows = load_aci_bench_split("train", data_dir=data_dir)

        self.assertEqual(len(rows), 3)
        flagged_ids = {
            row["conversation_id"]
            for row in rows
            if row["exclude_from_evaluation"]
        }
        self.assertEqual(flagged_ids, EXCLUDED_EVALUATION_CONVERSATION_IDS)

    def test_evaluation_loading_excludes_reserved_train_prompt_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            self.write_split_file(data_dir, "train", ["0-aci", "1-aci", "2-aci"])

            rows = load_aci_bench_evaluation_dataset(splits=("train",), data_dir=data_dir)

        self.assertEqual([row["conversation_id"] for row in rows], ["train:aci_asrcorr:2-aci"])

    def test_main_analysis_dataset_uses_test1_without_optional_test_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            self.write_split_file(data_dir, "test1", ["0-aci"])
            self.write_split_file(data_dir, "test2", ["0-aci"])

            rows = load_aci_bench_main_analysis_dataset(data_dir=data_dir)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["conversation_id"], "test1:aci_asrcorr:0-aci")


if __name__ == "__main__":
    unittest.main()
