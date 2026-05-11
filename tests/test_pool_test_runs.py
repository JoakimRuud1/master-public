import json
import tempfile
import unittest
from pathlib import Path

from src.pool_test_runs import pool_test_runs


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def row(run_id: str, split: str, conversation_no: int, strategy_id: str = "s1") -> dict:
    return {
        "run_id": run_id,
        "conversation_id": f"{split}:aci_asrcorr:{conversation_no}-aci",
        "strategy_id": strategy_id,
        "split": split,
        "transcript_variant": "aci_asrcorr",
        "source_id": f"{conversation_no}-aci",
    }


class PoolTestRunsTests(unittest.TestCase):
    def test_pool_rewrites_run_id_preserves_source_run_id_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dirs = []
            for idx, split in enumerate(("test1", "test2", "test3"), start=1):
                source_dir = root / f"{split}_run"
                source_dir.mkdir()
                source_dirs.append(source_dir)
                rows = [row(f"old-{split}", split, idx)]
                write_jsonl(source_dir / "summaries.jsonl", rows)
                write_jsonl(source_dir / "judgements.jsonl", rows)

            out_dir = root / "test_all"
            manifest = pool_test_runs(source_dirs, out_dir, "test_all_20260428")

            summaries = [json.loads(line) for line in (out_dir / "summaries.jsonl").read_text(encoding="utf-8").splitlines()]
            judgements = [json.loads(line) for line in (out_dir / "judgements.jsonl").read_text(encoding="utf-8").splitlines()]
            written_manifest = json.loads((out_dir / "pooled_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(len(summaries), 3)
            self.assertEqual(len(judgements), 3)
            self.assertTrue(all(item["run_id"] == "test_all_20260428" for item in summaries))
            self.assertEqual({item["source_run_id"] for item in summaries}, {"old-test1", "old-test2", "old-test3"})
            self.assertEqual(manifest["number_of_unique_conversations"], 3)
            self.assertEqual(manifest["number_of_strategies"], 1)
            self.assertEqual(manifest["splits_included"], ["test1", "test2", "test3"])
            self.assertEqual(written_manifest["pooled_run_id"], "test_all_20260428")
            self.assertEqual(len(written_manifest["warnings"]), 2)

    def test_pool_rejects_duplicate_conversation_strategy_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dirs = []
            for idx, split in enumerate(("test1", "test2", "test3"), start=1):
                source_dir = root / f"{split}_run"
                source_dir.mkdir()
                source_dirs.append(source_dir)
                rows = [row(f"old-{split}", split, idx)]
                if split == "test1":
                    rows.append(row(f"old-{split}", split, idx))
                write_jsonl(source_dir / "summaries.jsonl", rows)
                write_jsonl(source_dir / "judgements.jsonl", rows[:1])

            with self.assertRaisesRegex(ValueError, "Duplicate summaries.jsonl keys"):
                pool_test_runs(source_dirs, root / "out", "test_all_20260428")


if __name__ == "__main__":
    unittest.main()
