"""Compare validation results without consulting test during selection."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ExperimentTests(unittest.TestCase):
    def test_real_training_tracking_selection_generation_and_bundle(self):
        import os
        import torch
        os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
        from qwen3_translation.data import prepare_from_splits
        from qwen3_translation.experiments import run_suite

        torch.set_num_threads(4)
        raw = {"train": [{"translation": {"en": f"hi {i}", "pt": f"oi {i}"}} for i in range(8)],
               "validation": [{"translation": {"en": "thank you", "pt": "obrigado"}}],
               "test": [{"translation": {"en": "good day", "pt": "bom dia"}}]}
        settings = {"train_size": 8, "val_size": 1, "test_size": 1, "vocab_size": 280,
                    "max_source_length": 16, "max_target_length": 16,
                    "batch_size": 4, "max_epochs": 2,
                    "model": {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                              "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name, lr in (("a", 0.003), ("b", 0.001)):
                path = root / f"{name}.json"
                path.write_text(json.dumps(settings | {"learning_rate": lr}))
                paths.append(path)
            with patch("qwen3_translation.experiments.prepare_data",
                       side_effect=lambda dest, **kwargs: prepare_from_splits(raw, dest, **kwargs)):
                result = run_suite(paths, root / "data", root / "out", device="cpu",
                                   tracking_uri="sqlite:///" + (root / "tracking.db").as_posix())
            self.assertEqual(len(result["runs"]), 2)
            self.assertTrue(all(not r["test_evaluated"] for r in result["runs"]))
            self.assertEqual(result["final_test"]["examples"], 1)
            bundle = json.loads((root / "out/mlflow_bundle/bundle.json").read_text())
            self.assertEqual(len(bundle["runs"]), 3)
            for path in bundle["runs"][:2]:
                self.assertTrue((root / "out/mlflow_bundle" / Path(path).parent / "artifacts/best.pt").is_file())
            self.assertTrue((root / "out/loss_comparison.png").is_file())
            # MLflow caches SQLite engines; release Windows file handles before cleanup.
            from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
            for engine in SqlAlchemyStore._engine_map.values():
                engine.dispose()

    def test_selection_uses_validation_loss_and_rejects_incomplete_results(self):
        from qwen3_translation.experiments import select_winner

        runs = [{"run_name": "a", "best_validation_loss": 3.2, "validation_generation": {"bleu": 9}},
                {"run_name": "b", "best_validation_loss": 3.1, "validation_generation": {"bleu": 8}}]
        self.assertEqual(select_winner(runs)["run_name"], "b")
        for bad in ([], [{"best_validation_loss": float("nan")}], [{"best_validation_loss": float("inf")} ]):
            with self.assertRaises(ValueError):
                select_winner(bad)

    def test_suite_rejects_unmatched_data_and_conflicting_names_before_preparation(self):
        from qwen3_translation.experiments import run_suite

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.json", root / "b.json"
            a.write_text('{}')
            b.write_text('{"seed": 7}')
            with patch("qwen3_translation.experiments.prepare_data") as prepare:
                with self.assertRaisesRegex(ValueError, "same data"):
                    run_suite([a, b], root / "data", root / "out")
                self.assertFalse(prepare.called)
                with self.assertRaisesRegex(ValueError, "unique"):
                    run_suite([a, a], root / "data", root / "out")


if __name__ == "__main__":
    unittest.main()
