"""Exercise the shipped checkpoint, tokenizer and command-line interface."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qwen3_translation.inference import load_model

        torch.set_num_threads(4)
        cls.model, cls.tokenizer = load_model(ROOT / "artifacts", device="cpu")

    def test_saved_checkpoint_reproduces_recorded_translation(self):
        from qwen3_translation.inference import translate

        self.assertEqual(translate(self.model, self.tokenizer, "You can't tell what's inside!"),
                         "Não pode dizer o que está dentro!")
        self.assertFalse(self.model.training)
        self.assertIs(self.model.lm_head.weight, self.model.embed_tokens.weight)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 5_065_216)

    def test_invalid_inputs_and_context_budget_are_rejected(self):
        from qwen3_translation.inference import translate

        for text, budget in [("   ", 97), ("Hello", 0), ("Hello", -1),
                             ("Hello", 257), ("hello " * 200, 97)]:
            with self.subTest(text=text[:20], budget=budget):
                with self.assertRaises(ValueError):
                    translate(self.model, self.tokenizer, text, max_new_tokens=budget)

    def test_loading_rejects_a_different_tokenizer(self):
        from qwen3_translation.inference import load_model

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            shutil.copy2(ROOT / "artifacts" / "best.pt", destination / "best.pt")
            raw = (ROOT / "artifacts" / "tokenizer.json").read_bytes()
            (destination / "tokenizer.json").write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                load_model(destination, device="cpu")

    def test_cli_uses_the_shipped_artifacts(self):
        from qwen3_translation.inference import main

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = main(["You can't tell what's inside!", "--artifacts",
                           str(ROOT / "artifacts"), "--device", "cpu"])
        self.assertEqual(result, 0)
        self.assertEqual(captured.getvalue().strip(), "Não pode dizer o que está dentro!")


class ExperimentEvidenceTests(unittest.TestCase):
    def test_notebook_and_report_agree_on_the_completed_experiment(self):
        notebook = json.loads((ROOT / "notebooks" / "experiment.ipynb").read_text(encoding="utf-8"))
        metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
        counts = [c["execution_count"] for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(counts, list(range(1, 15)))
        self.assertEqual(len(notebook["cells"]), 33)
        output = "".join("".join(o.get("text", [])) for o in notebook["cells"][31]["outputs"])
        self.assertIn(str(metrics["autoregressive_corpus_bleu"]), output)
        self.assertIn(str(metrics["test_loss"]), output)
        self.assertTrue(any("image/png" in o.get("data", {})
                            for o in notebook["cells"][29]["outputs"]))


if __name__ == "__main__":
    unittest.main()
