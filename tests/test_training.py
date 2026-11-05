"""Small real training runs verify loss alignment, selection and reload."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_target_loss_shifts_once_and_weights_only_target_tokens(self):
        from qwen3_translation.train import batch_loss

        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.tensor(0.0))
                self.seen = None

            def forward(self, ids, attention_mask):
                self.seen = ids.clone()
                return torch.arange(ids.numel() * 7, dtype=torch.float32).reshape(*ids.shape, 7) / 5 + self.anchor

        model = Fixed()
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 4, 0, 0]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
                 "labels": torch.tensor([[-100, -100, -100, 4, 5], [-100, -100, 4, -100, -100]])}
        loss, count = batch_loss(model, batch)
        self.assertEqual(count, 3)
        torch.testing.assert_close(model.seen, batch["input_ids"][:, :-1])
        logits = model(batch["input_ids"][:, :-1], batch["attention_mask"][:, :-1])
        expected = F.cross_entropy(torch.stack((logits[0, 2], logits[0, 3], logits[1, 1])),
                                   torch.tensor([4, 5, 4]), reduction="sum")
        torch.testing.assert_close(loss, expected)

    def test_schedule_and_early_stopping_boundary(self):
        from qwen3_translation.train import EarlyStopping, learning_rate_multiplier

        stop = EarlyStopping(patience=2, min_delta=0.01)
        self.assertEqual(stop.update(3.0), (True, False))
        # Save every new minimum even when it doesn't reset patience.
        self.assertEqual(stop.update(2.995), (True, False))
        self.assertEqual(stop.update(2.994), (True, True))
        values = [learning_rate_multiplier(i, 10, "warmup_cosine", 0.2) for i in range(10)]
        self.assertAlmostEqual(values[0], 0.5)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertGreater(values[2], values[8])
        self.assertGreater(values[9], 0)
        self.assertEqual(learning_rate_multiplier(8, 10, "constant", 0), 1.0)

    def test_tiny_training_reproducibility_artifacts_and_no_test_evaluation(self):
        from qwen3_translation.data import prepare_from_splits
        from qwen3_translation.inference import load_model
        from qwen3_translation.train import TrainConfig, train

        def rows(pairs):
            return [{"translation": {"en": en, "pt": pt}} for en, pt in pairs]

        splits = {"train": rows([(f"hello friend {i}", f"ola amigo {i}") for i in range(10)]),
                  "validation": rows([("good morning", "bom dia"), ("thank you", "obrigado")]),
                  "test": rows([("good night", "boa noite")])}
        config = TrainConfig(train_size=10, val_size=2, test_size=1, vocab_size=280,
                             batch_size=5, max_epochs=3, patience=3, learning_rate=0.002,
                             model={"hidden_size": 16, "intermediate_size": 32,
                                    "num_hidden_layers": 1, "num_attention_heads": 2,
                                    "num_key_value_heads": 1, "head_dim": 8})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_from_splits(splits, root / "data", **config.data_settings())
            a = train(config, root / "data", root / "a", device="cpu", validation_bleu=False)
            b = train(copy.deepcopy(config), root / "data", root / "b", device="cpu", validation_bleu=False)
            self.assertEqual(a["best_validation_loss"], b["best_validation_loss"])
            self.assertNotIn("test_loss", a)
            history = json.loads((root / "a/history.json").read_text())
            self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
            self.assertEqual(a["selected_epoch"], min(history, key=lambda r: r["validation_loss"])["epoch"])
            self.assertTrue(math.isfinite(a["best_validation_loss"]))
            model, _ = load_model(root / "a", device="cpu")
            self.assertFalse(model.training)
            with self.assertRaisesRegex(FileExistsError, "empty"):
                train(config, root / "data", root / "a", device="cpu", validation_bleu=False)

    def test_bad_configuration_fails_before_training(self):
        from qwen3_translation.train import TrainConfig

        for kwargs in ({"learning_rate": -1}, {"batch_size": 0}, {"schedule": "unknown"},
                       {"warmup_fraction": 1.0}, {"model": {"hidden_size": 0}},
                       {"max_source_length": 1000}, {"max_source_length": 97}, {"vocab_size": 8},
                       {"model": {"rope_theta": 0}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
