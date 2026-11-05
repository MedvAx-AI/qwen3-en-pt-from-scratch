"""Network-free checks for the reproducible training corpus protocol."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer


def row(en, pt):
    return {"translation": {"en": en, "pt": pt}}


def raw_splits():
    return {
        "train": [row("train alpha", "treino alfa"), row("train beta", "treino beta"),
                  row(" TRAIN  ALPHA ", "TREINO ALFA"), row("", "vazio"),
                  row("ＨＯＬＤＯＵＴ  TWO", "unrelated"), row("new text", "RESERVADO DOIS"),
                  row("test clean", "unrelated target")],
        "validation": [row("holdout one", "reservado um"), row("holdout two", "reservado dois")],
        "test": [row("test clean", "teste limpo"), row("other source", "reservado um"),
                 row("holdout two", "test overlap")],
    }


class DataTests(unittest.TestCase):
    def prepare(self, directory, splits=None, **overrides):
        from qwen3_translation.data import prepare_from_splits
        settings = dict(train_size=2, val_size=1, test_size=1, seed=42,
                        vocab_size=300, max_source_length=96, max_target_length=96)
        settings.update(overrides)
        return prepare_from_splits(raw_splits() if splits is None else splits,
                                   Path(directory), **settings)

    def test_full_official_holdouts_exclude_source_or_target_before_sampling(self):
        from qwen3_translation.data import load_records
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.prepare(directory)
            train = load_records(directory, "train")
            self.assertEqual({" ".join(r["source"].casefold().split()) for r in train},
                             {"train alpha", "train beta"})
            self.assertEqual(load_records(directory, "test")[0]["source"], "test clean")
            self.assertEqual(manifest["sampled_counts"], {"train": 2, "validation": 1, "test": 1})
            self.assertEqual(manifest["source_split_sizes"], {"train": 7, "validation": 2, "test": 3})

    def test_training_target_overlap_is_excluded_even_when_source_is_new(self):
        from qwen3_translation.data import load_records
        splits = raw_splits()
        splits["train"] = [row("alpha", "alfa"), row("new English", "RESERVADO DOIS"),
                           row("beta", "beta")]
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory, splits)
            self.assertEqual({r["source"] for r in load_records(directory, "train")},
                             {"alpha", "beta"})

    def test_holdout_text_never_changes_training_only_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a", Path(directory) / "b"
            self.prepare(first)
            splits = raw_splits()
            splits["validation"].append(row("HOLDOUTNEVER " * 3, "SOHOLDOUT " * 3))
            self.prepare(second, splits)
            self.assertEqual((first / "tokenizer.json").read_bytes(),
                             (second / "tokenizer.json").read_bytes())

    def test_target_only_labels_and_collation_keep_shift_external(self):
        from qwen3_translation.data import collate_batch, load_records
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
            example = load_records(directory, "train")[0]
            prompt = [tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<en>")]
            prompt += tokenizer.encode(example["source"]).ids + [tokenizer.token_to_id("<pt>")]
            target = tokenizer.encode(example["target"]).ids + [tokenizer.token_to_id("<eos>")]
            self.assertEqual(example["input_ids"], prompt + target)
            self.assertEqual(example["labels"], [-100] * len(prompt) + target)
            short = {"input_ids": [2, 4, 5, 3], "labels": [-100, -100, -100, 3]}
            batch = collate_batch([example, short], tokenizer.token_to_id("<pad>"))
            self.assertEqual(batch["input_ids"][0].tolist(), example["input_ids"])
            self.assertEqual(batch["labels"][0].tolist(), example["labels"])
            self.assertTrue(torch.equal(batch["attention_mask"][1, :4], torch.ones(4, dtype=torch.long)))
            self.assertTrue((batch["attention_mask"][1, 4:] == 0).all())
            self.assertTrue((batch["labels"][1, 4:] == -100).all())
            self.assertTrue((batch["input_ids"][1, 4:] == 0).all())

    def test_length_filter_discards_complete_pairs_without_truncation(self):
        from qwen3_translation.data import load_records
        splits = raw_splits()
        splits["train"].append(row("a " * 300, "long source"))
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.prepare(directory, splits, train_size=3)
            self.assertEqual(manifest["sampled_counts"]["train"], 3)
            self.assertEqual(manifest["retained_counts"]["train"], 2)
            self.assertEqual(len(load_records(directory, "train")), 2)

    def test_reuse_requires_identical_config_and_unmodified_files(self):
        from qwen3_translation.data import load_prepared_data
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.prepare(directory)
            self.assertEqual(self.prepare(directory), manifest)
            self.assertEqual(load_prepared_data(directory, expected_config=manifest["config"]), manifest)
            with self.assertRaisesRegex(ValueError, "config|setting"):
                self.prepare(directory, seed=99)
            path = Path(directory) / "train.json"
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256|hash"):
                load_prepared_data(directory)
            self.assertEqual(path.read_bytes(), original + b" ")

    def test_partial_directory_and_modified_protocol_are_rejected(self):
        from qwen3_translation.data import load_prepared_data
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "partial|manifest"):
                self.prepare(directory)
        with tempfile.TemporaryDirectory() as directory:
            self.prepare(directory)
            path = Path(directory) / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["protocol"]["tokenizer_training"] = "all splits"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "protocol"):
                load_prepared_data(directory)

    def test_invalid_sizes_and_empty_post_filter_splits_fail_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data"
            for override in ({"train_size": 0}, {"vocab_size": 20}, {"seed": True},
                             {"max_source_length": -1}):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    self.prepare(output, **override)
            with self.assertRaisesRegex(ValueError, "empty|retained"):
                self.prepare(output, max_source_length=1, max_target_length=1)
            self.assertFalse(output.exists())
