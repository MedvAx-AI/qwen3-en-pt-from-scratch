"""The notebook's official-split, leakage-free OPUS100 preprocessing protocol."""
import hashlib
import json
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATASET_NAME = "Helsinki-NLP/opus-100"
DATASET_REVISION = "805090dc28bf78897da9641cdf08b61287580df9"
SPLITS = ("train", "validation", "test")
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>", "<en>", "<pt>")
PROTOCOL = {
    "version": 1,
    "normalization": "NFKC/casefold/collapse-whitespace for deduplication and exclusion; strip stored text",
    "selection": "shuffle each official split with seed; remove empty and duplicate normalized pairs before sampling",
    "train_exclusion": "source OR target in entire official validation OR test",
    "test_exclusion": "source OR target in entire official validation",
    "tokenizer_training": "sampled training source and target only, before length filtering",
    "tokenizer": "byte-level BPE, min_frequency=2, full byte alphabet, add_prefix_space=False",
    "sequence": "BOS EN source PT target EOS; prompt labels -100; external next-token shift",
    "length_filter": "reject whole pairs exceeding source or target token limits; never truncate",
}


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _config(train_size, val_size, test_size, seed, vocab_size, max_source_length, max_target_length):
    config = dict(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed,
                  vocab_size=vocab_size, max_source_length=max_source_length,
                  max_target_length=max_target_length)
    for key, value in config.items():
        minimum = 0 if key == "seed" else 262 if key == "vocab_size" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {key}: expected an integer >= {minimum}.")
    return config


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _existing(output_dir, config):
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Prepared data path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            return load_prepared_data(output_dir, expected_config=config)
    return None


def prepare_data(output_dir, train_size=50_000, val_size=1_000, test_size=1_000,
                 seed=42, vocab_size=8_000, max_source_length=96, max_target_length=96):
    """Download the pinned official corpus, or reuse exactly matching verified data."""
    config = _config(train_size, val_size, test_size, seed, vocab_size,
                     max_source_length, max_target_length)
    output_dir = Path(output_dir)
    existing = _existing(output_dir, config)
    if existing is not None:
        return existing
    from datasets import load_dataset
    raw_splits = load_dataset(DATASET_NAME, "en-pt", revision=DATASET_REVISION)
    return prepare_from_splits(raw_splits, output_dir, **config)


def _clean_pairs(rows, limit, seed, excluded_en, excluded_pt):
    if hasattr(rows, "shuffle"):
        shuffled = rows.shuffle(seed=seed)
    else:
        shuffled = (rows[int(i)] for i in np.random.default_rng(seed).permutation(len(rows)))
    seen, pairs = set(), []
    for row in shuffled:
        en, pt = (row["translation"][lang].strip() for lang in ("en", "pt"))
        en_key, pt_key = normalized_text(en), normalized_text(pt)
        key = (en_key, pt_key)
        if not en_key or not pt_key or key in seen:
            continue
        if en_key in excluded_en or pt_key in excluded_pt:
            continue
        seen.add(key)
        pairs.append((en, pt))
        if len(pairs) == limit:
            break
    if len(pairs) != limit:
        raise ValueError(f"Only {len(pairs)} clean pairs available; requested {limit}.")
    return pairs


def prepare_from_splits(raw_splits, output_dir, train_size=50_000, val_size=1_000,
                        test_size=1_000, seed=42, vocab_size=8_000,
                        max_source_length=96, max_target_length=96):
    """Prepare official-shaped datasets (also accepts lists for network-free checks).

    Callers providing raw_splits must supply the pinned official corpus when
    producing experiment data. Existing output is immutable: config/hash mismatches
    and partial directories fail instead of silently replacing an experiment.
    """
    config = _config(train_size, val_size, test_size, seed, vocab_size,
                     max_source_length, max_target_length)
    output_dir = Path(output_dir)
    existing = _existing(output_dir, config)
    if existing is not None:
        return existing
    if any(split not in raw_splits for split in SPLITS):
        raise ValueError("The official train, validation and test splits are required.")
    val_en, val_pt, holdout_en, holdout_pt = set(), set(), set(), set()
    for split in ("validation", "test"):
        for row in raw_splits[split]:
            en, pt = (normalized_text(row["translation"][lang]) for lang in ("en", "pt"))
            holdout_en.add(en)
            holdout_pt.add(pt)
            if split == "validation":
                val_en.add(en)
                val_pt.add(pt)
    pairs = {
        "train": _clean_pairs(raw_splits["train"], train_size, seed, holdout_en, holdout_pt),
        "validation": _clean_pairs(raw_splits["validation"], val_size, seed, set(), set()),
        "test": _clean_pairs(raw_splits["test"], test_size, seed, val_en, val_pt),
    }
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=list(SPECIAL_TOKENS), show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator((text for pair in pairs["train"] for text in pair), trainer=trainer)
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    encoded = {}
    for split, rows in pairs.items():
        examples = []
        for source, target in rows:
            source_ids, target_ids = tokenizer.encode(source).ids, tokenizer.encode(target).ids
            if not (0 < len(source_ids) <= max_source_length
                    and 0 < len(target_ids) <= max_target_length):
                continue
            prompt = [special_ids["<bos>"], special_ids["<en>"]] + source_ids + [special_ids["<pt>"]]
            continuation = target_ids + [special_ids["<eos>"]]
            examples.append({"input_ids": prompt + continuation,
                             "labels": [-100] * len(prompt) + continuation,
                             "source": source, "target": target})
        if not examples:
            raise ValueError(f"The retained {split} split is empty after token-length filtering.")
        encoded[split] = examples
    manifest = {
        "schema_version": 1,
        "dataset": {"name": DATASET_NAME, "subset": "en-pt", "revision": DATASET_REVISION},
        "protocol": PROTOCOL.copy(), "config": config,
        "source_split_sizes": {split: len(raw_splits[split]) for split in SPLITS},
        "source_fingerprints": {split: getattr(raw_splits[split], "_fingerprint", None) for split in SPLITS},
        "sampled_counts": {split: len(pairs[split]) for split in SPLITS},
        "retained_counts": {split: len(encoded[split]) for split in SPLITS},
        "vocab_size": tokenizer.get_vocab_size(), "special_token_ids": special_ids,
    }
    # Publish only a complete corpus; failed preparation leaves the destination untouched.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        tokenizer.save(str(staging / "tokenizer.json"))
        for split in SPLITS:
            (staging / f"{split}.json").write_text(json.dumps(encoded[split], ensure_ascii=False), encoding="utf-8")
        manifest["files"] = {name: _sha256(staging / name)
                             for name in ("tokenizer.json", *(f"{split}.json" for split in SPLITS))}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if output_dir.exists():
            output_dir.rmdir()  # Only an empty destination can be replaced.
        staging.rename(output_dir)
    return manifest


def load_prepared_data(output_dir, expected_config=None):
    """Validate the protocol, settings, tokenizer, and every prepared file's SHA256."""
    output_dir = Path(output_dir)
    try:
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Missing or invalid prepared data manifest (partial directory): {output_dir}") from error
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Prepared data protocol does not match this training pipeline.")
    if manifest.get("dataset") != {"name": DATASET_NAME, "subset": "en-pt", "revision": DATASET_REVISION}:
        raise ValueError("Prepared data uses a different dataset/revision.")
    try:
        config = _config(**manifest["config"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Prepared data config is invalid.") from error
    if expected_config is not None and config != expected_config:
        raise ValueError("Prepared data config differs from the requested settings; use a new data directory.")
    filenames = {"tokenizer.json", *(f"{split}.json" for split in SPLITS)}
    if set(manifest.get("files", {})) != filenames:
        raise ValueError("Prepared data manifest has missing or unexpected file hashes.")
    for name, expected_hash in manifest["files"].items():
        path = output_dir / name
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"Prepared data SHA256 mismatch or missing file: {name}")
    tokenizer = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if (None in ids.values() or ids != manifest.get("special_token_ids")
            or tokenizer.get_vocab_size() != manifest.get("vocab_size")):
        raise ValueError("Prepared tokenizer metadata does not match the manifest.")
    sizes = dict(zip(SPLITS, (config["train_size"], config["val_size"], config["test_size"])))
    if manifest.get("sampled_counts") != sizes:
        raise ValueError("Prepared sample counts do not match the config.")
    for split in SPLITS:
        retained = manifest.get("retained_counts", {}).get(split)
        if type(retained) is not int or not 0 < retained <= sizes[split]:
            raise ValueError(f"Invalid retained counts for {split}.")
    return manifest


def load_records(output_dir, split):
    """Read a verified split's unshifted training records."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    manifest = load_prepared_data(output_dir)
    records = json.loads((Path(output_dir) / f"{split}.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != manifest["retained_counts"][split]:
        raise ValueError(f"Prepared {split} record count differs from the manifest.")
    return records


def collate_batch(examples, pad_id):
    """Right-pad unshifted sequences; the training loss applies the one-token shift."""
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    length = max(len(example["input_ids"]) for example in examples)
    ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for row, example in enumerate(examples):
        size = len(example["input_ids"])
        if not size or len(example["labels"]) != size:
            raise ValueError("Each example needs equal nonempty input_ids and labels.")
        ids[row, :size] = torch.tensor(example["input_ids"], dtype=torch.long)
        labels[row, :size] = torch.tensor(example["labels"], dtype=torch.long)
        mask[row, :size] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}
