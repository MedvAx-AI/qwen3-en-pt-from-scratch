"""Seeded target-only training, validation selection and optional MLflow tracking."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import collate_batch, load_prepared_data, load_records, prepare_data
from .model import QwenDenseConfig, QwenDenseForCausalLM


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class TrainConfig:
    seed: int = 42
    train_size: int = 50000
    val_size: int = 1000
    test_size: int = 1000
    vocab_size: int = 8000
    max_source_length: int = 96
    max_target_length: int = 96
    batch_size: int = 32
    max_epochs: int = 10
    patience: int = 2
    min_delta: float = 0.001
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    schedule: str = "constant"
    warmup_fraction: float = 0.0
    model: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("train_size", "val_size", "test_size", "vocab_size", "max_source_length",
                     "max_target_length", "batch_size", "max_epochs", "patience"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.vocab_size < 262:
            raise ValueError("Byte-level BPE requires vocab_size >= 262")
        if self.max_source_length > 96:
            raise ValueError("max_source_length must be <= 96, matching the translation interface")
        for name in ("learning_rate", "gradient_clip_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("weight_decay", "min_delta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.schedule not in ("constant", "warmup_cosine"):
            raise ValueError("schedule must be constant or warmup_cosine")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.schedule == "constant" and self.warmup_fraction != 0:
            raise ValueError("constant schedule requires warmup_fraction=0")
        if "vocab_size" in self.model:
            raise ValueError("Set vocab_size at the top level; the actual tokenizer determines model vocabulary")
        allowed = set(QwenDenseConfig.__dataclass_fields__) - {"vocab_size"}
        if set(self.model) - allowed:
            raise ValueError(f"Unknown model settings: {set(self.model) - allowed}")
        values = asdict(QwenDenseConfig(vocab_size=self.vocab_size)) | self.model
        for name in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                     "num_key_value_heads", "head_dim", "max_position_embeddings"):
            if not isinstance(values[name], int) or values[name] <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if (values["num_attention_heads"] % values["num_key_value_heads"] or
                values["num_key_value_heads"] >= values["num_attention_heads"] or values["head_dim"] % 2):
            raise ValueError("Model requires even head_dim and divisible grouped-query heads")
        for name in ("rms_norm_eps", "rope_theta"):
            if not math.isfinite(values[name]) or values[name] <= 0:
                raise ValueError(f"model.{name} must be positive and finite")
        if self.max_source_length + self.max_target_length + 4 > values["max_position_embeddings"]:
            raise ValueError("Source/target limits exceed the model context")

    def data_settings(self):
        return {name: getattr(self, name) for name in ("train_size", "val_size", "test_size", "seed",
                "vocab_size", "max_source_length", "max_target_length")}

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class EarlyStopping:
    """Save every new minimum; only a substantial improvement resets patience."""
    def __init__(self, patience, min_delta):
        self.best = math.inf
        self.stale = 0
        self.patience, self.min_delta = patience, min_delta

    def update(self, loss):
        improved = loss < self.best
        significant = loss < self.best - self.min_delta
        if improved:
            self.best = loss
        self.stale = 0 if significant else self.stale + 1
        return improved, self.stale >= self.patience


def learning_rate_multiplier(step, total_steps, schedule, warmup_fraction):
    """Multiplier for the next zero-based optimizer update, bounded above zero."""
    if schedule == "constant":
        return 1.0
    warmup = math.ceil(total_steps * warmup_fraction)
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def precision_context(model):
    return (torch.autocast("cuda", dtype=torch.float16)
            if next(model.parameters()).device.type == "cuda" else nullcontext())


def batch_loss(model, batch):
    """Shift exactly once here: prompt/padding labels remain ignored."""
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with precision_context(model):
        logits = model(batch["input_ids"][:, :-1], attention_mask=batch["attention_mask"][:, :-1])
        labels = batch["labels"][:, 1:]
        count = int((labels != -100).sum())
        if not count:
            raise ValueError("Batch has no supervised target tokens")
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                               ignore_index=-100, reduction="sum")
    return loss, count


@torch.inference_mode()
def evaluate_loss(model, loader):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        loss, tokens = batch_loss(model, batch)
        total_loss += float(loss)
        total_tokens += tokens
    if not total_tokens:
        raise ValueError("Evaluation requires nonempty target tokens")
    return total_loss / total_tokens


def code_provenance():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src/qwen3_translation").glob("*.py"))
    paths += sorted((root / "configs").glob("*.json")) + [root / "pyproject.toml"]
    hashes = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if p.is_file()}
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": source_hash, "files": hashes}


def environment():
    import platform

    versions = {}
    for name in ("torch", "numpy", "tokenizers", "datasets", "sacrebleu", "mlflow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "versions": versions,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"}


@torch.inference_mode()
def evaluate_generation(artifact_dir, records, device="auto", max_new_tokens=97):
    from sacrebleu.metrics import BLEU
    from .inference import load_model, translate

    model, tokenizer = load_model(artifact_dir, None if device == "auto" else device)
    predictions = [{"source": row["source"], "reference": row["target"],
                    "hypothesis": translate(model, tokenizer, row["source"], max_new_tokens)} for row in records]
    scorer = BLEU()
    score = scorer.corpus_score([row["hypothesis"] for row in predictions],
                                [[row["reference"] for row in predictions]])
    return {"bleu": score.score, "bleu_signature": str(scorer.get_signature()),
            "examples": len(predictions), "empty_hypotheses": sum(not r["hypothesis"].strip() for r in predictions)}, predictions


def train(config, data_dir, output_dir, *, device="auto", tracker=None, run_name="training", validation_bleu=True):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    manifest = load_prepared_data(data_dir, expected_config=config.data_settings())
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Training output directory must be empty; choose a new run directory")
    selected = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select CPU or use a Colab GPU runtime")
    if selected.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if selected.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data_dir / "tokenizer.json", output_dir / "tokenizer.json")
    write_json(output_dir / "config.json", asdict(config))
    write_json(output_dir / "data_manifest.json", manifest)
    provenance = code_provenance()
    write_json(output_dir / "provenance.json", provenance)
    write_json(output_dir / "environment.json", environment())
    pad_id = manifest["special_token_ids"]["<pad>"]
    collate = partial(collate_batch, pad_id=pad_id)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(load_records(data_dir, "train"), batch_size=config.batch_size, shuffle=True,
                              generator=generator, collate_fn=collate)
    val_records = load_records(data_dir, "validation")
    val_loader = DataLoader(val_records, batch_size=config.batch_size, collate_fn=collate)
    model_config = QwenDenseConfig(vocab_size=manifest["vocab_size"], **config.model)
    model = QwenDenseForCausalLM(model_config).to(selected)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=selected.type == "cuda")
    total_steps = len(train_loader) * config.max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_multiplier(
        step, total_steps, config.schedule, config.warmup_fraction))
    stopper = EarlyStopping(config.patience, config.min_delta)
    params = asdict(config) | {"model": asdict(model_config), "parameter_count": sum(p.numel() for p in model.parameters()),
                              "tokenizer_sha256": manifest["files"]["tokenizer.json"],
                              "data_manifest_sha256": hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()}
    tags = {"source_sha256": provenance["source_sha256"], "git_commit": provenance["git_commit"],
            "git_dirty": str(provenance["git_dirty"]), "evaluation_protocol": "select-by-validation-loss-v1"}
    context = tracker.start_run(run_name, params, tags) if tracker else nullcontext(None)
    history, step, skipped, selected_epoch = [], 0, 0, None
    started = time.perf_counter()
    with context as run:
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            total_loss, total_tokens = 0.0, 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss_sum, tokens = batch_loss(model, batch)
                loss = loss_sum / tokens
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    step += 1
                else:
                    skipped += 1
                total_loss += float(loss_sum.detach())
                total_tokens += tokens
            validation_loss = evaluate_loss(model, val_loader)
            if not math.isfinite(validation_loss):
                raise RuntimeError("Non-finite validation loss")
            row = {"epoch": epoch, "train_loss": total_loss / total_tokens, "validation_loss": validation_loss,
                   "learning_rate": optimizer.param_groups[0]["lr"], "optimizer_steps": step,
                   "skipped_updates": skipped, "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            write_json(output_dir / "history.json", history)
            print(json.dumps({"run": run_name, **row}), flush=True)
            if run:
                run.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            improved, stop = stopper.update(validation_loss)
            if improved:
                selected_epoch = epoch
                torch.save({"model": model.state_dict(), "config": asdict(model_config), "epoch": epoch,
                            "val_loss": validation_loss, "seed": config.seed,
                            "tokenizer_sha256": manifest["files"]["tokenizer.json"],
                            "metadata": {"source_sha256": provenance["source_sha256"]}}, output_dir / "best.pt")
            if stop:
                break
        summary = {"run_name": run_name, "mlflow_run_id": run.run_id if run else None,
                   "best_validation_loss": stopper.best, "selected_epoch": selected_epoch,
                   "completed_epochs": len(history), "training_seconds": time.perf_counter() - started,
                   "optimizer_steps": step, "skipped_updates": skipped,
                   "parameter_count": params["parameter_count"], "selection_metric": "validation_loss",
                   "test_evaluated": False}
        del optimizer, scheduler, model
        if selected.type == "cuda":
            torch.cuda.empty_cache()
        if validation_bleu:
            metrics, predictions = evaluate_generation(output_dir, val_records, str(selected), config.max_target_length + 1)
            summary["validation_generation"] = metrics
            write_json(output_dir / "validation_translations.json", predictions)
            if run:
                run.log_metrics({"validation_bleu": metrics["bleu"]}, step=len(history))
        write_json(output_dir / "summary.json", summary)
        if run:
            run.log_metrics({"best_validation_loss": stopper.best, "selected_epoch": selected_epoch,
                             "training_seconds": summary["training_seconds"]}, step=len(history))
            run.log_artifacts(str(output_dir))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("runs/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="qwen3-en-pt")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    config = TrainConfig.from_file(args.config)
    prepare_data(args.data, **config.data_settings())
    if args.no_mlflow:
        tracker = None
    else:
        from .tracking import ExperimentTracker
        tracker = ExperimentTracker(args.tracking_uri, args.experiment)
    summary = train(config, args.data, args.output, device=args.device, tracker=tracker, run_name=args.config.stem)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
