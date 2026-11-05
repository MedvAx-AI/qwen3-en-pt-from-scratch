"""Offline checkpoint loading and the experiment's original greedy decoder."""
import argparse
import hashlib
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import QwenDenseConfig, QwenDenseForCausalLM


def load_model(artifact_dir="artifacts", device=None):
    """Return an evaluation-mode model and its exact, hash-checked tokenizer."""
    artifact_dir = Path(artifact_dir)
    for filename in ("best.pt", "tokenizer.json"):
        if not (artifact_dir / filename).is_file():
            raise FileNotFoundError(
                f"Missing {artifact_dir / filename}. Run from the repository root "
                "or set --artifacts to the directory containing both files."
            )
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation; use --device cpu.")
    checkpoint = torch.load(artifact_dir / "best.pt", map_location="cpu", weights_only=True)
    actual_hash = hashlib.sha256((artifact_dir / "tokenizer.json").read_bytes()).hexdigest()
    if checkpoint["tokenizer_sha256"] != actual_hash:
        raise ValueError("The tokenizer does not match this checkpoint (SHA256 mismatch).")
    tokenizer = Tokenizer.from_file(str(artifact_dir / "tokenizer.json"))
    config = QwenDenseConfig(**checkpoint["config"])
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("The tokenizer vocabulary does not match the model configuration.")
    if any(tokenizer.token_to_id(t) is None for t in ("<bos>", "<en>", "<pt>", "<eos>")):
        raise ValueError("The tokenizer is missing the translation boundary tokens.")
    model = QwenDenseForCausalLM(config)
    model.load_state_dict(checkpoint["model"])
    model.to(selected).eval()
    return model, tokenizer


@torch.inference_mode()
def translate(model, tokenizer, english, max_new_tokens=97):
    """Translate English alone; no reference, external service, or KV cache is used.

    Source strings are limited to the experiment's 96 BPE tokens. Output is
    educational baseline quality and can contain omissions or repetition.
    """
    if not isinstance(english, str) or not english.strip():
        raise ValueError("Provide a nonempty English sentence.")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")
    source_ids = tokenizer.encode(english).ids
    if len(source_ids) > 96:
        raise ValueError("The source exceeds the training limit of 96 BPE tokens; use a shorter sentence.")
    bos_id, en_id, pt_id, eos_id = [tokenizer.token_to_id(t)
                                   for t in ("<bos>", "<en>", "<pt>", "<eos>")]
    prompt = [bos_id, en_id] + source_ids + [pt_id]
    if len(prompt) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("The prompt and generation budget exceed the model context window.")
    model.eval()
    selected = next(model.parameters()).device
    ids = torch.tensor([prompt], dtype=torch.long, device=selected)
    generated = []
    for _ in range(max_new_tokens):
        precision = (torch.autocast("cuda", dtype=torch.float16)
                     if selected.type == "cuda" else nullcontext())
        with precision:
            logits = model(ids, attention_mask=torch.ones_like(ids))
        next_id = int(logits[0, -1].argmax().item())
        if next_id == eos_id:
            break
        generated.append(next_id)
        ids = torch.cat((ids, ids.new_tensor([[next_id]])), dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Translate English to Portuguese with a tiny model trained from scratch.")
    parser.add_argument("english", help="English sentence, enclosed in quotes")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Checkpoint/tokenizer directory (default: artifacts)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=97)
    args = parser.parse_args(argv)
    selected = None if args.device == "auto" else args.device
    if selected == "cpu" or (selected is None and not torch.cuda.is_available()):
        # Small matrices: avoid excessive CPU thread overhead in the CLI.
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    try:
        model, tokenizer = load_model(args.artifacts, device=selected)
        print(translate(model, tokenizer, args.english, args.max_new_tokens))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 0
