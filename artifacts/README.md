# Shipped baseline artifacts

- `best.pt`: selected epoch-7 PyTorch checkpoint (approximately 20MB), containing
  model state, configuration, validation loss, seed, tokenizer hash and run metadata.
- `tokenizer.json`: exact training-only shared 8k byte-level BPE tokenizer.
- `checksums.json`: SHA256 checksums for both files.

The architecture in `src/qwen3_translation/model.py` uses the original parameter
names and configuration, so the checkpoint can be loaded without conversion.
Inference validates the tokenizer hash against the checkpoint. No pretrained
Qwen weights are included; these weights were trained from random initialization.

`torch.load(..., weights_only=True)` is used by the loader. Keep checkpoint and
tokenizer together; replacing a tokenizer with another 8k vocabulary is not compatible.

The files are tracked directly in Git for a simple offline demo. Future experiment
checkpoints should be published separately rather than accumulating many binary
versions in Git history.
