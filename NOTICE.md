# Provenance and rights

This project originated from **PMLDL Assignment 2 (2026): Qwen3 Dense
Decoder-Only Transformer**. The notebooks retain the original assignment's
33-cell structure and its instructional context.

The course scaffold supplied the architecture contract, configuration, OPUS-100
preprocessing, tokenizer setup, batching, basic training/early-stopping loop,
greedy generation and evaluation structure. The completed work implements the
missing model components and adds checks, mixed precision, reproducibility
metadata, artifact exports, analysis and a reusable inference package.
The repository does not claim the supplied scaffold or architectural ideas as
an original research contribution.

Architecture references consulted:

- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).
- [Official Qwen3 release](https://qwenlm.github.io/blog/qwen3/).
- [Qwen3-0.6B configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json).
- [Transformers v4.51.3 Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)
  for convention checks; ready-made model layers are not imported.

Data source: [Helsinki-NLP/OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100),
English–Portuguese subset at the revision recorded in `reports/run_metadata.json`.
Saved generated translations and their corpus references are included as experiment
evidence. This repository does not relicense the source corpus, course materials,
papers or dependencies, and does not redistribute the full training corpus.

No blanket redistribution license is currently declared for this repository.
Public availability should not be interpreted as granting new rights over
third-party materials. The rights and licenses of those materials remain with
their respective owners. This project is not affiliated with or endorsed by Qwen.
