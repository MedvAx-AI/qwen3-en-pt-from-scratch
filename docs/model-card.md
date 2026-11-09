# Model card

## Identity and intended use

Tiny Qwen3 Dense-style English-to-Portuguese decoder-only language model,
version 0.1.0, trained from random initialization. Intended for learning,
architecture inspection, reproducibility exercises and portfolio demonstrations.
This card describes the original checkpoint shipped in `artifacts/`.
See the [training guide](training.md) for subsequent runs and selection criteria.
It is not an official Qwen checkpoint and is not suitable for reliable translation
of consequential content. The architecture follows the assignment's scaled-down
Qwen3 Dense contract; the weights and tokenizer are specific to this experiment.

## Architecture

5,065,216 unique parameters; 4 independent decoder layers; hidden width 256;
MLP width 768; 8 query heads; 2 KV heads; head dimension 32; context 256.
Per-head Q/K RMSNorm precedes split-half RoPE (`theta=1e6`). Sublayers are
pre-normalized with RMSNorm (`eps=1e-6`), with residual additions and final
RMSNorm. Attention and SwiGLU projections have no bias. The LM head and shared
8,000-token embedding use the same parameter. Dropout is zero.

Attention scores, grouping, masks, norms and blocks are explicit PyTorch tensor
operations. KV heads are repeated for clarity, so attention does not realize all
possible memory savings of a specialized GQA kernel. Generation recomputes the
entire prefix; there is no KV cache, encoder, cross-attention or expert routing.

## Training data and preprocessing

OPUS-100 en-pt, revision `805090dc28bf78897da9641cdf08b61287580df9`.
Official source split sizes are 1,000,000 train and 2,000 each validation/test.
The scaffold samples up to 50,000/1,000/1,000 with seed 42 and excludes normalized
exact holdout sentences from training. Empty/duplicate/overlength examples are
filtered. Retained counts: 49,567/984/991. English and Portuguese are each limited
to 96 tokens. Near-duplicates and noisy alignments can remain.

The byte-level BPE tokenizer is trained on training text only, with both languages
sharing one vocabulary. Local special tokens are `<pad>`, `<unk>`, `<bos>`,
`<eos>`, `<en>`, `<pt>`. The training sequence is
`<bos> <en> source <pt> target <eos>`. One external next-token shift is applied.
Only target/EOS labels are supervised; prompts and padding use ignore index -100.

## Optimization and checkpoint selection

Seed 42; AdamW lr 0.0003, weight decay 0.01; batch 32; clipping norm 1.
Normal initialization std 0.02 and unit RMSNorm scales. T4 training used CUDA
float16 autocast and gradient scaling, with sensitive reductions in float32.
Budget: 10 epochs maximum, patience 2, minimum validation improvement 0.001.
Every new minimum validation loss is saved. Actual run: 9 epochs, 13,941 batches,
565.1007 seconds of training. Best epoch: 7. The test model was selected only
using validation loss.

The recorded run was executed on 24 October 2026 using Python 3.13.16,
PyTorch 2.11.0+cu130, datasets 5.1.0, tokenizers 0.22.2, sacrebleu 2.6.0,
tqdm 4.67.3 and matplotlib 3.10.0. Full metadata are in
[`reports/run_metadata.json`](../reports/run_metadata.json).

## Evaluation

| Metric | Value |
| --- | ---: |
| Selected epoch validation loss | 3.2558591275 |
| Test target-token loss | 3.3053258282 |
| Test target-token perplexity | 27.2574213786 |
| Autoregressive corpus BLEU | 9.5189958463 |

Evaluation population: the 991 retained test examples. Loss averages target/EOS
tokens equally. Perplexity is exp(loss). Generation receives English only, uses
greedy argmax and ends at EOS or 97 new tokens. Corpus BLEU uses a single aligned
reference stream; it is not teacher-forced argmax scoring.

SacreBLEU signature:
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`.

## Observed limitations

- Poor vocabulary/idiom handling and partial source copying.
- Omissions and invented content in longer inputs.
- Repetition/degeneration, especially in the saved date/time example.
- Validation overfitting after epoch 7 despite falling training loss.
- Domain mixtures, Portuguese regional variation and noisy alignments.
- One reference can penalize valid paraphrases; BLEU does not verify meaning.
- Scores apply to the declared subset and evaluation protocol. Comparisons with
  other tokenizers or pretrained systems require a common evaluation design.

Surface diagnostics: 0 empty hypotheses, 17 normalized exact reference matches,
22 immediately repeated word-trigram cases. The repetition heuristic misses
some character/subword loops; none of these counts is semantic accuracy.

The CLI deliberately retains greedy decoding and numerical policy from the
experiment. It adds rejection of empty/overlength input and invalid generation
budgets. It does not add beam search, special-token suppression or a quality fix.

## Artifacts and checks

The selected checkpoint is [`artifacts/best.pt`](../artifacts/best.pt); tokenizer
is [`artifacts/tokenizer.json`](../artifacts/tokenizer.json). File hashes are in
[`artifacts/checksums.json`](../artifacts/checksums.json). Loading checks the
tokenizer hash recorded inside the checkpoint and uses `weights_only=True`.
The architecture source is extracted unchanged from the executed notebook.
CPU verification reproduces a saved generated translation and exercises masks,
grouping, shape flexibility, tying, gradients and learning.

GitHub Actions verifies CPU inference and tests; it does not rerun the full
GPU training experiment on every commit. Training reproducibility is supported
by the saved notebook, pinned corpus revision, tokenizer hash, metadata and outputs;
bitwise agreement across hardware is not promised.

## Provenance and rights

See [NOTICE.md](../NOTICE.md). The assignment scaffold, referenced papers,
dependencies and corpus are not claimed as original work by the repository author.
