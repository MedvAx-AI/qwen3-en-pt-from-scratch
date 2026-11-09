# Next experiments

The repository packages the original notebook baseline and a standalone pipeline
for matched learning-rate comparisons. The items below extend that work.

## 1. Compare architectures on the same split

Train a small encoder–decoder baseline with the same tokenizer, filtered splits,
sampling seed and a documented parameter/training budget. Compare validation
loss, autoregressive BLEU, parameter count and generation time. A comparison
with a pretrained translator would be a separate experiment with a different
training-data advantage, not a controlled architecture ablation.

**Evidence needed:** runnable configuration, selected checkpoints, matched
evaluation protocol, learning curves and measured results for both systems.

## 2. Check robustness and improve generalization

The original validation curve shows overfitting after epoch 7. Repeat the
learning-rate comparison with additional seeds, then try more training pairs
or documented alignment and duplicate cleaning. Use validation to select each change. An ablation
that removes QK-Norm can test its effect, but it would no longer meet the original
Qwen3 architecture contract and must be labeled as a separate experimental model.

**Evidence needed:** a small experiment table, validation-selected configuration,
data-retention counts and a clear explanation of both successful and failed trials.

## 3. Measure decoding and error tradeoffs

Compare greedy decoding with a small beam search selected on validation data.
Measure translation quality, omissions, repetitions and latency, including
short/long sentence groups. Add KV caching only after measuring prefix-recomputation
cost, and test cached logits against full-prefix recomputation.

The current test outputs have already been inspected. Do not repeatedly tune
beam widths, cleaning or training settings to this test score. Freeze the chosen
protocol using validation, and disclose any additional held-out evaluation set.

## 4. Optional interactive demonstration

After the core experiments, a small Gradio interface could accept an English
sentence, show generated Portuguese and display the input/output limits and known
failure examples. A UI demonstrates usability; it does not establish better
translation quality. A polished CLI and the current checkpoint already allow a
working portfolio demonstration without another service or API key.
