# Standalone training and MLflow

The original 33-cell assignment notebooks and the shipped `artifacts/` baseline
are preserved. The reusable pipeline trains fresh weights; it never loads the
shipped model as initialization.

## Install and train

Use Python 3.10+ with a compatible PyTorch installation. Keep Colab's GPU-enabled
PyTorch; do not install the CPU wheel there.

```bash
python -m pip install -e ".[train,tracking]"
python train.py --config configs/constant_3e4.json --data runs/prepared --output runs/single --device cuda
```

Equivalent entry points are `qwen3-train` and `python -m qwen3_translation.train`.
Use `--device cpu` for small debugging configurations. `--no-mlflow` allows training
with the `train` extra alone. Inference still needs only the base package.

All data selection, model, optimization and stopping settings are explicit JSON.
The dataset revision is pinned in `data.py`. Preparation saves SHA256 hashes for
the tokenizer and every encoded split. Reuse requires identical settings and
intact bytes. A partial or mismatched dataset is rejected. Training and suite
output directories must be empty to protect prior results.

Each run writes `best.pt`, its exact `tokenizer.json`, `config.json`, data manifest,
source hashes, package/device versions, per-epoch history, summary and validation
translations. `best.pt` works with the existing `qwen3-translate --artifacts ...`.
There is no optimizer-resume feature; an interrupted run needs a new output path.

## Controlled policy comparison

```bash
qwen3-experiments --configs configs/constant_3e4.json configs/constant_1e4.json configs/warmup_cosine_3e4.json --data runs/prepared --output runs/comparison --device cuda
```

The three configurations use the same sampled data, tokenizer, seed, fresh model
initialization, shuffle generator, batch size, epoch budget and early stopping.
Only the learning-rate policy changes. The warmup+cosine policy linearly warms up
for 10% of the maximum optimizer-update budget and decays toward 10% of the base
rate. This tests a combined policy; it cannot isolate the contribution of warmup.
AMP-overflow skipped updates do not advance the learning-rate schedule.

Every new minimum validation target-token loss is checkpointed. An improvement
smaller than `min_delta` still saves a checkpoint, but does not reset patience.
The comparison first writes a validation-only selection decision, then evaluates
the selected run on test. Validation BLEU is reported for each run; it does not
change the predeclared selection metric. Test is never used to choose settings.

The original assignment's test predictions were already inspected, so this is
not a newly unseen benchmark. One seed does not establish statistical robustness.
The refactored loader starts from a fresh generator rather than consuming the
original notebook's preview batch: use the new constant-policy run for matched
comparisons rather than attributing differences from the old baseline to policy.

Outputs include `comparison.json`, CSV, loss curves, test translations and a
portable `mlflow_bundle/`. The `test_evaluated: false` field on individual training
summaries means training did not access test; the suite's separate final-evaluation
run contains the selected model's test metrics.

## Inspect MLflow

MLflow 3.18.0 is an optional dependency. The default backend is a local SQLite
database, `sqlite:///mlflow.db`; experiment artifacts live under `mlartifacts/`
beside that database. No external tracking service or account is required.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. Select the experiment and runs, compare validation
loss/BLEU, then inspect parameters, metric histories and artifacts. For a different
database, pass the same URI to training and UI. These generated stores are ignored
by Git.

The run records configuration, dataset/tokenizer/source hashes, selected epoch,
loss, learning rate, elapsed time, update counts and validation BLEU. Artifact
logging stores checkpoint and tokenizer bytes directly, avoiding an unnecessary
MLflow model flavor or serving layer for this custom architecture.

## Move Colab runs to your computer

Copying a SQLite database alone leaves machine-specific artifact paths. The
suite exports JSON run metadata, every metric-history point and the actual
artifact files with relative paths. Restore the extracted bundle into your own
local store:

```bash
python -m qwen3_translation.tracking restore --bundle runs/comparison/mlflow_bundle --tracking-uri sqlite:///mlflow.db --experiment qwen3-colab-import
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Restoration creates new local run IDs and keeps `original_run_id` tags, original
timestamps and step numbers. Repeated restoration intentionally creates another
copy; import once per desired copy. MLflow's normal UI can inspect restored runs.
Tests restore after deleting the original source store and compare artifact bytes.

## Why no Airflow or ClearML deployment?

MLflow addresses experiment comparison and artifact traceability here. ClearML
could be an alternative tracker/remote-task system. Running both would duplicate
the current tracking role. Airflow would be useful once there is an actual
scheduled ingestion, retraining and promotion pipeline with dependency/retry
requirements. This project currently uses explicit local/Colab experiments and
GitHub Actions for CPU checks, so no scheduler service is deployed.
