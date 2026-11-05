"""Matched training-policy comparison; validation selection precedes test access."""
import argparse
import csv
import json
import math
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

from .data import collate_batch, load_prepared_data, load_records, prepare_data
from .inference import load_model
from .train import TrainConfig, evaluate_generation, evaluate_loss, train, write_json


def select_winner(runs):
    if not runs or any(not math.isfinite(run["best_validation_loss"]) for run in runs):
        raise ValueError("Selection requires nonempty, finite validation results")
    return min(runs, key=lambda run: run["best_validation_loss"])


def plot_comparison(output_dir, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in names:
        rows = json.loads((output_dir / name / "history.json").read_text(encoding="utf-8"))
        for ax, key in zip(axes, ("train_loss", "validation_loss")):
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o", label=name)
            ax.set(xlabel="Epoch", ylabel="Target-token cross-entropy", title=key.replace("_", " ").capitalize())
            ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_comparison.png", dpi=160)
    plt.close(fig)


def run_suite(config_paths, data_dir, output_dir, *, device="auto", tracking_uri="sqlite:///mlflow.db",
              experiment_name="qwen3-policy-comparison"):
    paths = [Path(p) for p in config_paths]
    if len(paths) < 2:
        raise ValueError("A comparison requires at least two configurations")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Configurations require unique file stems")
    configs = [TrainConfig.from_file(p) for p in paths]
    if any(c.data_settings() != configs[0].data_settings() or c.model != configs[0].model or
           c.batch_size != configs[0].batch_size or c.max_epochs != configs[0].max_epochs or
           c.patience != configs[0].patience or c.min_delta != configs[0].min_delta or
           c.weight_decay != configs[0].weight_decay or c.gradient_clip_norm != configs[0].gradient_clip_norm
           for c in configs[1:]):
        raise ValueError("Controlled policy comparison requires the same data, seed, model and training budget")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Suite output directory must be empty")
    prepare_data(data_dir, **configs[0].data_settings())
    from .tracking import ExperimentTracker
    tracker = ExperimentTracker(tracking_uri, experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for path, config in zip(paths, configs):
        runs.append(train(config, data_dir, output_dir / path.stem, device=device,
                          tracker=tracker, run_name=path.stem))
    winner = select_winner(runs)
    selected_dir = output_dir / winner["run_name"]
    # Persist the decision before reading the test split. No settings change after this point.
    write_json(output_dir / "selection.json", {"metric": "minimum validation target-token loss",
               "selected_run": winner["run_name"], "selected_run_id": winner["mlflow_run_id"], "runs": runs})
    manifest = load_prepared_data(data_dir)
    test_records = load_records(data_dir, "test")
    selected_config = configs[[p.stem for p in paths].index(winner["run_name"])]
    test_dir = output_dir / "final_evaluation"
    test_dir.mkdir()
    with tracker.start_run("final-test", {"selected_run": winner["run_name"],
                "selection_metric": "minimum validation target-token loss"},
                {"selected_run_id": winner["mlflow_run_id"], "split": "test"}) as evaluation:
        model, _ = load_model(selected_dir, None if device == "auto" else device)
        loader = DataLoader(test_records, batch_size=selected_config.batch_size,
                            collate_fn=partial(collate_batch, pad_id=manifest["special_token_ids"]["<pad>"]))
        test_loss = evaluate_loss(model, loader)
        del model
        generation, predictions = evaluate_generation(selected_dir, test_records, device,
                                                       selected_config.max_target_length + 1)
        test_metrics = {"test_loss": test_loss, "test_perplexity": math.exp(test_loss),
                        "test_bleu": generation["bleu"], "bleu_signature": generation["bleu_signature"],
                        "examples": generation["examples"], "empty_hypotheses": generation["empty_hypotheses"]}
        write_json(test_dir / "metrics.json", test_metrics)
        write_json(test_dir / "translations.json", predictions)
        evaluation.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}, step=0)
        evaluation.log_artifacts(test_dir)
        evaluation_id = evaluation.run_id
    summary = {"protocol": "matched-learning-rate-policies-v1", "selection_metric": "minimum validation target-token loss",
               "runs": runs, "selected_run": winner["run_name"], "final_test": test_metrics,
               "evaluation_run_id": evaluation_id, "dataset_revision": manifest["dataset"]["revision"],
               "retained_counts": manifest["retained_counts"],
               "limitations": ["One seed; no confidence interval or multi-seed robustness claim.",
                               "The original assignment test outputs were already inspected; this is not a newly unseen benchmark.",
                               "Warmup+cosine is one combined policy, not an isolated warmup ablation.",
                               "Refactored loader starts with a fresh shuffle generator; original notebook consumed a preview batch."]}
    write_json(output_dir / "comparison.json", summary)
    columns = ("run_name", "best_validation_loss", "validation_bleu", "selected_epoch", "completed_epochs", "training_seconds")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            writer.writerow({**{k: run[k] for k in columns if k != "validation_bleu"},
                             "validation_bleu": run["validation_generation"]["bleu"]})
    plot_comparison(output_dir, [p.stem for p in paths])
    tracker.export_runs([r["mlflow_run_id"] for r in runs] + [evaluation_id], output_dir / "mlflow_bundle")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("runs/prepared"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="qwen3-policy-comparison")
    args = parser.parse_args(argv)
    run_suite(args.configs, args.data, args.output, device=args.device,
              tracking_uri=args.tracking_uri, experiment_name=args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
