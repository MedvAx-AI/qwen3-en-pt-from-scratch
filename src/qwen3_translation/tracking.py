"""Optional MLflow tracking and portable JSON/artifact export.

The bundle contains run metadata and complete metric histories, never a tracking
SQLite database or its machine-specific artifact locations. Importing this
module does not require MLflow; install the project's ``tracking`` extra to use it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import time
from typing import Iterator
import uuid


class _TrackedRun:
    def __init__(self, client, run_id: str):
        self.client = client
        self.run_id = run_id

    def log_metrics(self, metrics: dict, step: int) -> None:
        timestamp = time.time_ns() // 1_000_000
        for key, value in metrics.items():
            self.client.log_metric(self.run_id, key, float(value), timestamp=timestamp,
                                   step=step, synchronous=True)

    def log_artifacts(self, directory: Path | str) -> None:
        self.client.log_artifacts(self.run_id, str(directory))

    def set_tags(self, tags: dict) -> None:
        for key, value in tags.items():
            self.client.set_tag(self.run_id, key, str(value))


class ExperimentTracker:
    """A client-scoped tracker; does not modify MLflow's global active run."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        try:
            from mlflow import MlflowClient
        except ModuleNotFoundError as error:
            if error.name != "mlflow":
                raise
            raise RuntimeError("MLflow tracking requires: pip install -e '.[train,tracking]'") from error

        self.client = MlflowClient(tracking_uri=tracking_uri)
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            self.experiment_id = experiment.experiment_id
        else:
            artifact_location = None
            if tracking_uri.startswith("sqlite:///"):
                database = tracking_uri[len("sqlite:///"):].split("?", 1)[0]
                artifact_location = (Path(database).resolve().parent / "mlartifacts" /
                                     uuid.uuid4().hex).as_uri()
            self.experiment_id = self.client.create_experiment(
                experiment_name, artifact_location=artifact_location
            )

    @contextmanager
    def start_run(self, run_name: str, params: dict, tags: dict | None = None) -> Iterator[_TrackedRun]:
        run = self.client.create_run(self.experiment_id, run_name=run_name)
        tracked = _TrackedRun(self.client, run.info.run_id)
        try:
            for key, value in params.items():
                self.client.log_param(tracked.run_id, key, value)
            tracked.set_tags(tags or {})
            yield tracked
        except BaseException:
            self.client.set_terminated(tracked.run_id, status="FAILED")
            raise
        else:
            self.client.set_terminated(tracked.run_id, status="FINISHED")

    def export_runs(self, run_ids: list[str], destination: Path) -> Path:
        """Export to a new or empty directory, writing bundle.json last."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError(f"Export destination must be empty: {destination}")
        manifests = []
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            relative = PurePosixPath("runs", run_id, "run.json")
            run_directory = destination / relative.parent
            artifacts = run_directory / "artifacts"
            artifacts.mkdir(parents=True)
            for artifact in self.client.list_artifacts(run_id):
                self.client.download_artifacts(run_id, artifact.path, str(artifacts.resolve()))
            metrics = [
                {"key": key, "value": metric.value, "step": metric.step,
                 "timestamp": metric.timestamp}
                for key in sorted(run.data.metrics)
                for metric in self.client.get_metric_history(run_id, key)
            ]
            metadata = {
                "run_id": run_id, "run_name": run.info.run_name,
                "status": run.info.status, "start_time": run.info.start_time,
                "end_time": run.info.end_time, "params": run.data.params,
                "tags": run.data.tags, "metrics": metrics,
            }
            (destination / relative).write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            manifests.append(relative.as_posix())
        (destination / "bundle.json").write_text(
            json.dumps({"format_version": 1, "runs": manifests}, indent=2) + "\n", encoding="utf-8"
        )
        return destination


def _bundle_path(bundle: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    resolved = (bundle / path).resolve()
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(bundle.resolve()):
        raise ValueError(f"Bundle paths must stay inside the bundle: {relative}")
    return resolved


def restore_bundle(bundle: Path, tracking_uri: str, experiment_name: str) -> list[str]:
    """Recreate exported runs in another store, preserving timestamps and history."""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported MLflow bundle format_version")
    # Validate all metadata before opening a destination store or creating runs.
    runs = []
    for relative in manifest["runs"]:
        metadata_path = _bundle_path(bundle, relative)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifacts = _bundle_path(bundle, (PurePosixPath(relative).parent / "artifacts").as_posix())
        if not artifacts.is_dir():
            raise ValueError(f"Missing artifacts directory: {artifacts}")
        runs.append((metadata, artifacts))
    tracker = ExperimentTracker(tracking_uri, experiment_name)
    restored_ids = []
    for metadata, artifacts in runs:
        tags = {**metadata["tags"], "original_run_id": metadata["run_id"]}
        run = tracker.client.create_run(tracker.experiment_id, start_time=metadata["start_time"],
                                        run_name=metadata["run_name"], tags=tags)
        run_id = run.info.run_id
        try:
            for key, value in metadata["params"].items():
                tracker.client.log_param(run_id, key, value)
            for metric in metadata["metrics"]:
                tracker.client.log_metric(run_id, metric["key"], metric["value"],
                                          timestamp=metric["timestamp"], step=metric["step"],
                                          synchronous=True)
            tracker.client.log_artifacts(run_id, str(artifacts))
            tracker.client.set_terminated(run_id, status=metadata["status"],
                                         end_time=metadata["end_time"])
        except BaseException:
            tracker.client.set_terminated(run_id, status="FAILED")
            raise
        restored_ids.append(run_id)
    return restored_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    restore = subcommands.add_parser("restore", help="Restore a portable MLflow bundle")
    restore.add_argument("--bundle", type=Path, required=True)
    restore.add_argument("--tracking-uri", required=True)
    restore.add_argument("--experiment", required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(restore_bundle(arguments.bundle, arguments.tracking_uri, arguments.experiment)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
