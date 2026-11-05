"""Round-trip real MLflow metadata and artifacts without a tracking server."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


class OptionalDependencyTests(unittest.TestCase):
    def test_import_does_not_load_mlflow(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import qwen3_translation.tracking; "
             "assert 'mlflow' not in sys.modules"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(importlib.util.find_spec("mlflow"), "install the train extra for MLflow tests")
class TrackingTests(unittest.TestCase):
    def setUp(self):
        from qwen3_translation.tracking import ExperimentTracker

        environment = patch.dict(os.environ, {"MLFLOW_DISABLE_TELEMETRY": "true",
                                               "MLFLOW_DISABLE_AGENT_HINT": "1",
                                               "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR": "false"})
        environment.start()
        self.addCleanup(environment.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.close_databases)
        self.source = self.root / "colab"
        self.source.mkdir()
        self.tracker = ExperimentTracker(
            "sqlite:///" + (self.source / "tracking.db").as_posix(), "translation"
        )

    def close_databases(self):
        # MLflow caches SQLite connection pools, which keep Windows files open.
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

        for engine in SqlAlchemyStore._engine_map.values():
            database = engine.url.database
            if database and Path(database).resolve().is_relative_to(self.root):
                engine.dispose()

    def test_bundle_restores_history_artifacts_and_metadata_after_source_is_removed(self):
        from mlflow import MlflowClient
        from qwen3_translation.tracking import restore_bundle

        artifacts = self.source / "outputs"
        (artifacts / "nested").mkdir(parents=True)
        payload = bytes(range(256)) + b"\x00checkpoint\xff"
        (artifacts / "nested" / "weights.pt").write_bytes(payload)
        with self.tracker.start_run("tiny-baseline", {"lr": 0.001, "seed": 7},
                                    tags={"device": "cpu"}) as run:
            run.log_metrics({"loss": 2.5, "bleu": 0.0}, step=0)
            run.log_metrics({"loss": 1.25}, step=10)
            run.set_tags({"checkpoint": "best"})
            run.log_artifacts(artifacts)
            run_id = run.run_id
        original = self.tracker.client.get_run(run_id)
        self.assertEqual(original.info.status, "FINISHED")
        original_history = self.tracker.client.get_metric_history(run_id, "loss")
        bundle = self.tracker.export_runs([run_id], self.root / "bundle")
        self.assertEqual(bundle, self.root / "bundle")
        manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["format_version"], 1)
        self.assertNotIn(str(self.source), (bundle / "bundle.json").read_text())
        self.close_databases()
        shutil.rmtree(self.source)
        destination_uri = "sqlite:///" + (self.root / "restored.db").as_posix()
        restored_ids = restore_bundle(bundle, destination_uri, "restored")
        self.assertEqual(len(restored_ids), 1)
        self.assertNotEqual(restored_ids[0], run_id)
        client = MlflowClient(tracking_uri=destination_uri)
        restored = client.get_run(restored_ids[0])
        self.assertEqual(restored.info.status, "FINISHED")
        self.assertEqual(restored.info.start_time, original.info.start_time)
        self.assertEqual(restored.info.end_time, original.info.end_time)
        self.assertEqual(restored.info.run_name, "tiny-baseline")
        self.assertEqual(restored.data.params, {"lr": "0.001", "seed": "7"})
        self.assertEqual(restored.data.tags["original_run_id"], run_id)
        self.assertEqual(restored.data.tags["checkpoint"], "best")
        self.assertEqual(restored.data.tags["device"], "cpu")
        history = client.get_metric_history(restored_ids[0], "loss")
        self.assertEqual([(m.value, m.step, m.timestamp) for m in history],
                         [(m.value, m.step, m.timestamp) for m in original_history])
        self.assertEqual([(m.value, m.step) for m in history], [(2.5, 0), (1.25, 10)])
        self.assertEqual(client.get_run(restored_ids[0]).data.metrics["bleu"], 0.0)
        downloaded = Path(client.download_artifacts(restored_ids[0], "nested/weights.pt"))
        self.assertEqual(downloaded.read_bytes(), payload)

    def test_exception_marks_run_failed_and_restore_preserves_failure(self):
        from qwen3_translation.tracking import ExperimentTracker, restore_bundle

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with self.tracker.start_run("failed", {"seed": 1}) as run:
                run_id = run.run_id
                raise RuntimeError("training failed")
        self.assertEqual(self.tracker.client.get_run(run_id).info.status, "FAILED")
        bundle = self.tracker.export_runs([run_id], self.root / "failed-bundle")
        uri = "sqlite:///" + (self.root / "failed-restored.db").as_posix()
        restored = restore_bundle(bundle, uri, "restored")
        self.assertEqual(ExperimentTracker(uri, "restored").client.get_run(restored[0]).info.status,
                         "FAILED")

    def test_restore_cli_prints_restored_run_ids(self):
        with self.tracker.start_run("cli", {}) as run:
            run_id = run.run_id
        bundle = self.tracker.export_runs([run_id], self.root / "cli-bundle")
        result = subprocess.run(
            [sys.executable, "-m", "qwen3_translation.tracking", "restore", "--bundle", str(bundle),
             "--tracking-uri", "sqlite:///" + (self.root / "cli.db").as_posix(),
             "--experiment", "cli-restored"],
            capture_output=True, text=True, check=False,
            env={**os.environ, "MLFLOW_DISABLE_TELEMETRY": "true"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)), 1)


if __name__ == "__main__":
    unittest.main()
