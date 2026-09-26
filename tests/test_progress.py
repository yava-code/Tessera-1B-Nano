import json
from pathlib import Path

from ncp_smol.progress import progress_from_files


def test_progress_from_files_reports_checkpoint_state(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "demo"
    checkpoint = run / "step-00000010"
    checkpoint.mkdir(parents=True)
    (run / "experiment.json").write_text(
        json.dumps({"train": {"hourly_cost_usd": 2.74}}),
        encoding="utf-8",
    )
    (run / "latest.json").write_text(
        json.dumps({"checkpoint": "step-00000010"}),
        encoding="utf-8",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"step": 10, "tokens_seen": 1310720, "billable_seconds": 3600.0}),
        encoding="utf-8",
    )

    payload = progress_from_files(run, max_steps=100)

    assert payload["status"] == "checkpointed"
    assert payload["checkpoint"] == "step-00000010"
    assert payload["step"] == 10
    assert payload["tokens_seen"] == 1_310_720
    assert payload["estimated_cost_usd"] == 2.74
    assert payload["max_steps"] == 100
    assert payload["finished"] is False


def test_progress_from_files_handles_missing_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "empty"
    run.mkdir(parents=True)

    payload = progress_from_files(run)

    assert payload["status"] == "no_checkpoint"
    assert payload["step"] == 0
    assert payload["tokens_seen"] == 0
    assert "estimated_cost_usd" not in payload


def test_progress_from_files_marks_finished_runs(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "done"
    checkpoint = run / "step-00000004"
    checkpoint.mkdir(parents=True)
    (run / "latest.json").write_text(
        json.dumps({"checkpoint": "step-00000004"}),
        encoding="utf-8",
    )
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"step": 4, "tokens_seen": 524288, "billable_seconds": 100.0}),
        encoding="utf-8",
    )

    payload = progress_from_files(run, max_steps=4)

    assert payload["status"] == "checkpointed"
    assert payload["finished"] is True
