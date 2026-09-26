from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def progress_from_files(run_dir: str | Path, *, max_steps: int | None = None) -> dict[str, object]:
    """Summarize a run directory: resume marker, trainer state, and billable cost."""
    root = Path(run_dir)
    experiment = _read_json(root / "experiment.json") or {}
    marker = _read_json(root / "latest.json") or {}
    checkpoint_name = marker.get("checkpoint")
    checkpoint_dir = root / checkpoint_name if isinstance(checkpoint_name, str) else root
    state = _read_json(checkpoint_dir / "trainer_state.json")
    hourly_cost = float(experiment.get("train", {}).get("hourly_cost_usd", 0.0) or 0.0)

    payload: dict[str, object] = {
        "run_dir": str(root),
        "checkpoint": checkpoint_name,
    }
    if state is None:
        payload.update({"status": "no_checkpoint", "step": 0, "tokens_seen": 0})
        return payload

    billable_seconds = float(state.get("billable_seconds", 0.0) or 0.0)
    payload.update(
        {
            "status": "checkpointed",
            "step": int(state.get("step", 0)),
            "tokens_seen": int(state.get("tokens_seen", 0)),
            "estimated_cost_usd": billable_seconds / 3600 * hourly_cost,
        }
    )
    if max_steps is not None:
        payload["max_steps"] = max_steps
        payload["finished"] = int(state.get("step", 0)) >= max_steps
    return payload
