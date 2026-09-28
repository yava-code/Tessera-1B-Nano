"""Spawn a durable publish_checkpoint call and record its id (git-ignored state)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

STATE_PATH = Path("state") / "submissions.json"


def main() -> None:
    config_name, eval_json, repo_id = sys.argv[1], sys.argv[2], sys.argv[3]
    key = f"publish:{config_name}"
    fn = modal.Function.from_name("ncp-smol-publish", "publish_checkpoint")
    call = fn.spawn(config_name, "latest", eval_json, repo_id, False)
    state: dict[str, object] = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state[key] = {
        "function_call_id": call.object_id,
        "repo_id": repo_id,
        "submitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "submitted", key: call.object_id}, indent=2))


if __name__ == "__main__":
    main()
