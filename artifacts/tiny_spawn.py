"""Spawn a durable modal_tiny train/eval call and record its id (git-ignored state)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

STATE_PATH = Path("state") / "submissions.json"


def main() -> None:
    function, key = sys.argv[1], sys.argv[2]
    raw_args = sys.argv[3:]
    # argv values are strings; pass through ints where the function expects them.
    args = [int(value) if value.isdigit() else value for value in raw_args]
    fn = modal.Function.from_name("ncp-smol-tiny", function)
    call = fn.spawn(*args)
    state: dict[str, object] = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state[key] = {
        "function_call_id": call.object_id,
        "function": function,
        "args": args,
        "submitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "submitted", key: call.object_id}, indent=2))


if __name__ == "__main__":
    main()
