from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import modal
from modal.exception import (
    NotFoundError,
    OutputExpiredError,
)
from modal.exception import TimeoutError as ModalTimeoutError

STATE_PATH = Path("state") / "submissions.json"


def _record_call_id(config: str, function_call_id: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state[config] = {
        "function_call_id": function_call_id,
        "submitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _saved_call_id(config: str) -> str | None:
    if not STATE_PATH.exists():
        return None
    entry = json.loads(STATE_PATH.read_text(encoding="utf-8")).get(config)
    if isinstance(entry, dict):
        value = entry.get("function_call_id")
        if isinstance(value, str):
            return value
    return None


def submit(config: str) -> dict[str, str]:
    train = modal.Function.from_name("ncp-smol", "train")
    call = train.spawn(config)
    _record_call_id(config, call.object_id)
    return {"status": "submitted", "config": config, "function_call_id": call.object_id}


def _call_status(call_id: str) -> dict[str, object]:
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except ModalTimeoutError:
        return {"status": "running", "function_call_id": call_id}
    except OutputExpiredError:
        return {
            "status": "complete_result_expired",
            "function_call_id": call_id,
            "hint": "the call finished and its stored result expired; check run outputs instead",
        }
    except NotFoundError:
        return {"status": "not_found", "function_call_id": call_id}
    except modal.exception.ExecutionError as error:
        return {
            "status": "failed",
            "function_call_id": call_id,
            "error": str(error),
        }
    return {"status": "complete", "function_call_id": call_id, "result": result}


def _call_logs(call_id: str, tail: int) -> dict[str, object]:
    call = modal.FunctionCall.from_id(call_id)
    lines: list[str] = []
    try:
        for item in call.logs.tail(entries=max(tail, 1)):
            message = getattr(item, "message", None)
            if isinstance(message, str) and message.strip():
                lines.extend(message.splitlines())
    except NotFoundError:
        return {"status": "not_found", "function_call_id": call_id}
    return {"function_call_id": call_id, "log_tail": lines[-tail:]}


def status(call_id: str) -> dict[str, object]:
    return _call_status(call_id)


def logs(call_id: str, tail: int = 40) -> dict[str, object]:
    return _call_logs(call_id, tail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit or inspect a deployed Modal run")
    commands = parser.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("config")
    status_parser = commands.add_parser("status")
    status_parser.add_argument("function_call_id", nargs="?")
    status_parser.add_argument("--config", help="look up the saved call id for this config")
    logs_parser = commands.add_parser("logs")
    logs_parser.add_argument("function_call_id", nargs="?")
    logs_parser.add_argument("--config", help="look up the saved call id for this config")
    logs_parser.add_argument("--tail", type=int, default=40)
    args = parser.parse_args()

    if args.command == "submit":
        result = submit(args.config)
    else:
        call_id = args.function_call_id or (
            _saved_call_id(args.config) if args.config else None
        )
        if not call_id:
            parser.error("provide function_call_id or --config with a saved submission")
        if args.command == "status":
            result = _call_status(call_id)
        else:
            result = _call_logs(call_id, args.tail)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
