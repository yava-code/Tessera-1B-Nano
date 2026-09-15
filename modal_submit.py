from __future__ import annotations

import argparse
import json

import modal


def submit(config: str) -> dict[str, str]:
    train = modal.Function.from_name("ncp-smol", "train")
    call = train.spawn(config)
    return {"status": "submitted", "config": config, "function_call_id": call.object_id}


def status(call_id: str) -> dict[str, object]:
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        return {"status": "running", "function_call_id": call_id}
    return {"status": "complete", "function_call_id": call_id, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit or inspect a deployed Modal run")
    commands = parser.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("config")
    status_parser = commands.add_parser("status")
    status_parser.add_argument("function_call_id")
    args = parser.parse_args()

    result = submit(args.config) if args.command == "submit" else status(args.function_call_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
