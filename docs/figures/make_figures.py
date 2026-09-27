"""Render paper figures from the FineWeb-Edu metrics logs.

Reads the committed metrics.jsonl files of both arms and writes PNG figures next to
this script. Matplotlib is intentionally not a project dependency, so run it in an
ephemeral environment:

    uv run --no-project --with matplotlib python docs/figures/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"


def read_eval_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "eval":
            records.append(record)
    records.sort(key=lambda r: r["tokens"])
    return records


def ntp_loss(record: dict[str, Any]) -> float:
    """Held-out NTP loss; the NTP-only arm logs it under the plain ``loss`` key."""
    return float(record.get("ntp_loss", record["loss"]))


def render_codebook(records: list[dict[str, Any]], path: Path) -> None:
    x = [r["tokens"] / 1e9 for r in records]
    fig, ax_ppl = plt.subplots(figsize=(7.0, 4.2))
    ax_use = ax_ppl.twinx()

    ax_ppl.plot(x, [r["codebook_perplexity"] for r in records], "o-", color="C0")
    ax_use.plot(x, [100 * r["codebook_usage"] for r in records], "s--", color="C1")

    # Reference: the low-entropy shortcut the TinyStories overfit gate fell into.
    ax_ppl.axhline(2.4, color="gray", lw=1.0, ls=":")
    ax_ppl.text(0.13, 2.47, "TinyStories gate shortcut (~2.4)", color="gray", fontsize=8)
    ax_use.axhspan(31, 41, color="gray", alpha=0.12, lw=0)
    ax_use.text(0.60, 33.5, "gate shortcut usage range", color="gray", fontsize=8)

    ax_ppl.set_xlabel("Tokens (billions)")
    ax_ppl.set_ylabel("Codebook effective perplexity")
    ax_use.set_ylabel("Codebook usage (%)")
    ax_ppl.set_ylim(0, 8.5)
    ax_use.set_ylim(0, 100)

    handles = ax_ppl.get_lines()[:1] + ax_use.get_lines()[:1]
    ax_ppl.legend(handles, ["Effective perplexity", "Usage"], loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_dynamics(
    records_ncp: list[dict[str, Any]],
    records_ntp: list[dict[str, Any]],
    path: Path,
) -> None:
    fig, (ax_ntp, ax_aux) = plt.subplots(1, 2, figsize=(10.0, 4.2))

    ax_ntp.plot(
        [r["tokens"] / 1e9 for r in records_ntp],
        [ntp_loss(r) for r in records_ntp],
        "o-",
        color="0.45",
        label="NTP-only",
    )
    ax_ntp.plot(
        [r["tokens"] / 1e9 for r in records_ncp],
        [r["ntp_loss"] for r in records_ncp],
        "s-",
        color="C0",
        label="NTP+NCP",
    )
    ax_ntp.set_xlabel("Tokens (billions)")
    ax_ntp.set_ylabel("Held-out NTP loss (nats)")
    ax_ntp.legend(loc="upper right")
    ax_ntp.set_title("(a) Token objective: neutral", fontsize=10)

    ax_aux.plot(
        [r["tokens"] / 1e9 for r in records_ncp],
        [r["ncp_loss"] for r in records_ncp],
        "o-",
        color="C0",
        label="Held-out NCP loss",
    )
    ax_aux.plot(
        [r["tokens"] / 1e9 for r in records_ncp],
        [r["vq_loss"] for r in records_ncp],
        "s--",
        color="C1",
        label="Held-out VQ loss",
    )
    ax_aux.set_xlabel("Tokens (billions)")
    ax_aux.set_ylabel("Held-out auxiliary loss (nats)")
    ax_aux.legend(loc="upper right")
    ax_aux.set_title("(b) Concept objectives: falling", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records_ncp = read_eval_records(ROOT / "results" / "fineweb-edu-ncp" / "metrics.jsonl")
    records_ntp = read_eval_records(ROOT / "results" / "fineweb-edu-ntp" / "metrics.jsonl")
    if not records_ncp or not records_ntp:
        raise SystemExit("no eval records found in one of the metrics logs")
    if "codebook_perplexity" not in records_ncp[0]:
        raise SystemExit("NCP-arm eval records lack codebook statistics")

    render_codebook(records_ncp, OUT / "codebook-growth.png")
    render_dynamics(records_ncp, records_ntp, OUT / "held-out-dynamics.png")
    print(f"wrote {(OUT / 'codebook-growth.png').relative_to(ROOT)}")
    print(f"wrote {(OUT / 'held-out-dynamics.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
