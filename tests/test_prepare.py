from pathlib import Path

import numpy as np
import pytest

from ncp_smol.prepare import _write_tokens


def broken_stream():
    yield 1
    yield 2
    raise RuntimeError("stream failed")


def test_interrupted_write_keeps_previous_cache(tmp_path: Path) -> None:
    target = tmp_path / "train.bin"
    previous = np.array([9, 9, 9], dtype=np.uint32).tobytes()
    target.write_bytes(previous)

    with pytest.raises(RuntimeError, match="stream failed"):
        _write_tokens(target, iter(broken_stream()), count=8)

    assert target.read_bytes() == previous
    assert target.with_suffix(".bin.partial").exists()


def test_complete_write_atomically_replaces_cache(tmp_path: Path) -> None:
    target = tmp_path / "train.bin"
    target.write_bytes(b"old")

    written = _write_tokens(target, iter(range(8)), count=8)

    assert written == 8
    assert np.fromfile(target, dtype=np.uint32).tolist() == list(range(8))
    assert not target.with_suffix(".bin.partial").exists()
