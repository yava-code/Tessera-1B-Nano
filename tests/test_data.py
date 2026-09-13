from pathlib import Path

import numpy as np

from ncp_smol.data import TokenBatcher, TokenCorpus


def make_corpus(path: Path) -> TokenCorpus:
    np.arange(96, dtype=np.uint32).tofile(path)
    return TokenCorpus(path, sequence_length=8)


def test_batcher_resume_replays_the_same_next_batch(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "tokens.bin")
    first = TokenBatcher(corpus, batch_size=2, seed=19, repeat=True)
    first.next()
    first.next()
    state = first.state_dict()

    resumed = TokenBatcher(corpus, batch_size=2, seed=19, repeat=True, **state)

    assert resumed.next().equal(first.next())


def test_batcher_epoch_order_is_deterministic(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path / "tokens.bin")
    first = TokenBatcher(corpus, batch_size=3, seed=23, repeat=True)
    second = TokenBatcher(corpus, batch_size=3, seed=23, repeat=True)

    for _ in range(6):
        assert first.next().equal(second.next())
