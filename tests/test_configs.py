from dataclasses import asdict
from pathlib import Path

from ncp_smol.experiment import load_experiment

ROOT = Path(__file__).resolve().parent.parent


def test_fineweb_pair_is_token_and_optimizer_matched(monkeypatch) -> None:
    monkeypatch.setenv("NCP_ROOT", str(ROOT))
    ncp = load_experiment(ROOT / "configs" / "fineweb-edu-ncp.yaml")
    ntp = load_experiment(ROOT / "configs" / "fineweb-edu-ntp.yaml")

    assert asdict(ncp.data) == asdict(ntp.data)
    assert asdict(ncp.optim) == asdict(ntp.optim)
    assert ncp.model.base_model == ntp.model.base_model
    assert ncp.model.revision == ntp.model.revision
    assert ncp.run.seed == ntp.run.seed
    assert ncp.tokens_per_step == ntp.tokens_per_step
    assert ncp.max_steps == ntp.max_steps
