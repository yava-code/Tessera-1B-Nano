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


def test_tinystories_target_ablation_is_matched(monkeypatch) -> None:
    monkeypatch.setenv("NCP_ROOT", str(ROOT))
    continuous = load_experiment(ROOT / "configs" / "tinystories-overfit.yaml")
    quantized = load_experiment(ROOT / "configs" / "tinystories-overfit-quantized.yaml")

    assert asdict(continuous.data) == asdict(quantized.data)
    assert asdict(continuous.optim) == asdict(quantized.optim)
    assert asdict(continuous.train) == asdict(quantized.train)
    assert continuous.run.seed == quantized.run.seed

    continuous_model = asdict(continuous.model)
    quantized_model = asdict(quantized.model)
    assert continuous_model.pop("ncp_target") == "continuous"
    assert quantized_model.pop("ncp_target") == "quantized"
    assert continuous_model == quantized_model
