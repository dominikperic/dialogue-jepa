"""Import smoke test for the editable research package."""

from importlib import import_module, metadata
from pathlib import Path


def test_package_and_planned_namespaces_import() -> None:
    names = [
        "dialogue_jepa",
        "dialogue_jepa.data.pairing",
        "dialogue_jepa.models.encoders.audio_encoder",
        "dialogue_jepa.models.encoders.text_encoder",
        "dialogue_jepa.models.jepa.predictor",
        "dialogue_jepa.models.alignment.procrustes",
        "dialogue_jepa.models.downstream.dialogue_state",
        "dialogue_jepa.training.trainer",
        "dialogue_jepa.evaluation.retrieval",
        "dialogue_jepa.utils.config",
    ]
    for name in names:
        assert import_module(name).__name__ == name


def test_editable_package_metadata_is_present() -> None:
    distribution = metadata.distribution("dialogue-jepa")
    assert distribution.version == "0.1.0"
    assert distribution.metadata["Requires-Python"] == ">=3.11"
    assert (Path(__file__).resolve().parents[1] / "pyproject.toml").is_file()
