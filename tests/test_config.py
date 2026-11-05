"""Structural and configuration contracts for the research scaffold."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = {
    "data": {"spokenwoz", "meld"},
    "model": {"audio_jepa", "text_jepa", "alignment", "dialogue_predictor"},
    "training": {"audio", "text"},
    "experiments": {"baseline", "good_views", "bad_views", "ablation"},
}


def test_all_planned_configs_are_valid_mappings() -> None:
    for directory, names in CONFIG_NAMES.items():
        for name in names:
            path = ROOT / "configs" / directory / f"{name}.yaml"
            assert path.is_file(), path
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert isinstance(config, dict) and config, path
            assert config.get("status") == "provisional", path


def test_datasets_require_conversation_disjoint_splits() -> None:
    for name in CONFIG_NAMES["data"]:
        config = yaml.safe_load(
            (ROOT / "configs" / "data" / f"{name}.yaml").read_text()
        )
        assert config["splits"]["unit"] == "conversation"
        assert set(config["splits"]["names"]) == {"train", "validation", "test"}
        assert config["splits"]["disjoint"] is True


def test_research_priority_configs() -> None:
    model_dir = ROOT / "configs" / "model"
    for modality in ("audio", "text"):
        config = yaml.safe_load((model_dir / f"{modality}_jepa.yaml").read_text())
        assert (
            config["context_encoder"]["selection_priority"] == "representation_quality"
        )
        assert len(config["predictive_objectives"]["candidates"]) >= 3
        assert config["predictive_objectives"]["selection"] == "validation_required"

    good = yaml.safe_load((ROOT / "configs/experiments/good_views.yaml").read_text())
    bad = yaml.safe_load((ROOT / "configs/experiments/bad_views.yaml").read_text())
    assert "aligned_audio_transcript" in good["pair_conditions"]
    assert "weaker_positive" in good["pair_conditions"]
    assert "intentionally_mismatched" in bad["pair_conditions"]
    assert bad["mismatch_scope"] == "within_split_only"

    baseline = yaml.safe_load((ROOT / "configs/experiments/baseline.yaml").read_text())
    assert {"text_only", "audio_only", "conventional_fusion"} <= set(
        baseline["baselines"]
    )
    assert baseline["selection_priority"] == "predictive_and_alignment_quality"
    assert baseline["efficiency_role"] == "reproducibility_only"
    assert {
        "unseen_conversations",
        "unseen_speakers",
        "cross_dataset_if_compatible",
    } <= set(baseline["generalization_evaluations"])


def test_expected_directories_exist() -> None:
    for directory in (
        "data/raw",
        "data/processed",
        "data/manifests",
        "data/splits",
        "notebooks",
        "experiments",
        "checkpoints",
        "tests",
        "demo",
        "docs",
        "src/dialogue_jepa/data",
        "src/dialogue_jepa/models/encoders",
        "src/dialogue_jepa/models/jepa",
        "src/dialogue_jepa/models/alignment",
        "src/dialogue_jepa/models/downstream",
        "src/dialogue_jepa/training",
        "src/dialogue_jepa/evaluation",
        "src/dialogue_jepa/utils",
    ):
        assert (ROOT / directory).is_dir(), directory
