"""Synthetic WAV and record tests for independent timing/channel review."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/prepare_timing_channel_review.py"
)


def module():
    spec = importlib.util.spec_from_file_location(
        "prepare_timing_channel_review", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def stereo_wav(path: Path, frames: int = 3000) -> Path:
    samples = np.column_stack(
        (
            np.full(frames, 0.25, dtype="float32"),
            np.full(frames, -0.25, dtype="float32"),
        )
    )
    sf.write(path, samples, 1000)
    return path


def test_extracts_both_isolated_channels_and_padded_stereo(tmp_path: Path) -> None:
    source = stereo_wav(tmp_path / "source.wav")
    clips = module().extract_review_clips(
        source, 1000, 2000, tmp_path / "clips", "sample"
    )
    ch0, sr0 = sf.read(clips["channel_0"], always_2d=True)
    ch1, sr1 = sf.read(clips["channel_1"], always_2d=True)
    context, src = sf.read(clips["stereo_context"], always_2d=True)
    assert sr0 == sr1 == src == 1000
    assert ch0.shape == ch1.shape == (1000, 1)
    assert context.shape == (2000, 2)
    assert np.all(ch0 > 0) and np.all(ch1 < 0)
    assert clips["context_frames"] == [500, 2500]
    assert clips["annotated_frames"] == [1000, 2000]


def test_padding_clamps_to_beginning_and_end_of_wav(tmp_path: Path) -> None:
    source = stereo_wav(tmp_path / "source.wav")
    clips = module().extract_review_clips(source, 0, 3000, tmp_path / "clips", "sample")
    assert clips["context_frames"] == [0, 3000]
    assert sf.info(clips["stereo_context"]).frames == 3000


def test_invalid_annotation_never_emits_clips(tmp_path: Path) -> None:
    source = stereo_wav(tmp_path / "source.wav")
    destination = tmp_path / "clips"
    with pytest.raises(ValueError, match="outside"):
        module().extract_review_clips(source, 2500, 3100, destination, "sample")
    assert not destination.exists()


def test_initializes_unreviewed_records_without_copying_transcript_pass(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reviews.jsonl"
    keys = [("MUL0003", 5), ("MUL0009", 2), ("MUL0013", 1)]
    module().initialize_review_records(path, keys)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["dialogue_id"], r["turn_id"]) for r in rows] == keys
    assert all(r["timing_review_status"] == "not_reviewed" for r in rows)
    assert all(r["role_channel_review_status"] == "not_reviewed" for r in rows)
    assert all(r["reviewed_role_channel"] is None for r in rows)
    assert all(r["timing_notes"] is None and r["channel_notes"] is None for r in rows)
    assert all("review_status" not in r for r in rows)


def test_repeated_initialization_preserves_manual_decisions(tmp_path: Path) -> None:
    path = tmp_path / "reviews.jsonl"
    keys = [("MUL0003", 5)]
    module().initialize_review_records(path, keys)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["timing_review_status"] = "pass"
    rows[0]["timing_notes"] = "Human-confirmed full utterance."
    edited = json.dumps(rows[0]) + "\n"
    path.write_text(edited)
    module().initialize_review_records(path, keys)
    assert path.read_text() == edited
