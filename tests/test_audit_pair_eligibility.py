"""Eligibility decisions use separate, synthetic evidence sources."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_pair_eligibility.py"


def audit():
    spec = importlib.util.spec_from_file_location("audit_pair_eligibility", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missing_review_is_not_classified_as_failure() -> None:
    result = audit().classify_turn(
        {"role": "user", "transcript": "hello", "validation_flags": ["missing_audio"]},
        {"in_bounds": True, "dominant_channel": 0, "flags": []},
        None,
        True,
        {"user": 0, "system": 1},
    )
    assert result["category"] == "needs_review"
    assert result["reasons"] == ["human_review_missing"]
    assert result["evidence"]["human_review"] == "not_reviewed"


def test_invalid_window_is_excluded_even_if_human_passed() -> None:
    result = audit().classify_turn(
        {"role": "user", "transcript": "hello", "validation_flags": []},
        {
            "in_bounds": False,
            "dominant_channel": None,
            "flags": ["out_of_bounds_or_invalid_time"],
        },
        {
            "review_status": "pass",
            "transcript": "hello",
            "role": "user",
            "selected_audio_channel": 0,
        },
        True,
        {"user": 0, "system": 1},
    )
    assert result["category"] == "excluded"
    assert "invalid_timing" in result["reasons"]


def test_missing_audio_is_excluded_despite_stale_manifest_flags() -> None:
    result = audit().classify_turn(
        {"role": "user", "transcript": "hello", "validation_flags": ["missing_audio"]},
        {"in_bounds": True, "dominant_channel": 0, "flags": []},
        None,
        False,
        {"user": 0, "system": 1},
    )
    assert result["category"] == "excluded"
    assert "audio_unavailable" in result["reasons"]


def test_ambiguous_channel_requires_review_even_with_transcript_pass() -> None:
    result = audit().classify_turn(
        {"role": "system", "transcript": "hello", "validation_flags": []},
        {"in_bounds": True, "dominant_channel": None, "flags": []},
        {
            "review_status": "pass",
            "transcript": "hello",
            "role": "system",
            "selected_audio_channel": "both",
        },
        True,
        {"user": 0, "system": 1},
    )
    assert result["category"] == "needs_review"
    assert "channel_ambiguous" in result["reasons"]


def test_overlap_is_separate_from_transcript_pass() -> None:
    result = audit().classify_turn(
        {
            "role": "user",
            "transcript": "hello",
            "validation_flags": ["overlapping_adjacent_turns"],
        },
        {
            "in_bounds": True,
            "dominant_channel": 0,
            "flags": ["overlapping_adjacent_turns"],
        },
        {
            "review_status": "pass",
            "transcript": "hello",
            "role": "user",
            "selected_audio_channel": 0,
            "overlap_listening_confirmation": "not_independently_confirmed",
        },
        True,
        {"user": 0, "system": 1},
    )
    assert result["category"] == "needs_review"
    assert "timestamp_overlap_unresolved" in result["reasons"]
    assert result["evidence"]["human_review"] == "pass"


def test_proper_noun_uncertainty_does_not_overwrite_source_transcript() -> None:
    result = audit().classify_turn(
        {"role": "user", "transcript": "papa", "validation_flags": []},
        {"in_bounds": True, "dominant_channel": 0, "flags": []},
        {
            "review_status": "needs_review",
            "transcript": "papa",
            "role": "user",
            "selected_audio_channel": 0,
            "human_perceived_transcript": "Pepasha",
            "initial_high_confidence_pair_eligibility": "exclude_until_resolved",
        },
        True,
        {"user": 0, "system": 1},
    )
    assert result["category"] == "needs_review"
    assert "transcript_uncertain" in result["reasons"]
    assert result["evidence"]["human_review"] == "needs_review"


def test_word_mismatch_requires_review_without_overriding_human_pass() -> None:
    result = audit().classify_turn(
        {
            "role": "user",
            "transcript": "hello",
            "validation_flags": ["transcript_word_mismatch"],
        },
        {
            "in_bounds": True,
            "dominant_channel": 0,
            "flags": ["transcript_word_mismatch"],
        },
        {
            "review_status": "pass",
            "transcript": "hello",
            "role": "user",
            "selected_audio_channel": 0,
        },
        True,
        {"user": 0, "system": 1},
    )
    assert result["category"] == "needs_review"
    assert "transcript_word_mismatch" in result["reasons"]


def test_audit_joins_only_downloaded_dialogues_without_promoting_other_turns(
    tmp_path: Path,
) -> None:
    (tmp_path / "MUL0003.wav").touch()
    rows = [
        {
            "dialogue_id": "MUL0003",
            "turn_id": 5,
            "role": "system",
            "transcript": "hello",
            "validation_flags": ["missing_audio"],
        },
        {
            "dialogue_id": "OTHER",
            "turn_id": 0,
            "role": "user",
            "transcript": "world",
            "validation_flags": [],
        },
    ]
    report = {
        "dialogues": [
            {
                "dialogue_id": "MUL0003",
                "turns": [
                    {
                        "turn_id": 5,
                        "role": "system",
                        "in_bounds": True,
                        "dominant_channel": 1,
                        "flags": [],
                    }
                ],
            }
        ],
        "channel_mapping": {"proposed_mapping": {"user": 0, "system": 1}},
    }
    reviews = [
        {
            "dialogue_id": "MUL0003",
            "turn_id": 5,
            "role": "system",
            "transcript": "hello",
            "review_status": "pass",
            "selected_audio_channel": 1,
            "timing_alignment_confirmation": "verified",
            "speaker_channel_confirmation": "verified",
        }
    ]
    audited = audit().audit_records(rows, report, reviews, tmp_path)
    assert len(audited) == 1
    assert audited[0]["category"] == "high_confidence"
    assert audited[0]["turn_id"] == 5


def test_audit_rejects_review_text_attached_to_wrong_turn(tmp_path: Path) -> None:
    (tmp_path / "MUL0003.wav").touch()
    rows = [
        {
            "dialogue_id": "MUL0003",
            "turn_id": 5,
            "role": "system",
            "transcript": "hello",
            "validation_flags": [],
        }
    ]
    report = {
        "dialogues": [
            {
                "dialogue_id": "MUL0003",
                "turns": [
                    {
                        "turn_id": 5,
                        "role": "system",
                        "in_bounds": True,
                        "dominant_channel": 1,
                        "flags": [],
                    }
                ],
            }
        ],
        "channel_mapping": {"proposed_mapping": {"user": 0, "system": 1}},
    }
    reviews = [
        {
            "dialogue_id": "MUL0003",
            "turn_id": 5,
            "role": "system",
            "transcript": "wrong turn",
            "review_status": "pass",
            "selected_audio_channel": 1,
        }
    ]
    import pytest

    with pytest.raises(ValueError, match="Review does not match"):
        audit().audit_records(rows, report, reviews, tmp_path)


def test_cli_writes_derived_audit_without_mutating_inputs(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    row = {
        "dialogue_id": "MUL0003",
        "turn_id": 5,
        "role": "system",
        "transcript": "hello",
        "validation_flags": [],
    }
    report = {
        "dialogues": [
            {
                "dialogue_id": "MUL0003",
                "turns": [
                    {
                        "turn_id": 5,
                        "role": "system",
                        "in_bounds": True,
                        "dominant_channel": 1,
                        "flags": [],
                    }
                ],
            }
        ],
        "channel_mapping": {"proposed_mapping": {"user": 0, "system": 1}},
    }
    review = dict(
        row,
        review_status="pass",
        selected_audio_channel=1,
        timing_alignment_confirmation="verified",
        speaker_channel_confirmation="verified",
    )
    inputs = (
        tmp_path / "manifest.jsonl",
        tmp_path / "report.json",
        tmp_path / "reviews.jsonl",
    )
    payloads = (json.dumps(row) + "\n", json.dumps(report), json.dumps(review) + "\n")
    for path, content in zip(inputs, payloads, strict=True):
        path.write_text(content)
    (tmp_path / "MUL0003.wav").touch()
    out = tmp_path / "audit.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(inputs[0]),
            "--validation",
            str(inputs[1]),
            "--reviews",
            str(inputs[2]),
            "--audio-dir",
            str(tmp_path),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["category"] == "high_confidence"
    assert [p.read_text() for p in inputs] == list(payloads)


def test_cli_refuses_output_on_human_review_manifest(tmp_path: Path) -> None:
    import subprocess
    import sys

    inputs = [
        tmp_path / name for name in ("turns.jsonl", "validation.json", "reviews.jsonl")
    ]
    for path in inputs:
        path.write_text("DO NOT OVERWRITE")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(inputs[0]),
            "--validation",
            str(inputs[1]),
            "--reviews",
            str(inputs[2]),
            "--output",
            str(inputs[2]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Refusing to overwrite input" in result.stderr
    assert inputs[2].read_text() == "DO NOT OVERWRITE"


def test_activity_only_channel_evidence_stays_pending() -> None:
    result = audit().classify_turn(
        {"role": "system", "transcript": "hello", "validation_flags": []},
        {"in_bounds": True, "dominant_channel": 1, "flags": []},
        {
            "review_status": "pass",
            "transcript": "hello",
            "role": "system",
            "selected_audio_channel": 1,
        },
        True,
        {"user": 0, "system": 1},
    )
    assert result["category"] == "needs_review"
    assert "timing_alignment_confirmation_missing" in result["reasons"]
    assert "speaker_channel_confirmation_missing" in result["reasons"]


def test_reviewed_unambiguous_pair_is_high_confidence() -> None:
    manifest = {
        "dialogue_id": "MUL0003",
        "turn_id": 5,
        "role": "system",
        "transcript": "hello",
        "validation_flags": ["missing_audio"],
    }
    validation = {"in_bounds": True, "dominant_channel": 1, "flags": []}
    review = {
        "review_status": "pass",
        "transcript": "hello",
        "selected_audio_channel": 1,
        "role": "system",
        "timing_alignment_confirmation": "verified",
        "speaker_channel_confirmation": "verified",
    }
    result = audit().classify_turn(
        manifest, validation, review, True, {"user": 0, "system": 1}
    )
    assert result["category"] == "high_confidence"
    assert result["reasons"] == []
    assert result["evidence"]["human_review"] == "pass"
    assert result["evidence"]["channel"] == "supported"
