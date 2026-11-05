"""Turn-manifest extraction uses only synthetic dialogue fixtures."""

import json
from pathlib import Path

import pytest

from dialogue_jepa.data.turn_manifest import build_turn_manifest, write_jsonl


def word(text: str, start: int | None, end: int | None, index: int = 0) -> dict:
    return {"Word": text, "BeginTime": start, "EndTime": end, "Word_index": index}


def test_extracts_original_order_boundaries_and_annotations(tmp_path: Path) -> None:
    sample = {
        "MUL0003": {
            "log": [
                {
                    "tag": "user",
                    "text": "hello there",
                    "words": [word("there", 40, 60, 7), word("hello", 10, 30, 0)],
                    "dialog_act": {"greet": [["none", "none"]]},
                    "metadata": {},
                },
                {
                    "tag": "system",
                    "text": "ok",
                    "words": [word("ok", 70, 90)],
                    "dialog_act": {},
                    "metadata": {"restaurant": {"semi": {"food": "indian"}}},
                },
            ]
        }
    }
    rows = build_turn_manifest(sample, tmp_path)
    assert [(r["dialogue_id"], r["turn_id"], r["role"]) for r in rows] == [
        ("MUL0003", 0, "user"),
        ("MUL0003", 1, "system"),
    ]
    assert (rows[0]["start_time_raw"], rows[0]["end_time_raw"]) == (10, 60)
    assert rows[0]["transcript"] == "hello there"
    assert rows[0]["expected_audio_filename"] == "MUL0003.wav"
    assert rows[0]["dialogue_act"] == {"greet": [["none", "none"]]}
    assert rows[0]["dialogue_state"] == {}
    assert rows[1]["dialogue_state"]["restaurant"]["semi"]["food"] == "indian"
    assert rows[0]["audio_validation_status"] == "missing_audio"
    assert rows[0]["timing_validation_status"] != "verified_audio_alignment"


def test_flags_adjacent_overlap_and_nonmonotonic_starts(tmp_path: Path) -> None:
    sample = {
        "MUL0005": {
            "log": [
                {"tag": "user", "text": "first", "words": [word("first", 10, 40)]},
                {"tag": "system", "text": "second", "words": [word("second", 30, 50)]},
                {"tag": "user", "text": "third", "words": [word("third", 20, 25)]},
            ]
        }
    }
    rows = build_turn_manifest(sample, tmp_path)
    assert "overlapping_adjacent_turns" in rows[1]["validation_flags"]
    assert "overlapping_adjacent_turns" in rows[2]["validation_flags"]
    assert "nonmonotonic_turn_start" in rows[2]["validation_flags"]
    assert len(rows) == 3  # overlapping records are preserved


def test_missing_invalid_and_negative_duration_word_times(tmp_path: Path) -> None:
    sample = {
        "MUL0006": {
            "log": [
                {"tag": "user", "text": "", "words": []},
                {"tag": "system", "text": "a", "words": [word("a", None, 20)]},
                {
                    "tag": "user",
                    "text": "b c",
                    "words": [word("b", 50, 40), word("c", 60, 80)],
                },
            ]
        }
    }
    rows = build_turn_manifest(sample, tmp_path)
    assert rows[0]["start_time_raw"] is None
    assert rows[0]["end_time_raw"] is None
    assert "empty_word_list" in rows[0]["validation_flags"]
    assert "missing_word_timestamps" in rows[1]["validation_flags"]
    assert rows[1]["timing_validation_status"] == "invalid_metadata"
    assert "negative_duration_interval" in rows[2]["validation_flags"]
    assert (rows[2]["start_time_raw"], rows[2]["end_time_raw"]) == (60, 80)
    assert rows[2]["timing_validation_status"] == "invalid_metadata"


def test_transcript_mismatch_and_missing_state_not_fabricated(tmp_path: Path) -> None:
    sample = {
        "MUL0007": {
            "log": [
                {
                    "tag": "user",
                    "text": "not matching",
                    "words": [word("different", 0, 10)],
                }
            ]
        }
    }
    row = build_turn_manifest(sample, tmp_path)[0]
    assert "transcript_word_mismatch" in row["validation_flags"]
    assert row["dialogue_state"] is None
    assert row["timing_validation_status"] == "valid_metadata_unverified_audio"


def test_existing_audio_remains_unverified(tmp_path: Path) -> None:
    (tmp_path / "MUL0008.wav").write_bytes(b"not decoded")
    row = build_turn_manifest(
        {
            "MUL0008": {
                "log": [{"tag": "user", "text": "a", "words": [word("a", 0, 1)]}]
            }
        },
        tmp_path,
    )[0]
    assert row["audio_validation_status"] == "file_present_alignment_unverified"
    assert row["timing_validation_status"] == "valid_metadata_unverified_audio"


def test_jsonl_rerun_is_identical_without_duplication(tmp_path: Path) -> None:
    sample = {
        "MUL0009": {"log": [{"tag": "user", "text": "a", "words": [word("a", 0, 10)]}]}
    }
    destination = tmp_path / "manifest.jsonl"
    rows = build_turn_manifest(sample, tmp_path)
    write_jsonl(rows, destination)
    before = destination.read_bytes()
    write_jsonl(build_turn_manifest(sample, tmp_path), destination)
    assert destination.read_bytes() == before
    assert len([json.loads(line) for line in destination.read_text().splitlines()]) == 1


def test_rejects_more_than_ten_conversations(tmp_path: Path) -> None:
    sample = {str(i): {"log": []} for i in range(11)}
    with pytest.raises(ValueError, match="10"):
        build_turn_manifest(sample, tmp_path)
