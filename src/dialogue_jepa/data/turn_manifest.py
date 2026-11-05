"""Validated turn-level manifest for a small SpokenWOZ text sample.

Times are raw source values. Metadata checks do not verify waveform alignment.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, TypeGuard

TIMING_FAULTS = {
    "empty_word_list",
    "missing_word_timestamps",
    "negative_duration_interval",
    "zero_duration_interval",
    "invalid_word_timestamp",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def _word_tokens(words: list[dict[str, Any]]) -> list[str]:
    return [
        token
        for word in words
        if isinstance(word, dict)
        if (token := re.sub(r"[^a-z0-9']", "", str(word.get("Word", "")).lower()))
    ]


def _valid_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def build_turn_manifest(
    dialogues: dict[str, Any], audio_dir: Path
) -> list[dict[str, Any]]:
    """Keep source dialogue/turn order and flag issues without dropping turns."""
    if len(dialogues) > 10:
        raise ValueError("Inspect only the 10 sampled conversations")
    rows: list[dict[str, Any]] = []
    for dialogue_id, dialogue in dialogues.items():
        if not isinstance(dialogue.get("log"), list):
            raise ValueError(f"{dialogue_id}: missing or invalid log")
        previous_start: int | float | None = None
        previous_end: int | float | None = None
        for turn_id, turn in enumerate(dialogue["log"]):
            flags: set[str] = set()
            words = turn.get("words")
            if not isinstance(words, list) or not words:
                flags.add("empty_word_list")
                words = []
            valid_intervals: list[tuple[int | float, int | float]] = []
            for word in words:
                if not isinstance(word, dict):
                    flags.add("missing_word_timestamps")
                    continue
                start, end = word.get("BeginTime"), word.get("EndTime")
                if not _valid_number(start) or not _valid_number(end):
                    flags.add("missing_word_timestamps")
                    continue
                if start < 0 or end < 0:
                    flags.add("invalid_word_timestamp")
                    continue
                if end < start:
                    flags.add("negative_duration_interval")
                    continue
                if end == start:
                    flags.add("zero_duration_interval")
                    continue
                valid_intervals.append((start, end))
            if words and _word_tokens(words) != _tokens(str(turn.get("text", ""))):
                flags.add("transcript_word_mismatch")
            start_time = min((start for start, _ in valid_intervals), default=None)
            end_time = max((end for _, end in valid_intervals), default=None)
            if start_time is not None and previous_start is not None:
                if start_time < previous_start:
                    flags.add("nonmonotonic_turn_start")
                if previous_end is not None and start_time < previous_end:
                    flags.add("overlapping_adjacent_turns")
            # Compare only adjacent turns; do not bridge a turn lacking timestamps.
            previous_start, previous_end = start_time, end_time
            audio_exists = (audio_dir / f"{dialogue_id}.wav").is_file()
            if not audio_exists:
                flags.add("missing_audio")
            if flags & TIMING_FAULTS or start_time is None:
                timing_status = "invalid_metadata"
            elif flags & {"overlapping_adjacent_turns", "nonmonotonic_turn_start"}:
                timing_status = "needs_review_unverified_audio"
            else:
                timing_status = "valid_metadata_unverified_audio"
            rows.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_id": turn_id,
                    "role": turn.get("tag"),
                    "transcript": turn.get("text"),
                    "start_time_raw": start_time,
                    "end_time_raw": end_time,
                    "expected_audio_filename": f"{dialogue_id}.wav",
                    "dialogue_act": turn.get("dialog_act"),
                    "dialogue_state": turn.get("metadata"),
                    "timing_validation_status": timing_status,
                    "audio_validation_status": (
                        "file_present_alignment_unverified"
                        if audio_exists
                        else "missing_audio"
                    ),
                    "validation_flags": sorted(flags),
                }
            )
    return rows


def write_jsonl(rows: list[dict[str, Any]], destination: Path) -> None:
    """Overwrite the existing manifest deterministically, never append."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
