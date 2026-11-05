"""Audit turn-level pairing eligibility without changing source records."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("data/processed/spokenwoz/sample_10_turns.jsonl")
DEFAULT_VALIDATION = Path("data/processed/spokenwoz/audio_validation_report.json")
DEFAULT_REVIEWS = Path("data/processed/spokenwoz/review_clips/review_manifest.jsonl")
DEFAULT_AUDIO_DIR = Path("data/raw/spokenwoz/audio_5700_train_dev")
DEFAULT_OUTPUT = Path("data/processed/spokenwoz/pair_eligibility_audit.jsonl")


def audit_records(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    reviews: list[dict[str, Any]],
    audio_dir: Path,
) -> list[dict[str, Any]]:
    """Require complete joins; never use historical missing-audio as current status."""
    groups = report["dialogues"]
    ids = [group["dialogue_id"] for group in groups]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate dialogue in validation report")
    selected = set(ids)
    source_rows = [row for row in rows if row["dialogue_id"] in selected]

    def unique(
        items: list[dict[str, Any]], label: str
    ) -> dict[tuple[str, int], dict[str, Any]]:
        indexed = {}
        for item in items:
            key = (item["dialogue_id"], item["turn_id"])
            if key in indexed:
                raise ValueError(f"Duplicate {label}: {key}")
            indexed[key] = item
        return indexed

    by_source = unique(source_rows, "manifest turn")
    validations = [
        dict(turn, dialogue_id=group["dialogue_id"])
        for group in groups
        for turn in group["turns"]
    ]
    by_validation = unique(validations, "validation turn")
    if by_source.keys() != by_validation.keys():
        raise ValueError("Manifest/validation turn set mismatch")
    by_review = unique(reviews, "review turn")
    if not by_review.keys() <= by_source.keys():
        raise ValueError("Review refers to a turn outside the audited set")
    mapping = report["channel_mapping"]["proposed_mapping"]
    results = []
    for key, row in by_source.items():
        validation = by_validation[key]
        review = by_review.get(key)
        if validation["role"] != row["role"]:
            raise ValueError(f"Validation role mismatch: {key}")
        if review is not None and (
            review.get("transcript") != row["transcript"]
            or review.get("role") != row["role"]
            or (
                "timestamp_window_raw" in review
                and review["timestamp_window_raw"]
                != [row["start_time_raw"], row["end_time_raw"]]
            )
        ):
            raise ValueError(f"Review does not match source turn: {key}")
        filename = f"{key[0]}.wav"
        if row.get("expected_audio_filename", filename) != filename:
            raise ValueError(f"Unexpected audio filename: {key}")
        path = audio_dir / filename
        audio_present = path.is_file()
        result = classify_turn(row, validation, review, audio_present, mapping)
        result.update(dialogue_id=key[0], turn_id=key[1], role=row["role"])
        result["evidence"].update(
            {
                "audio_file": str(path),
                "audio_present": audio_present,
                "timing_in_bounds": validation.get("in_bounds") is True,
                "channel_dominant": validation.get("dominant_channel"),
                "channel_role_expected": mapping.get(row["role"]),
                "timestamp_overlap_flag": "overlapping_adjacent_turns"
                in set(row.get("validation_flags", []))
                | set(validation.get("flags", [])),
                "transcript_word_mismatch_flag": "transcript_word_mismatch"
                in set(row.get("validation_flags", []))
                | set(validation.get("flags", [])),
                "review_selected_channel": review.get("selected_audio_channel")
                if review
                else None,
            }
        )
        results.append(result)
    return sorted(results, key=lambda item: (item["dialogue_id"], item["turn_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() in {
        args.manifest.resolve(),
        args.validation.resolve(),
        args.reviews.resolve(),
    }:
        parser.error("Refusing to overwrite input data or human review manifest")
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads(args.validation.read_text(encoding="utf-8"))
    reviews = [
        json.loads(line)
        for line in args.reviews.read_text(encoding="utf-8").splitlines()
    ]
    results = audit_records(rows, report, reviews, args.audio_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    counts = Counter(row["category"] for row in results)
    reasons = Counter(reason for row in results for reason in row["reasons"])
    print(
        json.dumps(
            {
                "turns": len(results),
                "categories": dict(counts),
                "overlapping_reasons": dict(reasons),
                "output": str(args.output),
            },
            indent=2,
        )
    )


def classify_turn(
    manifest: dict[str, Any],
    validation: dict[str, Any],
    review: dict[str, Any] | None,
    audio_present: bool,
    role_channels: dict[str, int | None],
) -> dict[str, Any]:
    """Classify one turn with independent evidence fields."""
    role = manifest["role"]
    channel = validation.get("dominant_channel")
    expected = role_channels.get(role)
    supported = channel is not None and channel == expected and expected is not None
    reviewed = review is not None and review.get("review_status") == "pass"
    reasons = [] if review is not None else ["human_review_missing"]
    if review is not None:
        if review.get("review_status") == "needs_review":
            reasons.append("transcript_uncertain")
        elif review.get("review_status") == "fail":
            reasons.append("human_review_failed")
        elif review.get("review_status") != "pass":
            reasons.append("human_review_unresolved")
        if reviewed and review.get("selected_audio_channel") != expected:
            reasons.append("channel_selection_unverified")
        if reviewed and review.get("timing_alignment_confirmation") != "verified":
            reasons.append("timing_alignment_confirmation_missing")
        if reviewed and review.get("speaker_channel_confirmation") != "verified":
            reasons.append("speaker_channel_confirmation_missing")
    if expected is None or channel is None:
        reasons.append("channel_ambiguous")
    elif channel != expected:
        reasons.append("channel_role_conflict")
    flags = set(manifest.get("validation_flags", [])) | set(validation.get("flags", []))
    if "overlapping_adjacent_turns" in flags:
        reasons.append("timestamp_overlap_unresolved")
    if "transcript_word_mismatch" in flags:
        reasons.append("transcript_word_mismatch")
    for flag in ("short_segment", "long_segment", "silent_segment"):
        if flag in flags:
            reasons.append(flag)
    if not audio_present:
        reasons.append("audio_unavailable")
    if not validation.get("in_bounds"):
        reasons.append("invalid_timing")
    excluded = any(
        reason in reasons
        for reason in ("audio_unavailable", "invalid_timing", "human_review_failed")
    )
    return {
        "category": "excluded"
        if excluded
        else ("high_confidence" if not reasons else "needs_review"),
        "reasons": reasons,
        "evidence": {
            "human_review": review.get("review_status") if review else "not_reviewed",
            "channel": "supported" if supported else "unresolved",
        },
    }


if __name__ == "__main__":
    main()
