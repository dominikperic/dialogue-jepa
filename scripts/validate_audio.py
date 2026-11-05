"""Validate five SpokenWOZ WAVs against raw turn timing; no ASR is performed."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SELECTED = ("MUL0003", "MUL0006", "MUL0008", "MUL0009", "MUL0013")
DEFAULT_AUDIO = Path("data/raw/spokenwoz/audio_5700_train_dev")
DEFAULT_MANIFEST = Path("data/processed/spokenwoz/sample_10_turns.jsonl")
DEFAULT_REPORT = Path("data/processed/spokenwoz/audio_validation_report.json")


def audio_metadata(path: Path) -> dict[str, Any]:
    """Read WAV metadata without decoding the full recording."""
    if not path.is_file():
        raise FileNotFoundError(path)
    info = sf.info(path)
    return {
        "file_size_bytes": path.stat().st_size,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frame_count": info.frames,
        "duration_seconds": info.duration,
    }


def validate_window(
    start_raw: int | float | None,
    end_raw: int | float | None,
    metadata: dict[str, Any],
    units_per_second: int = 1000,
) -> dict[str, Any]:
    """Convert only for a candidate-unit test; preserve raw source values."""
    if units_per_second <= 0:
        raise ValueError("units_per_second must be positive")
    if start_raw is None or end_raw is None or start_raw < 0 or end_raw <= start_raw:
        return {"in_bounds": False, "start_frame": None, "end_frame": None}
    start = round(start_raw * metadata["sample_rate"] / units_per_second)
    end = round(end_raw * metadata["sample_rate"] / units_per_second)
    return {
        "in_bounds": 0 <= start < end <= metadata["frame_count"],
        "start_frame": start,
        "end_frame": end,
    }


def channel_activity(path: Path, start_frame: int, end_frame: int) -> dict[str, Any]:
    """Compute channel RMS/peak only inside an already-validated window."""
    with sf.SoundFile(path) as audio:
        if not 0 <= start_frame < end_frame <= audio.frames:
            raise ValueError("Segment out of WAV bounds")
        audio.seek(start_frame)
        samples = audio.read(end_frame - start_frame, dtype="float32", always_2d=True)
    return {
        "rms": [float(v) for v in np.sqrt(np.mean(samples * samples, axis=0))],
        "peak": [float(v) for v in np.max(np.abs(samples), axis=0)],
    }


def dominant_channel(rms: list[float]) -> int | None:
    """Return a clear active channel only; ambiguous or silent means unresolved."""
    if not rms or max(rms) < 0.002:
        return None
    ranked = sorted(enumerate(rms), key=lambda pair: pair[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] < 3 * ranked[1][1]:
        return None
    return ranked[0][0]


def analyze_dialogue(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep all turn windows, including overlaps, and report evidence only."""
    meta = audio_metadata(path)
    turns = []
    for row in rows:
        candidate = validate_window(
            row["start_time_raw"], row["end_time_raw"], meta, 1000
        )
        manifest_flags = list(row.get("validation_flags", []))
        flags = [flag for flag in manifest_flags if flag != "missing_audio"]
        entry: dict[str, Any] = {
            "turn_id": row["turn_id"],
            "role": row["role"],
            "start_time_raw": row["start_time_raw"],
            "end_time_raw": row["end_time_raw"],
            **candidate,
            "manifest_flags": manifest_flags,
            "flags": flags,
        }
        if not candidate["in_bounds"]:
            entry["flags"] = sorted(set(flags + ["out_of_bounds_or_invalid_time"]))
            entry["rms"] = None
            entry["dominant_channel"] = None
        else:
            activity = channel_activity(
                path, candidate["start_frame"], candidate["end_frame"]
            )
            entry.update(activity)
            entry["dominant_channel"] = dominant_channel(activity["rms"])
            duration = (candidate["end_frame"] - candidate["start_frame"]) / meta[
                "sample_rate"
            ]
            entry["segment_seconds"] = duration
            if duration < 0.15:
                entry["flags"].append("short_segment")
            if duration > 30:
                entry["flags"].append("long_segment")
            if max(activity["rms"]) < 0.002:
                entry["flags"].append("silent_segment")
        turns.append(entry)
    return {
        "dialogue_id": path.stem,
        "audio": meta,
        "max_word_timestamp_raw": max(
            (r["end_time_raw"] for r in rows if r["end_time_raw"] is not None),
            default=None,
        ),
        "turns": turns,
    }


def infer_channel_mapping(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer role activity from nonoverlapping, unambiguous windows only."""
    votes: dict[str, Counter[int]] = defaultdict(Counter)
    ambiguous: Counter[str] = Counter()
    for report in reports:
        for turn in report["turns"]:
            if "overlapping_adjacent_turns" in turn["flags"] or not turn["in_bounds"]:
                continue
            if turn["role"] not in ("user", "system"):
                continue
            if turn["dominant_channel"] is None:
                ambiguous[turn["role"]] += 1
            else:
                votes[turn["role"]][turn["dominant_channel"]] += 1
    mapping = {}
    for role in ("user", "system"):
        counted = votes[role]
        total = counted.total()
        best, count = counted.most_common(1)[0] if total else (None, 0)
        mapping[role] = best if total >= 10 and count / total >= 0.95 else None
    if mapping["user"] == mapping["system"]:
        mapping = {"user": None, "system": None}
    return {
        "proposed_mapping": mapping,
        "nonoverlap_votes": {role: dict(votes[role]) for role in ("user", "system")},
        "ambiguous_nonoverlap": dict(ambiguous),
        "status": "activity_supported_not_listening_verified"
        if all(v is not None for v in mapping.values())
        else "unresolved",
    }


def export_clip(
    source: Path,
    start_frame: int,
    end_frame: int,
    destination: Path,
    channel: int | str,
) -> None:
    """Write exactly the annotated interval, without trimming overlapping speech."""
    with sf.SoundFile(source) as audio:
        if not 0 <= start_frame < end_frame <= audio.frames:
            raise ValueError("Review clip outside recording")
        if channel != "both" and not (
            isinstance(channel, int) and 0 <= channel < audio.channels
        ):
            raise ValueError("Invalid channel selection")
        audio.seek(start_frame)
        samples = audio.read(end_frame - start_frame, dtype="float32", always_2d=True)
        if channel != "both":
            samples = samples[:, channel]
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, samples, audio.samplerate)


REVIEW_CHOICES = (
    ("MUL0003", 5, "dialogue-state transition, system", "role"),
    ("MUL0006", 0, "user reference", "role"),
    ("MUL0008", 26, "annotated turn overlap and simultaneous channel activity", "both"),
    ("MUL0008", 1, "system window with activity on both channels", "both"),
    ("MUL0009", 2, "user, no annotated overlaps in dialogue", "role"),
    ("MUL0013", 1, "system reference", "role"),
)


def export_review_clips(
    reports: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    mapping: dict[str, Any],
    audio_dir: Path,
    destination: Path,
) -> Path:
    """Export six review-only clips and record provenance; no listening claim."""
    by_report = {d["dialogue_id"]: d for d in reports}
    by_row = {(r["dialogue_id"], r["turn_id"]): r for r in rows}
    destination.mkdir(parents=True, exist_ok=True)
    review_rows = []
    for dialogue_id, turn_id, reason, channel_policy in REVIEW_CHOICES:
        report = by_report[dialogue_id]
        turn = next(t for t in report["turns"] if t["turn_id"] == turn_id)
        if not turn["in_bounds"]:
            raise ValueError(f"Cannot export invalid turn: {dialogue_id}/{turn_id}")
        role = turn["role"]
        chosen = (
            mapping["proposed_mapping"].get(role) if channel_policy == "role" else None
        )
        channel = chosen if chosen is not None else "both"
        label = f"ch{channel}" if channel != "both" else "stereo"
        filename = f"{dialogue_id}_turn{turn_id:02d}_{label}.wav"
        path = destination / filename
        export_clip(
            audio_dir / f"{dialogue_id}.wav",
            turn["start_frame"],
            turn["end_frame"],
            path,
            channel,
        )
        review_rows.append(
            {
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "role": role,
                "transcript": by_row[(dialogue_id, turn_id)]["transcript"],
                "audio_clip_path": str(path),
                "timestamp_window_raw": [turn["start_time_raw"], turn["end_time_raw"]],
                "timestamp_window_seconds_candidate": [
                    turn["start_frame"] / report["audio"]["sample_rate"],
                    turn["end_frame"] / report["audio"]["sample_rate"],
                ],
                "selected_audio_channel": channel,
                "reason": reason,
                "review_status": "not_listened_to",
            }
        )
    manifest_path = destination / "review_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as out:
        for item in review_rows:
            out.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--export-review", action="store_true")
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("data/processed/spokenwoz/review_clips"),
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    reports = []
    for dialogue_id in SELECTED:
        dialogue_rows = [row for row in rows if row["dialogue_id"] == dialogue_id]
        if not dialogue_rows:
            raise ValueError(f"Missing manifest turns for {dialogue_id}")
        reports.append(
            analyze_dialogue(args.audio_dir / f"{dialogue_id}.wav", dialogue_rows)
        )
    mapping = infer_channel_mapping(reports)
    review_manifest = None
    if args.export_review:
        review_manifest = export_review_clips(
            reports, rows, mapping, args.audio_dir, args.review_dir
        )
    report = {
        "timestamp_candidate_units_per_second": 1000,
        "alignment_status": "bounds_checked_transcript_unverified",
        "dialogues": reports,
        "channel_mapping": mapping,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    flags = Counter(
        flag
        for dialogue in reports
        for turn in dialogue["turns"]
        for flag in turn["flags"]
    )
    print(
        json.dumps(
            {
                "dialogues": len(reports),
                "turns": sum(len(d["turns"]) for d in reports),
                "within_bounds": sum(
                    t["in_bounds"] for d in reports for t in d["turns"]
                ),
                "flags": dict(flags),
                "channel_mapping": mapping,
                "report": str(args.report),
                "review_manifest": str(review_manifest) if review_manifest else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
