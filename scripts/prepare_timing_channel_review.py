"""Prepare three independent human timing/channel checks; never record a decision."""

import argparse
import json
from pathlib import Path
from typing import Any

import soundfile as sf

SELECTED = (("MUL0003", 5), ("MUL0009", 2), ("MUL0013", 1))
DEFAULT_MANIFEST = Path("data/processed/spokenwoz/sample_10_turns.jsonl")
DEFAULT_TRANSCRIPT_REVIEWS = Path(
    "data/processed/spokenwoz/review_clips/review_manifest.jsonl"
)
DEFAULT_AUDIO = Path("data/raw/spokenwoz/audio_5700_train_dev")
DEFAULT_CLIPS = Path("data/processed/spokenwoz/timing_channel_review_clips")
DEFAULT_RECORDS = Path("data/processed/spokenwoz/timing_channel_reviews.jsonl")


def extract_review_clips(
    source: Path, start_raw: int, end_raw: int, destination: Path, label: str
) -> dict[str, Any]:
    """Copy exact annotated channels and ±0.5 s stereo context; clamp to WAV."""
    with sf.SoundFile(source) as audio:
        if audio.channels != 2:
            raise ValueError(f"Expected stereo WAV: {source}")
        rate, frames, subtype = audio.samplerate, audio.frames, audio.subtype
        start = round(start_raw * rate / 1000)
        end = round(end_raw * rate / 1000)
        if not 0 <= start < end <= frames:
            raise ValueError(f"Annotated window outside recording: {source}")
        pad = round(rate * 0.5)
        context_start, context_end = max(0, start - pad), min(frames, end + pad)
        audio.seek(start)
        isolated = audio.read(end - start, dtype="float32", always_2d=True)
        audio.seek(context_start)
        context = audio.read(
            context_end - context_start, dtype="float32", always_2d=True
        )

    paths = {
        "channel_0": destination / f"{label}_ch0.wav",
        "channel_1": destination / f"{label}_ch1.wav",
        "stereo_context": destination / f"{label}_stereo_context.wav",
    }
    samples = {
        "channel_0": isolated[:, 0],
        "channel_1": isolated[:, 1],
        "stereo_context": context,
    }
    for key, path in paths.items():
        if path.exists():
            info = sf.info(path)
            expected_channels = 2 if key == "stereo_context" else 1
            if (info.samplerate, info.channels, info.frames) != (
                rate,
                expected_channels,
                len(samples[key]),
            ):
                raise ValueError(
                    f"Existing review clip has unexpected dimensions: {path}"
                )
            continue  # Preserve any existing audio rather than overwriting it.
        destination.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            sf.write(output, samples[key], rate, format="WAV", subtype=subtype)
    return {
        **{key: str(path) for key, path in paths.items()},
        "annotated_frames": [start, end],
        "context_frames": [context_start, context_end],
        "sample_rate": rate,
    }


def initialize_review_records(path: Path, keys: list[tuple[str, int]]) -> None:
    """Create an untouched decision sheet once; reject mismatched reruns."""
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate candidate turn")
    if path.exists():
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if [(r["dialogue_id"], r["turn_id"]) for r in rows] != keys:
            raise ValueError("Existing independent review sheet has different turns")
        return  # Never reset manually entered statuses or notes.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for dialogue_id, turn_id in keys:
            row = {
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "timing_review_status": "not_reviewed",
                "role_channel_review_status": "not_reviewed",
                "reviewed_role_channel": None,
                "timing_notes": None,
                "channel_notes": None,
            }
            output.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_package(
    manifest: Path,
    transcript_reviews: Path,
    audio_dir: Path,
    clips_dir: Path,
    review_records: Path,
) -> list[dict[str, Any]]:
    """Join only selected source turns; preserve all prior review decisions."""
    inputs = [
        manifest,
        transcript_reviews,
        *[audio_dir / f"{did}.wav" for did, _ in SELECTED],
    ]
    if review_records.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("Independent review record path overlaps an input")
    rows = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    prior = [
        json.loads(line)
        for line in transcript_reviews.read_text(encoding="utf-8").splitlines()
    ]
    records = []
    for dialogue_id, turn_id in SELECTED:
        matched = [
            r
            for r in rows
            if (r["dialogue_id"], r["turn_id"]) == (dialogue_id, turn_id)
        ]
        human = [
            r
            for r in prior
            if (r["dialogue_id"], r["turn_id"]) == (dialogue_id, turn_id)
        ]
        if len(matched) != 1 or len(human) != 1:
            raise ValueError(
                f"Missing or duplicate turn/review: {dialogue_id}/{turn_id}"
            )
        row, reviewed = matched[0], human[0]
        if (
            reviewed["transcript"] != row["transcript"]
            or reviewed["role"] != row["role"]
            or reviewed["review_status"] != "pass"
            or reviewed["timestamp_window_raw"]
            != [row["start_time_raw"], row["end_time_raw"]]
        ):
            raise ValueError(
                f"Transcript review does not match source: {dialogue_id}/{turn_id}"
            )
        label = f"{dialogue_id}_turn{turn_id:02d}"
        clips = extract_review_clips(
            audio_dir / f"{dialogue_id}.wav",
            row["start_time_raw"],
            row["end_time_raw"],
            clips_dir,
            label,
        )
        records.append(
            {
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "transcript": row["transcript"],
                "role": row["role"],
                "start_time_raw": row["start_time_raw"],
                "end_time_raw": row["end_time_raw"],
                **clips,
            }
        )
    initialize_review_records(review_records, list(SELECTED))
    guide = clips_dir / "review_package.md"
    if not guide.exists():
        lines = [
            "# Independent timing and channel review",
            "",
            "These are unreviewed timing/channel checks, separate from prior "
            "transcript passes.",
            "Times are original candidate millisecond-like annotation units; "
            "1000 units/second is supported by prior bounds checks.",
            "",
            "For each turn: listen to ch0 and ch1 separately; identify which "
            "contains the stated role's complete utterance. Compare the exact "
            "clips against the original transcript and stated window. Play "
            "stereo_context for roughly 0.5 seconds before/after to detect "
            "clipped words or bad boundaries. Record timing and role-channel "
            "decisions independently in `../timing_channel_reviews.jsonl` "
            "(or send your observations to the agent). Do not infer a "
            "decision from loudness alone.",
            "",
        ]
        for item in records:
            lines += [
                f"## {item['dialogue_id']} turn {item['turn_id']} — {item['role']}",
                f"Original transcript: {item['transcript']}",
                f"Annotated raw window: {item['start_time_raw']}–"
                f"{item['end_time_raw']} (candidate "
                f"{item['start_time_raw'] / 1000:.3f}–"
                f"{item['end_time_raw'] / 1000:.3f} s).",
                f"Channel 0: `{Path(item['channel_0']).name}`",
                f"Channel 1: `{Path(item['channel_1']).name}`",
                f"Padded stereo context: `{Path(item['stereo_context']).name}`",
                "",
            ]
        with guide.open("x", encoding="utf-8") as output:
            output.write("\n".join(lines) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--transcript-reviews", type=Path, default=DEFAULT_TRANSCRIPT_REVIEWS
    )
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--review-records", type=Path, default=DEFAULT_RECORDS)
    args = parser.parse_args()
    records = prepare_package(
        args.manifest,
        args.transcript_reviews,
        args.audio_dir,
        args.clips_dir,
        args.review_records,
    )
    for item in records:
        print(
            f"{item['dialogue_id']}/{item['turn_id']} {item['role']}: "
            f"{item['start_time_raw']}–{item['end_time_raw']} raw; "
            f"ch0={item['channel_0']} ch1={item['channel_1']} "
            f"context={item['stereo_context']}"
        )
    print(f"Independent decision sheet: {args.review_records}")


if __name__ == "__main__":
    main()
