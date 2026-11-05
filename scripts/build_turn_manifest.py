"""Build a bounded SpokenWOZ turn manifest from the inspected ten-dialogue sample."""

import argparse
import json
from collections import Counter
from pathlib import Path

from dialogue_jepa.data.turn_manifest import build_turn_manifest, write_jsonl

DEFAULT_INPUT = Path("data/raw/spokenwoz/sample_10_train.json")
DEFAULT_OUTPUT = Path("data/processed/spokenwoz/sample_10_turns.jsonl")
DEFAULT_AUDIO_DIR = Path("data/raw/spokenwoz/audio_5700_train_dev")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    args = parser.parse_args()
    dialogues = json.loads(args.input.read_text(encoding="utf-8"))
    rows = build_turn_manifest(dialogues, args.audio_dir)
    write_jsonl(rows, args.output)
    flags = Counter(flag for row in rows for flag in row["validation_flags"])
    print(
        json.dumps(
            {
                "dialogues": len(dialogues),
                "total_turns": len(rows),
                "valid_timestamp_metadata": sum(
                    row["timing_validation_status"] != "invalid_metadata"
                    for row in rows
                ),
                "overlapping_adjacent_turns": flags["overlapping_adjacent_turns"],
                "transcript_word_mismatch": flags["transcript_word_mismatch"],
                "missing_audio": flags["missing_audio"],
                "other_flags": {
                    key: count
                    for key, count in flags.items()
                    if key
                    not in {
                        "overlapping_adjacent_turns",
                        "transcript_word_mismatch",
                        "missing_audio",
                    }
                },
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
