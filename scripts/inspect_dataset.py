"""Inspect at most ten official SpokenWOZ training conversations."""

import argparse
import codecs
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf

TEXT_REPO = "https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Text"
TEXT_REVISION = "d3aad10f2e5a37e7e1e84375f0db368b5872a044"
SAMPLE_PATH = Path("data/raw/spokenwoz/sample_10_train.json")
AUDIO_DIR = Path("data/raw/spokenwoz/audio_5700_train_dev")


def _compact(value: Any) -> Any:
    """Keep populated state values without hiding domain/section nesting."""
    if isinstance(value, dict):
        return {key: item for key, child in value.items() if (item := _compact(child))}
    if isinstance(value, list):
        return [item for child in value if (item := _compact(child))]
    return value if value not in ("", None) else None


def _clean_words(words: list[dict[str, Any]]) -> list[str]:
    return [
        re.sub(r"[^a-z0-9']", "", str(w.get("Word", "")).lower())
        for w in words
        if re.sub(r"[^a-z0-9']", "", str(w.get("Word", "")).lower())
    ]


def summarize(dialogues: dict[str, Any], audio_dir: Path) -> dict[str, Any]:
    """Report observable structure and possible alignment problems."""
    if len(dialogues) > 10:
        raise ValueError("Only 10 conversations may be inspected at a time")
    results: list[dict[str, Any]] = []
    for dialogue_id, dialogue in dialogues.items():
        turns = dialogue.get("log", [])
        issues: Counter[str] = Counter()
        speakers: set[str] = set()
        channels: set[int] = set()
        examples: list[dict[str, Any]] = []
        state_example: dict[str, Any] | None = None
        previous_end: float | None = None
        previous_start: float | None = None
        for index, turn in enumerate(turns):
            speakers.add(str(turn.get("tag", "unknown")))
            words = turn.get("words")
            if not isinstance(words, list) or not words:
                issues["missing_word_timestamps"] += 1
                words = []
            valid: list[tuple[str, float, float]] = []
            for word_index, word in enumerate(words):
                if not isinstance(word, dict):
                    issues["invalid_word_record"] += 1
                    continue
                begin, end = word.get("BeginTime"), word.get("EndTime")
                if not isinstance(begin, (int, float)) or not isinstance(
                    end, (int, float)
                ):
                    issues["missing_or_nonnumeric_timestamp"] += 1
                    continue
                if begin < 0 or end <= begin:
                    issues["invalid_word_interval"] += 1
                if valid and begin < valid[-1][1]:
                    issues["overlapping_word_intervals"] += 1
                if word.get("Word_index") != word_index:
                    issues["word_index_discrepancy"] += 1
                if isinstance(word.get("ChannelId"), int):
                    channels.add(word["ChannelId"])
                valid.append((str(word.get("Word", "")), begin, end))
            if valid:
                first, last = valid[0][1], valid[-1][2]
                if previous_start is not None and first < previous_start:
                    issues["out_of_order_turns"] += 1
                if previous_end is not None and first < previous_end:
                    issues["overlapping_turn_intervals"] += 1
                previous_start, previous_end = first, last
            transcript = re.findall(
                r"[a-z0-9]+(?:'[a-z0-9]+)?", turn.get("text", "").lower()
            )
            if words and _clean_words(words) != transcript:
                issues["text_word_token_mismatch"] += 1
            active_state = _compact(turn.get("metadata", {}))
            if state_example is None and active_state:
                state_example = {"turn": index, "state": active_state}
            if index < 3:
                examples.append(
                    {
                        "turn": index,
                        "speaker": turn.get("tag"),
                        "text": turn.get("text"),
                        "word_timestamps": [[w, b, e] for w, b, e in valid[:5]],
                        "turn_interval": [valid[0][1], valid[-1][2]] if valid else None,
                        "dialogue_state": _compact(turn.get("metadata", {})),
                        "dialogue_acts": turn.get("dialog_act", {}),
                    }
                )
        filename = f"{dialogue_id}.wav"
        audio_path = audio_dir / filename
        if not audio_path.is_file():
            issues["missing_audio"] += 1
        info = sf.info(audio_path) if audio_path.is_file() else None
        results.append(
            {
                "dialogue_id": dialogue_id,
                "number_of_turns": len(turns),
                "speakers": sorted(speakers),
                "speaker_ids_available": False,
                "channels_in_word_metadata": sorted(channels),
                "examples": examples,
                "state_turns": sum(bool(turn.get("metadata")) for turn in turns),
                "state_example": state_example,
                "dialogue_act_turns": sum(
                    bool(turn.get("dialog_act")) for turn in turns
                ),
                "audio": {
                    "filename": filename,
                    "available": bool(info),
                    "sample_rate": info.samplerate if info else None,
                    "channels": info.channels if info else None,
                    "duration_seconds": info.duration if info else None,
                },
                "issues": dict(issues),
            }
        )
    return {"sample_size": len(results), "dialogues": results}


def fetch_official_sample(destination: Path) -> None:
    """Stream only the first ten non-validation records from pinned official JSON."""
    base = f"{TEXT_REPO}/resolve/{TEXT_REVISION}/"
    with urllib.request.urlopen(base + "valListFile.json", timeout=45) as response:
        validation_ids = set(response.read().decode("utf-8").splitlines())
    decoder = json.JSONDecoder()
    utf8_decoder = codecs.getincrementaldecoder("utf-8")()
    with urllib.request.urlopen(base + "data.json", timeout=120) as response:
        buffer = ""
        sample: dict[str, Any] = {}
        started = False
        ended = False
        while len(sample) < 10 and not ended:
            chunk = response.read(65536)
            if not chunk:
                raise ValueError("Official text data ended before ten train records")
            buffer += utf8_decoder.decode(chunk)
            while len(sample) < 10:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer or buffer[0] != "{":
                        raise ValueError("Expected JSON object of dialogues")
                    started = True
                    buffer = buffer[1:]
                    continue
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("}"):
                    ended = True
                    break
                try:
                    dialogue_id, pos = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                suffix = buffer[pos:].lstrip()
                if not suffix:
                    break
                if suffix[0] != ":":
                    raise ValueError("Expected dialogue ID followed by record")
                try:
                    record, consumed = decoder.raw_decode(suffix[1:].lstrip())
                except json.JSONDecodeError:
                    break
                buffer = suffix[1:].lstrip()[consumed:]
                if dialogue_id not in validation_ids:
                    sample[dialogue_id] = record
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--fetch-official-sample", action="store_true")
    args = parser.parse_args()
    if args.fetch_official_sample:
        fetch_official_sample(args.input)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(summarize(data, args.audio_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
