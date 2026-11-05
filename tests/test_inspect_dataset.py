"""Contracts for the small SpokenWOZ inspection tool."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_dataset.py"


def load_script():
    spec = spec_from_file_location("inspect_dataset", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_flags_missing_audio_and_overlap(tmp_path: Path) -> None:
    inspect = load_script()
    dialogues = {
        "MUL0003": {
            "goal": {"restaurant": {}},
            "log": [
                {
                    "tag": "user",
                    "text": "hello there",
                    "metadata": {},
                    "dialog_act": {"general-greet": [["none", "none"]]},
                    "words": [
                        {
                            "Word": "hello",
                            "BeginTime": 10,
                            "EndTime": 50,
                            "ChannelId": 0,
                            "Word_index": 0,
                        },
                        {
                            "Word": "there",
                            "BeginTime": 50,
                            "EndTime": 80,
                            "ChannelId": 0,
                            "Word_index": 1,
                        },
                    ],
                },
                {
                    "tag": "system",
                    "text": "hi",
                    "metadata": {
                        "restaurant": {"semi": {"food": "indian"}, "book": {}}
                    },
                    "dialog_act": {},
                    "words": [
                        {
                            "Word": "hi",
                            "BeginTime": 70,
                            "EndTime": 100,
                            "ChannelId": 0,
                            "Word_index": 0,
                        }
                    ],
                },
            ],
        },
    }
    report = inspect.summarize(dialogues, tmp_path)
    assert report["sample_size"] == 1
    row = report["dialogues"][0]
    assert row["dialogue_id"] == "MUL0003"
    assert row["number_of_turns"] == 2
    assert row["speakers"] == ["system", "user"]
    assert row["audio"]["filename"] == "MUL0003.wav"
    assert row["audio"]["available"] is False
    assert row["audio"]["sample_rate"] is None
    assert row["audio"]["channels"] is None
    assert row["audio"]["duration_seconds"] is None
    assert row["issues"]["overlapping_turn_intervals"] == 1
    assert row["issues"]["missing_audio"] == 1
    assert row["examples"][0]["text"] == "hello there"
    assert row["examples"][0]["word_timestamps"][0] == ["hello", 10, 50]
    assert (
        row["examples"][1]["dialogue_state"]["restaurant"]["semi"]["food"] == "indian"
    )
    assert row["state_example"]["turn"] == 1
    assert row["state_example"]["state"]["restaurant"]["semi"]["food"] == "indian"


def test_rejects_inspection_of_more_than_ten(tmp_path: Path) -> None:
    inspect = load_script()
    with pytest.raises(ValueError, match="10"):
        inspect.summarize({f"MUL{i:04d}": {"log": []} for i in range(11)}, tmp_path)
