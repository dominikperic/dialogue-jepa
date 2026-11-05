"""Synthetic WAV tests for bounded audio validation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_audio.py"


def module():
    spec = spec_from_file_location("validate_audio", SCRIPT)
    assert spec is not None and spec.loader is not None
    result = module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def wav(tmp_path: Path) -> Path:
    p = tmp_path / "MUL0003.wav"
    x = np.zeros((8000, 2), dtype=np.float32)
    x[:4000, 0] = 0.2 * np.sin(2 * np.pi * 220 * np.arange(4000) / 8000)
    x[3200:7200, 1] = 0.1 * np.sin(2 * np.pi * 330 * np.arange(4000) / 8000)
    sf.write(p, x, 8000)
    return p


def test_audio_metadata_and_millisecond_window(tmp_path: Path) -> None:
    m = module()
    p = wav(tmp_path)
    info = m.audio_metadata(p)
    assert info["sample_rate"] == 8000
    assert info["channels"] == 2
    assert info["frame_count"] == 8000
    assert info["duration_seconds"] == 1.0
    window = m.validate_window(100, 500, info, units_per_second=1000)
    assert window["in_bounds"] is True
    assert window["start_frame"] == 800
    assert window["end_frame"] == 4000


def test_invalid_conversion_and_out_of_bounds(tmp_path: Path) -> None:
    m = module()
    info = m.audio_metadata(wav(tmp_path))
    assert not m.validate_window(100, 1500, info, 1000)["in_bounds"]
    assert not m.validate_window(-1, 10, info, 1000)["in_bounds"]
    assert not m.validate_window(None, 10, info, 1000)["in_bounds"]
    assert not m.validate_window(100, 500, info, 1)["in_bounds"]
    with pytest.raises(ValueError, match="positive"):
        m.validate_window(0, 1, info, 0)


def test_channel_selection_and_overlap_kept(tmp_path: Path) -> None:
    m = module()
    p = wav(tmp_path)
    a = m.channel_activity(p, 0, 3200)
    b = m.channel_activity(p, 4000, 6400)
    overlap = m.channel_activity(p, 3200, 4000)
    assert m.dominant_channel(a["rms"]) == 0
    assert m.dominant_channel(b["rms"]) == 1
    assert overlap["rms"][0] > 0 and overlap["rms"][1] > 0
    rows = [
        {
            "turn_id": 0,
            "role": "user",
            "start_time_raw": 0,
            "end_time_raw": 500,
            "validation_flags": [],
        },
        {
            "turn_id": 1,
            "role": "system",
            "start_time_raw": 400,
            "end_time_raw": 900,
            "validation_flags": ["overlapping_adjacent_turns"],
        },
    ]
    report = m.analyze_dialogue(p, rows)
    assert len(report["turns"]) == 2
    assert report["turns"][1]["start_frame"] == 3200
    assert "overlapping_adjacent_turns" in report["turns"][1]["flags"]


def test_present_audio_does_not_inherit_stale_missing_flag(tmp_path: Path) -> None:
    m = module()
    p = wav(tmp_path)
    row = {
        "turn_id": 0,
        "role": "user",
        "start_time_raw": 0,
        "end_time_raw": 500,
        "validation_flags": ["missing_audio"],
    }
    turn = m.analyze_dialogue(p, [row])["turns"][0]
    assert "missing_audio" not in turn["flags"]
    assert "missing_audio" in turn["manifest_flags"]


def test_export_clip_preserves_overlap_window_and_channel(tmp_path: Path) -> None:
    m = module()
    p = wav(tmp_path)
    mono = tmp_path / "mono.wav"
    stereo = tmp_path / "stereo.wav"
    m.export_clip(p, 3200, 4000, mono, channel=0)
    m.export_clip(p, 3200, 4000, stereo, channel="both")
    assert sf.info(mono).channels == 1
    assert sf.info(stereo).channels == 2
    assert sf.info(mono).frames == sf.info(stereo).frames == 800


def test_missing_file_and_silent_segment(tmp_path: Path) -> None:
    m = module()
    with pytest.raises(FileNotFoundError):
        m.audio_metadata(tmp_path / "absent.wav")
    p = wav(tmp_path)
    a = m.channel_activity(p, 7200, 8000)
    assert max(a["rms"]) < 0.002
    report = m.analyze_dialogue(
        p,
        [
            {
                "turn_id": 0,
                "role": "user",
                "start_time_raw": 900,
                "end_time_raw": 1000,
                "validation_flags": [],
            }
        ],
    )
    assert "silent_segment" in report["turns"][0]["flags"]
    assert "short_segment" in report["turns"][0]["flags"]
