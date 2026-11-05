# SpokenWOZ ten-dialogue turn manifest validation

## Scope and reproduction

Input: git-ignored `data/raw/spokenwoz/sample_10_train.json` from the official [SpokenWOZ training text release](https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Text), revision `d3aad10f2e5a37e7e1e84375f0db368b5872a044`, collected in the [initial inspection](dataset_inspection.md). This task reads the existing sample only; it does not fetch audio, generate embeddings or train anything.

Run from the repository root:

```bash
uv run python scripts/build_turn_manifest.py
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The generator **overwrites** (rather than appends to) `data/processed/spokenwoz/sample_10_turns.jsonl`, which is ignored by git. Running it twice produced 514 lines with the same SHA-256 (`15de8117f6edfd6df08908aebdf8a6a2c369b577d6c163671757250f75f3fc18`). The output is not a verified paired audio/text dataset.

## Schema and validation semantics

Each JSONL record contains `dialogue_id`, zero-based `turn_id` in original `log` order, `role` (`user` or `system` in this sample), unmodified `transcript`, `start_time_raw`, `end_time_raw`, `expected_audio_filename`, original `dialogue_act` and `dialogue_state` (`metadata`), `timing_validation_status`, `audio_validation_status`, and `validation_flags`. Missing `metadata` stays `null`, while an explicitly empty state remains `{}`. The expected filename is inferred as `<dialogue_id>.wav`; it is not a verified file match.

Turn boundaries are the minimum start and maximum end among **valid positive-duration word intervals**. Empty word lists, absent or nonnumeric timestamps, negative/reversed or zero-duration intervals and negative timestamps are flagged; a turn with only invalid intervals has null boundaries. Overlapping adjacent turns and nonmonotonic starts are flagged **without dropping or reordering** turns. Comparisons do not bridge a preceding turn lacking boundaries. A conservative case/punctuation-normalized `text` versus `words` check flags possible transcript mismatches, not confirmed transcription errors.

`timing_validation_status` distinguishes `valid_metadata_unverified_audio`, `needs_review_unverified_audio` (turn overlap or ordering issue), and `invalid_metadata` (bad/missing word timing). `audio_validation_status` is `missing_audio` or `file_present_alignment_unverified`; mere file presence never means the waveform is aligned. Raw timestamp values are retained **without conversion to seconds**, since their units have not been verified against audio.

## Observed results (these ten conversations only)

| Measure | Count |
| --- | ---: |
| Dialogues | 10 |
| Manifest turns | 514 |
| Turns with valid word-interval timestamp metadata | 514 |
| Turns with overlapping adjacent boundaries | 111 |
| Turns with nonmonotonic starts | 0 |
| Turns with transcript/word token mismatch flags | 13 |
| Turns with missing expected local audio | 514 |
| Turns with missing/invalid word timestamps or negative-duration intervals | 0 |
| Unexpected exceptions | 0 |

Of the 514 turns with valid *word-level interval metadata*, 403 are `valid_metadata_unverified_audio` and 111 are `needs_review_unverified_audio` due to overlap. Thus “valid timestamps” does **not** assert non-overlapping turns or verified audio. Some overlap may be genuine interruption; mismatches may be punctuation/time-expression tokenization differences. The initial inspection also found `Word_index` resets in some records, so the generator deliberately uses source turn order and never sorts by `Word_index`.

## Still unverified

The official train/dev audio is distributed as a single large archive, which was not downloaded. All 514 turns therefore lack local WAV verification: no sample rate, channel mapping, duration, word-offset units, diarization, or listening-based transcript check can be established. Conversation IDs and candidate filenames alone do not prove a synchronized recording. Any later waveform pairing must separately verify these properties and review the 111 overlap flags. SpokenWOZ data are distributed under **CC BY-NC 4.0**; rights review is required before commercial use or redistribution.
