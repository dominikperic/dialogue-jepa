# SpokenWOZ five-recording audio validation

## Scope and provenance

Five dialogue WAVs were selectively extracted from the publisher-linked [official training/development audio repository](https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Audio) at revision `5748a6e1b67b26a0e2fca5df061ed7ed2f782eab`. A guarded HTTP range request required `206`, `Content-Range: bytes 0-50331647/10341654034`, and the expected 48 MiB response length; a counting wrapper refused reads over 48 MiB. The streamed gzip tar was inspected in order, allowing only five exact `audio_5700_train_dev/<ID>.wav` members, rejecting unsafe paths/duplicates/wrong sizes/incomplete members. The stream closed after **29,102,080 compressed bytes** had been consumed. The **10,341,654,034-byte** archive was not saved. The extracted files are under git-ignored `data/raw/spokenwoz/audio_5700_train_dev/`; no WAV or review clip is committed.

The text source is the existing ten-dialogue `data/processed/spokenwoz/sample_10_turns.jsonl` (created before these WAVs existed). This task **did not rewrite that manifest**: its prior `missing_audio` flags are a historical snapshot, not the current audio status. The new separate report is `data/processed/spokenwoz/audio_validation_report.json`.

| Dialogue | Verified WAV bytes | Sample rate | Channels | Frames | WAV duration (s) | Maximum annotated end (raw) | Max / 1000 (s) | Turns in bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MUL0003 | 6,361,644 | 8,000 Hz | 2 | 1,590,400 | 198.800 | 196,910 | 196.910 | 42/42 |
| MUL0006 | 3,231,404 | 8,000 Hz | 2 | 807,840 | 100.980 | 99,160 | 99.160 | 24/24 |
| MUL0008 | 11,297,324 | 8,000 Hz | 2 | 2,824,320 | 353.040 | 351,730 | 351.730 | 60/60 |
| MUL0009 | 6,221,868 | 8,000 Hz | 2 | 1,555,456 | 194.432 | 193,430 | 193.430 | 36/36 |
| MUL0013 | 6,954,924 | 8,000 Hz | 2 | 1,738,720 | 217.340 | 216,230 | 216.230 | 54/54 |

File sizes matched tar headers exactly. `soundfile` opened all five WAVs. No full archive download, models, embeddings, or training were involved.

## Timing and segment checks

`uv run python scripts/validate_audio.py --export-review` loads only these five files and uses each manifest turn's raw start/end against a **candidate** `1000 raw units/second` conversion. All **216/216 turns (100%)** fit inside their WAV's frame count with this conversion; `1 unit/second` fits **0/216**, and `100 units/second` fits **24/216**. The last annotated time in each dialogue is within roughly 1–2 seconds of its recording duration. Windowed audio activity in the expected role channel further supports millisecond-like units. Taken together, these are substantially stronger evidence for milliseconds than numerical plausibility alone, but **not** a word-level forced alignment or transcript correctness test. Raw annotations and original manifest remain unchanged.

Segments use the exact annotated interval; **20/216 turns** carry an adjacent-turn overlap flag and are **not** clipped into disjoint windows. Example: `MUL0008` turn 26 (user) spans an overlap and has RMS ~0.0398 on channel 0 and ~0.0261 on channel 1, consistent with concurrent activity worth listening to. The tool flags durations below 0.15 s, above 30 s, RMS below 0.002, and out-of-range segments. None of those four flags occurred in these five recordings; observed segment lengths ranged about 0.92–23.51 s. Six turns retain a prior text/word-token mismatch review flag, which may reflect punctuation or time-expression normalization rather than a real transcription error.

## Channel-to-role evidence and limits

Each WAV is stereo. On **nonoverlapping, clearly dominant** intervals, the tool found **95 user windows** dominated by channel 0 and **98 system windows** dominated by channel 1; **three nonoverlapping system windows** had no channel at least 3× stronger. In `MUL0008` several overlapping windows have substantial energy on both channels, and a system window in an overlap can even be louder on channel 0. The proposed mapping for these five files is **channel 0 → user, channel 1 → system** for clear speech. This is a measured *activity-based* mapping, not an assumption about all SpokenWOZ files; the six reported human clip reviews do not independently validate the channel mapping throughout all five recordings. Mixed/overlapping turns should retain both channels until their overlap and channel assignment are specifically reviewed. `ChannelId` in text word records is not accepted as independent proof of speaker identity.

## Human review clips (user-reported)

Six clips were exported to `data/processed/spokenwoz/review_clips/`. The user reports listening to all six: **five passed transcript review; one needs review**. Their `review_manifest.jsonl` retains the dataset transcript and records dialogue/turn ID, path, raw and candidate-second window, selected channel, review status and source, and the reported uncertainty. A `pass` applies only to the reviewed clip and does not automatically validate any other turn, all channel assignments, or an overlap.

| Clip | Channel | Reported transcript review | Separate caveat |
| --- | --- | --- | --- |
| `MUL0003_turn05_ch1.wav` | 1 | pass | — |
| `MUL0006_turn00_ch0.wav` | 0 | **needs_review** | Uncertain restaurant name; exclude this turn from the initial high-confidence training-pair subset until resolved. |
| `MUL0008_turn26_stereo.wav` | both | pass | Timestamp overlap flagged; actual overlapping speech **not independently confirmed**. |
| `MUL0008_turn01_stereo.wav` | both | pass | Mixed channel activity is not itself speaker confirmation. |
| `MUL0009_turn02_ch0.wav` | 0 | pass | — |
| `MUL0013_turn01_ch1.wav` | 1 | pass | — |

**Transcript discrepancy, not a correction:** `MUL0006` turn 0 has original annotation “hi , i am looking for a restaurant called papa restaurant .”. The reported human-perceived alternative is “hi , i am looking for a restaurant called Pepasha restaurant .”. The proper noun is uncertain: neither spelling is established as the actual restaurant name. The dataset annotation and WAV are unchanged; the alternative is recorded only in the review manifest. Re-listen or consult independent evidence before treating this turn as a high-confidence training pair.

**Identifier correction:** The reported discrepancy was initially attributed to `MUL0008_turn26_stereo.wav`. The existing manifest and source annotations show that clip instead corresponds to `MUL0008` turn 26, “oh , no , i'm good .”. The user clarified the restaurant-name discrepancy belongs to `MUL0006_turn00_ch0.wav` and reports the other five clips passed. The **timestamp-derived overlap flag for MUL0008 turn 26** remains separate from transcription review; no listening-based confirmation of actual overlapping speech is claimed.

**Regeneration caution:** `scripts/validate_audio.py --export-review` currently recreates the review manifest with `not_listened_to` statuses. Do **not** rerun it over these review records without first preserving/merging the human-review fields. Running the validator *without* `--export-review` leaves this manifest untouched. Continue targeted review of the 20 timestamp-overlap windows and six prior text/word-token flags as needed; pass on five clips does not imply all 216 turns are correct.

Run tests and validate audio without replacing human review records:

```bash
uv run python scripts/validate_audio.py
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The dataset is [CC BY-NC 4.0](https://spokenwoz.github.io/) (attribution and noncommercial use). Neither downloaded WAVs nor review excerpts should be redistributed without confirming the applicable permissions.
