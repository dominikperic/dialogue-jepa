# SpokenWOZ dataset inspection — ten training conversations

**Scope:** Exactly ten multi-domain conversations from the official [SpokenWOZ training text repository](https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Text), at revision `d3aad10f2e5a37e7e1e84375f0db368b5872a044`. The official [project site](https://spokenwoz.github.io/) links that repository and the [training/development audio repository](https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Audio). No audio archive, model, or embedding was downloaded or generated. The sample is locally stored at the git-ignored `data/raw/spokenwoz/sample_10_train.json` (~728 kB); it is **not** committed or redistributed.

## Availability and acquisition

- The official train-text repository exposes `data.json` (246,209,221 bytes, top-level dialogue-ID dictionary), `valListFile.json` (newline-delimited development IDs despite its `.json` suffix), `ontology.json`, and `db/*.json`. The script streams just the first ten records *not* in the development-ID list from pinned `data.json`, stopping before the entire text file is transferred. It does not use a third-party mirror or the auto-converted Hugging Face viewer.
- The official [train/dev audio tree](https://huggingface.co/datasets/ssz1111/SpokenWOZ-Train-Audio/tree/main), revision `5748a6e1b67b26a0e2fca5df061ed7ed2f782eab`, exposes **one** `audio_5700_train_dev.tar.gz` (10,341,654,034 bytes), not individually downloadable WAV objects. The repository's preview references filenames inside that archive, and its viewer currently reports `DatasetGenerationError`. **The archive was not downloaded**. Per-file existence, waveform timing, sample rate, channel count, and duration cannot be independently verified here.
- Reproduce the small text sample with `uv run python scripts/inspect_dataset.py --fetch-official-sample > inspection.json`; inspect the existing sample with `uv run python scripts/inspect_dataset.py > inspection.json`. The command reports a maximum of ten dialogue records; `--audio-dir PATH` checks locally extracted matching WAVs using `soundfile.info` without decoding them. The default expected extracted location is `data/raw/spokenwoz/audio_5700_train_dev/`.

## Record organization

`data.json` maps IDs such as `MUL0003` to `{ "goal": {...}, "log": [...] }`. A log entry is one turn with `tag` (`user` or `system`), `text`, `words`, `dialog_act`, `span_info`, and `metadata`. Each word may contain `Word`, `BeginTime`, `EndTime`, `ChannelId`, and `Word_index`. `metadata` holds per-domain `semi` and `book` dialogue-state sections, mainly on system turns; `dialog_act` gives act/slot/value labels, while `span_info` contains annotated spans. `goal` describes the conversation-level task. There are no distinct speaker identity labels in these sampled text records: `tag` is a **role**, not an individual speaker ID; `ChannelId` is a word metadata field and cannot alone establish speaker identity.

| Training dialogue | Turns | Turns with `metadata` | Turns with dialogue acts | Expected WAV |
| --- | ---: | ---: | ---: | --- |
| MUL0003 | 42 | 21 | 40 | MUL0003.wav |
| MUL0005 | 70 | 35 | 51 | MUL0005.wav |
| MUL0006 | 24 | 12 | 23 | MUL0006.wav |
| MUL0007 | 58 | 29 | 49 | MUL0007.wav |
| MUL0008 | 60 | 30 | 49 | MUL0008.wav |
| MUL0009 | 36 | 18 | 34 | MUL0009.wav |
| MUL0010 | 52 | 26 | 51 | MUL0010.wav |
| MUL0013 | 54 | 27 | 50 | MUL0013.wav |
| MUL0014 | 88 | 44 | 65 | MUL0014.wav |
| MUL0018 | 30 | 15 | 29 | MUL0018.wav |

These ten dialogues contain **514 turns** and **4,238 word records**. The table reports a nonempty `metadata` dictionary, not necessarily populated slot values; most state templates contain blank fields. All expected WAVs are **unavailable locally**; their names come from the official documented ID-to-WAV naming rule, not verified archive contents. Audio properties are therefore null in the script report.

### Example: MUL0003 (first three of 42 turns)

| Turn | Role | `text` | First word timing (`BeginTime`–`EndTime`, raw units) | Dialogue act |
| ---: | --- | --- | --- | --- |
| 0 | user | “hello is this customer service center .” | `hello` 240–720 | `general-greet` |
| 1 | system | “yes , this is customer service center . how may i help .” | `yes` 2780–3007 | `general-reqmore` |
| 2 | user | “well , i'm looking for a place to dine . do you have any recommendation for me .” | `well` 6230–6448 | `restaurant-inform` |

The first populated dialogue-state slot in this conversation occurs at **turn 5 (system)**: `metadata.restaurant.semi.area = "centre"`. A sampled span annotation uses `["restaurant-inform", "area", "centre", 10, 11]`. Some system turns have a full but empty state template; user turns frequently have `metadata: {}`. The timestamps above are raw dataset values: **milliseconds are plausible but unverified without the WAV**.

## Pairing and quality checks

- The official text card documents a dialogue-level filename rule (e.g. `MUL0032` → `MUL0032.wav`). The sampled text offers word intervals that could define utterance windows within a **conversation-level** WAV, not separate per-turn WAV filenames. This is a promising pairing key, **not proof of correct alignment**.
- Every sampled turn has a nonempty word list with numeric intervals; there were no negative/reversed word intervals or out-of-order turn starts in this sample. But **111 adjacent turn pairs overlap in time** (e.g. MUL0003's first user ends at 2880 while the system starts at 2780). Some may reflect actual interruption; others may be annotation error. Do not crop turns under a strict non-overlap assumption.
- **436 word records** have a `Word_index` differing from their position within the turn (across 91 turns); observed examples reset to `0` mid-turn. Do not use this index alone as a reliable sequence/offset key.
- **13 turns** differ under a naive `text` versus `words` token comparison; some apparent mismatches arise from time expressions (`1:45`, `15:50`) or punctuation normalization, so this is a *review flag*, not a verified transcript error.
- **10/10 WAV files were unavailable locally.** No waveform-to-word offset check, sample rate/channel/duration measurement, speaker/channel mapping, or listening-based transcript check was possible. The Hugging Face audio viewer also fails to generate, though the archive itself is listed. In particular, matching a dialogue ID to an expected WAV does **not** demonstrate that word intervals fall within that file's duration or that role/channel assignments match the recording.

**Assessment:** SpokenWOZ appears *promising* for paired speech/text observations because it includes dialogue-level audio naming, transcript turns, word timing and dialogue-state labels. Alignment remains **unvalidated** until a bounded, authorized audio access route exists and sample recordings can be checked for existence, duration, channels, timing units, speaking roles and transcription consistency. Separate-speaker generalization also requires speaker identity metadata or another justified protocol; role tags are insufficient.

## License and usage

The [official site](https://spokenwoz.github.io/) says the dataset is distributed under **CC BY-NC 4.0**, requiring attribution and prohibiting commercial use absent separate permission. Check rights before using it in commercial work, redistributing audio/text, or distributing derived model weights; a repository MIT license does not override data terms. The locally sampled records remain under `data/raw/` and are ignored by git.
