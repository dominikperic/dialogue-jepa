# Pair-eligibility audit: five SpokenWOZ recordings

**Scope:** The five downloaded dialogues `MUL0003`, `MUL0006`, `MUL0008`, `MUL0009`, and `MUL0013`; **216 turns**. This is a snapshot of current evidence, not a dataset-wide transcript or speaker validation. Input: the existing turn manifest, audio-validation report, and six-clip human-review manifest. The audit **reads** these files; it does not regenerate or overwrite them. The per-turn machine-readable output is git-ignored `data/processed/spokenwoz/pair_eligibility_audit.jsonl`.

## Decision rule (from [good-views protocol](good_views_protocol.md))

- **`high_confidence` (initial):** current source WAV exists; independently computed time window is in bounds; per-turn audio activity clearly supports the proposed role channel (`user → 0`, `system → 1`), and the reviewed clip used that channel; its transcript-review status is `pass`; **separate, explicit confirmation of audible timing alignment and role-channel assignment is recorded**; no timestamp-overlap or transcript/word mismatch flags and no unresolved segment-quality flags. The original six review records confirm transcripts only: their clip channels were selected from the activity-based mapping, so agreement between those two fields is *not* an independent channel check. There are no such independent timing/channel confirmations yet. Even a future high-confidence label would be a candidate anchor, not proof of every word or speaker identity throughout a dialogue.
- **`needs_review`:** potentially usable, but no human review yet, an uncertain proper noun, ambiguous/conflicting channel evidence or selection, timestamp-derived overlap not independently resolved, or transcript/word mismatch. A missing review is **unknown**, neither a pass nor an audio failure. A text-review pass does not clear a separate overlap/channel issue. The explicit `exclude_until_resolved` note for `MUL0006` turn 0 means *exclude from the initial high-confidence subset*, not that its WAV is unusable forever.
- **`excluded`:** currently unusable WAV, invalid/out-of-bounds timing, or explicit failed human review. None was found here. This does **not** imply every non-excluded turn has been verified.

The original ten-dialogue manifest's `missing_audio` flags predate extraction, so current presence is checked against disk, not inherited from those flags. Validation/review turns must join exactly by dialogue and turn ID; reviewed text/role/windows must still match the source manifest. An audit error aborts rather than silently attaching a review to a different utterance. No annotation or review result is rewritten.

## Results

| Dialogue | Inspected | High confidence | Needs review | Excluded |
| --- | ---: | ---: | ---: | ---: |
| MUL0003 | 42 | 0 | 42 | 0 |
| MUL0006 | 24 | 0 | 24 | 0 |
| MUL0008 | 60 | 0 | 60 | 0 |
| MUL0009 | 36 | 0 | 36 | 0 |
| MUL0013 | 54 | 0 | 54 | 0 |
| **Total** | **216** | **0** | **216** | **0** |

**Zero turns meet the complete high-confidence gate.** The best three *provisional candidates* are `MUL0003/5`, `MUL0009/2`, and `MUL0013/1`: each has a transcript-review pass, in-bounds timing and strong activity in its selected role channel, but **none has separately recorded audible timing and role-channel confirmation**. Among the six user-reported transcript reviews, five were `pass` and one (`MUL0006/0`) has the unresolved “papa” / human-perceived “Pepasha” proper noun. `MUL0008/1` passed transcript review but has no clearly dominant role channel and was exported in stereo; `MUL0008/26` also passed transcript review but has mixed channel activity and a separate timestamp-overlap flag whose audible overlap is unconfirmed.

**Reason counts are overlapping, not a partition** (one turn can have several reasons):

| Independent review reason | Turns |
| --- | ---: |
| Human transcript review missing | 210 |
| Timestamp-overlap flag unresolved | 20 |
| Transcript/word-token mismatch flag | 6 |
| Independently recorded audible timing confirmation missing on transcript-passed clips | 5 |
| Independently recorded role-channel confirmation missing on transcript-passed clips | 5 |
| Ambiguous role-channel activity | 5 |
| Human-reviewed clip used both channels (single-role selection not verified) | 2 |
| Human-reported transcript uncertainty | 1 |
| Audio unavailable / invalid timing / failed human review | 0 |

For example, **19** overlap-flagged turns also lack a transcript review, and **all 6** transcript/word mismatch flags occur on turns lacking a human review; these counts must **not** be added to 216. The five ambiguous channel-activity windows include two overlap-flagged turns. All 216 selected WAV windows were in bounds under the tested millisecond-like conversion, and all five source WAVs were present; neither fact independently confirms the audible alignment or the speaker's channel.

**Further human verification:** **216 turns** currently need some kind of additional review to enter the initial high-confidence subset: **210** have no transcript review, **one** has the unresolved name, and **five** passed transcript review but lack independent timing and channel confirmation (two of these five also need targeted channel/overlap assessment). Within the 210 unreviewed turns, flags prioritize 19 overlapping windows and all six token-mismatch windows; other flagged windows may overlap these groups. Record audible timing and actual role-channel confirmation *separately* from transcript acceptance; review overlapping speech where flagged, resolve the proper noun without editing source annotations, and keep transcript, timing, channel, and overlap conclusions separate. More broadly, identity/disjoint-speaker claims require verified speaker information, which is currently absent.

**Alignment experiment readiness: no.** There are **no fully eligible anchors**; even the three best provisional candidates span only three conversations and cannot supply the protocol's disjoint encoder-training, model-selection, alignment-fit, and untouched-evaluation dialogue partitions with meaningful held-out retrieval pools. This is a structural split/evidence limitation, **not** an invented minimum number of training examples. The five recordings establish a workable audit method but not a defensible alignment result. A larger *licensed and verified* paired pool, enough dialogue-disjoint partitions/candidates, audited review coverage and independent timing/channel decisions are prerequisites before any alignment experiment. No data acquisition or model work was performed in this audit.

Reproduce only the derived audit (from the repository root):

```bash
uv run python scripts/audit_pair_eligibility.py
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The audit CLI refuses an output path identical to any source input. In particular, do **not** run `scripts/validate_audio.py --export-review`: that separate exporter would reset the recorded human-review statuses.
