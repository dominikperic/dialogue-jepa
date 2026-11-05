# Dataset candidates and rights review

**Status: candidate review only. No dataset acquired.** Dataset distribution pages and repository licenses are not substitutes for an asset-by-asset use-rights review. Keep raw audio, transcript, identifiers and checkpoints out of this repository.

## SpokenWOZ — primary task-oriented candidate

The official project reports 203k turns, 5.7k dialogues and 249 hours of audio across eight domains, with train/dev/test distribution and separately linked audio/text releases.[1][2] Its paper describes dialogue-state and dialogue-act annotations, including cross-turn and reasoning slots; check the actual release schema and correspondence between audio and text before designing labels/windows.[2] Official baseline material and evaluation instructions are linked from the project page.[1]

**License:** the official dataset site says **CC BY-NC 4.0**; the authors' paper also confirms this.[1][2] The Creative Commons deed requires attribution and prohibits commercial use; it cautions that privacy/publicity rights can remain outside the license.[5] An MIT license for *this repo* cannot override SpokenWOZ's noncommercial restriction. Do not redistribute samples, trained derivatives or claim commercial usability without a separate rights assessment/permission. The paper's reported dataset statistics are not project results.

**Limitations:** human-to-human task-oriented data do not automatically represent deployed human–agent interaction; ASR errors, audio/transcript timing and split leakage need inspection. The labels support task-oriented state tests, not an assertion that every nuance of conversational meaning is annotated. Official site: [SpokenWOZ](https://spokenwoz.github.io/); paper: [SpokenWOZ](https://arxiv.org/html/2305.13040v7).

## MELD — secondary conversational candidate

The official project describes >1,400 dialogues and ~13,000 utterances from *Friends*, with audio, video, text and per-utterance seven-way emotion plus three-way sentiment labels.[3][6] The official GitHub README provides train/dev/test statistics and annotation-field descriptions, including speaker, dialogue/utterance IDs and source episode/timing; verify whether the chosen distributed archive has all corresponding clips.[4] **These labels are not dialogue-state tracking slots** and cannot be substituted for SpokenWOZ state annotations. Scripted television speech, short clips, speaker/episode overlap and affect-label subjectivity limit real-world generalization.

**Rights ambiguity:** the official MELD GitHub project contains a GPL-3.0 license file.[4][7] That is not evidence that *Friends* recordings, captions, dataset bundles, and downstream model weights are all GPL-3.0 or commercially reusable. The project website identifies TV-series origin and provides downloads, but the pages reviewed do not establish clean commercial rights for those audiovisual assets.[3][4] Do not import clips or redistribute derivatives until rights have been checked with the owners and applicable terms documented. Official site: [MELD](https://affective-meld.github.io/); paper: [ACL Anthology](https://aclanthology.org/P19-1050/).

## Intake checklist (pending)

For each candidate record official source/revision, download URL, license for **each** transcript/audio/video/annotation asset, privacy/consent and commercial rights, permitted derivative distribution, modality-pair join keys, missing clips, speaker and dialogue/episode leakage, label schema, official splits and required attribution. Raw data remain outside git; follow least-privilege access and deletion requests. If permission or pairing cannot be established, choose another dataset or narrow claims rather than assuming authorization.

## Sources

[1] https://spokenwoz.github.io
[2] https://arxiv.org/html/2305.13040v7
[3] https://affective-meld.github.io
[4] https://github.com/declare-lab/MELD
[5] https://creativecommons.org/licenses/by-nc/4.0
[6] https://aclanthology.org/P19-1050
[7] https://raw.githubusercontent.com/declare-lab/MELD/master/LICENSE
