# dialogue-jepa

**Status: initial research phase.** This repository documents a proposed test of whether independently trained speech and text predictive dialogue representations converge and whether their shared structure is useful for dialogue-state modeling. It contains no downloaded datasets, trained models, validated results, or production voice agent.

## 1. Project overview and research questions
This project asks whether independently trained audio and text JEPAs can learn geometrically alignable conversation representations. Four questions guide it: (1) can the two independently learned spaces align; (2) how does the quality of paired observations ("good views" versus degraded or mismatched views) affect alignment; (3) can the representations predict future dialogue states or transitions; and (4) do they improve downstream prediction against strong unimodal and conventional multimodal baselines? These are open research questions. **Inference efficiency is not a primary optimization target**: record latency, memory and compute for reproducibility, without constraining model size to chase low inference cost. The Platonic Representation Hypothesis motivates the inquiry but does not establish that it holds for conversation.[9] Audio and text carry overlapping but nonidentical information; a shared component must not be assumed to recover modality-specific cues.

## 2. Hypotheses
- **H1 — emergent alignment:** independently trained speech/text dialogue embeddings align on held-out paired turns after a frozen orthogonal Procrustes map, more than untrained and shuffled-pair controls.
- **H2 — view quality:** verified aligned audio/transcript views outperform weaker positives and intentionally mismatched within-split controls on held-out retrieval.
- **H3 — predictive utility:** future-latent representations improve held-out dialogue-state/transition prediction over strong unimodal and conventional multimodal baselines where comparable labels exist.
- **H4 — generalization and limits:** alignment and prediction hold on unseen conversations and speakers, and ideally compatible separate datasets; failures or distribution shifts bound the claim. These are hypotheses, not findings.

## 3. Approach
Two separate JEPA-style pipelines predict latent representations of future conversational turns from preceding turns: one speech-only, one text-only. A stop-gradient/EMA target encoder and a context encoder/predictor are provisional design candidates, not proven collapse prevention for this setting.[8] No paired cross-modal objective, shared encoder, or shared learned parameters in the primary experiment. Fit orthogonal Procrustes **only after both models are frozen**, on a separate alignment split; evaluate on unseen pairs. See [architecture](docs/architecture.md) and [methodology](docs/methodology.md).

## 4. Research stages
- **A — data/protocol:** confirm rights and audio/text pairing, inspect annotations, define dialogue-level splits, leakage checks, fixed metrics and controls. No dataset acquisition is part of this initial phase.
- **B — unimodal modeling:** independently train speech and text JEPA candidates, establish non-collapse and within-modality prediction/utility before alignment.
- **C — frozen comparison:** fit Procrustes on training-only paired anchors; test held-out alignment, transfer, robustness, baselines and uncertainty. Stages B/C are future work, not completed. See [experiments](docs/experiments.md).

## 5. Candidate data
[SpokenWOZ](https://spokenwoz.github.io/) is the primary *candidate*: speech, text, dialogue-state/act annotations and official train/dev/test distributions; its official site explicitly distributes it under **CC BY-NC 4.0 (noncommercial)**.[1][2] [MELD](https://affective-meld.github.io/) is a secondary, qualitatively different emotion/sentiment candidate from *Friends* audio/video/text, not a substitute for SpokenWOZ's task-oriented state labels.[3][6] Rights, modality alignment and split details must be reviewed before use; see [datasets](docs/datasets.md).

## 6. Provisional decisions
Turn-level representations with causal dialogue context; separate modality-specific encoders and target branches; equal embedding dimensionality for orthogonal comparison; dialogue-level disjoint splits; training/selection/alignment/test partitions. Compare strong pretrained audio/text backbones and architecture families (alongside scratch controls) by representation and downstream quality; do not impose an inference-size cap. Audit pretrained cross-modal supervision before making an independent-convergence claim. Compare multiple predictive objectives, including cosine prediction, variance/covariance-regularized prediction and contrastive future prediction; check collapse and false negatives, rather than assuming mean-squared embedding error is sufficient. Encoder architecture, audio front-end, tokenizer, horizon, masking, EMA schedule, collapse remedy, dimension, seeds and hyperparameters remain **unresolved** and must be declared before evaluation. No implementation choice is claimed final.

## 7. Evaluation
Pre-register held-out paired retrieval (Recall@1, Recall@5, mean reciprocal rank, positive-pair cosine similarity, matched-versus-mismatched retrieval), orthogonal alignment and representation similarity (e.g. CKA), future-latent prediction versus trivial predictors, dialogue-state tracking, next dialogue-act and transition prediction where labels permit. Test verified aligned audio/transcript views, weaker positives and intentionally mismatched within-split controls. Compare text-only, audio-only, strong pretrained and conventional fusion baselines. Evaluate unseen conversations, speaker-disjoint partitions when feasible, and transfer across datasets only when labels and rights are compatible. Record latency, throughput, memory and training compute for reproducibility, **not as primary optimization criteria**. Report paired uncertainty by dialogue and seed, leakage controls, all failures and original units. A low latent loss alone is not evidence of meaningful representation. See [methodology](docs/methodology.md).

## 8. Baselines and falsification
Random/untrained encoders; shuffled alignment anchors; static pooled and nonpredictive unimodal encoders; **strong pretrained text-only and audio-only models**, conventional audio-text fusion, text lexical/ASR and speech-acoustic shortcuts. Supervised or jointly trained cross-modal models are comparison bounds, **not evidence of independent convergence**. Document parameter counts and data budgets and add capacity-matched controls where informative, but do not restrict JEPA or baseline size for inference-cost parity. If models collapse, if alignment vanishes on held-out dialogues, or if probes rely on shortcuts, the corresponding hypothesis is unsupported.

## 9. Reproducibility and scope
Record source revision, data permission, split IDs, pre-processing, seed, architecture, hardware, checkpoint selection and confidence intervals when experiments exist. Prevent same-dialogue and speaker overlap where feasible; do not use held-out labels for model selection. This is research documentation only: no trained checkpoint, benchmark score or claimed result exists. See [results](docs/results.md).

## 10. Privacy and ethics
Speech can reveal identity and sensitive attributes; transcript and emotion labels can expose private context or imperfect human judgments. Store licensed data outside version control with restricted access; minimize retained identifiers, avoid public raw audio or transcript examples, respect takedown/consent requirements, and avoid identity or emotion profiling deployments. Dataset publication is not proof of unrestricted downstream permission.

## 11. Licensing
Repository-authored code and documentation are offered under [MIT](LICENSE). **MIT does not relicense datasets, source recordings, third-party code, pretrained weights, or derived model/checkpoint rights.** SpokenWOZ's CC BY-NC 4.0 noncommercial restriction applies to its dataset; verify any intended commercial use with the rights holder.[1][5] MELD's GitHub repository has a GPL-3.0 license file, but that fact alone does not establish rights to the underlying *Friends* clips or every dataset artifact; review separately before use.[4][7]

## 12. Documentation map
- [Architecture](docs/architecture.md) — pipelines and separation boundary.
- [Methodology](docs/methodology.md) — hypotheses, metrics and safeguards.
- [Experiments](docs/experiments.md) — staged protocol, baselines and decision gates.
- [Datasets](docs/datasets.md) — source annotations, gaps and rights.
- [Results](docs/results.md) — empty, explicitly unrun results register.

## 13. References and current status
Research framing: Huh et al., *Position: The Platonic Representation Hypothesis*.[9] JEPA precedent: Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*.[8] Dataset primary sources are cited above and detailed in [datasets](docs/datasets.md). **Current state: initial research phase; no data acquired, experiments run, or results reported.**

## Repository structure

```text
configs/{data,model,training,experiments}/   # Provisional YAML protocols
src/dialogue_jepa/{data,models,training,evaluation,utils}/  # Importable placeholders
scripts/                                   # Nonfunctional future entry points
data/{raw,processed,manifests,splits}/     # Empty, ignored data locations
tests/                                     # Structural/import tests; model tests pending
docs/                                      # Architecture, methodology, experiments, datasets, results
demo/                                      # Not-implemented inference/demo placeholders
experiments/  checkpoints/  notebooks/    # Empty, ignored artifact locations
```

## Development environment

Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) are required. From the repository root:

```bash
uv sync                         # Creates .venv and installs runtime + dev dependencies
uv pip install -e .              # Explicit editable install (uv sync also installs the project editable)
uv run python -c "import dialogue_jepa; print(dialogue_jepa.__name__)"
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
# Optional demo dependencies only if future demo development is needed:
uv sync --extra demo
```

`uv.lock` records the resolved environment. CPU-only structural tests do not require CUDA; no dataset download or model training occurs in setup. The runnable scripts and demo entry points deliberately raise `NotImplementedError`. Keep `.env` private, and do not commit downloaded data, weights or experiment outputs.

## Planned milestones

1. Review per-asset licenses and availability; audit audio/transcript pairs and dialogue-disjoint splits without leaking test conversations.
2. Compare strong pretrained audio/text encoders and architecture families, with scratch controls; define context–target sampling and test several predictive objectives and collapse-prevention protocols; implement and test single-modality data ingestion and baselines.
3. Independently train audio and text JEPA candidates and evaluate non-collapse and held-out prediction.
4. Freeze both encoders; compare verified aligned pairs, weaker positives and intentionally mismatched controls, Procrustes retrieval and downstream tasks against strong unimodal and conventional multimodal baselines on unseen conversations, speakers and compatible datasets. Record latency and compute for reproducibility, not optimization.

## Sources

[1] https://spokenwoz.github.io
[2] https://arxiv.org/html/2305.13040v7
[3] https://affective-meld.github.io
[4] https://github.com/declare-lab/MELD
[5] https://creativecommons.org/licenses/by-nc/4.0
[6] https://aclanthology.org/P19-1050
[7] https://raw.githubusercontent.com/declare-lab/MELD/master/LICENSE
[8] https://arxiv.org/abs/2301.08243
[9] https://proceedings.mlr.press/v235/huh24a.html
