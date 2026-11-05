# Experiment protocol (planned, not executed)

## Stage A — source and evaluation readiness

1. Resolve per-asset data rights and access; no automatic download or redistribution. Confirm dialogue/turn/audio/transcript IDs, audio availability, gold versus ASR transcript, label definitions, split counts and missingness from official releases.[1][3][4]
2. Freeze dialogue-disjoint train, development, alignment-fit and untouched test protocol; document speaker/episode leakage risk and pre-register retrieval pool, downstream tasks, primary metric and uncertainty method.
3. Establish trivial baselines and leakage tests; stop if pairs, permission, or split isolation cannot be established. **Status: not run.**

## Stage B — independent predictive learning

1. Compare strong pretrained speech and text encoder families and architecture variants alongside scratch controls; check pretrained provenance and do not constrain parameter count for inference efficiency. Specify context window and horizons; train separate unimodal encoders/predictors from declared initial conditions. No paired cross-modal loss and no shared learned parameters.
2. Compare mean-squared embedding prediction against cosine, variance/covariance-regularized and contrastive future objectives (check false negatives). Validate each against static/nonpredictive encoders, shuffled-future and mean/last-state predictors. Diagnose collapse with variance, effective rank and covariance spectrum on held-out samples. EMA target branch, masking and any explicit variance/covariance regularization are competing choices, not a settled recipe.[8]
3. Keep architecture selection on dev only; publish seeds, compute and failures. If embeddings collapse or predictive gain fails, stop before claiming alignment. **Status: not run.**

## Stage C — post-hoc held-out comparison

1. Freeze both encoders. Compare verified same-utterance audio/transcript pairs, weaker positives (predefined noisy transcript or adjacent context), and intentionally mismatched within-split controls. Fit centering and orthogonal Procrustes on a disjoint paired alignment-fit subset, with no test labels or gradient updates. Evaluate bidirectional retrieval, alignment error and similarity on untouched dialogue pairs versus shuffled anchors, random encoders and identity map. Predefine positive/negative construction and control speaker, dialogue, duration and lexical overlap.
2. Freeze embeddings for state/act probes on SpokenWOZ, or emotion/sentiment probes on MELD; compare in-modality and cross-modal transfer against **strong pretrained text-only, audio-only and conventional multimodal fusion baselines**, direct lexical/acoustic shortcuts, and explicitly marked jointly trained references.[1][3][4]
3. Require unseen conversations; evaluate speaker-disjoint partitions where identifiers and coverage allow, and cross-dataset transfer only when licensing and label compatibility permit. Report overlaps if speaker isolation fails. Report seed and dialogue bootstrap uncertainty, speaker/domain and ASR-noise slices, negative results, latency and compute **for reproducibility rather than as optimization targets**. Do not infer universal Platonic convergence from a single benchmark. **Status: not run.**

## Decision register

Unresolved before implementation: legal permission for each asset/model; strong pretrained encoder provenance and architecture comparison; context/horizon, dimensions, target/EMA/masking and collapse remedy; objective selection and false-negative controls; verified aligned, weak-positive and mismatched view definitions; split cardinality and speaker isolation; cross-dataset label compatibility; primary effect size. No numeric experimental result has been generated. Results template: [results](results.md).

## Sources

[1] https://spokenwoz.github.io
[3] https://affective-meld.github.io
[4] https://github.com/declare-lab/MELD
[8] https://arxiv.org/abs/2301.08243
