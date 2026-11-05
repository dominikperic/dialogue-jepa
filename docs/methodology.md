# Methodology (proposed; data feasibility inspected, no model experiments)

## Falsifiable hypotheses

- **H1, emergent geometry:** separately optimized speech/text predictive embeddings align above random and shuffled-pair controls on held-out dialogues. A good fit only on alignment anchors is insufficient.
- **H2, view quality:** verified same-utterance audio/transcript views provide stronger held-out alignment than weaker positives, with deliberately mismatched views as a negative control.
- **H3, predictive utility:** learned representations improve future dialogue-state/act/transition prediction over strong pretrained text-only, audio-only and conventional multimodal baselines where labels permit; diagnose collapse and shortcut reliance.
- **H4, generalization/limits:** H1–H3 effects hold on unseen conversations, held-out speakers where feasible, and ideally a separate label-compatible dataset; negative or heterogeneous effects are reported, not averaged away.

The Platonic Representation Hypothesis is a motivating position, not a proof of these hypotheses.[9] JEPA is a design family whose image precedent does not settle collapse in dialogue.[8]

## Sampling and anti-leakage

Fix dataset version and official splits first. Isolate full dialogues before creating context/future windows; never let adjacent turns of one dialogue straddle train/validation/alignment/test. Fit preprocessing, normalization, dimensionality choices and Procrustes solely on allowed partitions. Use official test once after selecting architecture/hyperparameters on development data; if official dev is used for anchor fitting, reserve a separate dev-selection partition and never tune on test. Speaker-disjoint evaluation is conditional on usable speaker IDs and sufficient coverage; label overlap/shift explicitly. Avoid matching paired turns using annotation labels in the training objective.

## Measurements

- **Geometry and view quality:** compare exactly aligned audio/transcript pairs, weaker positive views (e.g. noisy ASR or adjacent context, defined before use), and deliberately mismatched controls sampled within the same split. Control for speaker, dialogue, text overlap and duration so retrieval cannot succeed on shortcuts. Measure paired speech-to-text and text-to-speech Recall@1, Recall@5, mean reciprocal rank, cosine matched-vs-mismatched distributions; orthogonal map error and CKA/RSA as complementary geometry diagnostics. Compare to a shuffled pairing null and a no-fit/identity mapping; retrieval candidate pool and duplicates must be stated.
- **Prediction/collapse:** compare mean-squared embedding error with cosine, variance/covariance-regularized and contrastive future objectives (auditing false negatives); select using held-out predictive utility, not loss magnitude across incompatible objectives. Check against mean/last-state/shuffled-future predictors; report per-dimension variance, effective rank/covariance spectrum, pairwise spread and condition number. Low loss with constant embeddings fails the non-collapse gate.
- **Dialogue utility:** compare JEPA against strong pretrained text-only and audio-only models plus conventional multimodal fusion, as well as frozen linear probes for slot/state/act where annotated (SpokenWOZ) and emotion/sentiment where annotated (MELD). Report dataset-appropriate macro F1/class support and, for task-oriented DST, joint goal accuracy and slot F1 under the dataset's schema. Do not treat MELD emotion as task-oriented dialogue state.[1][3][4]
- **Generalization:** untouched conversations are mandatory; hold out speakers where identifiers and sample size permit and report overlap otherwise. Evaluate cross-dataset transfer only with compatible tasks/labels and verified rights, without tuning on the transfer test set. Stratify by domain, dialogue length and turn position where metadata supports it. Compare clean transcript to defined ASR corruption or available ASR hypotheses.

Use per-dialogue bootstrap intervals and multiple training seeds; report the unit of resampling, denominators, uncertainty and all selected configurations. Before training, specify primary metric, minimum useful effect, statistical comparison, available compute budget and stopping rule. Log latency, memory and training compute under stated hardware for reproducibility, **not as primary optimization targets**; do not cap architecture size to reduce inference cost. No thresholds or numerical outcomes are asserted here.

## Baselines and interpretation

Untrained/random, constant and shuffled-pair/target controls; strong pretrained text-only and audio-only encoders, static pooled/nonpredictive variants, conventional multimodal fusion, lexical bag-of-words/ASR text and simple acoustics. Document parameter count, training data and compute; use capacity-matched controls for interpretation, **not** as a maximum model-size constraint. Supervised/joint multimodal references must remain clearly separated. Improvement over a weak null alone does not establish a shared semantic space. Failed collapse checks block alignment claims; success on training anchors but not held-out pairs falsifies generalization. External pretrained audio/text models must be traced for cross-modal supervision before serving as independent evidence.

## Sources

[1] https://spokenwoz.github.io
[3] https://affective-meld.github.io
[4] https://github.com/declare-lab/MELD
[8] https://arxiv.org/abs/2301.08243
[9] https://proceedings.mlr.press/v235/huh24a.html
