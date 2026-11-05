# Architecture (proposed; not implemented)

## Separation contract

Two independent pipelines, with distinct initializations and no shared learned weights, train on their respective training modalities. Paired transcript/audio turns can define a common evaluation index but **cannot enter a cross-modal training objective** in the primary independent-convergence claim. No joint projector, contrastive speech–text alignment, distillation, or text teacher for the speech branch. Pretrained front ends, if used, must be declared and analyzed separately because their previous multimodal or transcript supervision can contaminate the independence claim.

```text
Speech: audio turns -> speech context encoder -> context state -> speech predictor -> future speech latent
                                             future speech -> frozen/EMA target encoder -> target latent
Text: transcript turns -> text context encoder -> context state -> text predictor -> future text latent
                                             future text -> frozen/EMA target encoder -> target latent
After both training runs: freeze encoders -> anchor embeddings -> orthogonal map -> held-out evaluation
```

A JEPA objective predicts **representations** of missing/future material rather than reconstructing waveform or tokens; I-JEPA provides an architectural precedent, not evidence this dialogue variant works.[8] Proposed target for turn *t+h*: stop-gradient target-encoder embedding of that modality's future turn. Predictor receives causal prior context and horizon indicator only; no target content leaks into context. Record whether predicting an utterance, a pooled dialogue state, or several horizons. Compare strong pretrained encoder families and architectures, checking for prior multimodal supervision and without imposing an inference-size cap. Compare cosine prediction, variance/covariance-regularized prediction, contrastive future prediction (with false-negative controls), and a mean-squared-error reference; **do not assume MSE is sufficient**. EMA target update, asymmetric predictor and masking are candidates; the collapse-prevention choice remains unresolved. Measure embedding variance/effective rank, pairwise distances and trivial-predictor performance before interpreting any alignment.

## Frozen geometry

Encode paired turns from an **alignment-fit partition** after both models and preprocessing are fixed. Normalize consistently; center on fit anchors only; compute the orthogonal Procrustes transform (optionally with reflection policy fixed in advance), never update encoders or use test anchors. Apply fit means and transform unchanged to held-out dialogue pairs. Equal-dimensional embeddings are a provisional choice; any dimension matching projection must be fitted on non-test data and reported as learned alignment, not spontaneous identity.

## Boundary to downstream probes

Freeze embeddings for primary linear state/act probes; train probe weights on allowed training labels only. Report same-modality and cross-modal probes, lexical and acoustic shortcut controls, and direct unaligned scores. Supervised or jointly trained comparison models belong in a separate comparison lane, never the independent JEPA lane. See [methodology](methodology.md) and [experiments](experiments.md).

## Sources

[8] https://arxiv.org/abs/2301.08243
