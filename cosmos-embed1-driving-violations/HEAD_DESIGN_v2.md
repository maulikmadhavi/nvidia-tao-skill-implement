# Design review: ChatGPT architecture vs this kit, and the heads plan (v2)

Source: user's ChatGPT discussion (2026-07-17) on a scalable violation-detection system.
Verdict: the proposed architecture is sound, and most of it already exists here. The one
genuinely new, high-value piece is **per-violation classification heads over frozen
embeddings** — adopted below as the recommended middle tier between zero-shot prompts
and LoRA fine-tuning.

## What the kit already implements (no action needed)

| ChatGPT proposal | Status here |
|---|---|
| Stable shared multimodal encoder | ✅ Cosmos-Embed1-224p, frozen (`/model` snapshot) |
| 1–2s chunks, ~50% overlap | ✅ 2s chunks, 1s stride |
| Score every chunk, no fixed top-K | ✅ `10_infer_chunks.py` → full score matrix |
| Per-violation threshold calibration | ✅ `14_tune_thresholds.py` (θ_c, window, hysteresis per class, tuned on val) |
| Merge adjacent chunks → event timeline | ✅ glue mode (`11_glue_postprocess.py`) |
| Reviewer-centric timeline UI | ✅ `demo/app.py` (trajectories + event bars + seek) |
| High recall focus, honest metrics | ✅ per-class PR-AUC, P@R, event F1 @ IoU 0.3 |
| Modular taxonomy | ✅ `scripts/taxonomy.py` |

## What to adopt (recommended, in order)

### 1. Per-violation heads on frozen embeddings — YES, feasible with Cosmos-Embed1

- Each chunk embedding is a 256-d L2-normalized vector; a head = L2-regularized logistic
  regression per violation (sklearn, `class_weight='balanced'`) + probability calibration.
- Training cost: seconds on CPU for ~13K chunks. No GPU, no cosmos CLI, no encoder retraining.
- New violation = annotate → fit ONE head → ship the head file. Existing heads untouched.
- Integration is minimal because heads emit the SAME `S[T×C]` matrix shape (probabilities
  instead of cosines) — glue/tuning/eval/demo work unchanged.
- Overfitting caution at our scale (few positives per class): stick to linear heads (no MLP)
  with strong L2; keep temporal-jitter augmentation; judge only by the existing PR-AUC /
  event-F1 machinery on untouched test videos.
- **Evidence**: Recent work (FrEVL, 2025) shows frozen VL embeddings achieve 90.2% of SOTA
  performance with 69% fewer parameters and 2.3× speedup, confirming viability for our use case.

### 2. Enhanced abstention mechanism with uncertainty quantification

Two thresholds per head: predict positive if p >= t_hi, negative if p <= t_lo, else
**unknown** → reviewer queue. Tune (t_lo, t_hi) on val for target recall at manageable
review load; track the unknown-rate as a first-class metric.

**Enhancement**: Implement hybrid uncertainty quantification combining:
- Aleatoric uncertainty (from softmax entropy) for ambiguous in-distribution cases
- Epistemic uncertainty (via distance metrics in embedding space) for out-of-distribution cases
- This provides more principled abstention than simple thresholding, especially for
  borderline violations where confidence scores may be misleading.

### 3. Encoder versioning policy (resolves the LoRA-vs-heads tension)

The ChatGPT design says "don't retrain the encoder"; our RUNBOOK has a LoRA phase. Both
are right with a versioning rule:
- **encoder v1** = pretrained Cosmos-Embed1, frozen. Heads live on top.
- LoRA fine-tune ONLY if heads plateau (per-class AP stuck low with enough positives) —
  that produces **encoder v2**; re-embed all chunks once and refit ALL heads (cheap,
  seconds, because embeddings are cached).
- Never drift the encoder silently under existing heads.

Recommended experiment order on the air-gapped server therefore becomes:
**zero-shot prompts (Phase 2) → heads (minutes, new Phase 4a) → LoRA (hours, Phase 4b,
only if needed)**.

**Evidence**: F-VLM (2022) demonstrates that frozen VLMs retain strong localization and 
classification capabilities, supporting our approach of keeping the encoder frozen initially.

## What to defer (good ideas, wrong time)

- **Attribute layer + rules** (phone-visible AND hand-near-ear …): needs attribute-level
  labels we don't have. Cheap approximation available today: multiple prompts per concept
  scored zero-shot as weak attribute signals. Revisit when the violation list grows past
  ~10 or rules become obviously needed.
- **Crops (driver / trainer / face / torso)**: highest-value fix for trainer-vs-driver
  confusion, but a person/face detector adds a model to the air-gap bundle. CHEAP VARIANT
  FIRST: if the camera mount is fixed across the fleet, static crop rectangles per region
  (driver seat / passenger seat) need NO detector — just ffmpeg crop filters feeding the
  same encoder, doubling the embedding streams. (Ask: is the camera geometry fixed?)
- **VLM verifier on uncertain events**: sensible, but a capable VLM inflates the offline
  bundle materially. The unknown-queue → human reviewer covers this initially. If wanted
  later, the repo has `tao-finetune-cosmos-reason` (video QA) as a natural candidate.
- **Active learning**: becomes nearly free once heads exist (reviewer corrections → refit
  heads weekly). Design the reviewer output format now (event id, correct/incorrect, true
  class) so feedback is capturable from day one.

## Implementation plan (kit update BEFORE migration — small)

1. `10_infer_chunks.py`: also save the window embeddings `V` in each `scores_<stem>.npz`
   (one line) — makes every downstream head experiment free, no re-embedding.
2. New `15_train_heads.py` (container, sklearn): embeddings + chunk labels (from
   annotations, same overlap rule) → per-class calibrated logistic head + (t_lo, t_hi)
   tuned on val → `heads.npz`. Report per-class val AP vs the zero-shot prompt AP.
3. New `16_score_heads.py`: apply heads to saved embeddings → `scores_*.csv/npz` in the
   standard format (+ `unknown_*.csv` mask with uncertainty-based abstention) → existing
   11/13/14/demo just work; 11 gains an `--unknown` passthrough to emit an uncertain-events
   list for the reviewer queue.
4. Selftest: extend with a heads pass (train on the train lesson, verify event recovery
   on the test lesson).
5. RUNBOOK: insert Phase 4a (heads — minutes) before Phase 4b (LoRA — hours), with the
   decision rule for when 4b is worth running; rebuild the bundle kit.

**Enhanced implementation details**:
- Use calibrated probabilities (Platt scaling or isotonic regression) for more reliable 
  thresholding in the abstention mechanism
- Implement hierarchical head organization: train general violation detector first, 
  then specialized heads for subtypes if needed
- Add drift detection mechanism to monitor when encoder retraining (LoRA) becomes beneficial

Estimated effort: one short session; no new dependencies (sklearn already baked in).

## Future Enhancements (Beyond Current Scope)

### Hierarchical Head Architectures
For complex violations with sub-patterns (e.g., "trainer_phone_use" could involve 
different hand positions, phone visibility, etc.), consider:
- Hierarchical classification: general violation head → subtype heads
- Mixture of experts: lightweight gating network routing to specialized heads
- This maintains the efficiency benefits while capturing finer-grained patterns

### Test-Time Adaptation for Distribution Shift
In deployment scenarios where camera angles or lighting conditions may drift:
- Implement lightweight test-time adaptation (e.g., Tent-style entropy minimization)
- Only adapt normalization layers or very light adapters to preserve core knowledge
- Particularly relevant for long-term deployments in varying environments

### Uncertainty-Aware Active Learning
Enhance the reviewer queue with:
- Uncertainty sampling: prioritize examples with highest entropy or disagreement
- Diversity sampling: ensure varied examples in each batch for efficient learning
- Expected model change: select samples that would most improve the model if labeled

### Cross-Validation for Robust Threshold Selection
Instead of single validation set for threshold tuning:
- Use cross-validation to get more robust (t_lo, t_hi) estimates
- Particularly important given limited violation examples in safety-critical domains
- Provides confidence intervals on expected unknown rates

These extensions build upon the solid foundation established in this design while 
maintaining the core principles of efficiency, modularity, and deployability in 
air-gapped environments.