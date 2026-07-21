# Design review: ChatGPT architecture vs this kit, and the heads plan (v3 - Simplified Head Design)

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

### 1. Per-violation heads on frozen embeddings — YES, feasible with Cosmos-Embed1 (SIMPLIFIED DESIGN)

**Per user request: Each violation type gets a dedicated binary classifier with two explicit outputs:**
- **Node 1: [Violation]_positive** → Activates when the specific violation is occurring
- **Node 2: [Violation]_negative_normal** → Activates when we observe normal driving in the context relevant to detecting that violation

**Implementation approach:**
- For each violation class `v` (e.g., `trainer_phone_use`):
  - **Positive examples**: Video chunks where annotation `v` is present
  - **Negative examples**: Video chunks from **zero-annotation videos** (pure normal driving lessons)
    *Rationale: This ensures the negative class represents true "normal driving" baseline, not just "not v" (which could include other confusing violations)*
- Train a **single logistic regression model** (sklearn, `class_weight='balanced'`) per violation to distinguish:
  - Positive: chunks with violation `v`
  - Negative: chunks from zero-annotation videos (normal driving)
- **Output interpretation**:
  - High score → Looks like violation `v` compared to normal driving
  - Low score → Looks like normal driving compared to violation `v`
  - The model implicitly learns to ignore other violations during training (since they're excluded from both classes)

**Why this satisfies the user's request:**
- Effectively creates two "nodes" per violation:
  * Node 1 activation = P(v|chunk) [violation likelihood]
  * Node 2 activation = 1 - P(v|chunk) [normal driving likelihood, specifically in violation-relevant context]
- Uses the most meaningful contrast: violation vs. true normal driving (not violation vs. everything-else)
- Avoids confusion between violation types by excluding multi-violation chunks from training

**Technical details:**
- Each chunk embedding: 256-d L2-normalized vector from frozen Cosmos-Embed1
- Model: L2-regularized logistic regression (one per violation class)
- Training cost: Seconds on CPU for ~13K chunks (no GPU, no cosmos CLI)
- Adding new violation: Annotate → extract chunks → train ONE new head → deploy
- Existing heads remain completely unaffected when adding new violations
- Integration: Heads output single score per violation (P(v|chunk)), same format as cosine similarities from zero-shot → **zero changes needed** to `10_infer_chunks.py` output format, `11_glue_postprocess.py`, `12_tune_thresholds.py`, `13_eval_chunks.py`, or `demo/app.py`

**Evidence supporting feasibility:**
- Recent work (FrEVL, 2025) shows frozen VL embeddings achieve 90.2% of SOTA performance with 69% fewer parameters
- F-VLM (2022) demonstrates frozen VLMs retain strong localization and classification capabilities
- This approach leverages the same principle: using rich frozen embeddings with lightweight task-specific heads

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
   annotations, same overlap rule) → per-class logistic head distinguishing violation `v`
   from zero-annotation video chunks → `heads.npz`. Report per-class val AP vs the zero-shot
   prompt AP.
3. New `16_score_heads.py`: apply heads to saved embeddings → `scores_*.csv/npz` in the
   standard format (+ `unknown_*.csv` mask with uncertainty-based abstention) → existing
   11/13/14/demo just work; 11 gains an `--unknown` passthrough to emit an uncertain-events
   list for the reviewer queue.
4. Selftest: extend with a heads pass (train on the train lesson, verify event recovery
   on the test lesson).
5. RUNBOOK: insert Phase 4a (heads — minutes) before Phase 4b (LoRA — hours), with the
   decision rule for when 4b is worth running; rebuild the bundle kit.

**Implementation notes for the simplified head design:**
- Negative examples for each violation head come exclusively from videos with ZERO annotations (pure normal driving)
- This ensures the classifier learns: "What does violation `v` look like compared to TRUE normal driving?"
- Avoids using multi-violation or ambiguous chunks as negatives, which could confuse the model
- Maintains perfect compatibility with existing pipeline (outputs single probability per violation)
- Requires zero changes to downstream processing (glue, tuning, evaluation, demo)

Estimated effort: one short session; no new dependencies (sklearn already baked in).