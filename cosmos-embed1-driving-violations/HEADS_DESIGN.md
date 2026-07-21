# Per-violation heads — consolidated plan (design + implementation spec)

Origin: user's ChatGPT architecture review (2026-07-17), reconciled with this kit.
**STATUS: IMPLEMENTED + VALIDATED 2026-07-17** — full selftest passed end-to-end
including the heads chain (steps [9–11]; heads beat prompts on val AP for every
class; heads glue micro F1 0.44 on the test lesson from 4–11 positives/class).
Verdict: adopt **per-violation classification heads over frozen Cosmos-Embed1
embeddings** as a new middle tier. Order of experiments on the air-gapped server:

```
Phase 2  zero-shot prompts        (already built — minutes of GPU)
Phase 4a heads on frozen encoder  (NEW — seconds of CPU after embedding)
Phase 4b LoRA fine-tune           (already built — hours; ONLY if heads plateau)
```

Everything below fits the existing kit contracts: heads emit the same
`S[W×C]` score-matrix files as `10_infer_chunks.py`, so glue / tuning /
eval / demo (`11/12/13/14`, `demo/app.py`) run **unchanged**.

## Why heads (recap of the accepted arguments)

- A head = L2-regularized logistic regression per class on the 256-d
  L2-normalized window embeddings. Fits in seconds on CPU (~20K train windows);
  sklearn is already baked into the offline image; **no new dependencies**.
- New violation later = annotate → fit ONE head → ship one small file.
  Existing heads and the encoder are untouched.
- Probabilities enable a calibrated **positive / negative / UNKNOWN** band —
  the unknown queue is the reviewer entry point and replaces (for now) any
  VLM verifier.
- Linear only, strong L2 — at our positives-per-class scale an MLP would
  memorize. Judged strictly by the existing PR-AUC / event-F1 machinery on
  untouched test videos.

## Decision log (what the earlier sketch left open — now fixed)

| # | Decision | Rationale |
|---|---|---|
| D1 | Heads train on **full-lesson sliding windows** (train videos scored by `10_infer_chunks.py`), labels derived from annotations at the window grid with the same overlap ≥ 0.5s rule as `12_eval_chunk_level.py`. | One label convention everywhere; no dependency on the chunk-mp4 pipeline; natural class distribution + `class_weight='balanced'` instead of downsampling. |
| D2 | Train a head for **every class incl. `no_violation`** (C heads). | Output keeps the exact `class_ids` column set of `10`, so every downstream file format is byte-compatible. |
| D3 | **No `CalibratedClassifierCV`.** Plain LR sigmoid probabilities; per-class `C ∈ {0.01, 0.1, 1}` picked by val AP; **AP ties break toward the LARGER C**. | CV calibration needs positives in every fold — rare classes break it. The val-tuned thresholds absorb miscalibration where it matters. Tie-break (found in selftest): AP is rank-based, so heavy L2 + balanced weights can win a tie while pinning every sigmoid ≈ 0.5 — a range the 0.02-step threshold grid cannot resolve; the larger C keeps the same ranking with a usable probability scale. |
| D4 | Unknown band is **chunk-level**, computed in `15` from the val PR curve: `t_lo` = threshold at recall ≥ 0.95 (below → confident negative), `t_hi` = threshold at precision ≥ 0.90 (above → confident positive). Degenerate band (`t_lo ≥ t_hi`) → empty band + warning. | Two thresholds serve the reviewer queue; they are NOT the event-extraction thresholds. |
| D5 | **Event-extraction thresholds stay `14`'s job**, run on the heads probability trajectories with a probability grid `--thr-grid 0.05:0.95:0.02`. | `14` already takes `--thr-grid` — zero code change. Trajectory smoothing + hysteresis tuning is orthogonal to the unknown band. |
| D6 | The reviewer **unknown queue is emitted by `16`**, not by `11`. `11/12/13/14` are not modified at all. | The four validated post-processing scripts keep their gates; one new file owns all heads-specific output. |
| D7 | `15`/`16` run **in the container** (need numpy+sklearn; CPU is fine). Host scripts stay stdlib-pure. | Preserves the kit's "no host pip" invariant. |
| D8 | **Mechanical encoder-version check**: `heads.npz` stores the `model` string already embedded in every `scores_*.npz`; `16` refuses to apply heads to scores from a different encoder. | Enforces the versioning rule below by construction, not by discipline. |

## Encoder versioning rule (unchanged, now enforced by D8)

- **encoder v1** = pretrained `/model/Cosmos-Embed1-224p`, frozen. Heads live on top.
- Run LoRA (Phase 4b) only if a class with enough positives (≥ ~30 positive
  train windows) stays at low AP after heads. That produces **encoder v2** →
  re-run `10` (embeddings are then cached in the npz) → refit ALL heads in
  seconds (`heads_v2`). Never drift the encoder silently under existing heads.

## File-by-file implementation spec

### 1. `scripts/10_infer_chunks.py` — one-line change
Add `V=V` to the `np.savez(...)` call so every `scores_<stem>.npz` also carries
the window embeddings (`[W, 256]` float32 ≈ 300 KB per lesson). All existing
readers ignore the extra key. Every downstream head experiment becomes free —
no re-embedding, ever, per encoder version.

### 2. `scripts/15_train_heads.py` — NEW (container)
```
--scores <dir with scores_*.npz incl. V>  --annotations <csv>
--train-videos <stems.txt>  --val-videos <stems.txt>
--min-overlap 0.5  --out <heads.npz>  [--report <heads_report.json>]
```
Per class: assemble `(V, y)` from train stems (overlap rule of D1); fit
`LogisticRegression(class_weight='balanced', C=c, max_iter=2000)` for
`c ∈ {0.01, 0.1, 1}`; keep the `c` with best **val AP**; compute `t_lo`/`t_hi`
per D4 on val probabilities. Writes:
- `heads.npz`: `W [C,256]`, `b [C]`, `class_ids`, `t_lo [C]`, `t_hi [C]`,
  `C_reg [C]`, `encoder` (copied from the scores npz `model` field).
- `heads_report.json`: per class — train/val positive counts, chosen `C`,
  **val AP of the head vs val AP of the zero-shot prompt** (the go/no-go
  table), `t_lo`, `t_hi`, val unknown-rate.
Prints the report as a table. GATE: warn on any class with < 10 positive
train windows (head unreliable — keep the prompt score for that class).

### 3. `scripts/16_score_heads.py` — NEW (container, CPU)
```
--scores <dir with scores_*.npz>  --heads <heads.npz>  --out <dir>
```
Refuses to run if `heads.encoder != scores npz model` (D8). For each video:
`P = sigmoid(V @ W.T + b)` → writes `scores_<stem>.csv` + `.npz` in the exact
`10` format (`S`=probabilities, `starts`, `class_ids`, `model`, `chunk_sec`,
`stride_sec`, plus `V` passed through). Also writes:
- `unknown_<stem>.csv`: per window per class, 1 where `t_lo ≤ p < t_hi`.
- `review_queue.json`: contiguous unknown runs (≥ 2 windows) merged into
  `{video, class, start_sec, end_sec, peak_p}` items, sorted by `peak_p`
  descending — the reviewer's worklist.
Prints per-class unknown-rate; WARN above 20% (an unknown queue nobody can
clear helps nobody).

### 4. `scripts/phase4a_heads.sh` — NEW (RUNBOOK fast path)
```
./scripts/phase4a_heads.sh          # after phase2; optional TAG=heads_v2 SCORES=finetuned
```
1. `10` on `train_videos.txt` → `/results/scores/baseline` (same encoder,
   same dir as phase 2 — completes the embedding cache; ~1–2 min/lesson).
2. `15` → `/results/heads/heads_v1.npz` + report.
3. `16` → `/results/scores/heads_v1` (+ unknown files + review queue).
4. `14` on val with `--thr-grid 0.05:0.95:0.02` → `thresholds_heads_v1.json`.
5. `11` isolated + glue → `events/heads_v1_{isolated,glue}`.
6. `13` (test, both modes) + `12` (`--phase heads_v1`) → reports.
7. Prints the comparison rows: baseline vs heads_v1 (chunk mAP, event F1).

GATE + Phase 4b decision rule: proceed to LoRA only if some class with
≥ 30 positive train windows still has test AP clearly below the baseline
requirement after heads; otherwise heads_v1 is the shipping configuration.

### 5. `selftest/run_selftest.sh` / `.ps1` — extend
Append steps [9–11]: `15` on the selftest scores (train lesson) → `16` →
`14` (probability grid) → `11` glue → `13`. GATE: heads glue micro-F1 > 0 on
the test lesson and the per-class head-vs-prompt val AP table prints.
(Selftest scores dir is regenerated by step [4], so it will contain `V`.)

### 6. Docs + bundle
- `RUNBOOK.md`: insert **Phase 4a** between Phases 3 and 4 (rename current
  Phase 4 → **4b — LoRA (only if heads plateau)**); add the fast-path line;
  add Phase 5b note: after LoRA export + `10` on the finetuned model, refit
  heads via `TAG=heads_v2 SCORES=finetuned ./scripts/phase4a_heads.sh`.
- `README.md`: add `15/16` + `phase4a_heads.sh` to the layout table; extend
  the validated list after the selftest re-run.
- Re-run `scripts/check_bash_syntax.sh` (new .sh must be LF, parse-gated).
- Bundle: kit-only refresh (`make_bundle.ps1` without `-Full` — the image is
  unchanged, no new deps).

## Build order + acceptance gate (one session)

1. Edit `10` (V in npz) → 2. `15` → 3. `16` → 4. `phase4a_heads.sh` +
selftest extension → 5. docs. **Acceptance = full selftest re-run passes
end-to-end including the heads pass**, then bundle refresh.

## Deferred (unchanged from the original review)

- **Attribute layer + rules** — needs attribute labels we don't have; cheap
  interim: extra zero-shot prompts per concept. Revisit past ~10 classes.
- **Crops (driver / trainer / face)** — best fix for role confusion. Cheap
  variant first: **if the camera mount is fixed across the fleet**, static
  crop rectangles need no detector (ffmpeg crop → same encoder, second
  embedding stream). ⟵ open question to the user.
- **VLM verifier** — the unknown queue covers it initially; candidate later:
  `tao-finetune-cosmos-reason`.
- **Active learning** — nearly free once heads exist; the `review_queue.json`
  item format (`video, class, start_sec, end_sec, peak_p` + reviewer verdict)
  is the feedback record — refit heads weekly from accumulated verdicts.
