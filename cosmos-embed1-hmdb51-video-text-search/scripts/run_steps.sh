#!/usr/bin/env bash
# End-to-end driver for the Cosmos-Embed1 HMDB51 video-text-search experiment
# (Linux/macOS, zsh/bash). Runs the reproducible spine — data prep -> zero-shot
# baseline -> LoRA fine-tune -> evaluate/export/compare — through the canonical
# docker wrapper. Built to confirm PORTABILITY on a fresh GPU server: `preflight`
# checks every host dependency that tends to differ across machines before any
# long job starts.
#
# Usage:
#   ./run_steps.sh                 # default: preflight data baseline  (fast, no 3.5h train)
#   ./run_steps.sh all             # preflight data baseline train eval  (full, ~4h on 1 GPU)
#   ./run_steps.sh preflight       # just the environment checks
#   ./run_steps.sh data baseline   # pick phases explicitly (order is honored)
#   DRY_RUN=1 ./run_steps.sh all   # print every command without running it
#
# Phases:
#   preflight  host + container sanity: docker, GPU passthrough, image, ffmpeg,
#              ffprobe, 7z, python3, HF_TOKEN, and the run_container.sh wrapper
#   data       00 download -> 01 extract -> 02 sample/split -> 03 transcode -> 04 metadata
#   baseline   container zero-shot evaluate + custom metrics (scripts 10/11, phase=baseline)
#   train      container LoRA fine-tune (train.yaml, 600 iters) — the long pole
#   eval       evaluate ckpt + export HF + processor-config fix + metrics(finetuned) + compare
#
# Env knobs (all optional):
#   IMAGE       override the docker image (passed through to run_container.sh)
#   HF_TOKEN    HuggingFace token (exported into the container for weight pulls)
#   SEVENZIP    7-Zip binary for inner-rar extraction         (default: 7z)
#   PER_CLASS   clips sampled per class in step 02            (default: 30)
#   SEED        split seed                                     (default: 42)
#   WORKERS     ffmpeg transcode workers                       (default: 4)
#   CKPT        exact in-container checkpoint path (else auto-discovered)
#   SKIP_GPU_GATE=1   skip the `docker run --gpus all … nvidia-smi` check
#   DRY_RUN=1   print commands instead of executing them

set -euo pipefail

# ---- locations -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
EXP="$(dirname "$SCRIPT_DIR")"
WS="$EXP/workspace"
RUN_CONTAINER="$SCRIPT_DIR/run_container.sh"

IMAGE="${IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-embed}"
SEVENZIP="${SEVENZIP:-7z}"
PER_CLASS="${PER_CLASS:-30}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-4}"

# in-container checkpoint (max_iter=600 in train.yaml); auto-discovered in eval if unset
CKPT="${CKPT:-}"

# ---- helpers ---------------------------------------------------------------
c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_red=$'\033[1;31m'; c_yellow=$'\033[1;33m'
banner() { printf '\n%s==== %s ====%s\n' "$c_blue" "$1" "$c_reset"; }
info()   { printf '%s[i]%s %s\n' "$c_green" "$c_reset" "$1"; }
warn()   { printf '%s[!]%s %s\n' "$c_yellow" "$c_reset" "$1"; }
die()    { printf '%s[x]%s %s\n' "$c_red" "$c_reset" "$1" >&2; exit 1; }

# run a host command (echo under DRY_RUN)
host() {
    printf '%s+ %s%s\n' "$c_yellow" "$*" "$c_reset"
    [[ -n "${DRY_RUN:-}" ]] && return 0
    "$@"
}

# run an in-container command via the canonical wrapper
incontainer() {
    local cmd="$1"
    printf '%s+ run_container.sh %s%s\n' "$c_yellow" "$cmd" "$c_reset"
    [[ -n "${DRY_RUN:-}" ]] && return 0
    IMAGE="$IMAGE" "$RUN_CONTAINER" "$cmd"
}

have() { command -v "$1" >/dev/null 2>&1; }

# ---- phases ----------------------------------------------------------------
phase_preflight() {
    banner "preflight — portability checks"
    local fail=0

    [[ -x "$RUN_CONTAINER" ]] || { warn "run_container.sh not executable — fixing"; host chmod +x "$RUN_CONTAINER"; }

    for bin in python3 docker; do
        have "$bin" && info "found $bin ($(command -v $bin))" || { warn "MISSING: $bin"; fail=1; }
    done
    for bin in ffmpeg ffprobe "$SEVENZIP" unzip; do
        have "$bin" && info "found $bin" || warn "missing $bin (needed by the data phase: ffmpeg/ffprobe for transcode, $SEVENZIP/unzip for extract)"
    done

    if have docker; then
        if docker image inspect "$IMAGE" >/dev/null 2>&1; then
            info "docker image present: $IMAGE"
        else
            warn "docker image NOT pulled: $IMAGE  (run: docker pull $IMAGE)"
            fail=1
        fi
        if [[ -z "${SKIP_GPU_GATE:-}" ]]; then
            info "checking GPU passthrough into the container (docker run --gpus all … nvidia-smi)"
            if [[ -z "${DRY_RUN:-}" ]]; then
                if docker run --rm --gpus all "$IMAGE" nvidia-smi -L >/tmp/_gpu_gate.$$ 2>&1; then
                    sed 's/^/    /' /tmp/_gpu_gate.$$; rm -f /tmp/_gpu_gate.$$
                    info "GPU visible inside container"
                else
                    warn "GPU passthrough FAILED — is the NVIDIA Container Toolkit installed?"; cat /tmp/_gpu_gate.$$ >&2 || true; rm -f /tmp/_gpu_gate.$$; fail=1
                fi
            fi
        else
            warn "SKIP_GPU_GATE set — not verifying --gpus all"
        fi
    fi

    [[ -n "${HF_TOKEN:-}" ]] && info "HF_TOKEN is set" || warn "HF_TOKEN unset — Cosmos-Embed1-224p weights are ungated so this usually works, but set it if pulls 401"

    host mkdir -p "$WS/raw" "$WS/data" "$WS/specs" "$WS/results" "$WS/model" "$WS/hf_cache"

    [[ $fail -eq 0 ]] && info "preflight OK" || die "preflight found blocking issues (see [!] above)"
}

phase_data() {
    banner "data — download, extract, sample/split, transcode, metadata"
    host python3 "$SCRIPT_DIR/00_download_hmdb51.py" --out "$WS/raw"
    host python3 "$SCRIPT_DIR/01_extract_hmdb51.py"  --raw "$WS/raw" --sevenzip "$SEVENZIP"
    host python3 "$SCRIPT_DIR/02_sample_and_split.py" --raw "$WS/raw" --out "$WS/data" --seed "$SEED" --per-class "$PER_CLASS"
    host python3 "$SCRIPT_DIR/03_transcode.py" --manifest "$WS/data/split_manifest.csv" --raw "$WS/raw" --out "$WS/data/video" --workers "$WORKERS"
    host python3 "$SCRIPT_DIR/04_make_metadata.py" --manifest "$WS/data/split_manifest.csv" --data "$WS/data" --templates "$SCRIPT_DIR/spec_templates" --specs "$WS/specs"
    info "data phase complete — split_manifest.csv + video/*.mp4 + rendered specs are in $WS"
}

phase_baseline() {
    banner "baseline — zero-shot evaluate + custom metrics"
    incontainer "cosmos-embed1 evaluate -e /specs/evaluate_zeroshot.yaml results_dir=/results/evaluate_zeroshot"
    incontainer "python /exp/scripts/10_extract_embeddings.py --model nvidia/Cosmos-Embed1-224p --metadata /data/test.json --videos /data/video --prompts /data/class_prompts.json --out /results/metrics/baseline/embeddings.npz --phase baseline"
    incontainer "python /exp/scripts/11_compute_metrics.py --embeddings /results/metrics/baseline/embeddings.npz --phase baseline --out /results/metrics/baseline"
    info "baseline metrics -> $WS/results/metrics/baseline/ (expect top-1 ~65.7%)"
}

phase_train() {
    banner "train — LoRA fine-tune (600 iters, the long pole ~3.5h/GPU)"
    incontainer "cosmos-embed1 train -e /specs/train.yaml results_dir=/results/train"
    info "checkpoint(s) -> $WS/results/train/train/checkpoints/ (expect val top-1 climbing to ~78%)"
}

phase_eval() {
    banner "eval — evaluate ckpt, export HF, fix processor config, metrics(finetuned), compare"
    # discover the exact checkpoint inside the container unless CKPT was given
    local ckpt_expr
    if [[ -n "$CKPT" ]]; then
        ckpt_expr="$CKPT"
    else
        ckpt_expr='$(ls -1 /results/train/train/checkpoints/iter_*.pt 2>/dev/null | sort | tail -1)'
    fi

    incontainer "CKPT=$ckpt_expr; test -n \"\$CKPT\" || { echo 'no checkpoint found — run the train phase first' >&2; exit 1; }; echo \"using checkpoint \$CKPT\"; cosmos-embed1 evaluate -e /specs/evaluate_finetuned.yaml results_dir=/results/evaluate_finetuned evaluate.checkpoint=\"\$CKPT\""

    incontainer "CKPT=$ckpt_expr; cosmos-embed1 export -e /specs/export_hf.yaml results_dir=/results/export_hf export.checkpoint=\"\$CKPT\""

    # export omits processor_config.json -> copy it from the 224p snapshot in the HF cache (gotcha #1)
    incontainer "src=\$(find /hf_cache /root/.cache/huggingface -name processor_config.json -path '*Cosmos-Embed1-224p*' 2>/dev/null | head -1); test -n \"\$src\" || { echo 'processor_config.json not found in cache' >&2; exit 1; }; cp -v \"\$src\" /results/export_hf/cosmos_embed1_hmdb51_hf/"

    # sanity: the exported dir loads and embeds
    incontainer "python /exp/scripts/gate_hf_load.py /results/export_hf/cosmos_embed1_hmdb51_hf"

    incontainer "python /exp/scripts/10_extract_embeddings.py --model /results/export_hf/cosmos_embed1_hmdb51_hf --metadata /data/test.json --videos /data/video --prompts /data/class_prompts.json --out /results/metrics/finetuned/embeddings.npz --phase finetuned"
    incontainer "python /exp/scripts/11_compute_metrics.py --embeddings /results/metrics/finetuned/embeddings.npz --phase finetuned --out /results/metrics/finetuned"
    incontainer "python /exp/scripts/12_compare_phases.py --baseline /results/metrics/baseline/metrics.json --finetuned /results/metrics/finetuned/metrics.json --container-baseline /results/evaluate_zeroshot/evaluate/metrics.json --container-finetuned /results/evaluate_finetuned/evaluate/metrics.json --out /results/metrics/comparison.md"
    info "comparison -> $WS/results/metrics/comparison.md (expect top-1 ~65.7% -> ~77.5%)"
    info "deploy + web demo are manual follow-ups (see PIPELINE.md steps 6-7); their scripts live under deploy/ and demo/."
}

# ---- dispatch --------------------------------------------------------------
PHASES=("$@")
if [[ ${#PHASES[@]} -eq 0 ]]; then
    PHASES=(preflight data baseline)
    info "no phases given -> default: ${PHASES[*]}  (pass 'all' for the full run incl. training)"
elif [[ "${PHASES[0]}" == "all" ]]; then
    PHASES=(preflight data baseline train eval)
fi

for p in "${PHASES[@]}"; do
    case "$p" in
        preflight) phase_preflight ;;
        data)      phase_data ;;
        baseline)  phase_baseline ;;
        train)     phase_train ;;
        eval)      phase_eval ;;
        *) die "unknown phase '$p' (valid: preflight data baseline train eval all)" ;;
    esac
done

banner "done: ${PHASES[*]}"
