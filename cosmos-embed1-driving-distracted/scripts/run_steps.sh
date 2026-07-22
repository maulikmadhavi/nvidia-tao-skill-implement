#!/usr/bin/env bash
# End-to-end driver for the distracted-driving clip-classification experiment
# (Linux/macOS, zsh/bash). One host step (stratified split) then three container
# steps — embed -> zero-shot metrics -> linear probe — all sharing one cached
# embeddings.npz. Container steps go through the canonical run_container.sh
# wrapper (offline image cosmos-embed-offline:7.0.1). Built to confirm
# PORTABILITY on a fresh GPU server: `preflight` checks every dependency and
# mount that tends to differ across machines before any GPU job starts.
#
# Usage:
#   ./run_steps.sh                 # default: preflight split embed zeroshot probe  (full)
#   ./run_steps.sh preflight       # just the environment/mount checks
#   ./run_steps.sh embed zeroshot  # pick phases explicitly (order is honored)
#   DRY_RUN=1 ./run_steps.sh       # print every command without running it
#
# Phases:
#   preflight  host + container sanity: docker, GPU passthrough, offline image,
#              python3, the shared /model + /hf_cache snapshot, the dataset root,
#              and the run_container.sh wrapper
#   split      host: 00_prepare_split.py -> workspace/splits/{metadata,class_prompts,split_stats}.json
#   embed      container GPU: 10_embed.py -> workspace/results/baseline/embeddings.npz
#   zeroshot   container: 11_zeroshot_metrics.py (cosine argmax on the val split)
#   probe      container CPU: 20_linear_probe.py (LogReg on frozen embeddings)
#
# Env knobs (all optional):
#   IMAGE       override the docker image (passed through to run_container.sh)
#   DATA_ROOT   dataset root (shortclips_pos_neg)   [also consumed by run_container.sh]
#   SHARED      base for the default model/hf_cache locations              [run_container.sh]
#   MODEL_DIR   Cosmos-Embed1-224p snapshot dir    (default: $SHARED/model/Cosmos-Embed1-224p)
#   HF_CACHE    HF cache dir                       (default: $SHARED/hf_cache)
#   VAL_FRAC    val fraction for the split          (default: 0.2)
#   SEED        split seed                           (default: 0)
#   SPLIT       which split zero-shot reports on     (default: val; val|train|all)
#   CV          linear-probe CV folds                (default: 5)
#   DEVICE      embed device                         (default: cuda)
#   SKIP_GPU_GATE=1   skip the `docker run --gpus all … nvidia-smi` check
#   DRY_RUN=1   print commands instead of executing them

set -euo pipefail

# ---- locations -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
EXP="$(dirname "$SCRIPT_DIR")"
WS="$EXP/workspace"
RUN_CONTAINER="$SCRIPT_DIR/run_container.sh"

IMAGE="${IMAGE:-cosmos-embed-offline:7.0.1}"
DATA_ROOT="${DATA_ROOT:-/data/research_data/driving_violation/shortclips_pos_neg/shortclips_pos_neg}"
SHARED="${SHARED:-$(dirname "$EXP")/cosmos-embed1-driving-violations/workspace}"
MODEL_DIR="${MODEL_DIR:-$SHARED/model/Cosmos-Embed1-224p}"
HF_CACHE="${HF_CACHE:-$SHARED/hf_cache}"
VAL_FRAC="${VAL_FRAC:-0.2}"
SEED="${SEED:-0}"
SPLIT="${SPLIT:-val}"
CV="${CV:-5}"
DEVICE="${DEVICE:-cuda}"

# ---- helpers ---------------------------------------------------------------
c_reset=$'\033[0m'; c_blue=$'\033[1;34m'; c_green=$'\033[1;32m'
c_red=$'\033[1;31m'; c_yellow=$'\033[1;33m'
banner() { printf '\n%s==== %s ====%s\n' "$c_blue" "$1" "$c_reset"; }
info()   { printf '%s[i]%s %s\n' "$c_green" "$c_reset" "$1"; }
warn()   { printf '%s[!]%s %s\n' "$c_yellow" "$c_reset" "$1"; }
die()    { printf '%s[x]%s %s\n' "$c_red" "$c_reset" "$1" >&2; exit 1; }

host() {
    printf '%s+ %s%s\n' "$c_yellow" "$*" "$c_reset"
    [[ -n "${DRY_RUN:-}" ]] && return 0
    "$@"
}

# run an in-container command via the canonical wrapper (exports DATA_ROOT/SHARED/IMAGE)
incontainer() {
    local cmd="$1"
    printf '%s+ run_container.sh %s%s\n' "$c_yellow" "$cmd" "$c_reset"
    [[ -n "${DRY_RUN:-}" ]] && return 0
    IMAGE="$IMAGE" DATA_ROOT="$DATA_ROOT" SHARED="$SHARED" MODEL_DIR="$MODEL_DIR" HF_CACHE="$HF_CACHE" "$RUN_CONTAINER" "$cmd"
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

    # dataset root must exist on the host (mounted read-only at /data)
    if [[ -d "$DATA_ROOT" ]]; then
        local n; n=$(find "$DATA_ROOT" -name '*.mp4' 2>/dev/null | head -1000 | wc -l | tr -d ' ')
        info "dataset root OK: $DATA_ROOT (found >=$n mp4s in a quick scan)"
    else
        warn "dataset root NOT found: $DATA_ROOT  (set DATA_ROOT=/path/to/shortclips_pos_neg/shortclips_pos_neg)"; fail=1
    fi

    # model snapshot + hf cache (MODEL_DIR/HF_CACHE, defaulting under $SHARED)
    if [[ -f "$MODEL_DIR/config.json" ]]; then
        info "model snapshot OK: $MODEL_DIR"
    elif [[ -d "$MODEL_DIR" ]]; then
        warn "MODEL_DIR exists but has no config.json: $MODEL_DIR  (incomplete download? expect the full nvidia/Cosmos-Embed1-224p snapshot here)"; fail=1
    else
        warn "model snapshot NOT found: $MODEL_DIR  (download it: hf download nvidia/Cosmos-Embed1-224p --local-dir \"$MODEL_DIR\", or set MODEL_DIR=/path/to/your/Cosmos-Embed1-224p)"; fail=1
    fi
    if compgen -G "$HF_CACHE/models--bert-base-uncased/snapshots/*/config.json" >/dev/null 2>&1; then
        info "hf_cache OK (bert-base-uncased staged): $HF_CACHE"
    else
        warn "bert-base-uncased NOT in $HF_CACHE — the offline Q-Former load will fail. Stage it: hf download bert-base-uncased --cache-dir \"$HF_CACHE\"  (or just run ./download_model.sh)"; fail=1
    fi

    if have docker; then
        if docker image inspect "$IMAGE" >/dev/null 2>&1; then
            info "docker image present: $IMAGE"
        else
            warn "docker image NOT built/pulled: $IMAGE  (build the offline image via the driving-violations Dockerfile.offline)"; fail=1
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

    host mkdir -p "$WS/splits" "$WS/results/baseline" "$HF_CACHE"

    [[ $fail -eq 0 ]] && info "preflight OK" || die "preflight found blocking issues (see [!] above)"
}

phase_split() {
    banner "split — stratified 80/20 per class (host, stdlib)"
    host python3 "$SCRIPT_DIR/00_prepare_split.py" --data-root "$DATA_ROOT" --out-dir "$WS/splits" --val-frac "$VAL_FRAC" --seed "$SEED"
    info "wrote $WS/splits/{metadata,class_prompts,split_stats}.json"
}

phase_embed() {
    banner "embed — all clips + 7 prompts -> one cached NPZ (container GPU)"
    incontainer "python /exp/scripts/10_embed.py --model /model/Cosmos-Embed1-224p --metadata /splits/metadata.json --prompts /splits/class_prompts.json --videos /data --out /results/baseline/embeddings.npz --phase baseline --device $DEVICE"
    info "embeddings -> $WS/results/baseline/embeddings.npz"
}

phase_zeroshot() {
    banner "zeroshot — cosine argmax metrics on the '$SPLIT' split (container)"
    incontainer "python /exp/scripts/11_zeroshot_metrics.py --embeddings /results/baseline/embeddings.npz --split $SPLIT --out /results/baseline/zeroshot"
    info "zero-shot metrics -> $WS/results/baseline/zeroshot/ (expect val acc ~0.945)"
}

phase_probe() {
    banner "probe — linear probe (LogReg) on frozen embeddings (container CPU)"
    incontainer "python /exp/scripts/20_linear_probe.py --embeddings /results/baseline/embeddings.npz --out /results/baseline/probe --cv $CV"
    info "probe + metrics -> $WS/results/baseline/probe/ (expect val acc ~0.989)"
    warn "PowerShell reports the probe exit 1 on a harmless sklearn FutureWarning; under bash 'set -e' this only trips on docker's real exit code. Either way, confirm metrics_probe.json was written."
}

# ---- dispatch --------------------------------------------------------------
PHASES=("$@")
if [[ ${#PHASES[@]} -eq 0 || "${PHASES[0]}" == "all" ]]; then
    PHASES=(preflight split embed zeroshot probe)
    info "running full pipeline: ${PHASES[*]}"
fi

for p in "${PHASES[@]}"; do
    case "$p" in
        preflight) phase_preflight ;;
        split)     phase_split ;;
        embed)     phase_embed ;;
        zeroshot)  phase_zeroshot ;;
        probe)     phase_probe ;;
        *) die "unknown phase '$p' (valid: preflight split embed zeroshot probe all)" ;;
    esac
done

banner "done: ${PHASES[*]}"
