#!/bin/bash
# monitor_and_advance.sh
# Runs periodically (via cron). Checks E/F training, triggers ICC eval,
# and auto-launches next experiments when resources free up.
#
# State file: results/pipeline_state.json
# Log:        results/monitor_log.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")/results"
STATE_FILE="$EXP_DIR/pipeline_state.json"
LOG_FILE="$EXP_DIR/monitor_log.txt"
PYTHON="/home/vlm/minyoung/.venv/bin/python"
export UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

get_epochs() {
    local f="$1"
    [[ -f "$f" ]] || { echo 0; return; }
    $PYTHON -c "import json; d=json.load(open('$f')); print(len(d))" 2>/dev/null || echo 0
}

get_best_dice() {
    local f="$1"
    [[ -f "$f" ]] || { echo "N/A"; return; }
    $PYTHON -c "
import json; d=json.load(open('$f'))
print(f\"{max(e['val_dice'] for e in d):.4f}\")
" 2>/dev/null || echo "N/A"
}

is_running() {
    local exp="$1"
    ps aux | grep train_reproseg | grep -v grep | grep "$exp" | wc -l | grep -q "^[1-9]"
}

free_gpus() {
    # Returns space-separated list of GPU indices with <5% util and <10GB mem
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$2<5 && $3<10000 {print $1}' \
    | grep -v "^3$" || true  # exclude GPU3 (other user); || true prevents set -e on empty
}

log "=== Monitor run ==="

# ── 1. Training status ────────────────────────────────────────────────────────
LOG_E="$EXP_DIR/reproseg_v1_E_inv_pcgrad/logs/training_log.json"
LOG_F="$EXP_DIR/reproseg_v1_F_full_pcgrad/logs/training_log.json"

EP_E=$(get_epochs "$LOG_E")
EP_F=$(get_epochs "$LOG_F")
DICE_E=$(get_best_dice "$LOG_E")
DICE_F=$(get_best_dice "$LOG_F")

log "E: $EP_E/30 (best_dice=$DICE_E)"
log "F: $EP_F/30 (best_dice=$DICE_F)"

FREE=$(free_gpus)
log "Free GPUs: [${FREE:-none}]"

# ── 2. ICC eval on A (if not done and GPU free) ───────────────────────────────
ICC_OUT="$EXP_DIR/icc_comparison.json"
CKPT_A="$EXP_DIR/reproseg_v1_A_baseline/checkpoints/best.pt"

if [[ ! -f "$ICC_OUT" ]] && [[ -f "$CKPT_A" ]]; then
    # Need at least one free GPU
    GPU_A=$(echo "$FREE" | awk '{print $1}')
    if [[ -n "$GPU_A" ]]; then
        log "Starting ICC eval for A on GPU $GPU_A..."
        cd "$(dirname "$SCRIPT_DIR")" && \
        CUDA_VISIBLE_DEVICES=$GPU_A $PYTHON \
            "$SCRIPT_DIR/eval_reproducibility.py" \
            --checkpoints "A:$CKPT_A" \
            --output "$ICC_OUT" \
            >> "$LOG_FILE" 2>&1
        log "ICC eval for A complete."
    fi
fi

# ── 3. When E complete → add E to ICC comparison ─────────────────────────────
CKPT_E="$EXP_DIR/reproseg_v1_E_inv_pcgrad/checkpoints/best.pt"
ICC_AE="$EXP_DIR/icc_A_vs_E.json"

if [[ "$EP_E" -ge 30 ]] && [[ -f "$CKPT_E" ]] && [[ ! -f "$ICC_AE" ]]; then
    GPU=$(echo "$FREE" | awk '{print $1}')
    if [[ -n "$GPU" ]]; then
        log "E complete. Running ICC for A vs E on GPU $GPU..."
        cd "$(dirname "$SCRIPT_DIR")" && \
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
            "$SCRIPT_DIR/eval_reproducibility.py" \
            --checkpoints "A:$CKPT_A" "E:$CKPT_E" \
            --output "$ICC_AE" \
            >> "$LOG_FILE" 2>&1
        log "ICC A vs E complete → $ICC_AE"
    fi
fi

# ── 4. When F complete → full ICC comparison ─────────────────────────────────
CKPT_F="$EXP_DIR/reproseg_v1_F_full_pcgrad/checkpoints/best.pt"
ICC_ALL="$EXP_DIR/icc_all_models.json"

if [[ "$EP_F" -ge 30 ]] && [[ -f "$CKPT_F" ]] && [[ ! -f "$ICC_ALL" ]]; then
    # Wait for a free GPU
    GPU=$(echo "$FREE" | awk '{print $1}')
    if [[ -n "$GPU" ]]; then
        log "F complete. Running full ICC comparison on GPU $GPU..."
        # Build checkpoint list
        CKPTS="A:$CKPT_A"
        for EXP in B_tcl C_dualstream D_volume E_inv_pcgrad F_full_pcgrad; do
            CKPT="$EXP_DIR/reproseg_v1_${EXP}/checkpoints/best.pt"
            LABEL=$(echo $EXP | sed 's/_pcgrad//' | sed 's/reproseg_v1_//')
            [[ -f "$CKPT" ]] && CKPTS="$CKPTS ${EXP%%_*}:$CKPT"
        done
        cd "$(dirname "$SCRIPT_DIR")" && \
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
            "$SCRIPT_DIR/eval_reproducibility.py" \
            --checkpoints $CKPTS \
            --output "$ICC_ALL" \
            >> "$LOG_FILE" 2>&1
        log "Full ICC comparison complete → $ICC_ALL"
    fi
fi

# ── 5. Auto-launch v2 experiments if ICC improves ────────────────────────────
# v2 = CSG (Gram-Schmidt) + FeatCons (stop-gradient) fixes applied
# Only launch if F ICC shows mean_icc improvement > 0.01 vs A

if [[ -f "$ICC_ALL" ]]; then
    ICC_IMPROVED=$($PYTHON -c "
import json
data = json.load(open('$ICC_ALL'))
if len(data) < 2: exit(1)
baseline = next((d for d in data if d['label']=='A'), None)
full = next((d for d in data if 'F' in d['label']), None)
if not baseline or not full: exit(1)
delta = full['mean_icc'] - baseline['mean_icc']
print(f'delta_icc={delta:.4f}')
exit(0 if delta > 0.01 else 1)
" 2>/dev/null && echo "yes" || echo "no")

    if [[ "$ICC_IMPROVED" == "yes" ]]; then
        # Check for v2 experiments not yet launched
        V2_LOG="$EXP_DIR/reproseg_v2_F_full/logs/training_log.json"
        V2_EP=$(get_epochs "$V2_LOG")
        FREE2=$(free_gpus | tr '\n' ' ')
        GPU1=$(echo $FREE2 | awk '{print $1}')
        GPU2=$(echo $FREE2 | awk '{print $2}')

        if [[ "$V2_EP" -eq 0 ]] && [[ -n "$GPU1" ]] && [[ -n "$GPU2" ]]; then
            log "ICC improved! Launching v2 (fixed CSG+FeatCons) on GPU $GPU1,$GPU2..."
            mkdir -p "$EXP_DIR/reproseg_v2_F_full/"{logs,checkpoints}
            cd "$(dirname "$SCRIPT_DIR")" && \
            CUDA_VISIBLE_DEVICES=$GPU1,$GPU2 $PYTHON \
                -m torch.distributed.run --nproc_per_node=2 --master_port=29581 \
                "$SCRIPT_DIR/train_reproseg.py" \
                --mode full --epochs 30 --batch-size 1 \
                --exp-name reproseg_v2_F_full --pcgrad \
                >> "$EXP_DIR/reproseg_v2_F_full/stdout.log" 2>&1 &
            log "v2 launched PID=$!"
        else
            log "ICC improved but v2 already running or no free GPUs (ep=$V2_EP, free=$FREE2)"
        fi
    else
        log "ICC not improved enough for v2 auto-launch"
    fi
fi

log "=== Monitor done ==="
