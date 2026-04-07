#!/bin/bash
# post_train_watcher.sh
# Background daemon: watches for training completion, auto-runs ICC eval,
# compiles 3-seed statistics, then launches SynthSeg eval.
#
# Usage: nohup bash post_train_watcher.sh > results/watcher_log.txt 2>&1 &

set -euo pipefail
export UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache
PYTHON=/home/vlm/minyoung/.venv/bin/python
EXP_DIR=/home/vlm/minyoung2/experiments/ReproSeg/results
LOG="$EXP_DIR/watcher_log.txt"
SCRIPTS=/home/vlm/minyoung2/experiments/ReproSeg/scripts

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Models to watch: "label:exp_dir_name"
declare -A MODELS=(
    ["A"]="reproseg_v1_A_baseline"
    ["B"]="reproseg_v1_B_tcl"
    ["C"]="reproseg_v1_C_dualstream"
    ["D"]="reproseg_v1_D_volume"
    ["E"]="reproseg_v1_E_inv_pcgrad"
    ["F"]="reproseg_v1_F_full_pcgrad"
    ["A_seed0"]="reproseg_v1_A_seed0"
    ["A_seed1"]="reproseg_v1_A_seed1"
    ["C_seed0"]="reproseg_v1_C_seed0"
    ["C_seed1"]="reproseg_v1_C_seed1"
    ["G"]="reproseg_v1_G_grl_only"
    ["H"]="reproseg_v1_H_inv_only"
)

wait_free_gpu() {
    # Returns one free GPU index; blocks until available
    while true; do
        gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
            --format=csv,noheader,nounits \
            | awk -F', ' '$2<5 && $3<10000 {print $1}' \
            | grep -v "^3$" | head -1 || true)
        if [[ -n "$gpu" ]]; then
            echo "$gpu"
            return
        fi
        sleep 60
    done
}

is_training_done() {
    # Returns true if best.pt exists and no training process for this exp
    local exp_name="$1"
    local ckpt="$EXP_DIR/$exp_name/checkpoints/best.pt"
    [[ -f "$ckpt" ]] && ! (ps aux | grep train_reproseg | grep -v grep | grep -q "$exp_name")
}

icc_done() {
    local label="$1"
    [[ -f "$EXP_DIR/icc_${label}.json" ]]
}

run_icc_eval() {
    local label="$1"
    local exp_name="$2"
    local ckpt="$EXP_DIR/$exp_name/checkpoints/best.pt"
    local out="$EXP_DIR/icc_${label}.json"

    log "Running ICC eval for $label ($exp_name) on GPU $gpu..."
    cd /home/vlm/minyoung2 && \
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON \
        "$SCRIPTS/eval_reproducibility.py" \
        --checkpoints "${label}:${ckpt}" \
        --output "$out" \
        >> "$EXP_DIR/icc_${label}_eval.log" 2>&1
    log "ICC eval for $label done → $out"
}

compile_seed_stats() {
    # Called after A_seed0, A_seed1, C_seed0, C_seed1 all done
    local out="$EXP_DIR/icc_seed_stats.json"
    [[ -f "$out" ]] && { log "Seed stats already compiled."; return; }

    log "Compiling 3-seed statistics (A and C)..."
    $PYTHON - <<'PYEOF' >> "$LOG" 2>&1
import json, numpy as np
from pathlib import Path
EXP = Path('/home/vlm/minyoung2/experiments/ReproSeg/results')

def load_icc(label):
    p = EXP / f'icc_{label}.json'
    if not p.exists(): return None
    data = json.load(open(p))
    entry = next((d for d in data if d['label'] == label), data[0])
    return entry['mean_icc'], entry['mean_cv_pct']

results = {}
for model, seeds in [('A', ['A', 'A_seed0', 'A_seed1']),
                      ('C', ['C', 'C_seed0', 'C_seed1'])]:
    vals = []
    for s in seeds:
        r = load_icc(s)
        if r: vals.append(r)
    if len(vals) >= 2:
        iccs = [v[0] for v in vals]
        cvs  = [v[1] for v in vals]
        results[model] = {
            'n_seeds': len(vals),
            'icc_mean': round(float(np.mean(iccs)), 4),
            'icc_std':  round(float(np.std(iccs)),  4),
            'icc_vals': [round(v, 4) for v in iccs],
            'cv_mean':  round(float(np.mean(cvs)), 2),
            'cv_std':   round(float(np.std(cvs)),  2),
        }

delta_icc = results.get('C', {}).get('icc_mean', 0) - results.get('A', {}).get('icc_mean', 0)
results['delta_icc_C_vs_A'] = round(delta_icc, 4)
print(json.dumps(results, indent=2))
json.dump(results, open('/home/vlm/minyoung2/experiments/ReproSeg/results/icc_seed_stats.json', 'w'), indent=2)
print("Saved icc_seed_stats.json")
PYEOF
    log "Seed stats compiled → $out"
}

compile_full_icc_table() {
    local out="$EXP_DIR/icc_all_models_fixed.json"
    [[ -f "$out" ]] && { log "Full ICC table already exists."; return; }

    log "Building full ICC comparison table from individual eval files..."
    # Merge all per-model icc_*.json into one list
    $PYTHON - <<'PYEOF' >> "$LOG" 2>&1
import json
from pathlib import Path
EXP = Path('/home/vlm/minyoung2/experiments/ReproSeg/results')
labels = ['A','B','C','D','E','F','G','H']
results = []
for lbl in labels:
    p = EXP / f'icc_{lbl}.json'
    if not p.exists(): continue
    data = json.load(open(p))
    entry = next((d for d in data if d['label'] == lbl), data[0])
    entry['label'] = lbl
    results.append(entry)
json.dump(results, open(EXP / 'icc_all_models_fixed.json', 'w'), indent=2)
print(f"Merged {len(results)} models into icc_all_models_fixed.json")
PYEOF
}

launch_synthseg_eval() {
    local out="$EXP_DIR/icc_synthseg.json"
    [[ -f "$out" ]] && { log "SynthSeg eval already done."; return; }
    local gpu=$(wait_free_gpu)
    log "Launching SynthSeg eval on GPU $gpu..."
    cd /home/vlm/minyoung2 && \
    FREESURFER_HOME=/home/vlm/hyerin/tools/freesurfer \
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON \
        "$SCRIPTS/eval_synthseg.py" \
        --output "$EXP_DIR/icc_synthseg.json" \
        --cache-dir /tmp/synthseg_cache \
        >> "$EXP_DIR/icc_synthseg_eval.log" 2>&1
    log "SynthSeg eval done → $out"
}

extract_individual_iccs() {
    # Split icc_all_models_fixed.json → icc_A.json, icc_B.json, etc.
    local src="$EXP_DIR/icc_all_models_fixed.json"
    [[ -f "$src" ]] || { log "ERROR: $src missing"; return 1; }
    log "Extracting individual model ICC files from icc_all_models_fixed.json..."
    $PYTHON - <<PYEOF >> "$LOG" 2>&1
import json
from pathlib import Path
src = Path('$src')
EXP = Path('$EXP_DIR')
data = json.load(open(src))
for entry in data:
    lbl = entry['label']
    out = EXP / f'icc_{lbl}.json'
    if not out.exists():
        json.dump([entry], open(out, 'w'), indent=2)
        print(f"  Extracted icc_{lbl}.json (mean_icc={entry.get('mean_icc','?')})")
    else:
        print(f"  icc_{lbl}.json already exists, skipping")
PYEOF
    log "Individual ICC extraction done."
}

log "=== Post-train watcher started ==="
log "Watching: ${!MODELS[*]}"

# Wait for the current re-eval (icc_all_models_fixed.json) to finish first
log "Waiting for icc_all_models_fixed.json (current re-eval)..."
while [[ ! -f "$EXP_DIR/icc_all_models_fixed.json" ]]; do
    sleep 120
done
log "icc_all_models_fixed.json found."

# Extract A, B, C, D, E, F into individual files for seed-stats
extract_individual_iccs

log "Starting per-seed ICC evals."

# Main watch loop: run ICC eval for each experiment as it completes
REMAINING=(A_seed0 A_seed1 C_seed0 C_seed1 G H)

while [[ ${#REMAINING[@]} -gt 0 ]]; do
    NEW_REMAINING=()
    for label in "${REMAINING[@]}"; do
        exp_name="${MODELS[$label]}"
        if icc_done "$label"; then
            log "$label ICC already done, skipping."
            continue
        fi
        if is_training_done "$exp_name"; then
            gpu=$(wait_free_gpu)
            run_icc_eval "$label" "$exp_name"
        else
            NEW_REMAINING+=("$label")
        fi
    done
    REMAINING=("${NEW_REMAINING[@]+"${NEW_REMAINING[@]}"}")
    [[ ${#REMAINING[@]} -gt 0 ]] && sleep 300
done

log "All per-seed ICC evals done."

# Compile 3-seed statistics if A_seed0/1 and C_seed0/1 are all done
all_seeds_done=true
for label in A_seed0 A_seed1 C_seed0 C_seed1; do
    icc_done "$label" || { all_seeds_done=false; break; }
done

if $all_seeds_done; then
    compile_seed_stats
fi

# Compile full table if all A-F + G + H done
compile_full_icc_table

# Launch SynthSeg eval
launch_synthseg_eval

log "=== Post-train watcher complete ==="
