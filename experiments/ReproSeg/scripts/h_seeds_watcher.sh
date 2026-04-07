#!/bin/bash
# Watches H_seed0 and H_seed1 training, runs ICC eval on completion.
set -uo pipefail
export UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache
PYTHON=/home/vlm/minyoung/.venv/bin/python
EXP_DIR=/home/vlm/minyoung2/experiments/ReproSeg/results
SCRIPTS=/home/vlm/minyoung2/experiments/ReproSeg/scripts
LOG="$EXP_DIR/h_seeds_watcher.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

get_epochs() {
  local logf=$(find "$EXP_DIR/$1/logs/" -name "*.json" 2>/dev/null | head -1)
  [[ -f "$logf" ]] && $PYTHON -c "import json; d=json.load(open('$logf')); print(len(d))" 2>/dev/null || echo 0
}

wait_free_gpu() {
  while true; do
    gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits \
      | awk -F', ' '$2<5 && $3<10000 {print $1}' \
      | grep -v "^3$" | head -1 || true)
    [[ -n "$gpu" ]] && echo "$gpu" && return
    sleep 120
  done
}

run_icc() {
  local label=$1 exp=$2
  local ckpt="$EXP_DIR/$exp/checkpoints/best.pt"
  local out="$EXP_DIR/icc_${label}_final.json"
  [[ -f "$out" ]] && { log "$label ICC already done"; return 0; }
  local gpu=$(wait_free_gpu)
  log "ICC eval: $label on GPU $gpu"
  local attempt=0
  while [[ $attempt -lt 3 ]]; do
    attempt=$((attempt+1))
    cd /home/vlm/minyoung2 && CUDA_VISIBLE_DEVICES=$gpu $PYTHON \
      "$SCRIPTS/eval_reproducibility.py" \
      --checkpoints "${label}:${ckpt}" \
      --output "$out" >> "$EXP_DIR/icc_${label}_final_eval.log" 2>&1 && break
    log "WARN: $label eval attempt $attempt failed, retrying in 30s..."
    sleep 30
  done
  if [[ -f "$out" ]]; then
    log "Done: $label → mean_icc=$(python3 -c "import json; d=json.load(open('$out')); print(d[0]['mean_icc'])" 2>/dev/null)"
  else
    log "ERROR: $label eval failed after $attempt attempts"
  fi
}

log "=== H seeds watcher started (H_seed0, H_seed1) ==="

declare -A JOBS=(
  ["H_seed0"]="reproseg_v1_H_seed0"
  ["H_seed1"]="reproseg_v1_H_seed1"
)

while true; do
  all_done=true
  for label in H_seed0 H_seed1; do
    exp="${JOBS[$label]}"
    out="$EXP_DIR/icc_${label}_final.json"
    [[ -f "$out" ]] && continue
    ep=$(get_epochs "$exp")
    if [[ "$ep" -ge 30 ]] && ! ps aux | grep train_reproseg | grep -v grep | grep -q "$exp"; then
      run_icc "$label" "$exp"
    else
      all_done=false
      log "$label: ep=$ep/30, still training..."
    fi
  done
  $all_done && break
  sleep 600
done

# Compile 3-seed stats for H
log "Compiling H 3-seed stats..."
$PYTHON - << 'PYEOF' >> "$LOG" 2>&1
import json, numpy as np
from pathlib import Path
EXP = Path('/home/vlm/minyoung2/experiments/ReproSeg/results')

def load_icc(label):
    for suffix in ['_final', '']:
        p = EXP / f'icc_{label}{suffix}.json'
        if p.exists():
            data = json.load(open(p))
            e = next((d for d in data if d['label'] == label), data[0])
            return e['mean_icc'], e.get('per_structure', {})
    return None

# H: seed42, seed0, seed1
h_seeds = ['H', 'H_seed0', 'H_seed1']
h_results = [load_icc(s) for s in h_seeds]
h_results = [r for r in h_results if r is not None]

if h_results:
    iccs = [r[0] for r in h_results]
    print(f"H 3-seed ICC: {iccs}")
    print(f"  mean={np.mean(iccs):.4f}, std={np.std(iccs):.4f}, min={np.min(iccs):.4f}, max={np.max(iccs):.4f}")

    # Per-structure mean across seeds
    all_structs = {}
    for _, per_struct in h_results:
        for s, v in per_struct.items():
            all_structs.setdefault(s, []).append(v['icc'])
    print("\nPer-structure mean ICC (H, 3 seeds):")
    for s, vals in sorted(all_structs.items()):
        print(f"  {s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    stats = {
        'model': 'H_CSG_only',
        'n_seeds': len(iccs),
        'icc_vals': [round(v, 4) for v in iccs],
        'icc_mean': round(float(np.mean(iccs)), 4),
        'icc_std': round(float(np.std(iccs)), 4),
        'icc_min': round(float(np.min(iccs)), 4),
        'icc_max': round(float(np.max(iccs)), 4),
        'per_structure_mean': {s: round(float(np.mean(vals)), 4) for s, vals in all_structs.items()},
    }
    out = EXP / 'icc_H_seed_stats.json'
    json.dump(stats, open(out, 'w'), indent=2)
    print(f"\nSaved {out}")
else:
    print("ERROR: No H ICC results found")
PYEOF

log "=== H seeds watcher complete ==="
