#!/bin/bash
# Auto-launch queue: C seeds + Task #4 ablations as GPUs free up
set -euo pipefail
export UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache
PYTHON=/home/vlm/minyoung/.venv/bin/python
LOG=/home/vlm/minyoung2/experiments/ReproSeg/results/queue_log.txt
EXP_DIR=/home/vlm/minyoung2/experiments/ReproSeg/results

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

wait_for_free_gpus() {
    local n=$1
    while true; do
        free=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
            --format=csv,noheader,nounits \
            | awk -F', ' '$2<10 && $3<10000 {print $1}' \
            | grep -v "^3$" | head -"$n" | tr '\n' ',' | sed 's/,$//')
        count=$(echo "$free" | tr ',' '\n' | grep -c '[0-9]' || true)
        if [[ "$count" -ge "$n" ]]; then
            echo "$free"
            return
        fi
        sleep 60
    done
}

launch_train() {
    local mode=$1 seed=$2 name=$3 gpus=$4 port=$5
    log "Launching $name on GPU $gpus (mode=$mode seed=$seed)"
    nohup bash -c "
export UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache
CUDA_VISIBLE_DEVICES=$gpus $PYTHON -m torch.distributed.run \
  --nproc_per_node=2 --master_port=$port \
  experiments/ReproSeg/scripts/train_reproseg.py \
  --mode $mode --epochs 30 --batch-size 1 --seed $seed \
  --exp-name $name
" > "$EXP_DIR/${name}_launch.log" 2>&1 &
    log "$name PID: $!"
}

cd /home/vlm/minyoung2

# Queue (in order, 2 GPUs each)
declare -a QUEUE=(
    "inv:0:reproseg_v1_C_seed0"
    "inv:1:reproseg_v1_C_seed1"
    "grl_only:42:reproseg_v1_G_grl_only"
    "inv_only:42:reproseg_v1_H_inv_only"
)

PORTS=(29582 29583 29584 29585)
idx=0

for item in "${QUEUE[@]}"; do
    IFS=':' read -r mode seed name <<< "$item"
    log "Waiting for 2 free GPUs for: $name"
    gpus=$(wait_for_free_gpus 2)
    g1=$(echo "$gpus" | cut -d',' -f1)
    g2=$(echo "$gpus" | cut -d',' -f2)
    launch_train "$mode" "$seed" "$name" "${g1},${g2}" "${PORTS[$idx]}"
    sleep 30  # Give job time to claim GPU before next iteration
    idx=$((idx+1))
done

log "All jobs queued."
