#!/bin/bash
# Waits for C/D to finish (30 epochs), then launches E and F, then monitors.

LOG_C="/home/vlm/minyoung2/experiments/ReproSeg/results/reproseg_v1_C_dualstream/logs/training_log.json"
LOG_D="/home/vlm/minyoung2/experiments/ReproSeg/results/reproseg_v1_D_volume/logs/training_log.json"
SCRIPT_DIR="/home/vlm/minyoung2/experiments/ReproSeg/scripts"
LAUNCH_LOG="/home/vlm/minyoung2/experiments/ReproSeg/results/launch_EF.log"

cd /home/vlm/minyoung2

get_epochs() { python3 -c "import json; d=json.load(open('$1')); print(len(d))" 2>/dev/null || echo 0; }

echo "[$(date '+%H:%M:%S')] Watcher started. Waiting for C(30ep) and D(30ep)..." | tee -a "$LAUNCH_LOG"

while true; do
    ep_c=$(get_epochs "$LOG_C")
    ep_d=$(get_epochs "$LOG_D")
    echo "[$(date '+%H:%M:%S')] C=$ep_c/30  D=$ep_d/30" | tee -a "$LAUNCH_LOG"

    if [ "$ep_c" -ge 30 ] && [ "$ep_d" -ge 30 ]; then
        echo "[$(date '+%H:%M:%S')] C and D complete. Launching E and F..." | tee -a "$LAUNCH_LOG"
        break
    fi
    sleep 300  # check every 5 min
done

# Launch E: repro+inv+PCGrad on GPU 2,4
echo "[$(date '+%H:%M:%S')] Starting E (inv mode + PCGrad) on GPU 2,4..." | tee -a "$LAUNCH_LOG"
CUDA_VISIBLE_DEVICES=2,4 UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    /home/vlm/minyoung/.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29561 \
    "$SCRIPT_DIR/train_reproseg.py" \
    --mode inv --epochs 30 --batch-size 1 \
    --exp-name reproseg_v1_E_inv_pcgrad --pcgrad \
    >> /home/vlm/minyoung2/experiments/ReproSeg/results/reproseg_v1_E_inv_pcgrad/stdout.log 2>&1 &
PID_E=$!
echo "[$(date '+%H:%M:%S')] E launched PID=$PID_E" | tee -a "$LAUNCH_LOG"

# Launch F: full+PCGrad on GPU 6,7
echo "[$(date '+%H:%M:%S')] Starting F (full mode + PCGrad) on GPU 6,7..." | tee -a "$LAUNCH_LOG"
CUDA_VISIBLE_DEVICES=6,7 UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    /home/vlm/minyoung/.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29562 \
    "$SCRIPT_DIR/train_reproseg.py" \
    --mode full --epochs 30 --batch-size 1 \
    --exp-name reproseg_v1_F_full_pcgrad --pcgrad \
    >> /home/vlm/minyoung2/experiments/ReproSeg/results/reproseg_v1_F_full_pcgrad/stdout.log 2>&1 &
PID_F=$!
echo "[$(date '+%H:%M:%S')] F launched PID=$PID_F" | tee -a "$LAUNCH_LOG"

# Monitor E and F every 10 min
LOG_E="/home/vlm/minyoung2/experiments/ReproSeg/results/reproseg_v1_E_inv_pcgrad/logs/training_log.json"
LOG_F="/home/vlm/minyoung2/experiments/ReproSeg/results/reproseg_v1_F_full_pcgrad/logs/training_log.json"

echo "[$(date '+%H:%M:%S')] Monitoring E and F..." | tee -a "$LAUNCH_LOG"
while true; do
    ep_e=$(get_epochs "$LOG_E")
    ep_f=$(get_epochs "$LOG_F")

    # Last dice for each
    dice_e=$(python3 -c "import json; d=json.load(open('$LOG_E')); print(f\"{d[-1]['val_dice']:.4f}\")" 2>/dev/null || echo "N/A")
    dice_f=$(python3 -c "import json; d=json.load(open('$LOG_F')); print(f\"{d[-1]['val_dice']:.4f}\")" 2>/dev/null || echo "N/A")
    cos_e=$(python3 -c "import json; d=json.load(open('$LOG_E')); c=d[-1].get('grad_cos',{}); print(' '.join(f'{k}={v:+.3f}' for k,v in c.items()))" 2>/dev/null || echo "")
    cos_f=$(python3 -c "import json; d=json.load(open('$LOG_F')); c=d[-1].get('grad_cos',{}); print(' '.join(f'{k}={v:+.3f}' for k,v in c.items()))" 2>/dev/null || echo "")

    echo "[$(date '+%H:%M:%S')] E=$ep_e/30 dice=$dice_e  cos=[$cos_e]" | tee -a "$LAUNCH_LOG"
    echo "[$(date '+%H:%M:%S')] F=$ep_f/30 dice=$dice_f  cos=[$cos_f]" | tee -a "$LAUNCH_LOG"
    echo "---" | tee -a "$LAUNCH_LOG"

    if [ "$ep_e" -ge 30 ] && [ "$ep_f" -ge 30 ]; then
        echo "[$(date '+%H:%M:%S')] E and F complete." | tee -a "$LAUNCH_LOG"
        break
    fi
    sleep 600  # report every 10 min
done
