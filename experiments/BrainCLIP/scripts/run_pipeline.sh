#!/bin/bash
# run_pipeline.sh
# BrainCLIP 전체 실험 파이프라인 — nohup 환경에서 세션 독립 실행
#
# 실행법:
#   cd /home/vlm/minyoung2
#   nohup bash experiments/BrainCLIP/scripts/run_pipeline.sh > \
#       experiments/BrainCLIP/results/pipeline_stdout.log 2>&1 &
#   echo "PID: $!"
#
# 재시작 안전: STATE_FILE 기반 멱등 실행 (완료된 phase 스킵)
# 중단 후 재실행하면 마지막 완료 phase 다음부터 이어서 진행

set -euo pipefail

# ── 환경 ──────────────────────────────────────────────────────────────────────
export UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache
PYTHON="uv run python"
ROOT=/home/vlm/minyoung2
SCRIPT_DIR="$ROOT/experiments/BrainCLIP/scripts"
DATA_DIR="$ROOT/experiments/BrainCLIP/data"
RESULT_DIR="$ROOT/experiments/BrainCLIP/results"
STATE_FILE="$RESULT_DIR/pipeline_state.json"
LOG="$RESULT_DIR/pipeline.log"

mkdir -p "$DATA_DIR" "$RESULT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
fail() { log "FAILED: $*"; exit 1; }

# ── GPU 선택 ──────────────────────────────────────────────────────────────────
# GPU 3은 타 사용자 예약 → 항상 제외
RESERVED_GPUS="3"
MEM_FREE_THRESHOLD=8000  # MB

pick_all_free_gpus() {
    # free GPU 전체를 쉼표 구분 문자열로 반환 e.g. "0,6"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits \
    | awk -F', ' -v reserved="$RESERVED_GPUS" -v thresh="$MEM_FREE_THRESHOLD" '
        BEGIN { split(reserved, r, ","); for (k in r) excl[r[k]]=1 }
        !excl[$1] && $2+0 < thresh && $3+0 < 10 { printf "%s,", $1 }
    ' | sed 's/,$//'
}

pick_best_gpu() {
    # free GPU 중 메모리 사용이 가장 적은 단일 GPU
    nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits \
    | awk -F', ' -v reserved="$RESERVED_GPUS" '
        BEGIN { split(reserved, r, ","); for (k in r) excl[r[k]]=1 }
        !excl[$1] { print $2+0, $1 }
    ' | sort -n | head -1 | awk '{print $2}'
}

wait_free_gpus() {
    # 최소 1개 free GPU 대기 → 모든 free GPU 반환
    log "GPU 대기 중 (가용 GPU 탐색)..."
    while true; do
        GPUS=$(pick_all_free_gpus)
        COUNT=$(echo "$GPUS" | tr ',' '\n' | grep -c '[0-9]' || true)
        if [[ "$COUNT" -ge 1 ]]; then
            log "가용 GPU: [$GPUS] (${COUNT}개)"
            echo "$GPUS"
            return
        fi
        log "Free GPU 없음. 5분 후 재시도..."
        sleep 300
    done
}

VENV_PYTHON="$ROOT/.venv/bin/python"
VENV_TORCHRUN="$ROOT/.venv/bin/torchrun"
# BrainCLIP은 소형 모델(~12M trainable) → DDP 최대 2 GPU
# 2 초과 시 통신 오버헤드 > 병렬화 이득
MAX_DDP_GPUS=2

launch_train() {
    # launch_train <config_path> <log_path>
    # GPU 1,2 DDP — B200 183GB × 2, --standalone (single-node torchrun)
    local config=$1
    local train_log=$2

    # GPU 1,2 가용 대기
    log "GPU 1,2 가용 여부 확인 중..."
    while true; do
        MEM1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        MEM2=$(nvidia-smi --id=2 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if [[ "${MEM1:-99999}" -lt "$MEM_FREE_THRESHOLD" ]] && \
           [[ "${MEM2:-99999}" -lt "$MEM_FREE_THRESHOLD" ]]; then
            log "GPU 1 (${MEM1}MB), GPU 2 (${MEM2}MB) — 가용"
            break
        fi
        log "GPU 1 (${MEM1}MB) or GPU 2 (${MEM2}MB) 사용 중. 5분 후 재시도..."
        sleep 300
    done

    log "학습 시작: GPU=1,2 (DDP/2) config=$config"
    CUDA_VISIBLE_DEVICES=1,2 \
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_TORCHRUN" \
        --nproc_per_node=2 \
        --master_port=29601 \
        --standalone \
        "$SCRIPT_DIR/train_brainclip.py" \
        --config "$config" \
        >> "$train_log" 2>&1
}

# ── State 관리 ────────────────────────────────────────────────────────────────
get_phase() {
    if [[ ! -f "$STATE_FILE" ]]; then
        echo "setup"
        return
    fi
    python3 -c "
import json; d=json.load(open('$STATE_FILE'))
print(d.get('phase','setup'))
" 2>/dev/null || echo "setup"
}

set_phase() {
    local phase=$1
    python3 -c "
import json, os
f='$STATE_FILE'
d = json.load(open(f)) if os.path.exists(f) else {}
d['phase'] = '$phase'
d['updated_at'] = __import__('datetime').datetime.now().isoformat()
json.dump(d, open(f,'w'), indent=2)
"
    log "Phase → $phase"
}

set_key() {
    python3 -c "
import json, os
f='$STATE_FILE'
d = json.load(open(f)) if os.path.exists(f) else {}
d['$1'] = $2
json.dump(d, open(f,'w'), indent=2)
"
}

# ── Phase 실행 함수들 ──────────────────────────────────────────────────────────

phase_setup() {
    log "=== Phase: setup (의존성 설치) ==="
    cd "$ROOT"
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache uv sync || fail "uv sync 실패"
    log "의존성 설치 완료"
    set_phase "text_build"
}

phase_text_build() {
    log "=== Phase: text_build (임상 텍스트 생성) ==="
    cd "$ROOT"

    # EXP01용 (diagnosis 포함)
    if [[ ! -f "$DATA_DIR/clinical_texts.csv" ]]; then
        log "clinical_texts.csv 생성 중..."
        $PYTHON "$SCRIPT_DIR/build_clinical_text.py" \
            --output "$DATA_DIR/clinical_texts.csv" --preview 3 \
            >> "$LOG" 2>&1 || fail "build_clinical_text (w/ diag) 실패"
    else
        log "clinical_texts.csv 이미 존재, 스킵"
    fi

    # EXP02용 (diagnosis 제외)
    if [[ ! -f "$DATA_DIR/clinical_texts_no_diag.csv" ]]; then
        log "clinical_texts_no_diag.csv 생성 중..."
        $PYTHON "$SCRIPT_DIR/build_clinical_text.py" \
            --output "$DATA_DIR/clinical_texts_no_diag.csv" \
            --no_diagnosis --preview 3 \
            >> "$LOG" 2>&1 || fail "build_clinical_text (no diag) 실패"
    else
        log "clinical_texts_no_diag.csv 이미 존재, 스킵"
    fi

    # 텍스트 품질 확인
    N01=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$DATA_DIR/clinical_texts.csv')))" 2>/dev/null || echo 0)
    N02=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$DATA_DIR/clinical_texts_no_diag.csv')))" 2>/dev/null || echo 0)
    log "텍스트 생성 완료: EXP01=${N01}건, EXP02=${N02}건"
    [[ "$N01" -gt 100 ]] || fail "clinical_texts.csv 건수 이상: $N01"

    set_phase "embed_extract"
}

phase_embed_extract() {
    log "=== Phase: embed_extract (BrainIAC embedding 추출) ==="
    EMB_DIR="$DATA_DIR/mri_embeddings"
    DONE_MARK="$EMB_DIR/.extraction_complete"

    if [[ -f "$DONE_MARK" ]]; then
        N=$(ls "$EMB_DIR"/*.npy 2>/dev/null | wc -l)
        log "embedding 캐시 이미 존재 (${N}개), 스킵"
        set_phase "train_exp01"
        return
    fi

    # 추출은 단일 GPU로 충분 (직렬 I/O, DDP 불필요)
    GPU=$(pick_best_gpu)
    # GPU가 없으면 wait
    while [[ -z "$GPU" ]]; do
        log "Free GPU 없음. 5분 후 재시도..."
        sleep 300
        GPU=$(pick_best_gpu)
    done
    log "GPU $GPU 에서 embedding 추출 시작..."
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES=$GPU \
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_PYTHON" "$SCRIPT_DIR/extract_mri_embeddings.py" \
        --output_dir "$EMB_DIR" --split all --batch_size 8 --num_workers 4 \
        >> "$LOG" 2>&1 || fail "extract_mri_embeddings 실패"

    N=$(ls "$EMB_DIR"/*.npy 2>/dev/null | wc -l)
    log "embedding 추출 완료: ${N}개"
    [[ "$N" -gt 100 ]] || fail "embedding 건수 이상: $N"
    touch "$DONE_MARK"
    set_phase "train_exp01"
}

phase_train_exp01() {
    log "=== Phase: train_exp01 ==="
    CKPT="$RESULT_DIR/exp01_baseline/checkpoints/brainclip_best.pt"

    if [[ -f "$CKPT" ]]; then
        log "EXP01 best checkpoint 이미 존재, 스킵"
        set_phase "eval_exp01"
        return
    fi

    cd "$ROOT"

    # embedding 캐시 활성화: config에 경로 주입
    CONFIG="$ROOT/experiments/BrainCLIP/configs/exp01_baseline.toml"
    CONFIG_RUN="$RESULT_DIR/exp01_baseline_run.toml"
    python3 -c "
import re
txt = open('$CONFIG').read()
txt = re.sub(r'^# (embedding_cache_dir\s*=.*)', r'\1', txt, flags=re.MULTILINE)
open('$CONFIG_RUN','w').write(txt)
"
    launch_train "$CONFIG_RUN" "$LOG" || fail "train EXP01 실패"
    log "EXP01 학습 완료"
    set_phase "eval_exp01"
}

phase_eval_exp01() {
    log "=== Phase: eval_exp01 ==="
    CKPT="$RESULT_DIR/exp01_baseline/checkpoints/brainclip_best.pt"
    EVAL_DONE="$RESULT_DIR/exp01_baseline/logs/.eval_complete"

    if [[ -f "$EVAL_DONE" ]]; then
        log "EXP01 eval 이미 완료, 스킵"
        set_phase "train_exp02"
        return
    fi

    [[ -f "$CKPT" ]] || fail "EXP01 checkpoint 없음: $CKPT"

    GPU=$(pick_best_gpu)
    cd "$ROOT"
    log "EXP01 retrieval 평가 (GPU $GPU)..."
    CUDA_VISIBLE_DEVICES=$GPU UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_PYTHON" "$SCRIPT_DIR/eval_retrieval.py" \
        --ckpt "$CKPT" \
        --texts_csv "$DATA_DIR/clinical_texts.csv" \
        --embedding_cache_dir "$DATA_DIR/mri_embeddings" \
        --split test --topk 1 5 10 \
        >> "$LOG" 2>&1 || fail "eval_retrieval EXP01 실패"

    log "EXP01 linear probe 평가 (GPU $GPU)..."
    CUDA_VISIBLE_DEVICES=$GPU UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_PYTHON" "$SCRIPT_DIR/eval_linear_probe.py" \
        --ckpt "$CKPT" \
        --texts_csv "$DATA_DIR/clinical_texts.csv" \
        --embedding_cache_dir "$DATA_DIR/mri_embeddings" \
        --mode both \
        >> "$LOG" 2>&1 || fail "eval_linear_probe EXP01 실패"

    touch "$EVAL_DONE"
    log "EXP01 평가 완료"
    set_phase "train_exp02"
}

phase_train_exp02() {
    log "=== Phase: train_exp02 ==="
    CKPT="$RESULT_DIR/exp02_no_diag_text/checkpoints/brainclip_best.pt"

    if [[ -f "$CKPT" ]]; then
        log "EXP02 best checkpoint 이미 존재, 스킵"
        set_phase "eval_exp02"
        return
    fi

    cd "$ROOT"

    CONFIG="$ROOT/experiments/BrainCLIP/configs/exp02_no_diag_text.toml"
    CONFIG_RUN="$RESULT_DIR/exp02_no_diag_text_run.toml"
    python3 -c "
import re
txt = open('$CONFIG').read()
txt = re.sub(r'^# (embedding_cache_dir\s*=.*)', r'\1', txt, flags=re.MULTILINE)
open('$CONFIG_RUN','w').write(txt)
"
    launch_train "$CONFIG_RUN" "$LOG" || fail "train EXP02 실패"
    log "EXP02 학습 완료"
    set_phase "eval_exp02"
}

phase_eval_exp02() {
    log "=== Phase: eval_exp02 ==="
    CKPT="$RESULT_DIR/exp02_no_diag_text/checkpoints/brainclip_best.pt"
    EVAL_DONE="$RESULT_DIR/exp02_no_diag_text/logs/.eval_complete"

    if [[ -f "$EVAL_DONE" ]]; then
        log "EXP02 eval 이미 완료, 스킵"
        set_phase "report"
        return
    fi

    [[ -f "$CKPT" ]] || fail "EXP02 checkpoint 없음: $CKPT"

    GPU=$(pick_best_gpu)
    cd "$ROOT"
    log "EXP02 retrieval 평가 (GPU $GPU)..."
    CUDA_VISIBLE_DEVICES=$GPU UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_PYTHON" "$SCRIPT_DIR/eval_retrieval.py" \
        --ckpt "$CKPT" \
        --texts_csv "$DATA_DIR/clinical_texts_no_diag.csv" \
        --embedding_cache_dir "$DATA_DIR/mri_embeddings" \
        --split test --topk 1 5 10 \
        >> "$LOG" 2>&1 || fail "eval_retrieval EXP02 실패"

    log "EXP02 linear probe 평가 (GPU $GPU)..."
    CUDA_VISIBLE_DEVICES=$GPU UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_PYTHON" "$SCRIPT_DIR/eval_linear_probe.py" \
        --ckpt "$CKPT" \
        --texts_csv "$DATA_DIR/clinical_texts_no_diag.csv" \
        --embedding_cache_dir "$DATA_DIR/mri_embeddings" \
        --mode both \
        >> "$LOG" 2>&1 || fail "eval_linear_probe EXP02 실패"

    touch "$EVAL_DONE"
    log "EXP02 평가 완료"
    set_phase "smoke_exp03"
}

phase_smoke_exp03() {
    log "=== Phase: smoke_exp03 (EXP03 1-epoch 속도/메모리 검증) ==="
    SMOKE_LOG="$RESULT_DIR/exp03_mri_finetune/smoke_test.log"
    mkdir -p "$RESULT_DIR/exp03_mri_finetune"

    # 1-epoch smoke config
    SMOKE_CFG="$RESULT_DIR/exp03_smoke_run.toml"
    python3 -c "
import re
txt = open('$ROOT/experiments/BrainCLIP/configs/exp03_mri_finetune.toml').read()
txt = re.sub(r'^epochs\s*=\s*\d+', 'epochs = 1', txt, flags=re.MULTILINE)
open('$SMOKE_CFG', 'w').write(txt)
"
    log "GPU 1,2로 1-epoch smoke test 시작..."
    CUDA_VISIBLE_DEVICES=1,2 \
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_TORCHRUN" \
        --nproc_per_node=2 --master_port=29602 --standalone \
        "$SCRIPT_DIR/train_brainclip.py" \
        --config "$SMOKE_CFG" \
        > "$SMOKE_LOG" 2>&1 || fail "smoke_exp03 실패 — 로그: $SMOKE_LOG"

    # Go/Kill gate: 1-epoch 후 val_loss < 5.5 (EXP01 초기 4.71 참고)
    VAL_LOSS=$(grep "\-1,val," "$RESULT_DIR/exp03_mri_finetune/logs/train_log.csv" \
               | tail -1 | awk -F',' '{print $4}')
    log "Smoke test val_loss=$VAL_LOSS"
    python3 -c "
import sys
v = float('$VAL_LOSS')
if v > 5.5:
    print(f'  KILL: val_loss={v:.4f} > 5.5 — 학습 신호 없음')
    sys.exit(1)
print(f'  GO: val_loss={v:.4f} <= 5.5 — EXP03 본 학습 진행')
" || fail "smoke_exp03 Kill gate: val_loss 기준 미달"

    # smoke ckpt 제거 (본 학습이 덮어씀)
    rm -rf "$RESULT_DIR/exp03_mri_finetune/checkpoints"
    rm -f  "$RESULT_DIR/exp03_mri_finetune/logs/train_log.csv"

    log "Smoke test 통과 → train_exp03 진행"
    set_phase "train_exp03"
}

phase_train_exp03() {
    log "=== Phase: train_exp03 (BrainIAC 2-block unfreeze) ==="
    CKPT="$RESULT_DIR/exp03_mri_finetune/checkpoints/brainclip_best.pt"

    if [[ -f "$CKPT" ]]; then
        log "EXP03 best checkpoint 이미 존재, 스킵"
        set_phase "eval_exp03"
        return
    fi

    cd "$ROOT"
    # EXP03은 embedding_cache_dir 없이 raw NIfTI 사용 (fine_tune_layers > 0)
    CONFIG="$ROOT/experiments/BrainCLIP/configs/exp03_mri_finetune.toml"
    log "학습 시작: GPU=1,2 (DDP/2) config=$CONFIG"
    CUDA_VISIBLE_DEVICES=1,2 \
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_TORCHRUN" \
        --nproc_per_node=2 --master_port=29601 --standalone \
        "$SCRIPT_DIR/train_brainclip.py" \
        --config "$CONFIG" \
        >> "$LOG" 2>&1 || fail "train EXP03 실패"
    log "EXP03 학습 완료"
    set_phase "eval_exp03"
}

phase_eval_exp03() {
    log "=== Phase: eval_exp03 ==="
    CKPT="$RESULT_DIR/exp03_mri_finetune/checkpoints/brainclip_best.pt"
    EVAL_DONE="$RESULT_DIR/exp03_mri_finetune/logs/.eval_complete"

    if [[ -f "$EVAL_DONE" ]]; then
        log "EXP03 eval 이미 완료, 스킵"
        set_phase "report"
        return
    fi

    [[ -f "$CKPT" ]] || fail "EXP03 checkpoint 없음: $CKPT"

    GPU=$(pick_best_gpu)
    cd "$ROOT"
    # EXP03: backbone fine-tuned → embedding cache 사용 불가 (cached는 frozen backbone 기준)
    # raw NIfTI로 직접 추론해야 fine-tuned backbone feature 반영됨
    log "EXP03 retrieval 평가 (GPU $GPU, raw NIfTI)..."
    CUDA_VISIBLE_DEVICES=$GPU UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_PYTHON" "$SCRIPT_DIR/eval_retrieval.py" \
        --ckpt "$CKPT" \
        --texts_csv "$DATA_DIR/clinical_texts.csv" \
        --split test --topk 1 5 10 \
        >> "$LOG" 2>&1 || fail "eval_retrieval EXP03 실패"

    log "EXP03 linear probe 평가 (GPU $GPU, raw NIfTI)..."
    CUDA_VISIBLE_DEVICES=$GPU UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_PYTHON" "$SCRIPT_DIR/eval_linear_probe.py" \
        --ckpt "$CKPT" \
        --texts_csv "$DATA_DIR/clinical_texts.csv" \
        --mode both \
        >> "$LOG" 2>&1 || fail "eval_linear_probe EXP03 실패"

    touch "$EVAL_DONE"
    log "EXP03 평가 완료"
    set_phase "train_exp04"
}

phase_train_exp04() {
    log "=== Phase: train_exp04 (Fixed τ=0.02) ==="
    CKPT="$RESULT_DIR/exp04_fixed_temp/checkpoints/brainclip_best.pt"

    if [[ -f "$CKPT" ]]; then
        log "EXP04 best checkpoint 이미 존재, 스킵"
        set_phase "eval_exp04"
        return
    fi

    cd "$ROOT"
    CONFIG="$ROOT/experiments/BrainCLIP/configs/exp04_fixed_temp.toml"
    CONFIG_RUN="$RESULT_DIR/exp04_fixed_temp_run.toml"
    python3 -c "
import re
txt = open('$CONFIG').read()
txt = re.sub(r'^# (embedding_cache_dir\s*=.*)', r'\1', txt, flags=re.MULTILINE)
open('$CONFIG_RUN','w').write(txt)
"
    log "학습 시작: GPU=1,2 (DDP/2) config=$CONFIG_RUN"
    CUDA_VISIBLE_DEVICES=1,2 \
    UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_TORCHRUN" \
        --nproc_per_node=2 --master_port=29601 --standalone \
        "$SCRIPT_DIR/train_brainclip.py" \
        --config "$CONFIG_RUN" \
        >> "$LOG" 2>&1 || fail "train EXP04 실패"
    log "EXP04 학습 완료"
    set_phase "eval_exp04"
}

phase_eval_exp04() {
    log "=== Phase: eval_exp04 ==="
    CKPT="$RESULT_DIR/exp04_fixed_temp/checkpoints/brainclip_best.pt"
    EVAL_DONE="$RESULT_DIR/exp04_fixed_temp/logs/.eval_complete"

    if [[ -f "$EVAL_DONE" ]]; then
        log "EXP04 eval 이미 완료, 스킵"
        set_phase "report"
        return
    fi

    [[ -f "$CKPT" ]] || fail "EXP04 checkpoint 없음: $CKPT"

    GPU=$(pick_best_gpu)
    cd "$ROOT"
    log "EXP04 retrieval 평가 (GPU $GPU)..."
    CUDA_VISIBLE_DEVICES=$GPU UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_PYTHON" "$SCRIPT_DIR/eval_retrieval.py" \
        --ckpt "$CKPT" \
        --texts_csv "$DATA_DIR/clinical_texts.csv" \
        --embedding_cache_dir "$DATA_DIR/mri_embeddings" \
        --split test --topk 1 5 10 \
        >> "$LOG" 2>&1 || fail "eval_retrieval EXP04 실패"

    log "EXP04 linear probe 평가 (GPU $GPU)..."
    CUDA_VISIBLE_DEVICES=$GPU UV_CACHE_DIR=/home/vlm/minyoung/.uv_cache \
    "$VENV_PYTHON" "$SCRIPT_DIR/eval_linear_probe.py" \
        --ckpt "$CKPT" \
        --texts_csv "$DATA_DIR/clinical_texts.csv" \
        --embedding_cache_dir "$DATA_DIR/mri_embeddings" \
        --mode both \
        >> "$LOG" 2>&1 || fail "eval_linear_probe EXP04 실패"

    touch "$EVAL_DONE"
    log "EXP04 평가 완료"
    set_phase "report"
}

phase_report() {
    log "=== Phase: report (결과 정리 및 보고서 생성) ==="
    cd "$ROOT"
    $PYTHON "$SCRIPT_DIR/generate_report.py" \
        --result_dir "$RESULT_DIR" \
        --output "$RESULT_DIR/BrainCLIP_Report.md" \
        >> "$LOG" 2>&1 || fail "generate_report 실패"

    log "보고서 생성 완료: $RESULT_DIR/BrainCLIP_Report.md"
    set_phase "done"
}

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
cd "$ROOT"
log "====== BrainCLIP Pipeline 시작 ======"
log "ROOT=$ROOT"
log "PID=$$"

while true; do
    PHASE=$(get_phase)
    log "현재 phase: $PHASE"

    case "$PHASE" in
        setup)         phase_setup ;;
        text_build)    phase_text_build ;;
        embed_extract) phase_embed_extract ;;
        train_exp01)   phase_train_exp01 ;;
        eval_exp01)    phase_eval_exp01 ;;
        train_exp02)   phase_train_exp02 ;;
        eval_exp02)    phase_eval_exp02 ;;
        smoke_exp03)   phase_smoke_exp03 ;;
        train_exp03)   phase_train_exp03 ;;
        eval_exp03)    phase_eval_exp03 ;;
        train_exp04)   phase_train_exp04 ;;
        eval_exp04)    phase_eval_exp04 ;;
        report)        phase_report ;;
        done)
            log "====== 파이프라인 완료 ======"
            log "보고서: $RESULT_DIR/BrainCLIP_Report.md"
            exit 0
            ;;
        *)
            fail "알 수 없는 phase: $PHASE"
            ;;
    esac
done
