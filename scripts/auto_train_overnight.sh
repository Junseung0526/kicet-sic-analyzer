#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

LOG_FILE="$DIR/runs/overnight_train.log"
mkdir -p "$DIR/runs"

echo "==========================================================" > "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pure-TIF Golden Training Pipeline Started" >> "$LOG_FILE"
echo "Compiling Pure Lossless TIF Ground-Truth Dataset (9 Wafers)..." >> "$LOG_FILE"

# Step 1: Prepare pure TIF dataset
./venv/bin/python main.py --mode dataset >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pure TIF dataset compiled successfully!" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 20-Epoch Fine-Tuning (exp_tsd_master_v1) on base weights..." >> "$LOG_FILE"

# Step 2: Run 20 epochs fine-tuning from models/best_2d_base.pt
./venv/bin/python main.py --mode train --name exp_tsd_master_v1 --epochs 20 >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==========================================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PURE TIF TRAINING & ONNX EXPORT COMPLETED SUCCESSFULLY!" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deployment Model: models/exp_tsd_master_v1.onnx" >> "$LOG_FILE"
echo "==========================================================" >> "$LOG_FILE"
