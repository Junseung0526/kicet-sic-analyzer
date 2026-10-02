#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

LOG_FILE="$DIR/runs/overnight_train.log"
mkdir -p "$DIR/runs"

echo "==========================================================" > "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 60-Epoch Pure-TIF Golden Training Pipeline Started" >> "$LOG_FILE"
echo "Target completion: ~06:45 - 07:00 AM" >> "$LOG_FILE"
echo "Starting 60-Epoch Fine-Tuning (exp_tsd_master_v1) on base weights..." >> "$LOG_FILE"

# Run 60 epochs training with baseline weights (pure TIF dataset)
./venv/bin/python main.py --mode train --name exp_tsd_master_v1 --epochs 60 >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==========================================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL 60 EPOCHS TRAINING & ONNX EXPORT COMPLETED SUCCESSFULLY!" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Final Deployment Model: models/exp_tsd_master_v1.onnx" >> "$LOG_FILE"
echo "==========================================================" >> "$LOG_FILE"
