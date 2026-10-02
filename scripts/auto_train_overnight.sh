#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

LOG_FILE="$DIR/runs/overnight_train.log"
mkdir -p "$DIR/runs"

echo "==========================================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-Chained Overnight Pipeline Started" >> "$LOG_FILE"
echo "Waiting for current dataset compilation to complete..." >> "$LOG_FILE"

# Wait for any active 'main.py --mode dataset' processes to exit
while pgrep -f "main.py --mode dataset" > /dev/null 2>&1; do
    sleep 5
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dataset compilation finished successfully!" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 20-Epoch YOLO Fine-tuning (exp_tsd_master_v1)..." >> "$LOG_FILE"

# Run 20 epochs training with baseline weights
./venv/bin/python main.py --mode train --name exp_tsd_master_v1 --epochs 20 >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==========================================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL OVERNIGHT TRAINING & ONNX EXPORT COMPLETED!" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check weights in: models/exp_tsd_master_v1.onnx" >> "$LOG_FILE"
echo "==========================================================" >> "$LOG_FILE"
