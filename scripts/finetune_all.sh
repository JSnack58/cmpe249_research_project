#!/usr/bin/env bash
# Fine-tune every main model for every seed, one after another on GPU 0.
#
# Safe to stop at any time and safe to re-run: exported models are skipped and an interrupted
# model resumes from its last finished epoch (see src/plate_cloak/models/finetune.py).
#
#   bash scripts/finetune_all.sh --start     start (or resume) as a background service
#   systemctl --user stop plate-finetune     stop; resume later with --start
#   journalctl --user -u plate-finetune -f   follow the log
#   bash scripts/finetune_all.sh             run in this terminal instead
#
# Before starting, take the unstable CPU core 6 offline (lasts until reboot):
#   echo 0 | sudo tee /sys/devices/system/cpu/cpu6/online /sys/devices/system/cpu/cpu22/online
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/anaconda3/envs/plate-cloak/bin/python}"
CPUS="0-5,7-21,23-31"  # every logical CPU except core 6 (threads 6 and 22)

if [[ "${1:-}" == "--start" ]]; then
    if [[ "$(cat /sys/devices/system/cpu/cpu6/online)" != "0" ]]; then
        echo "CPU core 6 is online; take it offline first (see the top of this script)." >&2
        exit 1
    fi
    systemctl --user reset-failed plate-finetune 2>/dev/null || true
    exec systemd-run --user --unit plate-finetune --working-directory "$REPO" \
        -p MemoryMax=20G -p MemorySwapMax=0 \
        bash "$REPO/scripts/finetune_all.sh"
fi

# Seed 0 of every model first, so a complete set exists as early as possible;
# yolov8s first because the attack stages need the surrogate.
RUNS=(
    "yolov8s 0" "rtdetr-l 0" "yolov10l 0"
    "yolov8s 1" "rtdetr-l 1" "yolov10l 1"
    "yolov8s 2" "rtdetr-l 2" "yolov10l 2"
)

cd "$REPO"
for run in "${RUNS[@]}"; do
    read -r model seed <<< "$run"
    echo "===== $model seed $seed: $(date '+%F %T')"
    taskset -c "$CPUS" "$PYTHON" -m plate_cloak finetune --model "$model" --seed "$seed" --device 0
done
echo "===== all fine-tunes done: $(date '+%F %T')"
