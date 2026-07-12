#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# kv_cache_read_only_sweep.sh -- pure-read sweep across concurrency levels
# and optional L1/L2 cache-size variants.
#
# Requires a pre-populated file set (run kv_cache_e2e_stress WITHOUT
# --read-only once first, or with --skip-delete, so the files exist on
# disk with valid data).  This script only issues --read-only runs that
# reuse the existing set.
#
# Usage:
#   sudo scripts/kv_cache_read_only_sweep.sh \
#       --service 127.0.0.1:50051 --dev-id 0 --dev-id 1 --cuda 0 \
#       [--n 2000000] [--data-gb 0] \
#       [--batch-list "1000 2000"] \
#       [--l1-list "32 512"] [--l2-mib 1024] \
#       [--log-dir sweep_logs]
#
# Output: one log per (batch, l1) combination under --log-dir, plus a
# summary.txt with the THROUGHPUT / CACHE STATE / CACHE ACTIVITY lines.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO_ROOT/build/bin/kv_cache_e2e_stress"
LOG_DIR="$REPO_ROOT/sweep_logs"

# Defaults
N_FILES=2000000
DATA_GB=0
BATCH_LIST="1000 2000"
L1_LIST="512"
L2_MIB=1024
SERVICE=""
CUDA_DEV=0
DEV_IDS=()

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --service)  SERVICE="$2"; shift 2;;
        --cuda)     CUDA_DEV="$2"; shift 2;;
        --dev-id)   DEV_IDS+=(--dev-id "$2"); shift 2;;
        --n)        N_FILES="$2"; shift 2;;
        --data-gb)  DATA_GB="$2"; shift 2;;
        --batch-list) BATCH_LIST="$2"; shift 2;;
        --l1-list)  L1_LIST="$2"; shift 2;;
        --l2-mib)   L2_MIB="$2"; shift 2;;
        --log-dir)  LOG_DIR="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

if [[ -z "$SERVICE" ]]; then
    echo "ERROR: --service is required" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binary not found: $BIN" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/read_only_summary.txt"
: > "$SUMMARY"

echo "=== kv_cache read-only sweep ===" >> "$SUMMARY"
echo "n_files=$N_FILES  data_gb=$DATA_GB  l2_mib=$L2_MIB" >> "$SUMMARY"
echo "batch_list=[$BATCH_LIST]  l1_list=[$L1_LIST]" >> "$SUMMARY"
echo "" >> "$SUMMARY"

run_idx=0
for L1_MIB in $L1_LIST; do
    for BATCH in $BATCH_LIST; do
        run_idx=$((run_idx + 1))
        TAG="ro_l1_${L1_MIB}_b${BATCH}"
        LOG="$LOG_DIR/${TAG}.log"

        echo "[$run_idx] read-only: L1=${L1_MIB}MiB L2=${L2_MIB}MiB batch=${BATCH}"
        echo "--- log: $LOG"

        "$BIN" --cuda "$CUDA_DEV" --service "$SERVICE" "${DEV_IDS[@]}" \
            --n "$N_FILES" --batch "$BATCH" --data-gb "$DATA_GB" \
            --l1-mib "$L1_MIB" --l2-mib "$L2_MIB" \
            --read-only --skip-crash-consistency --skip-delete \
            --log "$LOG" 2>&1 | tee "$LOG_DIR/${TAG}.stdout"

        echo "" >> "$SUMMARY"
        echo "[$run_idx] L1=${L1_MIB}MiB batch=${BATCH}" >> "$SUMMARY"
        grep -E 'Phase 2 (THROUGHPUT|READ|CACHE)' "$LOG" >> "$SUMMARY" 2>/dev/null || \
            echo "  (no Phase 2 summary found -- check $LOG for errors)" >> "$SUMMARY"
    done
done

echo "" >> "$SUMMARY"
echo "=== sweep complete: $run_idx runs ===" >> "$SUMMARY"
echo ""
echo "Summary written to $SUMMARY"
cat "$SUMMARY"
