#!/bin/bash
# kv_cache_e2e_sweep.sh -- concurrency x metadata-cache-size sweep driver
# for examples/adapters/kv_cache_e2e_stress.
#
# Runs the SAME 8M-GpuFile / full-coverage (--data-gb 0) R/W workload
# at each of --batch in {256,512,1024,2048} crossed with L1==L2 cache
# size in {512,1024} MiB, so the effect of concurrency and of L1/L2
# metadata-cache replacement on throughput can be compared directly.
#
# The file set (created once by the FIRST run) is reused by every
# subsequent run (--skip-crash-consistency --skip-delete), so only the
# very LAST combination pays the crash-consistency + bulk-delete cost
# and leaves the storage clean afterwards.
#
# Each run gets its own log file so results can be diffed/greped
# after the fact; a summary table is printed (and saved) once the
# whole sweep finishes.
#
# Usage:
#   sudo scripts/kv_cache_e2e_sweep.sh [--service <endpoint>] \
#       [--dev-id N ...] [--cuda N] [--n 8000000] \
#       [--concurrency "256 512 1024 2048"] [--cache-mib "512 1024"] \
#       [--out-dir <dir>]
#
# Requires a running tutti_daemon (see examples/tutti_daemon.cpp) at
# --service; this script does not start one.

set -eu

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$PROJECT_ROOT/build/bin/kv_cache_e2e_stress"

SERVICE="127.0.0.1:50051"
DEV_IDS=(0 1)
CUDA_DEV=0
N_FILES=8000000
CONCURRENCIES=(256 512 1024 2048)
CACHE_MIBS=(512 1024)
OUT_DIR="$PROJECT_ROOT/sweep_logs"

while [ $# -gt 0 ]; do
    case "$1" in
        --service)     SERVICE="$2"; shift 2 ;;
        --dev-id)      DEV_IDS+=("$2"); shift 2 ;;
        --cuda)        CUDA_DEV="$2"; shift 2 ;;
        --n)           N_FILES="$2"; shift 2 ;;
        --concurrency) read -r -a CONCURRENCIES <<< "$2"; shift 2 ;;
        --cache-mib)   read -r -a CACHE_MIBS <<< "$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found/executable -- build target kv_cache_e2e_stress first" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

DEV_ID_ARGS=()
for d in "${DEV_IDS[@]}"; do DEV_ID_ARGS+=(--dev-id "$d"); done

SUMMARY="$OUT_DIR/summary.txt"
: > "$SUMMARY"

n_combos=$(( ${#CACHE_MIBS[@]} * ${#CONCURRENCIES[@]} ))
combo_idx=0

echo "=== kv_cache_e2e_stress sweep: n=$N_FILES, cache_mib in {${CACHE_MIBS[*]}}, batch in {${CONCURRENCIES[*]}} ($n_combos runs) ==="

for cache_mib in "${CACHE_MIBS[@]}"; do
    for batch in "${CONCURRENCIES[@]}"; do
        combo_idx=$((combo_idx + 1))
        is_last=0
        if [ "$combo_idx" -eq "$n_combos" ]; then is_last=1; fi

        log="$OUT_DIR/run_l${cache_mib}_b${batch}.log"
        skip_flags=(--skip-crash-consistency --skip-delete)
        if [ "$is_last" -eq 1 ]; then skip_flags=(); fi

        echo "--- [$combo_idx/$n_combos] cache_mib=$cache_mib batch=$batch (log: $log) ---"
        set +e
        "$BIN" --cuda "$CUDA_DEV" --service "$SERVICE" "${DEV_ID_ARGS[@]}" \
            --n "$N_FILES" --batch "$batch" \
            --l1-mib "$cache_mib" --l2-mib "$cache_mib" \
            --data-gb 0 --log "$log" \
            "${skip_flags[@]}"
        rc=$?
        set -e
        if [ "$rc" -ne 0 ]; then
            echo "FAILED: cache_mib=$cache_mib batch=$batch (exit $rc), see $log" | tee -a "$SUMMARY"
            exit "$rc"
        fi

        # Pull the two throughput lines + cache-activity line for the summary.
        {
            echo "== cache_mib=$cache_mib batch=$batch =="
            grep -E "^\[ OK \].*Phase 2:" "$log" || true
            grep -E "^\[ OK \].*Phase 2 cache activity" "$log" || true
            echo
        } >> "$SUMMARY"
    done
done

echo "=== sweep complete: $n_combos runs === "
echo "Summary: $SUMMARY"
cat "$SUMMARY"
