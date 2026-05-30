#!/bin/bash
#
# reset_snvme.sh -- attempt a clean snvme.ko / snvme_core.ko reload.
#
# Behaviour change vs. earlier revision: this script is now strictly
# fail-fast.  Any rmmod failure aborts the script *before* attempting
# insmod, because re-inserting a half-unloaded module is the exact
# failure mode that produced the "sysfs cannot create duplicate
# filename '/devices/virtual/libsnvm helper'" wedge that requires a
# reboot to recover from.
#
# When rmmod fails this script will:
#   1. Print which processes (if any) are holding /dev/snvm_control
#      or /dev/ssnvme* file descriptors.
#   2. Print the kernel module reference count so the operator can
#      tell whether the leak is in userspace (kill the holder) or
#      kernel-space (reboot is the only safe option).
#   3. Exit non-zero without touching insmod.
#
# Pass --force-cleanup to additionally SIGKILL anything holding
# /dev/snvm* fds before rmmod.  Use sparingly; if a NVMeService
# daemon is running this WILL kill it.
#
# Pass --no-insmod to only do unbind+rmmod (rebuild flow: edit code,
# rmmod, build, insmod manually).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"

FORCE_CLEANUP=0
DO_INSMOD=1
for arg in "$@"; do
    case "$arg" in
        --force-cleanup) FORCE_CLEANUP=1 ;;
        --no-insmod)     DO_INSMOD=0 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--force-cleanup] [--no-insmod]

  --force-cleanup   SIGKILL any process holding /dev/snvm* before rmmod.
                    Will kill running NVMeService daemons; use with care.
  --no-insmod       Stop after rmmod; do not reload the module.
EOF
            exit 0
            ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "=== snvme reset (force_cleanup=$FORCE_CLEANUP, do_insmod=$DO_INSMOD) ==="

# ----------------------------------------------------------------
# Step 1: Unbind any controllers currently owned by snvme so the
# in-tree nvme driver can take them back over.  unbind.sh is
# idempotent and safe to run when no controllers are bound.
# ----------------------------------------------------------------
echo "[1/4] Unbinding all snvme-owned controllers..."
bash "$SCRIPT_DIR/unbind.sh"

# ----------------------------------------------------------------
# Step 2: If anything is still holding /dev/snvm* fds, surface it.
# ----------------------------------------------------------------
echo "[2/4] Checking for fd holders..."
HOLDERS=$(sudo lsof /dev/snvm_control /dev/ssnvme* 2>/dev/null || true)
if [ -n "$HOLDERS" ]; then
    echo "  Currently held by:"
    echo "$HOLDERS" | sed 's/^/    /'
    if [ "$FORCE_CLEANUP" = "1" ]; then
        echo "  --force-cleanup set, killing holders..."
        # Skip the lsof header by stripping the first line; column 2 is PID.
        echo "$HOLDERS" | tail -n +2 | awk '{print $2}' | sort -u \
            | xargs -r sudo kill -9
        sleep 1
    else
        echo "  Refusing to rmmod with live fd holders; rerun with --force-cleanup"
        echo "  to SIGKILL them, or stop the holders manually first." >&2
        exit 1
    fi
else
    echo "  No fd holders."
fi

# ----------------------------------------------------------------
# Step 3: rmmod, fail-fast on error.
# ----------------------------------------------------------------
echo "[3/4] Removing kernel module..."

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory $BUILD_DIR does not exist" >&2
    exit 1
fi

# Skip rmmod entirely if the module isn't loaded -- saves
# time on first boot and avoids a noisy "module not found".
if ! lsmod | grep -qE '^snvme(\s|$)' && ! lsmod | grep -qE '^snvme_core(\s|$)'; then
    echo "  Module not loaded, skipping rmmod."
else
    cd "$BUILD_DIR"
    if ! make rmmod; then
        echo
        echo "rmmod failed.  Refcount snapshot:"
        lsmod | grep -E '^(snvme|snvme_core)\s' | sed 's/^/    /' || true
        echo
        echo "Common causes:"
        echo "  * A user-space process still has /dev/ssnvme* open"
        echo "    (lsof check above missed it; race).  Re-run with"
        echo "    --force-cleanup to kill all holders, then retry."
        echo "  * A controller is still bound to snvme.  Check"
        echo "    'lspci -k -d ::0108' for any 'Kernel driver in use:"
        echo "    snvme' lines."
        echo "  * Kernel-side leak inside the module (rare).  In this"
        echo "    case rmmod will keep failing until reboot." >&2
        exit 1
    fi
fi

# ----------------------------------------------------------------
# Step 4: insmod (unless --no-insmod).
# ----------------------------------------------------------------
if [ "$DO_INSMOD" = "0" ]; then
    echo "[4/4] --no-insmod set, stopping here.  Module is unloaded."
    echo "snvme reset (rmmod-only) completed!"
    exit 0
fi

echo "[4/4] Re-inserting kernel module..."
cd "$BUILD_DIR"
if ! make insmod; then
    echo
    echo "insmod failed.  See dmesg for the kernel-side reason." >&2
    echo "If dmesg says 'cannot create duplicate filename" >&2
    echo "/devices/virtual/libsnvm helper' or 'failed to create" >&2
    echo "/dev/snvm_control: -17', the module's last unload was" >&2
    echo "incomplete and a reboot is required.  Do NOT keep" >&2
    echo "retrying insmod -- repeated attempts have been observed" >&2
    echo "to crash the host." >&2
    exit 1
fi

echo "Kernel module reloaded successfully."
echo "snvme reset completed!"
