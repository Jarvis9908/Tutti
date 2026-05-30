#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="/mnt/nvme_layer"

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root." >&2
    exit 1
fi

if [[ ! -d "$TARGET_ROOT" ]]; then
    echo "Target root directory $TARGET_ROOT does not exist."
    exit 0
fi

mapfile -t subdirs < <(find "$TARGET_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#subdirs[@]} -eq 0 ]]; then
    echo "No subdirectories found under $TARGET_ROOT."
else
    for dir in "${subdirs[@]}"; do
        if mountpoint -q "$dir"; then
            echo "Unmounting $dir ..."
            umount "$dir"
        else
            echo "Skipping $dir (not a mount point)."
        fi
    done
fi

echo "Calling reset_snvme.sh to reset the driver..."
bash "$SCRIPT_DIR/reset_snvme.sh"

echo "Completed NVMe layer unmount and driver reset."
