#!/bin/bash

# ==============================================================================
# Script Name:  create_nvme_raid0.sh
# Description:  Creates a RAID 0 array from multiple unpartitioned,
#               unmounted NVMe devices. It automatically finds an available
#               /dev/mdX device name.
# Usage:        sudo ./create_nvme_raid0.sh /dev/nvme0n1 /dev/nvme1n1 ...
# Example:      sudo ./create_nvme_raid0.sh /dev/nvme0n1 /dev/nvme1n1
# ==============================================================================

# --- Configuration ---
CHUNK_SIZE="64K" # Stripe size

# --- Color Definitions ---
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Step 1: Root and Tool Checks ---
# Ensure the script is run as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root. Please use sudo.${NC}"
   exit 1
fi

# Ensure mdadm is installed
if ! command -v mdadm &> /dev/null; then
    echo -e "${RED}Error: 'mdadm' is not installed. Please install it first (e.g., 'sudo apt install mdadm').${NC}"
    exit 1
fi

# --- Step 2: Input Validation ---
# Check if at least two devices are provided
if [ "$#" -lt 2 ]; then
    echo -e "${RED}Error: You must provide at least two NVMe devices.${NC}"
    echo "Usage: $0 /dev/nvme0n1 /dev/nvme1n1 ..."
    exit 1
fi

# --- Step 3: Device Safety Checks ---
CLEAN_DEVICES=()
echo "--- Starting safety checks for all provided devices ---"

for device in "$@"; do
    echo -n "Checking device ${device}... "

    # Check 1: Does the block device exist?
    if [ ! -b "$device" ]; then
        echo -e "${RED}FAIL! Device does not exist or is not a block device.${NC}"
        exit 1
    fi

    # Check 2: Is the device already mounted?
    if findmnt -n --source "$device" --first-only > /dev/null; then
        echo -e "${RED}FAIL! Device is currently mounted.${NC}"
        exit 1
    fi

    # Check 3: Does the device contain a filesystem?
    if blkid -p -o value -s TYPE "$device" &> /dev/null; then
        echo -e "${RED}FAIL! Device appears to contain a filesystem.${NC}"
        exit 1
    fi
    
    # Check 4: Is the device part of another RAID array?
    if mdadm --examine "$device" | grep -q "mdadm"; then
        echo -e "${RED}FAIL! Device appears to be part of an existing RAID array.${NC}"
        exit 1
    fi

    echo -e "${GREEN}OK!${NC}"
    CLEAN_DEVICES+=("$device")
done

echo "--- All safety checks passed ---"

# --- Step 4: Find an Available RAID Device Name ---
RAID_DEVICE=""
i=0
while true; do
    # Check if /dev/mdX exists
    if [ -e "/dev/md${i}" ]; then
        i=$((i + 1))
        # Add a safeguard to prevent an infinite loop in an extreme edge case
        if [ $i -gt 255 ]; then
            echo -e "${RED}Error: Could not find an available /dev/mdX device name between 0 and 255.${NC}"
            exit 1
        fi
    else
        RAID_DEVICE="/dev/md${i}"
        break
    fi
done

echo "Found available RAID device name: ${GREEN}${RAID_DEVICE}${NC}"


# --- Step 5: User Confirmation ---
echo
echo -e "${YELLOW}The following devices will be permanently erased and combined into a RAID 0 array:${NC}"
for device in "${CLEAN_DEVICES[@]}"; do
    echo "  - $device"
done
echo
echo -e "A new RAID 0 array will be created at: ${YELLOW}${RAID_DEVICE}${NC}"
echo -e "${YELLOW}This action is IRREVERSIBLE.${NC}"
read -p "Are you absolutely sure you want to continue? [y/N] " -n 1 -r REPLY
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled by user."
    exit 1
fi

# --- Step 6: Create the RAID 0 Array ---
echo "Creating RAID 0 array at ${RAID_DEVICE}..."
NUM_DEVICES=${#CLEAN_DEVICES[@]}
DEVICE_LIST="${CLEAN_DEVICES[*]}"

# --run: starts the array after creating it.
# --verbose: provides detailed output.
mdadm --create "${RAID_DEVICE}" \
      --level=0 \
      --raid-devices=${NUM_DEVICES} \
      ${DEVICE_LIST} \
      --chunk=${CHUNK_SIZE} \
      --run \
      --verbose

# Check if mdadm command was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create the RAID array. Please check the output above.${NC}"
    exit 1
fi

echo -e "${GREEN}Successfully created RAID 0 array at ${RAID_DEVICE}!${NC}"
echo
echo "--- Next Steps ---"
echo "1. Create a filesystem: sudo mkfs.ext4 ${RAID_DEVICE}"
echo "2. Save the configuration to make it persistent: sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf"
echo "3. Create a mount point: sudo mkdir /mnt/raid0_disk"
echo "4. Mount the array: sudo mount ${RAID_DEVICE} /mnt/raid0_disk"

exit 0