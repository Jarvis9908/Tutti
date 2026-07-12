#!/bin/bash

# ==============================================================================
# Script Name:  remove_raid.sh
# Description:  Checks for a given software RAID (md) device and stops it.
# Usage:        ./remove_raid.sh <raid_name>
# Example:      ./remove_raid.sh md127
# ==============================================================================

# Define colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Step 1: Check for input argument ---
if [ "$#" -ne 1 ]; then
    echo -e "${RED}Error: Please provide a RAID device name as an argument.${NC}"
    echo "Usage: $0 <raid_device_name>"
    echo "Example: $0 md127"
    exit 1
fi

RAID_NAME=$1
DEVICE_PATH="/dev/${RAID_NAME}"

echo "Checking for device: ${DEVICE_PATH}"

# --- Step 2: Check if the device exists ---
# Use -b to check if it's a block device
if [ -b "${DEVICE_PATH}" ]; then
    echo -e "${GREEN}Successfully found RAID device: ${DEVICE_PATH}${NC}"

    # --- Step 3: Add a confirmation prompt for safety ---
    echo -e "${YELLOW}Warning! This will stop the RAID array and may make data inaccessible.${NC}"
    read -p "Are you sure you want to stop and remove this RAID array? [y/N] " -n 1 -r REPLY
    echo # Move to a new line

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # --- Step 4: Execute the stop command ---
        echo "Stopping RAID array ${DEVICE_PATH}..."

        # mdadm requires root privileges, so use sudo
        sudo mdadm --stop "${DEVICE_PATH}"

        # Check if the command executed successfully
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully stopped RAID array ${DEVICE_PATH}. The device node has been removed.${NC}"
        else
            echo -e "${RED}Error: Failed to stop RAID array ${DEVICE_PATH}. Please check mdadm output.${NC}"
            exit 1
        fi
    else
        echo "Operation cancelled."
    fi
else
    echo -e "${RED}Error: RAID device named ${RAID_NAME} not found under /dev/.${NC}"
    exit 1
fi

exit 0