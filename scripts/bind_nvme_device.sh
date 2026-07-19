#!/bin/bash

# Script to check if a PCI device is an NVMe device and bind it to nvme driver
# Usage: ./bind_nvme_device.sh <PCI_BDF>
# Example: ./bind_nvme_device.sh 0000:50:00.0

set -e

# Function to display usage
usage() {
    echo "Usage: $0 <PCI_BDF>"
    echo "Example: $0 0000:50:00.0"
    echo ""
    echo "This script will:"
    echo "1. Check if the specified PCI device exists"
    echo "2. Verify if it's an NVMe device (class 01:08:02)"
    echo "3. Bind it to the nvme driver if it's an NVMe device"
    exit 1
}

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "Error: This script must be run as root (use sudo)"
        exit 1
    fi
}

# Function to validate PCI BDF format
validate_pci_bdf() {
    local pci_bdf="$1"
    if [[ ! "$pci_bdf" =~ ^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$ ]]; then
        echo "Error: Invalid PCI BDF format. Expected format: XXXX:XX:XX.X (e.g., 0000:50:00.0)"
        exit 1
    fi
}

# Function to check if PCI device exists
check_device_exists() {
    local pci_bdf="$1"
    if [[ ! -d "/sys/bus/pci/devices/$pci_bdf" ]]; then
        echo "Error: PCI device $pci_bdf not found in system"
        exit 1
    fi
    echo "✓ PCI device $pci_bdf found"
}

# Function to get device class
get_device_class() {
    local pci_bdf="$1"
    local class_file="/sys/bus/pci/devices/$pci_bdf/class"
    
    if [[ ! -f "$class_file" ]]; then
        echo "Error: Cannot read device class for $pci_bdf"
        exit 1
    fi
    
    cat "$class_file"
}

# Function to check if device is NVMe
is_nvme_device() {
    local pci_bdf="$1"
    local device_class
    device_class=$(get_device_class "$pci_bdf")
    
    # NVMe devices have class 0x010802 (Mass storage controller: NVM Express)
    if [[ "$device_class" == "0x010802" ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Function to get current driver
get_current_driver() {
    local pci_bdf="$1"
    local driver_link="/sys/bus/pci/devices/$pci_bdf/driver"
    
    if [[ -L "$driver_link" ]]; then
        basename "$(readlink "$driver_link")"
    else
        echo "none"
    fi
}

# Function to unbind from current driver
unbind_current_driver() {
    local pci_bdf="$1"
    local current_driver="$2"
    
    if [[ "$current_driver" != "none" ]]; then
        echo "Unbinding $pci_bdf from current driver: $current_driver"
        echo "$pci_bdf" > "/sys/bus/pci/drivers/$current_driver/unbind"
        echo "✓ Successfully unbound from $current_driver"
    fi
}

# Function to bind to nvme driver
bind_to_nvme() {
    local pci_bdf="$1"

    # Ensure the nvme driver is actually available
    if [[ ! -d /sys/bus/pci/drivers/nvme ]]; then
        echo "✗ nvme driver is not loaded; cannot bind $pci_bdf"
        echo "  Try: modprobe nvme"
        exit 1
    fi

    echo "Binding $pci_bdf to nvme driver..."

    # Get vendor and device IDs
    local vendor_id
    local device_id
    vendor_id=$(cat "/sys/bus/pci/devices/$pci_bdf/vendor")
    device_id=$(cat "/sys/bus/pci/devices/$pci_bdf/device")

    # Remove 0x prefix
    vendor_id=${vendor_id#0x}
    device_id=${device_id#0x}

    # Register the device ID with the nvme driver (so it can probe this device).
    # This is NOT the same as binding; report it separately and do not claim success yet.
    if ! grep -q "$vendor_id $device_id" /sys/bus/pci/drivers/nvme/new_id 2>/dev/null; then
        if echo "$vendor_id $device_id" > /sys/bus/pci/drivers/nvme/new_id 2>/dev/null; then
            echo "✓ Registered device ID $vendor_id:$device_id with nvme driver"
        else
            echo "⚠ Could not register device ID with nvme driver (continuing anyway)"
        fi
    fi

    # Bind the device to the nvme driver. Capture the result instead of letting
    # set -e abort with a raw kernel error.
    if echo "$pci_bdf" > /sys/bus/pci/drivers/nvme/bind 2>/dev/null; then
        echo "✓ Successfully bound $pci_bdf to nvme driver"
    else
        echo "✗ Failed to bind $pci_bdf to nvme driver"
        echo "  (kernel rejected the bind, e.g. ENODEV: device not currently enumerable/bindable)"
        echo "  Check device state with: lspci -k -s $pci_bdf   and   lspci -vvv -s $pci_bdf"
        exit 1
    fi
}

# Function to verify binding
verify_binding() {
    local pci_bdf="$1"
    local new_driver
    new_driver=$(get_current_driver "$pci_bdf")
    
    if [[ "$new_driver" == "nvme" ]]; then
        echo "✓ Verification successful: $pci_bdf is now bound to nvme driver"
        
        # Show NVMe device information if available
        sleep 1  # Give the driver time to initialize
        local nvme_dev
        nvme_dev=$(find /sys/bus/pci/devices/"$pci_bdf"/nvme -name "nvme*" 2>/dev/null | head -1)
        if [[ -n "$nvme_dev" ]]; then
            local nvme_name
            nvme_name=$(basename "$nvme_dev")
            echo "✓ NVMe device registered as: $nvme_name"
            
            # Show device info
            if command -v nvme >/dev/null 2>&1; then
                echo "Device information:"
                nvme id-ctrl "/dev/$nvme_name" 2>/dev/null | head -5 || true
            fi
        fi
    else
        echo "⚠ Warning: Device is bound to '$new_driver' instead of 'nvme'"
        exit 1
    fi
}

# Main function
main() {
    local pci_bdf="$1"
    
    # Check if PCI BDF is provided
    if [[ -z "$pci_bdf" ]]; then
        echo "Error: PCI BDF not provided"
        usage
    fi
    
    echo "=== NVMe Device Binding Script ==="
    echo "Target device: $pci_bdf"
    echo ""
    
    # Validate input and check permissions
    check_root
    validate_pci_bdf "$pci_bdf"
    check_device_exists "$pci_bdf"
    
    # Check if device is NVMe
    if is_nvme_device "$pci_bdf"; then
        local device_class
        device_class=$(get_device_class "$pci_bdf")
        echo "✓ Device $pci_bdf is an NVMe device (class: $device_class)"
    else
        local device_class
        device_class=$(get_device_class "$pci_bdf")
        echo "✗ Device $pci_bdf is NOT an NVMe device (class: $device_class)"
        echo "Only NVMe devices (class 0x010802) are supported by this script"
        exit 1
    fi
    
    # Check current driver
    local current_driver
    current_driver=$(get_current_driver "$pci_bdf")
    echo "Current driver: $current_driver"
    
    if [[ "$current_driver" == "nvme" ]]; then
        echo "✓ Device is already bound to nvme driver"
        verify_binding "$pci_bdf"
        exit 0
    fi
    
    # Proceed with binding
    echo ""
    echo "Proceeding with driver binding..."
    
    # Unbind from current driver if necessary
    unbind_current_driver "$pci_bdf" "$current_driver"
    
    # Bind to nvme driver
    bind_to_nvme "$pci_bdf"
    
    # Verify the binding
    verify_binding "$pci_bdf"
    
    echo ""
    echo "=== Binding completed successfully ==="
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
