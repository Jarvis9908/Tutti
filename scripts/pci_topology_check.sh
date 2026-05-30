#!/bin/bash

# ==============================================================================
# Script Name:  pci_topology_check.sh
# Description:  Finds all NVIDIA GPUs and NVMe devices, calculates the topological
#               distance between each pair, prints a matrix to the console,
#               and saves the results to a JSON file.
#
# Version:      1.8
# Changes:      - Reworked console output as a true 2D matrix:
#                 X-axis = GPUs, Y-axis = NVMe devices, cell = distance.
#               - Color-coded cells by distance (green/yellow/red).
#               - Devices are sorted (GPU by index, NVMe by BDF) for stable layout.
#
# Distance Metric:
#   - 0: Same Primary Bus (likely under the same PCIe Switch/Root Complex).
#   - 1: Same NUMA node, but on different primary buses.
#   - 2: Different NUMA nodes (requires cross-socket communication).
#
# Dependencies: pciutils (for lspci), nvidia-smi
# Usage:        sudo ./pci_topology_check.sh
# ==============================================================================

# --- Configuration ---
JSON_OUTPUT_FILE="/mnt/sys_GPU_NVMe_topology.json"

# --- Color Definitions ---
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Step 1: Dependency and Permission Checks ---
# Ensure the script is run as root for full accuracy and file write permissions
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root. Please use sudo.${NC}"
   exit 1
fi

# Check for dependencies
if ! command -v lspci &> /dev/null || ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: Required command 'lspci' or 'nvidia-smi' not found.${NC}"
    echo "Please install 'pciutils' and the NVIDIA driver."
    exit 1
fi

# Check if output file is writable
if ! touch "$JSON_OUTPUT_FILE" &>/dev/null; then
    echo -e "${RED}Error: Cannot write to output file ${JSON_OUTPUT_FILE}. Check permissions on /mnt/.${NC}"
    exit 1
fi

# --- Step 2: Find Devices and Collect Their Info ---

# Use associative arrays to store device data: BDF -> "NUMA_NODE|UPSTREAM_BRIDGE_BDF|DEVICE_ID"
declare -A gpus
declare -A nvmes

echo -e "${YELLOW}--- Finding NVIDIA GPU Devices ---${NC}"
while read -r index bdf; do
    if [ -z "$bdf" ]; then continue; fi
    full_bdf=$bdf
    numa_node_path="/sys/bus/pci/devices/${full_bdf}/numa_node"
    numa_node=$(cat "$numa_node_path" 2>/dev/null || echo "N/A")
    upstream_bdf=$(basename "$(readlink -f "/sys/bus/pci/devices/${full_bdf}/..")")
    gpus["$full_bdf"]="${numa_node}|${upstream_bdf}|${index}"
    echo -e "Found GPU ${index}: ${CYAN}${full_bdf}${NC} (NUMA: ${numa_node}, Upstream: ${upstream_bdf})"
done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F',' '{bdf=$2; gsub(/^ /, "", bdf); gsub(/^00000000:/, "0000:", bdf); print $1, tolower(bdf)}')

echo ""
echo -e "${YELLOW}--- Finding NVMe Devices ---${NC}"
while read -r line; do
    if [ -z "$line" ]; then continue; fi
    full_bdf=$(echo "$line" | awk '{print $1}')
    numa_node_path="/sys/bus/pci/devices/${full_bdf}/numa_node"
    numa_node=$(cat "$numa_node_path" 2>/dev/null || echo "N/A")
    upstream_bdf=$(basename "$(readlink -f "/sys/bus/pci/devices/${full_bdf}/..")")
    device_path=$(find "/sys/bus/pci/devices/${full_bdf}/nvme" -name "nvme*n*" -print -quit 2>/dev/null)
    if [ -n "$device_path" ]; then
        device_name="/dev/$(basename "$device_path")"
    else
        device_name="N/A"
    fi
    nvmes["$full_bdf"]="${numa_node}|${upstream_bdf}|${device_name}"
    echo -e "Found NVMe: ${CYAN}${full_bdf}${NC} -> ${device_name} (NUMA: ${numa_node}, Upstream: ${upstream_bdf})"
done < <(lspci -D | grep -i 'Non-Volatile memory controller' | tr '[:upper:]' '[:lower:]')


# --- Step 3: Calculate Distances and Prepare Outputs ---
echo ""

if [ ${#gpus[@]} -eq 0 ] || [ ${#nvmes[@]} -eq 0 ]; then
    echo -e "${RED}Could not find both GPU and NVMe devices to compare.${NC}"
    echo "[]" > "$JSON_OUTPUT_FILE" # Write an empty JSON array
    exit 0
fi

# --- Step 3a: Sort devices for stable matrix layout ---
# GPUs: sorted by gpu_index (numeric)
mapfile -t sorted_gpu_bdfs < <(
    for bdf in "${!gpus[@]}"; do
        idx=$(echo "${gpus[$bdf]}" | cut -d'|' -f3)
        printf "%s\t%s\n" "$idx" "$bdf"
    done | sort -k1,1n | cut -f2
)
# NVMes: sorted by BDF string
mapfile -t sorted_nvme_bdfs < <(printf "%s\n" "${!nvmes[@]}" | sort)

# --- Step 3b: Compute the full distance matrix into an associative array ---
declare -A dist_matrix   # key: "${gpu_bdf}|${nvme_bdf}" -> distance

for gpu_bdf in "${sorted_gpu_bdfs[@]}"; do
    gpu_info=${gpus[$gpu_bdf]}
    gpu_numa=$(echo "$gpu_info" | cut -d'|' -f1)
    gpu_upstream_bdf=$(echo "$gpu_info" | cut -d'|' -f2)
    gpu_primary_bus=$(echo "$gpu_upstream_bdf" | cut -d':' -f1-2)

    for nvme_bdf in "${sorted_nvme_bdfs[@]}"; do
        nvme_info=${nvmes[$nvme_bdf]}
        nvme_numa=$(echo "$nvme_info" | cut -d'|' -f1)
        nvme_upstream_bdf=$(echo "$nvme_info" | cut -d'|' -f2)
        nvme_primary_bus=$(echo "$nvme_upstream_bdf" | cut -d':' -f1-2)

        distance=-1
        if [[ "$gpu_primary_bus" == "$nvme_primary_bus" ]]; then
            distance=0
        elif [[ "$gpu_numa" == "$nvme_numa" && "$gpu_numa" != "-1" && "$gpu_numa" != "N/A" ]]; then
            distance=1
        else
            distance=2
        fi
        dist_matrix["${gpu_bdf}|${nvme_bdf}"]=$distance
    done
done

# --- Step 3c: Pretty-print the matrix (X-axis = GPUs, Y-axis = NVMes) ---
echo -e "${YELLOW}--- Topological Distance Matrix (rows: NVMe, cols: GPU) ---${NC}"
echo "  Legend: 0 = same PCIe switch/root complex   1 = same NUMA node   2 = cross-NUMA"
echo ""

# Column width for the NVMe row-header (left side)
row_header_width=32
# Column width for each GPU column
col_width=10

# Header row 1: GPU index
printf "%-${row_header_width}s" "NVMe \\ GPU"
for gpu_bdf in "${sorted_gpu_bdfs[@]}"; do
    gpu_index=$(echo "${gpus[$gpu_bdf]}" | cut -d'|' -f3)
    printf "${CYAN}%-${col_width}s${NC}" "GPU${gpu_index}"
done
echo ""

# Header row 2: GPU BDF (short form: bus:dev.func)
printf "%-${row_header_width}s" ""
for gpu_bdf in "${sorted_gpu_bdfs[@]}"; do
    short_bdf=${gpu_bdf#0000:}
    printf "%-${col_width}s" "$short_bdf"
done
echo ""

# Separator line
total_width=$(( row_header_width + col_width * ${#sorted_gpu_bdfs[@]} ))
printf '%*s\n' "$total_width" '' | tr ' ' '-'

# Data rows: one per NVMe device
for nvme_bdf in "${sorted_nvme_bdfs[@]}"; do
    nvme_info=${nvmes[$nvme_bdf]}
    nvme_name=$(echo "$nvme_info" | cut -d'|' -f3)
    short_nvme_bdf=${nvme_bdf#0000:}
    # Row label: "<device_name> (<short_bdf>)"
    row_label="${nvme_name} (${short_nvme_bdf})"
    printf "%-${row_header_width}s" "$row_label"

    for gpu_bdf in "${sorted_gpu_bdfs[@]}"; do
        d=${dist_matrix["${gpu_bdf}|${nvme_bdf}"]}
        # Color the cell based on distance
        case "$d" in
            0) color="$GREEN"  ;;
            1) color="$YELLOW" ;;
            2) color="$RED"    ;;
            *) color="$NC"     ;;
        esac
        printf "${color}%-${col_width}s${NC}" "$d"
    done
    echo ""
done
echo ""

# --- Step 3d: Write JSON file ---
echo "[" > "$JSON_OUTPUT_FILE"
is_first_json_entry=true

for gpu_bdf in "${sorted_gpu_bdfs[@]}"; do
    gpu_index=$(echo "${gpus[$gpu_bdf]}" | cut -d'|' -f3)
    for nvme_bdf in "${sorted_nvme_bdfs[@]}"; do
        nvme_name=$(echo "${nvmes[$nvme_bdf]}" | cut -d'|' -f3)
        distance=${dist_matrix["${gpu_bdf}|${nvme_bdf}"]}

        if [ "$is_first_json_entry" = true ]; then
            is_first_json_entry=false
        else
            echo "," >> "$JSON_OUTPUT_FILE"
        fi

        printf '  {\n    "gpu_index": %s,\n    "gpu_bdf": "%s",\n    "nvme_device": "%s",\n    "nvme_bdf": "%s",\n    "distance": %s\n  }' \
            "$gpu_index" "$gpu_bdf" "$nvme_name" "$nvme_bdf" "$distance" >> "$JSON_OUTPUT_FILE"
    done
done

# Finalize JSON file
echo -e "\n]" >> "$JSON_OUTPUT_FILE"

echo -e "\n${GREEN}Topology analysis complete.${NC}"
echo -e "Results have been saved to: ${CYAN}${JSON_OUTPUT_FILE}${NC}"