#!/bin/bash
# Hailo PCIe Driver Patch — find_vma mmap_read_lock fix
#
# Bug: hailo_vdma_buffer_map() calls find_vma() without mmap_read_lock(),
#      causing kernel WARNING floods that freeze the Pi 5 and kill USB/WiFi.
# Fix: Wrap find_vma() with mmap_read_lock/unlock on all exit paths.
#
# Applies to: hailo_pci 4.21.0 (also reported on 4.23.0)
# Platform: Raspberry Pi 5, kernel 6.12.x, Debian Bookworm
#
# Usage:
#   sudo bash patches/apply_hailo_fix.sh
#
# After apt upgrade overwrites the driver:
#   sudo bash patches/apply_hailo_fix.sh

set -e

HAILO_VERSION=$(ls /usr/src/ | grep hailo_pci | head -1)
if [ -z "$HAILO_VERSION" ]; then
    echo "ERROR: No hailo_pci source found in /usr/src/"
    exit 1
fi

SRC_DIR="/usr/src/${HAILO_VERSION}"
MEMORY_C="${SRC_DIR}/linux/vdma/memory.c"
PATCH_FILE="$(dirname "$0")/hailo_pci_find_vma_fix.patch"

echo "=== Hailo PCIe Driver Patch ==="
echo "Driver: ${HAILO_VERSION}"
echo "Source: ${MEMORY_C}"
echo ""

# Check if already patched
if grep -q "mmap_read_lock" "$MEMORY_C" 2>/dev/null; then
    echo "Already patched — mmap_read_lock found in source."
    echo "To force re-patch, restore from .bak first:"
    echo "  sudo cp ${MEMORY_C}.bak ${MEMORY_C}"
    exit 0
fi

# Check source exists
if [ ! -f "$MEMORY_C" ]; then
    echo "ERROR: ${MEMORY_C} not found"
    exit 1
fi

# Backup
echo "1. Backing up original source..."
cp "$MEMORY_C" "${MEMORY_C}.bak"
echo "   Saved to ${MEMORY_C}.bak"

# Apply patch
echo "2. Applying mmap_read_lock patch..."
cd "$SRC_DIR"
patch -p1 < "$PATCH_FILE"
echo "   Patch applied"

# Extract version number for DKMS
DKMS_VERSION=$(echo "$HAILO_VERSION" | sed 's/hailo_pci-//')

# Rebuild
echo "3. Rebuilding with DKMS..."
dkms build --force "hailo_pci/${DKMS_VERSION}"
echo "   Build complete"

echo "4. Installing patched module..."
dkms install --force "hailo_pci/${DKMS_VERSION}"
echo "   Install complete"

# Reload
echo "5. Reloading module..."
if lsmod | grep -q hailo_pci; then
    # Try to unload — may fail if device is in use
    if modprobe -r hailo_pci 2>/dev/null; then
        modprobe hailo_pci
        echo "   Module reloaded"
    else
        echo "   Module in use — reboot required to activate patch"
    fi
else
    modprobe hailo_pci
    echo "   Module loaded"
fi

# Verify
echo ""
echo "=== Verification ==="
modinfo hailo_pci | head -3
echo ""

if dmesg | tail -20 | grep -q "find_vma"; then
    echo "WARNING: find_vma warnings detected in recent dmesg (may be from before patch)"
else
    echo "No find_vma warnings in recent dmesg"
fi

echo ""
echo "=== Done ==="
echo "Monitor with: sudo dmesg -w | grep hailo"
echo ""
echo "NOTE: If 'apt upgrade' updates hailo_pci, re-run this script."
