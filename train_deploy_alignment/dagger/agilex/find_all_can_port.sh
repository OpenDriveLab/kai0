#!/bin/bash
# List all CAN interfaces and their USB bus-info (use after plugging in USB-CAN adapters).
# Prerequisite: sudo apt install ethtool can-utils

if ! dpkg -l | grep -q "ethtool"; then
    echo -e "\033[31mError: ethtool not found.\033[0m"
    echo "Install with: sudo apt update && sudo apt install ethtool"
    exit 1
fi

if ! dpkg -l | grep -q "can-utils"; then
    echo -e "\033[31mError: can-utils not found.\033[0m"
    echo "Install with: sudo apt update && sudo apt install can-utils"
    exit 1
fi

echo "ethtool and can-utils are installed."

for iface in $(ip -br link show type can | awk '{print $1}'); do
    BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
    if [ -z "$BUS_INFO" ]; then
        echo "Error: Could not get bus-info for interface $iface."
        continue
    fi
    echo "Interface $iface is on USB port $BUS_INFO"
done
