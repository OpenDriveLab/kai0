#!/bin/bash
# Activate a single CAN interface: optional name, bitrate, and USB bus-info.
# Usage: ./can_activate.sh <can_name> <bitrate> [usb_bus_info]
# Example: ./can_activate.sh can_left_slave 1000000 "1-13:1.0"

DEFAULT_CAN_NAME="${1:-can0}"
DEFAULT_BITRATE="${2:-1000000}"
USB_ADDRESS="${3}"

echo "-------------------START-----------------------"

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

CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can")

if [ "$CURRENT_CAN_COUNT" -ne "1" ]; then
    if [ -z "$USB_ADDRESS" ]; then
        for iface in $(ip -br link show type can | awk '{print $1}'); do
            BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
            if [ -z "$BUS_INFO" ]; then
                echo "Error: Could not get bus-info for interface $iface."
                continue
            fi
            echo "Interface $iface is on USB port $BUS_INFO"
        done
        echo -e "\033[31mError: Detected $CURRENT_CAN_COUNT CAN interface(s), expected 1. Specify USB address, e.g.:\033[0m"
        echo "  bash can_activate.sh can0 1000000 1-2:1.0"
        echo "-------------------ERROR-----------------------"
        exit 1
    fi
fi

if [ -n "$USB_ADDRESS" ]; then
    echo "Using USB address: $USB_ADDRESS"
    INTERFACE_NAME=""
    for iface in $(ip -br link show type can | awk '{print $1}'); do
        BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
        if [ "$BUS_INFO" == "$USB_ADDRESS" ]; then
            INTERFACE_NAME="$iface"
            break
        fi
    done
    if [ -z "$INTERFACE_NAME" ]; then
        echo "Error: No CAN interface found for USB address $USB_ADDRESS."
        exit 1
    fi
    echo "Found interface $INTERFACE_NAME for USB $USB_ADDRESS"
else
    INTERFACE_NAME=$(ip -br link show type can | awk '{print $1}')
    if [ -z "$INTERFACE_NAME" ]; then
        echo "Error: No CAN interface detected."
        exit 1
    fi
    BUS_INFO=$(sudo ethtool -i "$INTERFACE_NAME" | grep "bus-info" | awk '{print $2}')
    echo "Single CAN: interface $INTERFACE_NAME on USB $BUS_INFO"
fi

IS_LINK_UP=$(ip link show "$INTERFACE_NAME" | grep -q "UP" && echo "yes" || echo "no")
CURRENT_BITRATE=$(ip -details link show "$INTERFACE_NAME" | grep -oP 'bitrate \K\d+')

if [ "$IS_LINK_UP" == "yes" ] && [ "$CURRENT_BITRATE" -eq "$DEFAULT_BITRATE" ]; then
    echo "Interface $INTERFACE_NAME is already up with bitrate $DEFAULT_BITRATE"
    if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
        echo "Renaming $INTERFACE_NAME to $DEFAULT_CAN_NAME"
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
        sudo ip link set "$DEFAULT_CAN_NAME" up
        echo "Renamed to $DEFAULT_CAN_NAME and reactivated."
    else
        echo "Interface name is already $DEFAULT_CAN_NAME"
    fi
else
    if [ "$IS_LINK_UP" == "yes" ]; then
        echo "Interface $INTERFACE_NAME is up but bitrate $CURRENT_BITRATE != $DEFAULT_BITRATE."
    else
        echo "Interface $INTERFACE_NAME is down or bitrate not set."
    fi
    sudo ip link set "$INTERFACE_NAME" down
    sudo ip link set "$INTERFACE_NAME" type can bitrate $DEFAULT_BITRATE
    sudo ip link set "$INTERFACE_NAME" up
    echo "Interface $INTERFACE_NAME set to bitrate $DEFAULT_BITRATE and brought up."
    if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
        echo "Renaming $INTERFACE_NAME to $DEFAULT_CAN_NAME"
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
        sudo ip link set "$DEFAULT_CAN_NAME" up
        echo "Renamed to $DEFAULT_CAN_NAME and reactivated."
    fi
fi

echo "-------------------OVER------------------------"
