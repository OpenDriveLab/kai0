#!/bin/bash
# Enable ARX master + slave controller nodes (arm enable only).
# Run from dagger/arx after CAN is configured and up (per ARX official repo).
# Then run the DAGGER collection script in another terminal (see README).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNNING=0
_find_pids() {
    ps -ef | awk -v pat="$1" '$0~pat{print $2}' | tr '\n' ' '
}

master_pids=$(_find_pids 'open_remote_master\.launch\.py')
slave_pids=$(_find_pids  'open_remote_slave\.launch\.py')

[[ -n "$master_pids" ]] && { echo "Master node already running, PID(s): $master_pids"; RUNNING=1; }
[[ -n "$slave_pids"  ]] && { echo "Slave node already running, PID(s): $slave_pids";  RUNNING=1; }

if [[ $RUNNING -eq 1 ]]; then
    echo "======================================="
    echo "Please stop existing nodes first (e.g. kill -9 <pid>)."
    echo "Nodes may take a moment to shut down."
    echo "======================================="
    exit 1
fi

# Optional: restore terminator config from backup if it looks truncated
TERMINATOR_CONFIG="${TERMINATOR_CONFIG:-$HOME/.config/terminator/config}"
TERMINATOR_CONFIG_BP="${TERMINATOR_CONFIG}_bp"
if [[ -f "$TERMINATOR_CONFIG_BP" ]] && [[ -f "$TERMINATOR_CONFIG" ]]; then
    FILE_SIZE=$(stat -c%s "$TERMINATOR_CONFIG" 2>/dev/null || echo 0)
    if [[ "$FILE_SIZE" -le 2000 ]]; then
        cp -p "$TERMINATOR_CONFIG_BP" "$TERMINATOR_CONFIG"
        echo "Restored terminator config from backup."
    fi
fi

# Start master and slave in separate terminals if available; otherwise in background
if command -v gnome-terminal &>/dev/null; then
    echo "Starting master and slave in new terminals..."
    gnome-terminal --tab -t "master" -e "bash -c 'source \"$SCRIPT_DIR/X5_ws/install/setup.bash\"; ros2 launch arx_x5_controller open_remote_master.launch.py; exec bash'"
    sleep 2
    gnome-terminal --tab -t "slave" -e "bash -c 'source \"$SCRIPT_DIR/X5_ws/install/setup.bash\"; ros2 launch arx_x5_controller open_remote_slave.launch.py; exec bash'"
else
    source "$SCRIPT_DIR/X5_ws/install/setup.bash"
    ros2 launch arx_x5_controller open_remote_master.launch.py &
    sleep 2
    ros2 launch arx_x5_controller open_remote_slave.launch.py &
fi

echo "Arms enabled. In another terminal run the DAGGER script (see README)."
