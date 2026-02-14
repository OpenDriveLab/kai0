#!/bin/bash
# Activate all four USB-CAN interfaces for dual master + dual slave (Lngxiao arms).
# Update the USB bus-info strings below using ./find_all_can_port.sh, then run:
#   ./activate_can_arms.sh
# Required names: can_left_slave, can_right_slave, can_left_mas, can_right_mas (see start_ms_piper_new.launch).

bash ./can_activate.sh can_left_slave 1000000  "1-13:1.0"
bash ./can_activate.sh can_right_slave 1000000 "1-12:1.0"
bash ./can_activate.sh can_left_mas 1000000   "1-6:1.0"
bash ./can_activate.sh can_right_mas 1000000  "1-5:1.0"
