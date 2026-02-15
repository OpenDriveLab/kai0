#!/bin/bash
# Source this before running the DAGGER script so bimanual .so can be loaded.
# Usage: source setup.sh   (from train_deploy_alignment/dagger/arx)
export LD_LIBRARY_PATH=$(pwd)/bimanual/api/arx_x5_src:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
