#!/usr/bin/env bash
# run.sh – activate the virtual‑env and start run_servers.py
set -euo pipefail

# Directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------- 1. activate the virtual environment --------
# If your venv is in a different place, edit the line below.
source ~/venvs/yolo-mps/bin/activate

# -------- 2. launch the servers --------
sudo python "${SCRIPT_DIR}/unity_backend/run_servers.py" "$@"
