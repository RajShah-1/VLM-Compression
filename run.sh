#!/bin/bash

# Check if the script argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <script_name.py>"
  exit 1
fi

# Extract the script name without the extension for log and PID naming
SCRIPT_NAME=$(basename "$1" .py)

# Log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Timestamp for log files
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")

# Log file path
LOG_FILE="$LOG_DIR/out.$SCRIPT_NAME.$TIMESTAMP.log"

# PID file path
PID_FILE="$LOG_DIR/out.pid.$SCRIPT_NAME.pid"

# Activate Conda environment
# conda activate bda-project-py38

# Run the Python script, log to file and stdout, and write PID to a file
{
  echo "Running $1 with PID $$"
  python "$1" 2>&1
} | tee "$LOG_FILE"

# Save the process ID to the PID file
echo $! > "$PID_FILE"

# Inform the user
echo "Logs are written to $LOG_FILE"
echo "PID is saved in $PID_FILE"
