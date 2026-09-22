#!/bin/sh

# This script simply wraps the client launch with a check to ensure
# that the virtual-environment Python is available before running the client. 
# This is handy for the sake of convenience during development iterations

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_PYTHON="$SCRIPT_DIR/whisper_env/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
	echo "Virtual-environment Python not found: $VENV_PYTHON" >&2
	exit 1
fi

exec "$VENV_PYTHON" "$SCRIPT_DIR/live_poc.py"
