#!/bin/sh
# This script simply wraps the server launch with a check to ensure
# that the virtual-environment Python is available before running the server. 
# This is handy for the sake of convenience during development iterations


SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_PYTHON="$SCRIPT_DIR/whisper_env/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
	echo "Virtual-environment Python not found: $VENV_PYTHON" >&2
	exit 1
fi

exec "$VENV_PYTHON" "$SCRIPT_DIR/run_server.py" --port 9090 --backend faster_whisper --max_clients 4 --max_connection_time 600 --enable_rest --cors-origins="http://localhost:8080,http://127.0.0.1:8080"
