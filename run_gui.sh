#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_PYTHON="$SCRIPT_DIR/whisper_env/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
	echo "Virtual-environment Python not found: $VENV_PYTHON" >&2
	exit 1
fi

exec "$VENV_PYTHON" -m gui "$@"