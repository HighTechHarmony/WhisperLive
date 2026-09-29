#!/bin/sh

# This script simply wraps the client launch with a check to ensure
# that the virtual-environment Python is available before running the client. 
# This is handy for the sake of convenience during development iterations

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_PYTHON="$SCRIPT_DIR/whisper_env/bin/python"


# Some assumed command line args
# Nothing is written to disk unless asked for. --output-srt writes a timestamped
# transcript; --enable-summaries (which implies --output-srt) also writes a
# summary every --auto-summary-minutes minutes to the summaries/ folder.
#
# The capture node keeps a fixed identity in the PipeWire patchbay
# (qpwgraph/Helvum): live_poc.py pins it through PIPEWIRE_ALSA (node name) and
# PIPEWIRE_PROPS (the label patchbays display), so it is the same however the
# client is launched. The label applies on both the PipeWire and the JACK
# capture paths; a direct hw: PCM bypasses PipeWire entirely and never shows up
# in the patchbay. See --node-name / --node-description in live_poc.py.
ARGS="--n-display-segments 40 --enable-timestamps --output-srt --enable-summaries --auto-summary-minutes 10"

if [ ! -x "$VENV_PYTHON" ]; then
	echo "Virtual-environment Python not found: $VENV_PYTHON" >&2
	exit 1
fi

exec "$VENV_PYTHON" "$SCRIPT_DIR/live_poc.py" $ARGS "$@"
