# Instructions for AI Coding Agent: Local Meeting Transcription GUI

In this phase, we will create a separate GUI for the existing implementation demonstrated by the fileset: live_poc.py, run_sjm_client.sh, and run_sjm_server.sh.

Do not modify the existing `live_poc.py` or the other files specified above. They must remain untouched as the working Proof of Concept. This phase will build a standalone PyQt6 GUI application that functionally replaces `live_poc.py` as the primary user interface.

## Server Systemd service

* Generate a systemd user unit file (whisper-server.service) that executes run_server.py using the Python 3.12 virtual environment.Generate a systemd user unit file (whisper-server.service) that executes run_server.py using the Python 3.12 virtual environment.  This will later be edited with correct paths as needed, and placed at /etc/systemd/system, and tested.

## Dual-Source PipeWire Auto-Routing

To capture both local mic audio (`capture_AUX0`) and remote meeting audio (`monitor_AUX0`) simultaneously without manual `qpwgraph` interaction:

* Once PyAudio opens the input stream, use `pw-link` via `subprocess` to query existing input ports registered under the application's node.
* Iterate through `audio.sources` defined in `config.toml` and execute `pw-link <source_port> <app_input_port>` for each listed port.
* Provide a fallback or manual "Re-link Audio" trigger in the GUI header to execute `pw-link` on demand if audio devices reconnect mid-session.

## Configuration Management

* Implement a parser using standard library tomllib to load a config.toml file on application launch.
* The config file will maintain all of the parameters that are currently specified at the top of live_poc.py (environment variables, etc.) as well as any command line args that are specifed in run_sjm_client.py
Example (values should be taken from the PoC fileset):

```
[audio]
# Array of PipeWire ports/monitors to sum into the capture stream
sources = ["capture_AUX0", "monitor_AUX0"]
sample_rate = 16000
vad_enabled = true

[server]
host = "127.0.0.1"
port = 9090
model = "small.en"
auto_start_systemd = true

[llm]
endpoint = "http://127.0.0.1:11434/api/generate"
model = "llama3:8b"
summary_interval_seconds = 60

[export]
save_directory = "~/Documents/Meetings"
filename_template = "{date}_{time}_{tag}.md"
```

##  PyQt6 Threading Architecture

Implement a Supervisor Pattern using `QThread` and `pyqtSignal` to prevent main thread blocking:

* **Main Thread:** Build the GUI, parse configuration, handle file I/O, execute `pw-link` routing calls, and manage the `systemd` subprocess.
* **STT Worker (`QThread`):**
* Run the `WhisperLive` WebSocket client.
* Emit a `pyqtSignal` containing text chunks received from the server.


* **LLM Worker (`QThread`):**
* Run an asynchronous timer based on `summary_interval_seconds`.
* Read the current STT buffer state and execute HTTP requests to the Ollama endpoint.
* Emit a `pyqtSignal` with the returned summary text.

##  GUI Layout Construction

Construct a dark-mode optimized PyQt6 window with the following elements:

* **Header Area:** Display a Server Status indicator (checking `systemctl --user is-active`), the active audio target, and numerical counters showing the elapsed time since the last STT and LLM updates.
* **Split Pane Body:** Provide two auto-scrolling, read-only text areas side-by-side. Left pane for Live STT; right pane for LLM Summaries.
* **Footer Area:** Include a `QLineEdit` for "Meeting Tag" input and a `QPushButton` for the "Wrap & Export" action.

## Wrap & Export Routine

Bind the "Wrap & Export" button to a function that executes the following atomic sequence:

1. Suspend UI text buffer updates from the worker threads.
2. Generate the output filename using the `filename_template` and the "Meeting Tag" input. Write the contents of both text panes to disk.
3. Terminate and immediately recreate the WebSocket connection to the WhisperLive server to force a flush of the server's context window.
4. Clear the GUI text panes, reset the timer variables, and resume UI buffer updates.

