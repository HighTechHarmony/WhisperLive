# WhisperLive Meeting GUI

On Linux with Python 3.10+, install the project using PyTorch's CUDA 12.8
index so the PyTorch and faster-whisper CUDA libraries stay compatible:

```sh
./whisper_env/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e .
```

The CUDA-enabled wheels also support CPU operation. The server service reads
`server.device` from `config.toml` (`cpu`, `cuda`, or `auto`); an explicit
`--device` command-line argument overrides it.

Install the GUI dependency into the existing Python 3.12 environment:

```sh
./whisper_env/bin/python -m pip install -r requirements/gui.txt
```

Validate configuration without opening the window:

```sh
./whisper_env/bin/python -m gui --validate-config
```

Start the application with the root `config.toml`:

```sh
./run_gui.sh
```

The GUI captures at the input device selected in `[audio]`, resamples to 16 kHz
for WhisperLive, and links each configured `audio.sources` port to the named
capture node. Use **Re-link Audio** after a PipeWire device reconnects.

The generated `whisper-server.service` is a system-unit template. Review its
absolute paths, then install it at `/etc/systemd/system/` and run:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now whisper-server.service
```

The GUI retains and displays the complete transcript for the active meeting;
`server.display_segments` does not limit that history. Automatic summaries and
**Summarize Now** use the latest `llm.summary_interval_seconds` of transcript,
with overlapping windows allowed. **Wrap & Export** generates its final summary
from the complete meeting transcript and exports that complete transcript.

Choose a group template from the **Template** selector to use it as the
summarizer's complete system prompt. Templates are Markdown files in
`whisper_live/summarizer_templates/` named
`SUMMARIZER_TEMPLATE-<group name>.md`. The packaged default template contains
general meeting-summary instructions and comments suggesting where to add
relevant names, acronyms, and house terminology. Copy and adapt it for group
specific prompts. **Automatic** selects the first filename containing `default`
alphabetically (case-insensitive), or uses no template if none is marked.
**None** leaves the system prompt empty. Set `[llm] summary_template` in
`config.toml` to `auto`, `none`, or an exact filename for the initial selection.
Template files are scanned at startup, so restart the GUI after adding or
removing them; changing the selection applies to the next summary request.