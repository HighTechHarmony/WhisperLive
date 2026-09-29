# WhisperLive Meeting GUI

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