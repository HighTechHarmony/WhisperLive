"""Command-line entry point for the meeting transcription GUI."""

from __future__ import annotations

import argparse
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WhisperLive meeting transcription GUI")
    parser.add_argument("--config", default="config.toml", help="Path to the TOML configuration")
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Load and validate the configuration, then exit",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List PortAudio input devices, then exit",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.validate_config:
        print(f"Configuration is valid: {args.config}")
        return 0
    if args.list_devices:
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            for index in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(index)
                if info["maxInputChannels"] > 0:
                    print(f"[{index}] {info['name']} ({info['defaultSampleRate']} Hz)")
        finally:
            pa.terminate()
        return 0
    from PyQt6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .pipewire import configure_pipewire_node

    configure_pipewire_node(config.audio.node_name, config.audio.node_description)
    application = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())