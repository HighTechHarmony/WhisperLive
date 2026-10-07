"""Configuration loading for the meeting transcription GUI."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from whisper_live.summarizer import resolve_summary_template


@dataclass(frozen=True)
class AudioConfig:
    sources: list[str] = field(default_factory=lambda: ["capture_AUX0", "monitor_AUX0"])
    sample_rate: int = 16000
    channels: int = 1
    gain: float = 1.0
    device: str | int | None = None
    node_name: str = "whisperlive-capture"
    node_description: str = "WhisperLive Capture"
    vad_enabled: bool = True
    no_speech_thresh: float = 0.45
    chunk_seconds: float = 60.0
    send_block_seconds: float = 0.25
    recording: bool = False
    output_recording: str = "./output_recording.wav"


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9090
    lang: str = "en"
    model: str = "small"
    ready_timeout: float = 30.0
    display_segments: int = 40
    enable_timestamps: bool = True
    auto_start_systemd: bool = True
    unit: str = "whisper-server.service"
    backend: str = "faster_whisper"
    device: str = "cpu"
    max_clients: int = 4
    max_connection_time: int = 0
    enable_rest: bool = True
    rest_port: int = 8000
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:8080", "http://127.0.0.1:8080"]
    )


@dataclass(frozen=True)
class LlmConfig:
    enabled: bool = True
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    model: str = "ornith-1.5:9b"
    summary_interval_seconds: float = 600.0
    request_timeout: float = 300.0
    summary_template: str = "auto"


@dataclass(frozen=True)
class ExportConfig:
    save_directory: str = "~/Documents/Meetings"
    filename_template: str = "{date}_{time}_{tag}.md"


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _validate(config: AppConfig) -> AppConfig:
    if not config.audio.sources:
        raise ValueError("audio.sources must contain at least one PipeWire port")
    if config.audio.sample_rate <= 0:
        raise ValueError("audio.sample_rate must be positive")
    if config.audio.channels < 1:
        raise ValueError("audio.channels must be at least 1")
    if config.audio.gain <= 0:
        raise ValueError("audio.gain must be positive")
    if config.server.port < 1 or config.server.port > 65535:
        raise ValueError("server.port must be between 1 and 65535")
    if config.server.device not in {"auto", "cuda", "cpu"}:
        raise ValueError("server.device must be 'auto', 'cuda', or 'cpu'")
    if config.server.display_segments < 1:
        raise ValueError("server.display_segments must be positive")
    if config.server.max_clients < 1:
        raise ValueError("server.max_clients must be positive")
    if config.server.rest_port < 1 or config.server.rest_port > 65535:
        raise ValueError("server.rest_port must be between 1 and 65535")
    if config.llm.enabled and config.llm.summary_interval_seconds <= 0:
        raise ValueError("llm.summary_interval_seconds must be positive when enabled")
    if config.llm.enabled:
        resolve_summary_template(config.llm.summary_template)
    if config.export.filename_template == "":
        raise ValueError("export.filename_template must not be empty")
    fields = set(re.findall(r"\{([^{}]+)\}", config.export.filename_template))
    unknown = fields - {"date", "time", "tag"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"export.filename_template contains unknown fields: {names}")
    return config


def load_config(path: str | os.PathLike[str] = "config.toml") -> AppConfig:
    """Load and validate an application configuration from ``path``."""
    config_path = Path(path).expanduser()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    config = AppConfig(
        audio=AudioConfig(**_section(data, "audio")),
        server=ServerConfig(**_section(data, "server")),
        llm=LlmConfig(**_section(data, "llm")),
        export=ExportConfig(**_section(data, "export")),
    )
    return _validate(config)