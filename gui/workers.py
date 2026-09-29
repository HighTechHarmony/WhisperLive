"""Background workers for audio capture and live summarization."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request

import numpy as np
import pyaudio
from PyQt6.QtCore import QThread, pyqtSignal
from scipy.signal import resample_poly

from whisper_live.client import Client, TranscriptionClient
from whisper_live.summarizer import strip_reasoning

from .config import AudioConfig, LlmConfig, ServerConfig
from .pipewire import configure_pipewire_node

SERVER_RATE = 16000


class TranscriptBuffer:
    """A small synchronized full-text buffer shared by the workers and UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._text = ""

    def set(self, text: str) -> None:
        with self._lock:
            self._text = text

    def snapshot(self) -> str:
        with self._lock:
            return self._text

    def clear(self) -> None:
        self.set("")


class SttWorker(QThread):
    transcript_updated = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    audio_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, audio: AudioConfig, server: ServerConfig, buffer: TranscriptBuffer):
        super().__init__()
        self.audio_config = audio
        self.server_config = server
        self.buffer = buffer
        self._stop = threading.Event()
        self._reconnect = threading.Event()
        self._audio = queue.Queue()
        self._tc = None
        self._stream = None
        self._segments: list[dict] = []
        self._rate = 0
        self._pa = None

    def stop(self) -> None:
        self._stop.set()
        self._reconnect.set()

    def request_reconnect(self) -> None:
        self._reconnect.set()

    def run(self) -> None:
        try:
            self._connect()
            self._capture()
        except Exception as exc:
            if not self._stop.is_set():
                self.error.emit(str(exc))
        finally:
            self._close_connection()
            self.status_changed.emit("Stopped")

    def _make_client(self):
        return TranscriptionClient(
            host=self.server_config.host,
            port=self.server_config.port,
            lang=self.server_config.lang,
            model=self.server_config.model,
            use_vad=self.audio_config.vad_enabled,
            no_speech_thresh=self.audio_config.no_speech_thresh,
            output_transcription_path=os.path.join(os.getcwd(), ".gui-session.srt"),
            log_transcription=False,
            enable_timestamps=self.server_config.enable_timestamps,
            display_segments=self.server_config.display_segments,
            transcription_callback=self._on_transcription,
        )

    def _connect(self) -> None:
        self.status_changed.emit("Connecting")
        configure_pipewire_node(
            self.audio_config.node_name, self.audio_config.node_description
        )
        self._tc = self._make_client()
        self._pa = self._tc.p
        old_stream = getattr(self._tc, "stream", None)
        if old_stream is not None:
            old_stream.stop_stream()
            old_stream.close()
            self._tc.stream = None
        self._open_stream()
        self.audio_ready.emit(self.audio_config.node_name)
        deadline = time.monotonic() + self.server_config.ready_timeout
        while not self._stop.is_set() and not self._tc.client.recording:
            if self._tc.client.server_error:
                raise RuntimeError(getattr(self._tc.client, "error_message", "server error"))
            if self._tc.client.waiting:
                raise RuntimeError("WhisperLive server is full")
            if time.monotonic() >= deadline:
                raise TimeoutError("WhisperLive server did not become ready")
            self._stop.wait(0.05)
        if self._stop.is_set():
            return
        self.status_changed.emit("Listening")

    def _open_stream(self) -> None:
        device_index = self._resolve_device(self._pa, self.audio_config.device)
        info = self._pa.get_device_info_by_index(device_index)
        self._rate = int(info["defaultSampleRate"])
        if self.audio_config.channels > int(info["maxInputChannels"]):
            raise ValueError(f"Audio device has only {info['maxInputChannels']} input channel(s)")
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.audio_config.channels,
            rate=self._rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=4096,
            stream_callback=self._on_audio,
        )
        self.audio_ready.emit(str(info["name"]))

    @staticmethod
    def _resolve_device(pa, spec):
        if spec is None:
            return int(pa.get_default_input_device_info()["index"])
        if isinstance(spec, int) or str(spec).isdigit():
            return int(spec)
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if str(spec).lower() in info["name"].lower() and info["maxInputChannels"] > 0:
                return index
        raise ValueError(f"No input device matches {spec!r}")

    def _on_audio(self, data, _frame_count, _time_info, _status):
        if not self._stop.is_set():
            self._audio.put_nowait(data)
        return None, pyaudio.paComplete if self._stop.is_set() else pyaudio.paContinue

    def _capture(self) -> None:
        send_buffer = np.empty(0, dtype=np.int16)
        block_size = max(1, int(self.audio_config.send_block_seconds * self._rate))
        while not self._stop.is_set():
            if self._reconnect.is_set():
                self._reconnect.clear()
                self._close_connection()
                self._segments.clear()
                self.buffer.clear()
                if not self._stop.is_set():
                    self._connect()
                send_buffer = np.empty(0, dtype=np.int16)
                continue
            try:
                raw = self._audio.get(timeout=0.2)
            except queue.Empty:
                continue
            samples = np.frombuffer(raw, dtype=np.int16)
            if self.audio_config.gain != 1.0:
                samples = np.clip(
                    samples.astype(np.float32) * self.audio_config.gain,
                    -32768.0,
                    32767.0,
                ).astype(np.int16)
            send_buffer = np.concatenate((send_buffer, samples))
            if send_buffer.size >= block_size and self._tc.client.recording:
                self._send_to_server(send_buffer)
                send_buffer = np.empty(0, dtype=np.int16)

    def _send_to_server(self, samples: np.ndarray) -> None:
        signal_block = samples.astype(np.float32) / 32768.0
        if self.audio_config.channels > 1:
            signal_block = signal_block.reshape(-1, self.audio_config.channels).mean(axis=1)
        if self._rate != SERVER_RATE:
            signal_block = resample_poly(signal_block, SERVER_RATE, self._rate)
        self._tc.multicast_packet(np.clip(signal_block, -1.0, 1.0).astype(np.float32).tobytes())

    def _on_transcription(self, _text, segments) -> None:
        for segment in segments:
            start = float(segment.get("start", 0))
            existing = next((item for item in self._segments if item["start"] == start), None)
            clean = {
                "start": start,
                "end": float(segment.get("end", start)),
                "text": str(segment.get("text", "")).strip(),
                "completed": segment.get("completed", False),
            }
            if existing is None:
                self._segments.append(clean)
            else:
                existing.update(clean)
        lines = []
        for segment in self._segments[-self.server_config.display_segments :]:
            text = segment["text"]
            if self.server_config.enable_timestamps:
                text = f"[{segment['start']:.1f} -> {segment['end']:.1f}] {text}"
            lines.append(text)
        transcript = "\n".join(line for line in lines if line.strip())
        self.buffer.set(transcript)
        self.transcript_updated.emit(transcript)

    def _close_connection(self) -> None:
        stream, tc = self._stream, self._tc
        self._stream = None
        self._tc = None
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if tc is not None:
            try:
                if tc.client.recording:
                    tc.client.send_packet_to_server(Client.END_OF_AUDIO.encode("utf-8"))
            except Exception:
                pass
            try:
                tc.close_all_clients()
            except Exception:
                pass
            try:
                tc.p.terminate()
            except Exception:
                pass
        self._pa = None


class LlmWorker(QThread):
    summary_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, config: LlmConfig, buffer: TranscriptBuffer):
        super().__init__()
        self.config = config
        self.buffer = buffer
        self._stop = threading.Event()
        self._previous_text = ""

    def stop(self) -> None:
        self._stop.set()

    def reset(self) -> None:
        self._previous_text = ""

    def run(self) -> None:
        while not self._stop.wait(self.config.summary_interval_seconds):
            text = self.buffer.snapshot()
            if not text:
                continue
            delta = text[len(self._previous_text) :] if text.startswith(self._previous_text) else text
            if not delta.strip():
                continue
            try:
                summary = self._request_summary(delta)
            except Exception as exc:
                self.error.emit(str(exc))
                continue
            self._previous_text = text
            if summary:
                self.summary_ready.emit(summary)

    def _request_summary(self, text: str) -> str:
        payload = json.dumps(
            {
                "model": self.config.model,
                "prompt": (
                    "Summarize this live meeting transcript excerpt in concise bullet points. "
                    "Cover topics, decisions, and action items without inventing details.\n\n"
                    f"{text}"
                ),
                "stream": True,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks = []
        with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                event = json.loads(raw_line.decode("utf-8"))
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("response"):
                    chunks.append(event["response"])
                if event.get("done"):
                    break
        return strip_reasoning("".join(chunks))