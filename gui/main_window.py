"""Main window for the local meeting transcription application."""

from __future__ import annotations

import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig
from .pipewire import PipeWireError, PipeWireRouter
from .systemd import service_is_active, start_service
from .workers import LlmWorker, SttWorker, TranscriptBuffer


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.buffer = TranscriptBuffer()
        self.router = PipeWireRouter(config.audio.node_name, config.audio.sources)
        self.stt_worker = SttWorker(config.audio, config.server, self.buffer)
        self.llm_worker = LlmWorker(config.llm, self.buffer) if config.llm.enabled else None
        self._updates_suspended = False
        self._last_stt_at = time.monotonic()
        self._last_llm_at = time.monotonic()
        self._llm_error = None
        self._exporting = False
        self._build_ui()
        self._connect_workers()
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(1000)
        self._start_server_if_needed()
        self.stt_worker.start()
        if self.llm_worker is not None:
            self.llm_worker.start()

    def _build_ui(self) -> None:
        self.setWindowTitle("WhisperLive Meeting Capture")
        self.resize(1200, 760)
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        self.server_status = QLabel("Server: checking")
        self.audio_target = QLabel("Audio: waiting for capture")
        self.stt_age = QLabel("STT: --")
        self.llm_age = QLabel("LLM: --")
        self.relink_button = QPushButton("Re-link Audio")
        self.relink_button.clicked.connect(self._relink_audio)
        for widget in (self.server_status, self.audio_target, self.stt_age, self.llm_age):
            header.addWidget(widget)
        header.addStretch(1)
        header.addWidget(self.relink_button)
        layout.addLayout(header)

        splitter = QSplitter()
        self.transcript_view = self._make_text_view("Live STT")
        self.summary_view = self._make_text_view("LLM Summaries")
        splitter.addWidget(self.transcript_view)
        splitter.addWidget(self.summary_view)
        splitter.setSizes([600, 600])
        layout.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.meeting_tag = QLineEdit()
        self.meeting_tag.setPlaceholderText("Meeting Tag")
        self.export_button = QPushButton("Wrap & Export")
        self.export_button.clicked.connect(self._wrap_and_export)
        footer.addWidget(self.meeting_tag, 1)
        footer.addWidget(self.export_button)
        layout.addLayout(footer)
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #11161b; color: #e7edf2; }
            QLabel { color: #9eacb8; font-size: 13px; }
            QPlainTextEdit { background: #0b0f13; border: 1px solid #2a353e;
                border-radius: 4px; color: #dce6ed; padding: 10px;
                selection-background-color: #286b80; }
            QLineEdit { background: #0b0f13; border: 1px solid #2a353e;
                border-radius: 4px; color: #e7edf2; padding: 9px; }
            QPushButton { background: #1d7180; border: 0; border-radius: 4px;
                color: #f4fbfc; padding: 9px 15px; }
            QPushButton:hover { background: #258b9d; }
            QPushButton:disabled { background: #344149; color: #81909a; }
            QSplitter::handle { background: #26323a; }
            """
        )

    @staticmethod
    def _make_text_view(title: str) -> QPlainTextEdit:
        view = QPlainTextEdit()
        view.setObjectName(title.replace(" ", ""))
        view.setReadOnly(True)
        view.setPlaceholderText(title)
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        return view

    def _connect_workers(self) -> None:
        self.stt_worker.transcript_updated.connect(self._on_transcript)
        self.stt_worker.status_changed.connect(self._on_stt_status)
        self.stt_worker.audio_ready.connect(self._on_audio_ready)
        self.stt_worker.error.connect(self._on_worker_error)
        if self.llm_worker is not None:
            self.llm_worker.summary_ready.connect(self._on_summary)
            self.llm_worker.error.connect(self._on_llm_error)

    def _start_server_if_needed(self) -> None:
        if service_is_active(self.config.server.unit):
            return
        if not self.config.server.auto_start_systemd:
            self.server_status.setText("Server: inactive")
            return
        started, message = start_service(self.config.server.unit)
        if not started:
            self.server_status.setText(f"Server: start failed ({message or 'permission denied'})")

    def _refresh_status(self) -> None:
        if service_is_active(self.config.server.unit):
            self.server_status.setText("Server: active")
        elif "start failed" not in self.server_status.text():
            self.server_status.setText("Server: inactive")
        self.stt_age.setText(f"STT: {self._age(self._last_stt_at)}")
        if self._llm_error:
            self.llm_age.setText(f"LLM error: {self._llm_error}")
        else:
            self.llm_age.setText(f"LLM: {self._age(self._last_llm_at)}")

    @staticmethod
    def _age(timestamp: float) -> str:
        seconds = max(0, int(time.monotonic() - timestamp))
        return f"{seconds}s ago"

    def _on_stt_status(self, status: str) -> None:
        self.server_status.setText(f"STT: {status}")

    def _on_audio_ready(self, target: str) -> None:
        if target == self.config.audio.node_name:
            self._relink_audio()
            QTimer.singleShot(500, self._relink_audio)
            return
        self.audio_target.setText(f"Audio: {target}")

    def _on_transcript(self, text: str) -> None:
        if self._updates_suspended:
            return
        self._last_stt_at = time.monotonic()
        self._replace_view(self.transcript_view, text)

    def _on_summary(self, text: str) -> None:
        if self._updates_suspended:
            return
        self._llm_error = None
        self._last_llm_at = time.monotonic()
        current = self.summary_view.toPlainText().strip()
        combined = f"{current}\n\n{text}" if current else text
        self._replace_view(self.summary_view, combined)

    @staticmethod
    def _replace_view(view: QPlainTextEdit, text: str) -> None:
        view.setPlainText(text)
        cursor = view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        view.setTextCursor(cursor)

    def _relink_audio(self) -> None:
        try:
            linked, missing = self.router.relink()
        except PipeWireError as exc:
            self.audio_target.setText(f"Audio: {exc}")
            return
        label = f"Audio: linked {len(linked)} source(s)"
        if missing:
            label += f"; missing {', '.join(missing)}"
        self.audio_target.setText(label)

    def _on_worker_error(self, message: str) -> None:
        self.server_status.setText(f"Error: {message}")

    def _on_llm_error(self, message: str) -> None:
        self._llm_error = message

    def _wrap_and_export(self) -> None:
        if self._exporting:
            return
        self._exporting = True
        self._updates_suspended = True
        self.export_button.setEnabled(False)
        try:
            path = self._export_path()
            content = self._export_content()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, path)
            temporary = None
            self.stt_worker.request_reconnect()
            self.buffer.clear()
            if self.llm_worker is not None:
                self.llm_worker.reset()
            self.transcript_view.clear()
            self.summary_view.clear()
            now = time.monotonic()
            self._last_stt_at = now
            self._last_llm_at = now
            self.server_status.setText(f"Exported: {path}")
        except (OSError, ValueError, KeyError) as exc:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass
            QMessageBox.critical(self, "Export failed", str(exc))
        finally:
            self._updates_suspended = False
            self._exporting = False
            self.export_button.setEnabled(True)

    def _export_path(self) -> Path:
        now = datetime.now()
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.meeting_tag.text().strip()) or "meeting"
        filename = self.config.export.filename_template.format(
            date=now.strftime("%Y-%m-%d"), time=now.strftime("%H%M%S"), tag=tag
        )
        filename = os.path.basename(filename)
        return Path(os.path.expanduser(self.config.export.save_directory)) / filename

    def _export_content(self) -> str:
        return (
            f"# Meeting: {self.meeting_tag.text().strip() or 'meeting'}\n\n"
            "## Transcript\n\n"
            f"{self.transcript_view.toPlainText().strip()}\n\n"
            "## Summaries\n\n"
            f"{self.summary_view.toPlainText().strip()}\n"
        )

    def closeEvent(self, event) -> None:
        self._status_timer.stop()
        self.stt_worker.stop()
        self.stt_worker.wait(3000)
        if self.llm_worker is not None:
            self.llm_worker.stop()
            self.llm_worker.wait(3000)
        event.accept()