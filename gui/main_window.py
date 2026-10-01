"""Main window for the local meeting transcription application."""

from __future__ import annotations

import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QTimer, QSize, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap, QColor, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStyle,
    QSplitter,
    QToolButton,
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
        self._server_start_failed = False
        self._first_stt_at: float | None = None
        self._last_summary_at: float | None = None
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
        self.server_status = self._make_status_button(
            QStyle.StandardPixmap.SP_ComputerIcon, "Server status: checking"
        )
        self.audio_target = QLabel("Audio: waiting for capture")
        self.audio_target.setObjectName("audioStatus")
        self.audio_target.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.audio_target.setWordWrap(True)
        self.stt_age = self._make_status_button(
            QStyle.StandardPixmap.SP_MediaPlay, "STT: waiting for updates"
        )
        self.llm_age = self._make_status_button(
            QStyle.StandardPixmap.SP_MessageBoxInformation,
            "LLM: waiting for summaries",
        )
        self.meeting_timer = self._make_timer_button(
            QStyle.StandardPixmap.SP_MediaPlay, "Meeting duration"
        )
        self.summary_timer = self._make_timer_button(
            QStyle.StandardPixmap.SP_MessageBoxInformation, "Time since last summary"
        )
        self.autosummary_timer = self._make_timer_button(
            QStyle.StandardPixmap.SP_BrowserReload, "Time until next autosummary"
        )
        self.relink_button = self._make_tool_button(
            QStyle.StandardPixmap.SP_MediaVolume, "Re-link Audio"
        )
        self.relink_button.clicked.connect(self._relink_audio)
        audio_panel = QVBoxLayout()
        audio_panel.setSpacing(4)
        audio_panel.addWidget(self.relink_button, alignment=Qt.AlignmentFlag.AlignCenter)
        audio_panel.addWidget(self.audio_target)
        for widget in (
            self.server_status,
            self.stt_age,
            self.llm_age,
            self.meeting_timer,
            self.summary_timer,
            self.autosummary_timer,
        ):
            header.addWidget(widget)
        header.addLayout(audio_panel)
        header.addStretch(1)
        layout.addLayout(header)

        controls = QHBoxLayout()
        self.listen_button = QPushButton("Listening")
        self.listen_button.setObjectName("toggleButton")
        self.listen_button.setCheckable(True)
        self.listen_button.setChecked(True)
        self.listen_button.toggled.connect(self._toggle_listening)
        self.auto_summary_button = QPushButton("Auto Summarizing")
        self.auto_summary_button.setObjectName("toggleButton")
        self.auto_summary_button.setCheckable(True)
        self.auto_summary_button.setChecked(True)
        self.auto_summary_button.setEnabled(self.llm_worker is not None)
        self.auto_summary_button.toggled.connect(self._toggle_auto_summary)
        self.summarize_button = QPushButton("Summarize Now")
        self.summarize_button.setObjectName("primaryButton")
        self.summarize_button.setEnabled(self.llm_worker is not None)
        self.summarize_button.clicked.connect(self._summarize_now)
        self.clear_transcript_button = QPushButton("Clear Transcript")
        self.clear_transcript_button.clicked.connect(self._clear_transcript)
        self.clear_summaries_button = QPushButton("Clear Summaries")
        self.clear_summaries_button.clicked.connect(self._clear_summaries)
        for widget in (
            self.listen_button,
            self.auto_summary_button,
            self.summarize_button,
            self.clear_transcript_button,
            self.clear_summaries_button,
        ):
            controls.addWidget(widget)
        controls.addStretch(1)
        layout.addLayout(controls)
        self._update_toggle_style(self.listen_button, True)
        self._update_toggle_style(self.auto_summary_button, True)
        if self.llm_worker is None:
            self.autosummary_timer.hide()

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
        self.export_button = QPushButton("Wrap && Export")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self._wrap_and_export)
        footer.addWidget(self.meeting_tag, 1)
        footer.addWidget(self.export_button)
        layout.addLayout(footer)
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #11161b; color: #e7edf2; }
            QPlainTextEdit { background: #0b0f13; border: 1px solid #2a353e;
                border-radius: 4px; color: #dce6ed; padding: 10px;
                selection-background-color: #286b80; }
            QLineEdit { background: #0b0f13; border: 1px solid #2a353e;
                border-radius: 4px; color: #e7edf2; padding: 9px; }
            QLabel#audioStatus { color: #9eacb8; font-size: 11px; }
            QLabel#audioStatus[status="ok"] { color: #75e0bb; }
            QLabel#audioStatus[status="warning"] { color: #f0c76a; }
            QLabel#audioStatus[status="error"] { color: #ff9aa3; }
            QToolButton#statusIndicator, QToolButton#toolButton {
                background: #1a242b; border: 1px solid #31414a; border-radius: 18px;
                color: #dce6ed; }
            QToolButton#timerIndicator {
                min-width: 124px; max-width: 124px; min-height: 36px;
                background: #1a242b; border: 1px solid #31414a; border-radius: 6px;
                color: #dce6ed; font-family: monospace; font-size: 11px; }
            QToolButton#timerIndicator:hover {
                background: #25343d; border-color: #4d7f8c; }
            QToolButton#timerIndicator[status="warning"] {
                background: #4a3a1f; border-color: #b58b42; }
            QToolButton#statusIndicator:hover, QToolButton#toolButton:hover {
                background: #25343d; border-color: #4d7f8c; }
            QToolButton#statusIndicator[status="ok"] {
                background: #193d35; border-color: #3d947e; }
            QToolButton#statusIndicator[status="warning"] {
                background: #4a3a1f; border-color: #b58b42; }
            QToolButton#statusIndicator[status="error"] {
                background: #4a2529; border-color: #c56b73; }
            QPushButton { min-height: 36px; background: #1a242b;
                border: 1px solid #31414a; border-radius: 6px;
                color: #dce6ed; padding: 7px 14px; }
            QPushButton:hover { background: #25343d; border-color: #4d7f8c; }
            QPushButton:pressed { background: #142027; }
            QPushButton:disabled { background: #1b2227; color: #68757d; }
            QPushButton#primaryButton { background: #1d7180; border-color: #2f9aaa;
                color: #f4fbfc; }
            QPushButton#primaryButton:hover { background: #258b9d; }
            QPushButton#toggleButton:checked { background: #2e7d32; }
            QPushButton#toggleButton:unchecked { background: #4c565d;
                color: #d0d6da; text-decoration: line-through; }
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

    def _make_status_button(self, icon: QStyle.StandardPixmap, tooltip: str) -> QToolButton:
        button = self._make_tool_button(icon, tooltip)
        button.setObjectName("statusIndicator")
        return button

    @staticmethod
    def _make_timer_button(icon: QStyle.StandardPixmap, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("timerIndicator")
        button.setAutoRaise(True)
        button.setFixedSize(124, 36)
        button.setIconSize(QSize(18, 18))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setText("--:--:--")
        button.setIcon(MainWindow._colored_icon(button, icon, "#8dd7e5"))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        return button

    @staticmethod
    def _make_tool_button(icon: QStyle.StandardPixmap, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("toolButton")
        button.setAutoRaise(True)
        button.setFixedSize(36, 36)
        button.setIconSize(QSize(20, 20))
        button.setIcon(MainWindow._colored_icon(button, icon, "#dce6ed"))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        return button

    @staticmethod
    def _colored_icon(
        button: QToolButton, icon: QStyle.StandardPixmap, color: str
    ) -> QIcon:
        source = button.style().standardIcon(icon).pixmap(QSize(24, 24))
        colored = QPixmap(source.size())
        colored.fill(QColor(0, 0, 0, 0))
        painter = QPainter(colored)
        painter.drawPixmap(0, 0, source)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(colored.rect(), QColor(color))
        painter.end()
        return QIcon(colored)

    def _set_status(
        self,
        button: QToolButton,
        message: str,
        state: str = "info",
        icon: QStyle.StandardPixmap | None = None,
    ) -> None:
        if icon is not None:
            icon_color = {
                "ok": "#75e0bb",
                "warning": "#f0c76a",
                "error": "#ff9aa3",
                "info": "#8dd7e5",
            }.get(state, "#dce6ed")
            button.setIcon(self._colored_icon(button, icon, icon_color))
        button.setToolTip(message)
        button.setStatusTip(message)
        button.setProperty("status", state)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    @staticmethod
    def _set_audio_status(label: QLabel, message: str, state: str = "info") -> None:
        label.setText(message)
        label.setProperty("status", state)
        label.setToolTip(message)
        label.style().unpolish(label)
        label.style().polish(label)
        label.update()

    def _connect_workers(self) -> None:
        self.stt_worker.transcript_updated.connect(self._on_transcript)
        self.stt_worker.status_changed.connect(self._on_stt_status)
        self.stt_worker.audio_ready.connect(self._on_audio_ready)
        self.stt_worker.error.connect(self._on_worker_error)
        if self.llm_worker is not None:
            self.llm_worker.summary_ready.connect(self._on_summary)
            self.llm_worker.summary_finished.connect(self._on_summary_finished)
            self.llm_worker.final_summary_ready.connect(self._on_final_summary)
            self.llm_worker.final_summary_failed.connect(self._on_final_summary_failed)
            self.llm_worker.error.connect(self._on_llm_error)

    def _start_server_if_needed(self) -> None:
        if service_is_active(self.config.server.unit):
            self._set_status(
                self.server_status,
                f"Whisper server {self.config.server.unit} is active",
                "ok",
                QStyle.StandardPixmap.SP_DialogApplyButton,
            )
            return
        if not self.config.server.auto_start_systemd:
            self._set_status(
                self.server_status,
                f"Whisper server {self.config.server.unit} is inactive",
                "warning",
                QStyle.StandardPixmap.SP_MessageBoxWarning,
            )
            return
        started, message = start_service(self.config.server.unit)
        if not started:
            self._server_start_failed = True
            self._set_status(
                self.server_status,
                f"Whisper server start failed: {message or 'permission denied'}",
                "error",
                QStyle.StandardPixmap.SP_MessageBoxCritical,
            )

    def _refresh_status(self) -> None:
        if service_is_active(self.config.server.unit):
            self._set_status(
                self.server_status,
                f"Whisper server {self.config.server.unit} is active",
                "ok",
                QStyle.StandardPixmap.SP_DialogApplyButton,
            )
        elif not self._server_start_failed:
            self._set_status(
                self.server_status,
                f"Whisper server {self.config.server.unit} is inactive",
                "warning",
                QStyle.StandardPixmap.SP_MessageBoxWarning,
            )
        stt_age = self._age(self._last_stt_at)
        self._set_status(
            self.stt_age,
            f"STT last updated {stt_age}",
            "ok" if time.monotonic() - self._last_stt_at < 10 else "warning",
        )
        if self._llm_error:
            self._set_status(
                self.llm_age,
                f"LLM error: {self._llm_error}",
                "error",
                QStyle.StandardPixmap.SP_MessageBoxCritical,
            )
        else:
            llm_age = self._age(self._last_llm_at)
            self._set_status(
                self.llm_age,
                f"LLM last updated {llm_age}",
                "ok" if time.monotonic() - self._last_llm_at < 10 else "warning",
            )
        self._update_timer_displays()

    @staticmethod
    def _age(timestamp: float) -> str:
        seconds = max(0, int(time.monotonic() - timestamp))
        return f"{seconds}s ago"

    @staticmethod
    def _format_stopwatch(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _set_timer_display(
        self, button: QToolButton, value: str, tooltip: str, state: str = "info"
    ) -> None:
        button.setText(value)
        button.setToolTip(tooltip)
        button.setStatusTip(tooltip)
        button.setProperty("status", state)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def _update_timer_displays(self) -> None:
        now = time.monotonic()
        if self._first_stt_at is None:
            self._set_timer_display(
                self.meeting_timer,
                "--:--:--",
                "Meeting duration: waiting for the first STT content",
            )
        else:
            duration = self._format_stopwatch(now - self._first_stt_at)
            self._set_timer_display(
                self.meeting_timer,
                duration,
                f"Meeting duration since first STT content: {duration}",
                "ok",
            )

        if self._last_summary_at is None:
            self._set_timer_display(
                self.summary_timer,
                "--:--:--",
                "Time since last summary: no summary generated yet",
            )
        else:
            elapsed = self._format_stopwatch(now - self._last_summary_at)
            self._set_timer_display(
                self.summary_timer,
                elapsed,
                f"Time since last summary: {elapsed}",
                "ok" if now - self._last_summary_at < 10 else "warning",
            )

        if self.llm_worker is None or not self.llm_worker.auto_summary_enabled():
            self.autosummary_timer.hide()
            return
        remaining = self.llm_worker.seconds_until_next_summary()
        if remaining is None:
            self.autosummary_timer.hide()
            return
        countdown = self._format_stopwatch(remaining)
        self.autosummary_timer.show()
        self._set_timer_display(
            self.autosummary_timer,
            countdown,
            f"Time until next autosummary: {countdown}",
            "warning" if remaining <= 10 else "info",
        )

    def _on_stt_status(self, status: str) -> None:
        state = {"Listening": "ok", "Connecting": "info", "Stopped": "warning"}.get(
            status, "error"
        )
        self._set_status(self.stt_age, f"STT status: {status}", state)

    def _on_audio_ready(self, target: str) -> None:
        if target == self.config.audio.node_name:
            self._relink_audio()
            QTimer.singleShot(500, self._relink_audio)
            return
        self._set_audio_status(self.audio_target, f"Audio capture device: {target}", "ok")

    def _on_transcript(self, text: str) -> None:
        if self._updates_suspended:
            return
        now = time.monotonic()
        self._last_stt_at = now
        if self._first_stt_at is None and text.strip():
            self._first_stt_at = now
        if text.strip():
            self._update_meeting_duration()
        self._replace_view(self.transcript_view, text)

    def _on_summary(self, text: str) -> None:
        if self._updates_suspended:
            return
        self._llm_error = None
        self._last_llm_at = time.monotonic()
        self._last_summary_at = self._last_llm_at
        self._update_meeting_duration()
        self._set_status(self.llm_age, "LLM summary updated", "ok")
        current = self.summary_view.toPlainText().strip()
        combined = f"{current}\n\n{text}" if current else text
        self._replace_view(self.summary_view, combined)

    def _on_summary_finished(self, generated: bool) -> None:
        if self._exporting:
            return
        self.summarize_button.setEnabled(True)
        if not generated:
            self._set_status(self.llm_age, "LLM: no new transcript to summarize", "warning")

    def _summarize_now(self) -> None:
        if self.llm_worker is None or self._exporting:
            return
        self.summarize_button.setEnabled(False)
        self._set_status(self.llm_age, "LLM summary in progress", "info")
        self._update_meeting_duration()
        self.llm_worker.request_summary()

    def _on_final_summary(self, text: str) -> None:
        if text:
            self._llm_error = None
            self._last_llm_at = time.monotonic()
            self._last_summary_at = self._last_llm_at
            self._update_meeting_duration()
            self._set_status(self.llm_age, "Final LLM summary generated", "ok")
            current = self.summary_view.toPlainText().strip()
            combined = f"{current}\n\n{text}" if current else text
            self._replace_view(self.summary_view, combined)
        self._finish_export()

    def _on_final_summary_failed(self, message: str) -> None:
        self._llm_error = message
        self._finish_export()

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
            self._set_audio_status(
                self.audio_target,
                f"Audio routing error: {exc}",
                "error",
            )
            return
        label = f"Audio: linked {len(linked)} source(s)"
        if missing:
            label += f"; missing {', '.join(missing)}"
        self._set_audio_status(self.audio_target, label, "warning" if missing else "ok")

    def _toggle_listening(self, checked: bool) -> None:
        self._update_toggle_style(self.listen_button, checked)
        self.stt_worker.set_paused(not checked)
        self._set_status(
            self.stt_age,
            "STT listening for audio" if checked else "STT paused; incoming audio is discarded",
            "ok" if checked else "warning",
            QStyle.StandardPixmap.SP_MediaPlay if checked else QStyle.StandardPixmap.SP_MediaPause,
        )

    def _toggle_auto_summary(self, checked: bool) -> None:
        self._update_toggle_style(self.auto_summary_button, checked)
        if self.llm_worker is not None:
            self.llm_worker.set_auto_enabled(checked)
        self._update_timer_displays()

    def _update_meeting_duration(self) -> None:
        if self.llm_worker is not None and self._first_stt_at is not None:
            self.llm_worker.set_meeting_duration(time.monotonic() - self._first_stt_at)

    @staticmethod
    def _update_toggle_style(button: QPushButton, checked: bool) -> None:
        font = button.font()
        font.setStrikeOut(not checked)
        button.setFont(font)

    def _clear_transcript(self) -> None:
        self.stt_worker.clear_transcript()
        self.transcript_view.clear()
        if self.llm_worker is not None:
            self.llm_worker.reset()
        self._last_stt_at = time.monotonic()
        self._first_stt_at = None
        self._update_timer_displays()

    def _clear_summaries(self) -> None:
        self.summary_view.clear()
        self._llm_error = None
        self._last_llm_at = time.monotonic()
        self._last_summary_at = None
        self._set_status(self.llm_age, "LLM summaries cleared", "info")
        self._update_timer_displays()

    def _on_worker_error(self, message: str) -> None:
        self._set_status(
            self.server_status,
            f"Server error: {message}",
            "error",
            QStyle.StandardPixmap.SP_MessageBoxCritical,
        )

    def _on_llm_error(self, message: str) -> None:
        self._llm_error = message

    def _wrap_and_export(self) -> None:
        if self._exporting:
            return
        self._exporting = True
        self._updates_suspended = True
        self.summarize_button.setEnabled(False)
        self.export_button.setEnabled(False)
        if self.llm_worker is not None:
            self._set_status(self.llm_age, "Finalizing LLM summary before export", "info")
            self._update_meeting_duration()
            self.llm_worker.request_final_summary()
            return
        self._finish_export()

    def _finish_export(self) -> None:
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
            self._first_stt_at = None
            self._last_summary_at = None
            self._set_status(self.server_status, f"Meeting exported to {path}", "ok")
            self._update_timer_displays()
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
            self.summarize_button.setEnabled(self.llm_worker is not None)
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
            "## Summaries\n\n"
            f"{self.summary_view.toPlainText().strip()}\n\n"
            "## Transcript\n\n"
            f"{self.buffer.snapshot().strip()}\n"
        )

    def closeEvent(self, event) -> None:
        if self.transcript_view.toPlainText().strip():
            answer = QMessageBox.question(
                self,
                "Confirm Exit",
                "Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._status_timer.stop()
        self.stt_worker.stop()
        self.stt_worker.wait(3000)
        if self.llm_worker is not None:
            self.llm_worker.stop()
            self.llm_worker.wait(3000)
        event.accept()