"""Automatic transcript summarization through a local Ollama server.

Long meetings produce far more transcript than anyone will read.  This module
watches a :class:`whisper_live.client.Client`'s growing transcript and, every
``interval_minutes``, sends the text transcribed since the previous summary to
Ollama's HTTP API.  Each block is written to its own file, so a session yields an
ordered series of summary files instead of one unwieldy document.

The implementation deliberately uses only the standard library (``urllib``) so it
adds no dependency to the client requirements.  Ollama failures are reported as
warnings and never interrupt transcription.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from importlib import resources
from typing import Optional

DEFAULT_MODEL = "ornith-1.5:9b"
DEFAULT_URL = "http://ollama:11434"
DEFAULT_REQUEST_TIMEOUT = 300.0
TEMPLATE_DIRECTORY = "summarizer_templates"
TEMPLATE_PREFIX = "SUMMARIZER_TEMPLATE-"


def discover_summary_templates(template_directory=None):
    """Return matching template filenames in deterministic display order."""
    directory = template_directory
    if directory is None:
        directory = resources.files(__package__).joinpath(TEMPLATE_DIRECTORY)
    try:
        entries = directory.iterdir()
        filenames = [
            entry.name
            for entry in entries
            if entry.is_file()
            and re.fullmatch(r"SUMMARIZER_TEMPLATE-.+\.md", entry.name)
        ]
    except (FileNotFoundError, NotADirectoryError):
        return []
    return sorted(filenames, key=lambda filename: (filename.casefold(), filename))


def resolve_summary_template(template=None, template_directory=None):
    """Resolve ``None``/``auto``, ``none``, or an exact discovered filename."""
    filenames = discover_summary_templates(template_directory)
    selection = "auto" if template is None else str(template)
    if selection.casefold() in {"auto", "automatic"}:
        return next(
            (filename for filename in filenames if "default" in filename.casefold()),
            None,
        )
    if selection.casefold() == "none":
        return None
    if selection in filenames:
        return selection
    available = ", ".join(filenames) if filenames else "none found"
    raise ValueError(
        f"Unknown summarizer template {selection!r}; choose 'auto', 'none', or "
        f"one of: {available}. Templates must be named "
        f"{TEMPLATE_PREFIX}<group name>.md in {TEMPLATE_DIRECTORY}/."
    )


def build_system_prompt(prompt=None, template=None, template_directory=None):
    """Return an explicit prompt or the selected template contents."""
    if prompt is not None:
        return prompt.strip()

    filename = resolve_summary_template(template, template_directory)
    if filename is None:
        return ""

    directory = template_directory
    if directory is None:
        directory = resources.files(__package__).joinpath(TEMPLATE_DIRECTORY)
    try:
        template_text = directory.joinpath(filename).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"Could not read summarizer template {filename!r}: {exc}"
        ) from exc
    return re.sub(r"<!--.*?-->", "", template_text, flags=re.DOTALL).strip()

_DISABLED_WORDS = {"0", "false", "off", "no", "none", "disable", "disabled", ""}


def parse_interval_minutes(value):
    """Interpret the ``--auto-summary-minutes`` value.

    Returns the interval in minutes as a float, or ``None`` when summaries are
    disabled.  Accepts numbers as well as the words ``0``/``false``/``off``/
    ``no``/``none``/``disable`` (case-insensitive).

    Raises:
        ValueError: if the value is neither a positive number nor a recognised
            "disabled" word.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip().lower()
    if text in _DISABLED_WORDS:
        return None
    try:
        minutes = float(text)
    except ValueError:
        raise ValueError(
            f"Invalid --auto-summary-minutes value {value!r}: expected a number of "
            "minutes, or 0/false to disable."
        ) from None
    return minutes if minutes > 0 else None


def format_offset(seconds):
    """Format a segment offset (in seconds) as ``HH:MM:SS``."""
    total = int(float(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


# Markers that reasoning models emit after their chain-of-thought, inline in
# Ollama's ``response`` field. Only the text after the last marker is the answer.
_REASONING_MARKERS = ("</think>", "</thinking>", "<｜end▁of▁thinking｜>")


def strip_reasoning(text):
    """Drop a model's leaked chain-of-thought, keeping only the final answer.

    Reasoning models (e.g. ``ornith``) emit their working out inline in the
    ``response`` field and terminate it with a marker such as ``</think>``, so a
    raw summary is roughly half internal monologue. Everything up to and
    including the last marker is removed. When no marker is present the text is
    returned unchanged (stripped of surrounding whitespace).
    """
    cut = -1
    for marker in _REASONING_MARKERS:
        index = text.rfind(marker)
        if index != -1:
            cut = max(cut, index + len(marker))
    return text[cut:].strip() if cut != -1 else text.strip()


class AutoSummarizer:
    """Summarize the newest transcript text every N minutes via Ollama.

    Args:
        client: The ``whisper_live.client.Client`` whose ``transcript`` list is
            watched.  Segments are dicts with ``start``, ``end`` and ``text``.
        interval_minutes (float): Minutes between summaries. Must be > 0.
        model (str): Ollama model name.
        url (str): Base URL of the Ollama server.
        output_dir (str): Directory the summary files are written to.
        base_timestamp (str, optional): Shared timestamp (``YYYYMMDD_HHMMSS``)
            used in every filename for this session. Defaults to "now".
        prompt (str, optional): Complete system-prompt override. When provided,
            it replaces the selected group template.
        request_timeout (float, optional): HTTP timeout for a single summary
            request, in seconds.
        summary_template (str, optional): ``auto``, ``none``, or a discovered
            template filename. ``None`` selects automatically.

    Each block is summarized independently in arrival order (non-overlapping):
    only text that arrived since the previous summary is included.  The final,
    partial block is summarized when :meth:`stop` is called.
    """

    def __init__(
        self,
        client,
        interval_minutes,
        model=DEFAULT_MODEL,
        url=DEFAULT_URL,
        output_dir="summaries",
        base_timestamp=None,
        prompt=None,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
        summary_template=None,
    ):
        self.client = client
        self.interval_seconds = float(interval_minutes) * 60.0
        self.model = model
        self.url = url.rstrip("/")
        self.output_dir = output_dir
        self.base_timestamp = base_timestamp or time.strftime(
            "%Y%m%d_%H%M%S", time.localtime()
        )
        self.prompt = prompt
        self.summary_template = summary_template
        self.system_prompt = build_system_prompt(prompt, summary_template)
        self.request_timeout = request_timeout

        self._index = 0  # number of transcript segments already summarized
        self._sequence = 0  # summaries written this session
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        """Start the background worker. No-op when the interval is not positive."""
        if self.interval_seconds <= 0:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="auto-summarizer", daemon=True
        )
        self._thread.start()

    def stop(self, timeout=DEFAULT_REQUEST_TIMEOUT + 5.0):
        """Ask the worker to stop and wait for it to summarize the final block.

        The worker summarizes any remaining text before exiting, so a session
        that is cut short still yields a last (partial) summary.  Summarizing a
        full block can take a minute or more, so ``timeout`` must be generous.
        If the worker does not finish within ``timeout`` seconds the daemon
        thread is abandoned; files are written atomically, so no half-written
        summary is left behind.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                print(
                    f"[WARN] Final auto-summary still running after {timeout:.0f}s; "
                    "abandoning it. Use --auto-summary-stop-timeout to wait longer.",
                    flush=True,
                )

    # -- worker ------------------------------------------------------------

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self._summarize_new_segments(final=False)
        self._summarize_new_segments(final=True)

    @staticmethod
    def _segments(client):
        return list(getattr(client, "transcript", None) or [])

    def has_pending(self):
        """Return True when transcript text has not yet been summarized."""
        return self._index < len(self._segments(self.client))

    def _summarize_new_segments(self, final):
        segments = self._segments(self.client)
        if self._index >= len(segments):
            return None

        block = segments[self._index:]
        text = " ".join(str(seg.get("text", "")).strip() for seg in block).strip()
        if not text:
            # Nothing usable arrived (e.g. only empty segments); skip the block
            # but do consume it so it is not recounted next time.
            self._index = len(segments)
            return None

        start = block[0].get("start", 0)
        end = block[-1].get("end", 0)
        try:
            summary = self._request_summary(text, start, end)
        except Exception as exc:  # network/HTTP/JSON errors are all non-fatal
            print(f"[WARN] Auto-summary request to {self.url} failed: {exc}", flush=True)
            # Do not advance the index: this text rolls into the next block.
            return None
        summary = strip_reasoning(summary)
        if not summary:
            print("[WARN] Auto-summary returned an empty response; skipping.", flush=True)
            return None

        path = self._write_summary(summary, start, end)
        self._index = len(segments)
        self._sequence += 1
        label = "Final summary" if final else "Summary"
        print(f"[*] {label} written to {path}", flush=True)
        return path

    # -- Ollama ------------------------------------------------------------

    def _request_summary(self, text, start, end):
        """Send one block to Ollama and return the generated summary text.

        The response is streamed (``stream: true``) and the ``response`` fields
        are concatenated.  Streaming keeps data flowing over the socket, so a
        slow model is not cut off by an idle-socket timeout part way through a
        long generation; ``request_timeout`` only bounds a genuine stall.
        """
        prompt = (
            f"Transcript excerpt, {format_offset(start)} to {format_offset(end)}:\n\n"
            f"{text}\n"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "system": self.system_prompt,
                "prompt": prompt,
                "stream": True,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks = []
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                piece = event.get("response")
                if piece:
                    chunks.append(piece)
                if event.get("done"):
                    break
        return "".join(chunks).strip()

    # -- output ------------------------------------------------------------

    def _write_summary(self, summary, start, end):
        """Write one summary file atomically and return its path."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(
            self.output_dir,
            f"summary_{self.base_timestamp}_{self.sequence_number:03d}.md",
        )
        generated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        content = (
            f"# Summary generated {generated}\n"
            f"# Block: {format_offset(start)} - {format_offset(end)}\n"
            f"# Model: {self.model}\n\n"
            f"{summary}\n"
        )
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
        return path

    @property
    def sequence_number(self):
        """The 1-based sequence number of the next summary file."""
        return self._sequence + 1
