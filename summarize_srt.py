#!/usr/bin/env python3
"""Summarize existing .srt transcripts with :mod:`whisper_live.summarizer`.

The live client summarizes a transcript as it arrives.  This is the offline
equivalent: point it at one or more .srt files and it writes the same
``summary_<name>_NNN.md`` series that a live session produces.

``AutoSummarizer`` is duck-typed - it reads ``client.transcript``, a list of
``{"start", "end", "text"}`` dicts with the times in seconds - so the only work
needed here is parsing SRT clock times and deciding where one block ends and the
next begins.

Defaults come from the live client's own configuration, so this matches whatever
``run_sjm_client.sh`` and ``live_poc.py`` are set up to do without repeating any
flags.  In precedence order, lowest first:

1. the library defaults in ``whisper_live.summarizer``;
2. the constants in ``live_poc.py``;
3. the ``ARGS=`` line of ``run_sjm_client.sh``, which is what the launcher
   actually passes to ``live_poc.py`` and therefore wins over (2);
4. anything given on this command line.

Each run prints the settings it resolved and the files they came from, so there
is no guessing about which value is in play.

Run it with the same interpreter the launcher uses (``whisper_env/bin/python``):
reading step (2) imports ``live_poc``, which pulls in numpy, scipy and PortAudio.
Importing that module only defines constants and functions, so it has no side
effects - and if it cannot be imported the tool says so and falls back to step
(1), so it still runs standalone.

Usage:
    whisper_env/bin/python summarize_srt.py transcripts/transcript_20260928_180506.srt
    whisper_env/bin/python summarize_srt.py output.srt --block-minutes 5
    whisper_env/bin/python summarize_srt.py meeting.srt --block-minutes 0
    whisper_env/bin/python summarize_srt.py *.srt --dry-run     # show the blocks
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys

from whisper_live.summarizer import (
    DEFAULT_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_URL,
    AutoSummarizer,
    format_offset,
    parse_interval_minutes,
)

# Last-resort values, used only when live_poc.py cannot be read. Ordinarily both
# come from the live configuration (see _resolve_live_defaults).
FALLBACK_BLOCK_MINUTES = 10.0
FALLBACK_OUTPUT_DIR = "summaries"

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_LAUNCHER_NAME = "run_sjm_client.sh"
# The launcher's command line: ARGS="--output-srt --auto-summary-minutes 10 ..."
_ARGS_LINE_RE = re.compile(
    r'^[ \t]*ARGS=(?P<quote>["\'])(?P<value>.*?)(?P=quote)[ \t]*$', re.MULTILINE
)

# "01:23:45,678", "01:23:45.678" (SRT) and "23:45.678" (WebVTT, no hours).
_CLOCK_RE = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d{1,3}):(?P<seconds>\d{1,2})"
    r"(?:[.,](?P<fraction>\d{1,3}))?$"
)


# -- defaults from the live setup ------------------------------------------


def _import_live_poc():
    """Import ``live_poc`` to read its configuration constants.

    Returns:
        tuple: ``(module, error)``. ``module`` is ``None`` when the import
            failed and ``error`` then explains why - typically a missing numpy,
            scipy or PortAudio, or no virtual environment. Importing the module
            only defines constants and functions, so it has no side effects.
    """
    if _REPO_DIR not in sys.path:
        sys.path.insert(0, _REPO_DIR)
    try:
        import live_poc
    except Exception as exc:  # ImportError, or a dependency failing to load
        return None, f"{type(exc).__name__}: {exc}"
    return live_poc, None


def _read_launcher_args(path):
    """Extract the arguments a launcher script passes to ``live_poc.py``.

    Only the ``ARGS="..."`` assignment is read, and it is parsed as shell words
    so quoting is handled the same way the shell would.

    Args:
        path (str): Path to the launcher script.

    Returns:
        list | None: The argument tokens, or ``None`` when the file, the
            assignment or its closing quote is missing.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    match = _ARGS_LINE_RE.search(text)
    if match is None:
        return None
    try:
        return shlex.split(match.group("value"))
    except ValueError:
        return None


def _flag_value(tokens, name):
    """Return the value of ``--name value`` or ``--name=value`` in ``tokens``."""
    prefix = f"{name}="
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _interval_or_none(value, source):
    """Normalise a configured summary interval, warning if it is unusable."""
    try:
        return parse_interval_minutes(value)
    except ValueError as exc:
        print(f"[!] {source} is not a valid interval ({exc}); ignoring it.", flush=True)
        return None


def _resolve_live_defaults():
    """Resolve this tool's defaults from the live client's configuration.

    The live settings live in two places: the constants in ``live_poc.py`` and
    the command line ``run_sjm_client.sh`` builds for it. The launcher is read
    last because its flags are what actually win at runtime, so they override
    the constants they were written against.

    Returns:
        tuple: ``(defaults, sources, error)``. ``defaults`` holds ``model``,
            ``url``, ``block_minutes`` and ``output_dir``; ``sources`` lists the
            files the values came from, lowest precedence first; ``error``
            describes a failed ``live_poc`` import, or is ``None``.
    """
    defaults = {
        "model": DEFAULT_MODEL,
        "url": DEFAULT_URL,
        "block_minutes": FALLBACK_BLOCK_MINUTES,
        "output_dir": FALLBACK_OUTPUT_DIR,
    }
    sources = ["library defaults"]

    live_poc, error = _import_live_poc()
    if live_poc is not None:
        sources.append("live_poc.py")
        defaults["model"] = getattr(live_poc, "AUTO_SUMMARY_MODEL", defaults["model"])
        defaults["url"] = getattr(live_poc, "AUTO_SUMMARY_URL", defaults["url"])
        defaults["output_dir"] = getattr(
            live_poc, "SUMMARY_DIRNAME", defaults["output_dir"]
        )
        configured = getattr(live_poc, "AUTO_SUMMARY_MINUTES", None)
        if configured is not None:
            minutes = _interval_or_none(configured, "live_poc.py AUTO_SUMMARY_MINUTES")
            if minutes is not None:
                defaults["block_minutes"] = minutes

    tokens = _read_launcher_args(os.path.join(_REPO_DIR, _LAUNCHER_NAME))
    if tokens:
        sources.append(_LAUNCHER_NAME)
        # A launcher value of 0 disables summaries for a live session, which
        # says nothing about how to block an offline transcript, so it is left
        # to mean "no opinion" and the live_poc.py value stands.
        minutes = _flag_value(tokens, "--auto-summary-minutes")
        if minutes is not None:
            parsed = _interval_or_none(
                minutes, f"{_LAUNCHER_NAME} --auto-summary-minutes"
            )
            if parsed is not None:
                defaults["block_minutes"] = parsed
        for key, flag in (
            ("model", "--auto-summary-model"),
            ("url", "--auto-summary-url"),
        ):
            value = _flag_value(tokens, flag)
            if value:
                defaults[key] = value

    return defaults, sources, error


# -- SRT parsing -----------------------------------------------------------


def clock_to_seconds(text):
    """Convert an SRT/WebVTT timestamp to seconds.

    Args:
        text (str): A timestamp such as ``01:23:45,678``. Both ``,`` and ``.``
            are accepted as the millisecond separator, and the leading hours
            field is optional.

    Returns:
        float: The offset in seconds.

    Raises:
        ValueError: If the timestamp cannot be parsed.
    """
    match = _CLOCK_RE.match(text.strip())
    if match is None:
        raise ValueError(f"Unrecognised timestamp: {text!r}")
    fraction = (match.group("fraction") or "0").ljust(3, "0")[:3]
    return (
        int(match.group("hours") or 0) * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
        + int(fraction) / 1000.0
    )


def parse_srt(text):
    """Parse SRT text into the segment shape the summarizer expects.

    Cue numbers are ignored, blank lines separate cues, and multi-line cue text
    is collapsed onto a single line.  Cues that cannot be parsed are skipped
    rather than failing the whole file.

    Args:
        text (str): The contents of an .srt (or .vtt) file.

    Returns:
        list: ``{"start", "end", "text"}`` dicts with times in seconds, sorted
            by start time so blocks are built in transcript order even if the
            file is not.
    """
    # Normalise line endings and a UTF-8 BOM, then split on blank lines.
    normalised = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n[ \t]*\n+", normalised):
        lines = block.split("\n")
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue

        start_text, _, end_text = lines[timing].partition("-->")
        # WebVTT appends cue settings ("align:start position:50%") after the
        # end time; only the first token is the timestamp.
        end_token = end_text.strip().split()[0] if end_text.strip() else ""
        cue_text = " ".join(line.strip() for line in lines[timing + 1:] if line.strip())
        if not end_token or not cue_text:
            continue

        try:
            start = clock_to_seconds(start_text)
            end = clock_to_seconds(end_token)
        except ValueError:
            continue
        cues.append({"start": start, "end": end, "text": cue_text})

    cues.sort(key=lambda cue: cue["start"])
    return cues


def split_into_blocks(cues, minutes):
    """Group cues into consecutive blocks of roughly ``minutes`` of transcript.

    Boundaries fall on transcript time (the first cue's offset plus N minutes),
    matching the wall-clock blocks a live session produces.  A cue that spans a
    boundary stays in the earlier block.

    Args:
        cues (list): Parsed cues, sorted by start time.
        minutes (float | None): Minutes per block, or ``None``/``0`` for a
            single block covering the whole file.

    Returns:
        list: A list of non-empty cue lists.
    """
    if not minutes or minutes <= 0:
        return [list(cues)]

    blocks = []
    current = []
    block_start = None
    for cue in cues:
        if current and cue["start"] - block_start >= minutes * 60.0:
            blocks.append(current)
            current = []
        if not current:
            block_start = cue["start"]
        current.append(cue)
    if current:
        blocks.append(current)
    return blocks


# -- driving the summarizer ------------------------------------------------


class _TranscriptStub:
    """Minimal stand-in for a live ``Client``: only ``transcript`` is read."""

    def __init__(self):
        self.transcript = []


def _worker(summarizer):
    """Return ``AutoSummarizer``'s per-block worker, with a clear failure if it moves.

    The class has no public "summarize what you have now" entry point: the only
    caller of its worker is the timer thread started by :meth:`AutoSummarizer.start`,
    which cannot be driven from a static file without replaying the transcript in
    real time.  Calling the worker directly keeps this a single file with no
    change to the library; the guard turns a future rename into a readable error
    instead of a bare ``AttributeError``.
    """
    worker = getattr(summarizer, "_summarize_new_segments", None)
    if worker is None:
        raise RuntimeError(
            "AutoSummarizer._summarize_new_segments no longer exists; "
            "summarizer.py changed and summarize_srt.py needs updating."
        )
    return worker


def _safe_stem(path):
    """Derive a filename-safe label from a transcript path."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(path))[0])
    return stem.strip("_") or "transcript"


def process(path, args):
    """Summarize one .srt file.

    Args:
        path (str): The transcript to read.
        args (argparse.Namespace): Parsed command line options.

    Returns:
        int: 0 on success, 1 if the file was unusable or text was left
            unsummarized.
    """
    if not os.path.isfile(path):
        print(f"[!] Not a file: {path}", flush=True)
        return 1

    with open(path, encoding="utf-8-sig") as handle:
        cues = parse_srt(handle.read())

    if not cues:
        print(f"[!] No usable cues in {path}; nothing to summarize.", flush=True)
        return 1

    span = cues[-1]["end"] - cues[0]["start"]
    blocks = split_into_blocks(cues, args.block_minutes)
    print(
        f"[*] {path}: {len(cues)} cues, {format_offset(span)} of audio, "
        f"{len(blocks)} block(s)",
        flush=True,
    )

    if args.dry_run:
        for number, block in enumerate(blocks, 1):
            characters = sum(len(cue["text"]) for cue in block)
            print(
                f"    block {number:03d}: {format_offset(block[0]['start'])} - "
                f"{format_offset(block[-1]['end'])}, {len(block)} cues, "
                f"{characters} chars",
                flush=True,
            )
        return 0

    stub = _TranscriptStub()
    summarizer = AutoSummarizer(
        client=stub,
        # Inert: this utility never calls start(), so the interval is not used.
        # It is passed for honesty about the block size, and so that a future
        # start() would behave sensibly.
        interval_minutes=args.block_minutes or 1.0,
        model=args.model,
        url=args.url,
        output_dir=args.output_dir,
        base_timestamp=args.timestamp or _safe_stem(path),
        prompt=args.prompt,
        request_timeout=args.timeout,
    )
    worker = _worker(summarizer)

    written_before = summarizer.sequence_number - 1
    carried = 0
    for number, block in enumerate(blocks, 1):
        stub.transcript.extend(block)
        worker(final=False)
        if summarizer.has_pending():
            # The request failed. _index was not advanced, so this text rolls
            # into the next block - the same behaviour as a live session.
            carried += 1
            print(
                f"[!] Block {number}/{len(blocks)} was not summarized; its text "
                "rolls into the next block.",
                flush=True,
            )

    # Mirror the live shutdown path: one final attempt at anything left over.
    worker(final=True)
    pending = summarizer.has_pending()

    written = summarizer.sequence_number - 1 - written_before
    print(f"[*] {written} summary file(s) written to {args.output_dir}", flush=True)
    if pending:
        print(
            "[!] Some transcript text is still unsummarized after the final "
            "attempt; re-run once Ollama is reachable.",
            flush=True,
        )
        if carried:
            print(f"[*] {carried} block(s) had failed earlier in the run.", flush=True)
        return 1
    if carried:
        print(
            f"[*] {carried} block(s) failed and were merged into a later block.",
            flush=True,
        )
    return 0


# -- command line ----------------------------------------------------------


def _interval(value):
    """argparse type: minutes per block, or ``None`` for the whole file."""
    try:
        return parse_interval_minutes(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parse_args(argv=None):
    defaults, sources, error = _resolve_live_defaults()
    origin = " + ".join(sources)
    parser = argparse.ArgumentParser(
        description="Summarize existing .srt transcripts through Ollama, one "
        "summary file per block. Requires no server or audio device.",
        epilog="Defaults are read from the live setup ("
        f"{origin})"
        + (f", except live_poc.py: {error}" if error else "")
        + ". Summaries are written as <output-dir>/summary_<timestamp>_NNN.md, "
        "where <timestamp> is the transcript's own filename by default, so the "
        "files sort next to the transcript they came from. Re-running the same "
        "transcript reuses those names and overwrites them, so pass a different "
        "--timestamp or --output-dir to keep an earlier set.",
    )
    parser.add_argument(
        "transcripts",
        nargs="+",
        metavar="TRANSCRIPT.srt",
        help="One or more .srt (or .vtt) files to summarize.",
    )
    parser.add_argument(
        "--block-minutes",
        type=_interval,
        default=defaults["block_minutes"],
        help="Minutes of transcript per summary (default: "
        f"{defaults['block_minutes']:g}, the live client's summary interval). "
        "Use 0/false/off for a single summary covering the whole file. Large "
        "blocks need more model context.",
    )
    parser.add_argument(
        "--model",
        default=defaults["model"],
        help="Ollama model used for summaries (default: "
        f"{defaults['model']}, from the live setup).",
    )
    parser.add_argument(
        "--url",
        default=defaults["url"],
        help="Base URL of the Ollama server (default: "
        f"{defaults['url']}, from the live setup).",
    )
    parser.add_argument(
        "--output-dir",
        default=defaults["output_dir"],
        help="Directory for the summary files (default: "
        f"{defaults['output_dir']}, from the live setup).",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Label used in the output filenames, replacing the transcript's "
        "own filename. Only useful when summarizing a single file.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Read the instruction prefix sent before each excerpt from this "
        "file instead of the built-in meeting-summary prompt.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP timeout for one summary request, in seconds (default: "
        f"{DEFAULT_REQUEST_TIMEOUT:g}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the blocks each file would produce and exit without calling "
        "Ollama.",
    )

    args = parser.parse_args(argv)
    args.defaults_origin = origin
    args.live_poc_error = error
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    args.prompt = None
    if args.prompt_file:
        try:
            with open(args.prompt_file, encoding="utf-8") as handle:
                args.prompt = handle.read()
        except OSError as exc:
            parser.error(f"could not read --prompt-file: {exc}")
        if not args.prompt.strip():
            parser.error(f"--prompt-file {args.prompt_file} is empty")
    return args


def main(argv=None):
    args = _parse_args(argv)
    if args.live_poc_error:
        print(
            f"[!] Could not read the live configuration from live_poc.py "
            f"({args.live_poc_error}); anything it would have set falls back to "
            "the library defaults. Run this with whisper_env/bin/python to pick "
            "up live_poc.py and " + _LAUNCHER_NAME + ".",
            flush=True,
        )
    block = f"{args.block_minutes:g} min" if args.block_minutes else "whole file"
    print(
        f"[*] Settings ({args.defaults_origin}): model={args.model}, "
        f"url={args.url}, block={block}, output={args.output_dir}",
        flush=True,
    )
    failures = sum(1 for path in args.transcripts if process(path, args))
    if failures:
        print(f"[!] {failures} of {len(args.transcripts)} file(s) failed.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
