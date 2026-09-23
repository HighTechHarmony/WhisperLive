#!/usr/bin/env python3
"""WhisperLive single-channel capture PoC.

Design notes
------------
**No blocking reads.**  ``TranscriptionTeeClient.record()`` calls
``stream.read()`` in a loop, which is a blocking PortAudio C call.  CPython only
runs signal handlers between bytecodes, so when the device stops delivering
audio that read never returns, the pending SIGINT is never turned into a
``KeyboardInterrupt``, and Ctrl+C appears to do nothing.  Aborting such a read
from another thread corrupts PortAudio's ALSA state (``alsa_snd_pcm_mmap_begin``
failures) and can end in a glibc heap abort.

This PoC therefore opens a **callback stream**: PortAudio pushes audio into a
queue from its own thread, and the main thread drains that queue with a
``queue.get(timeout=...)``.  Nothing ever blocks inside PortAudio, the audio
device is only ever touched by one thread, and Ctrl+C is handled within ~200 ms.

**Sample rate.**  WhisperLive expects mono 16 kHz float32.  Most interfaces
(including the MR18 and the JACK/PipeWire node) only open at their native rate,
e.g. 48 kHz - ``paInvalidSampleRate`` otherwise.  Audio is captured at the
device's own rate and resampled to 16 kHz with ``scipy.signal.resample_poly``.

**Device selection.**  PortAudio's default input is the ALSA ``default`` PCM,
which on a PipeWire/JACK host frequently points at nothing usable (reads simply
never deliver frames).  Use ``--list-devices`` and ``--device`` to pick the one
your qpwgraph wiring feeds.

Chunk files are written to ``<output>.wav.chunks/`` next to the output file and
removed on exit; the library's ``chunks/`` directory in the current working
directory is not used.  Recording is off by default: pass ``--recording`` to
write the captured audio to disk. Transcription is unaffected either way,
because it goes to the server over the websocket and never needs the files.
"""

import argparse
import os
import queue
import shutil
import signal
import sys
import threading
import time
import wave

import numpy as np
import pyaudio
from scipy.signal import resample_poly

from whisper_live.client import Client, TranscriptionClient

SERVER_RATE = 16000  # rate WhisperLive expects from clients
SERVER_CHANNELS = 1

HOST = "127.0.0.1"
PORT = 9090
LANG = "en"
MODEL = "small"
CHANNELS = 1
GAIN = 1.0
NO_SPEECH_THRESH = 0.45
CHUNK_SECONDS = 60.0
SEND_BLOCK_SECONDS = 0.25
READY_TIMEOUT = 30.0
SERVER_FLUSH_TIMEOUT = 10.0
STOP_TIMEOUT = 2.0
STALL_WARNING = 5.0
SHUTDOWN_TIMEOUT = 20.0
DISPLAY_SEGMENTS = 140  # transcript lines kept on screen (client display_segments)


class _Aborted(Exception):
    """Internal signal that a stop request arrived before capture started."""


def _write_wav(path, frames, channels, rate):
    """Write 16-bit PCM frames to ``path``."""
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(frames)


def _merge_wavs(chunk_paths, out_path, channels, rate):
    """Concatenate chunk WAVs into a single output file."""
    with wave.open(out_path, "wb") as out_file:
        out_file.setnchannels(channels)
        out_file.setsampwidth(2)
        out_file.setframerate(rate)
        for chunk_path in chunk_paths:
            with wave.open(chunk_path, "rb") as chunk_file:
                out_file.writeframes(chunk_file.readframes(chunk_file.getnframes()))


def _arm_watchdog(timeout):
    """Force-exit if shutdown wedges, so Ctrl+C always returns the shell."""

    def bail():
        print(f"[!] Shutdown exceeded {timeout:.0f}s; forcing exit.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    timer = threading.Timer(timeout, bail)
    timer.daemon = True
    timer.start()
    return timer


def _input_devices(pa):
    """Yield ``(index, name, max_input_channels, default_rate)`` for inputs."""
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        if info["maxInputChannels"] < 1:
            continue
        rate = int(info["defaultSampleRate"])
        yield index, info["name"], int(info["maxInputChannels"]), rate


def _resolve_device(pa, spec):
    """Resolve ``spec`` (index or name substring) to an input device index."""
    if spec is None:
        try:
            return int(pa.get_default_input_device_info()["index"])
        except Exception as exc:
            raise RuntimeError(f"PortAudio has no default input device: {exc}") from exc

    text = str(spec)
    if text.isdigit():
        index = int(text)
        if not 0 <= index < pa.get_device_count():
            raise ValueError(
                f"Device index {index} is out of range "
                f"(0-{pa.get_device_count() - 1}); use --list-devices."
            )
        if pa.get_device_info_by_index(index)["maxInputChannels"] < 1:
            raise ValueError(f"Device {index} has no input channels.")
        return index

    for index, name, _channels, _rate in _input_devices(pa):
        if text.lower() in name.lower():
            return index
    raise ValueError(f"No input device matches {spec!r}; use --list-devices.")


def _list_input_devices():
    pa = pyaudio.PyAudio()
    try:
        try:
            default = int(pa.get_default_input_device_info()["index"])
        except Exception:
            default = -1
        print("Input devices:")
        for index, name, channels, rate in _input_devices(pa):
            marker = "  <-- default" if index == default else ""
            print(f"  [{index:2}] {name}  ch={channels} rate={rate}{marker}")
    finally:
        pa.terminate()


class CaptureSession:
    """Owns the client, the audio stream and the shutdown path."""

    def __init__(self, args):
        self.args = args
        self.output_wav = (
            os.path.abspath(args.output_recording) if args.recording else None
        )
        self.output_srt = os.path.abspath(args.output_srt)
        self.chunk_dir = f"{self.output_wav}.chunks" if self.output_wav else None

        self.tc = None  # TranscriptionClient
        self.pa = None  # PyAudio instance reused from the client
        self.stream = None  # callback-mode capture stream
        self.device_index = None
        self.device_name = "?"
        self.rate = 0

        self.interrupted = threading.Event()
        self.audio = queue.Queue()
        self.frames = np.empty(0, dtype=np.int16)  # raw capture buffered for the WAV
        self.exit_code = 0

        self._send_buffer = np.empty(0, dtype=np.int16)
        self._chunks_written = 0
        self._last_callback_at = 0.0
        self._stall_warned = False
        self._session_started = False

        self._level_peak = 0
        self._level_sumsq = 0.0
        self._level_samples = 0

    # -- signal handling ---------------------------------------------------

    def install_signal_handlers(self):
        """Route termination signals to a handler that only sets a flag.

        ``SIGHUP`` matters for terminal windows: closing the terminal would
        otherwise kill the process outright, leaving the staging directory
        behind and never writing the merged recording.
        """
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum, frame):
        if not self.interrupted.is_set():
            name = signal.Signals(signum).name
            print(f"\n[*] {name} received. Finalizing recording ...", flush=True)
        self.interrupted.set()

    # -- audio callback ----------------------------------------------------

    def _on_audio(self, in_data, frame_count, time_info, status):
        """PortAudio callback: never block, never allocate more than needed."""
        self._last_callback_at = time.monotonic()
        self.audio.put_nowait(in_data)
        if self.interrupted.is_set():
            return (None, pyaudio.paComplete)
        return (None, pyaudio.paContinue)

    # -- setup -------------------------------------------------------------

    def _connect(self):
        for path in filter(None, (self.output_wav, self.output_srt)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.chunk_dir:
            shutil.rmtree(self.chunk_dir, ignore_errors=True)
            os.makedirs(self.chunk_dir, exist_ok=True)

        self.tc = TranscriptionClient(
            host=self.args.host,
            port=self.args.port,
            lang=self.args.lang,
            model=self.args.model,
            use_vad=True,
            no_speech_thresh=self.args.no_speech_thresh,
            save_output_recording=False,  # this script writes the WAV itself
            output_recording_filename=self.output_wav or self.args.output_recording,
            output_transcription_path=self.output_srt,
            enable_timestamps=self.args.enable_timestamps,
            display_segments=self.args.n_display_segments,
        )
        self.pa = self.tc.p
        self._release_library_stream()
        self._open_stream()

        if self.output_wav is None:
            print(
                "[*] Recording disabled (the default): no audio is written to "
                "disk, not even temporarily. Pass --recording to keep a WAV.",
                flush=True,
            )
        else:
            mb_per_minute = self.rate * self.args.channels * 2 * 60 / 1e6
            print(
                f"[*] Recording to {self.output_wav} ({self.rate} Hz, "
                f"{self.args.channels} ch, ~{mb_per_minute:.1f} MB/min); staged in "
                f"{self.chunk_dir} and merged then removed on exit.",
                flush=True,
            )

    def _release_library_stream(self):
        """Close the blocking-read stream the library opened for us."""
        stream = getattr(self.tc, "stream", None)
        if stream is None:
            return
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        self.tc.stream = None

    def _open_stream(self):
        self.device_index = _resolve_device(self.pa, self.args.device)
        info = self.pa.get_device_info_by_index(self.device_index)
        self.device_name = info["name"]
        self.rate = int(self.args.rate or info["defaultSampleRate"])

        if self.args.channels > int(info["maxInputChannels"]):
            raise ValueError(
                f"Device [{self.device_index}] {self.device_name} has "
                f"{int(info['maxInputChannels'])} input channel(s); "
                f"--channels {self.args.channels} is not available."
            )

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.args.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.tc.chunk,
            stream_callback=self._on_audio,
        )
        self._last_callback_at = time.monotonic()

        print(
            f"[*] Capturing from [{self.device_index}] {self.device_name} "
            f"@ {self.rate} Hz, {self.args.channels} ch"
        )
        if self.rate != SERVER_RATE:
            print(f"[*] Resampling {self.rate} Hz -> {SERVER_RATE} Hz for the server")

    def _wait_for_server(self):
        client = self.tc.client
        print(
            f"[*] Waiting for the server at {self.args.host}:{self.args.port} ...",
            flush=True,
        )
        deadline = time.monotonic() + self.args.ready_timeout
        while True:
            if self.interrupted.is_set():
                raise _Aborted
            if client.recording:
                self._session_started = True
                return True
            if client.waiting:
                raise RuntimeError("Server is full.")
            if client.server_error:
                error = getattr(client, "error_message", "connection failed")
                raise RuntimeError(
                    f"Could not reach the server: {error}. Is run_server.py running?"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Server did not report ready within {self.args.ready_timeout}s."
                )
            self.interrupted.wait(0.05)

    # -- capture -----------------------------------------------------------

    def _capture(self):
        while not self.interrupted.is_set():
            if not any(client.recording for client in self.tc.clients):
                print("[*] Server closed the session.", flush=True)
                break
            try:
                data = self.audio.get(timeout=0.2)
            except queue.Empty:
                self._check_stall()
                continue
            self._handle_audio_block(data)

    def _handle_audio_block(self, data):
        if self._stall_warned:
            self._stall_warned = False
            print("[*] Audio flow resumed.", flush=True)

        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size:
            # Level is measured before gain so the report describes the source.
            self._level_peak = max(self._level_peak, int(np.abs(samples).max()))
            self._level_sumsq += float(np.square(samples.astype(np.float64)).sum())
            self._level_samples += samples.size
        if self.args.gain != 1.0:
            samples = np.clip(
                samples.astype(np.float32) * self.args.gain, -32768.0, 32767.0
            ).astype(np.int16)
        if self.output_wav is not None:
            # Only buffer audio when a recording is actually being written.
            self.frames = np.concatenate((self.frames, samples))
        self._send_buffer = np.concatenate((self._send_buffer, samples))

        if self._send_buffer.size >= int(SEND_BLOCK_SECONDS * self.rate):
            self._send_to_server(self._send_buffer)
            self._send_buffer = np.empty(0, dtype=np.int16)

        if self.output_wav is not None and self.frames.size >= (
            self.args.chunk_seconds * self.rate * self.args.channels
        ):
            self._flush_chunk()

    def _send_to_server(self, samples):
        """Downmix, resample to 16 kHz and forward one block of audio.

        The payload must be float32 samples in [-1, 1]: the server reads them with
        ``get_audio_from_websocket``, matching the library's own
        ``bytes_to_float_array``.  Sending raw int16 magnitudes instead shifts the
        log-mel spectrogram far outside the range Whisper was trained on, and every
        segment then comes back with a high no-speech probability and is dropped.
        """
        if samples.size == 0:
            return
        signal_block = samples.astype(np.float32) / 32768.0
        if self.args.channels > 1:
            signal_block = signal_block.reshape(-1, self.args.channels).mean(axis=1)
        if self.rate != SERVER_RATE:
            signal_block = resample_poly(signal_block, SERVER_RATE, self.rate)
        if SERVER_CHANNELS == 1 and signal_block.ndim > 1:
            signal_block = signal_block.mean(axis=1)
        self.tc.multicast_packet(np.clip(signal_block, -1.0, 1.0).astype(np.float32).tobytes())

    def _check_stall(self):
        if self._stall_warned:
            return
        if time.monotonic() - self._last_callback_at < STALL_WARNING:
            return
        self._stall_warned = True
        print(
            f"[!] No audio from [{self.device_index}] {self.device_name} for "
            f"{STALL_WARNING:.0f}s. Check --device and your PipeWire/JACK wiring "
            "(--list-devices shows what PortAudio can see).",
            flush=True,
        )

    def _flush_chunk(self):
        if self.output_wav is None or self.frames.size == 0:
            return
        path = os.path.join(self.chunk_dir, f"{self._chunks_written}.wav")
        _write_wav(path, self.frames.tobytes(), self.args.channels, self.rate)
        self._chunks_written += 1
        self.frames = np.empty(0, dtype=np.int16)

    # -- shutdown ----------------------------------------------------------

    def _finalize(self):
        """Persist the recording, then release the connection and the device."""
        timer = _arm_watchdog(SHUTDOWN_TIMEOUT)
        try:
            self.interrupted.set()
            self._stop_stream()
            self._drain_audio()

            if self.tc is None:
                return

            # 1. Audio to disk first: nothing that follows can lose it.
            try:
                self._flush_chunk()
            except Exception as exc:
                print(f"[!] Could not flush the buffered audio: {exc}", flush=True)
            self._write_recording()

            # 2. Let the server flush its final segments, then save the SRT.
            if self._session_started:
                self._close_session()
                try:
                    self.tc.write_all_clients_srt()
                    print(f"[*] Transcript written to {self.output_srt}", flush=True)
                except Exception as exc:
                    print(f"[!] Could not write {self.output_srt}: {exc}", flush=True)

            # 3. Release the audio device.
            self._close_stream()

            # 4. Report what was captured, so a silent run is diagnosable.
            self._report_level()
        finally:
            timer.cancel()

    def _stop_stream(self):
        """Stop capture without stopping mid-callback.

        The callback returns ``paComplete`` once ``interrupted`` is set, so a
        healthy stream stops itself.  ``stop_stream()`` is only needed for a
        stalled device, which is exactly the case where no callback can be in
        flight and the call is therefore safe.
        """
        if self.stream is None:
            return
        deadline = time.monotonic() + STOP_TIMEOUT
        while self.stream.is_active() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self.stream.is_active():
            return
        if time.monotonic() - self._last_callback_at < STOP_TIMEOUT:
            print("[!] Capture stream is still running; stopping it anyway.", flush=True)
        try:
            self.stream.stop_stream()
        except Exception as exc:
            print(f"[!] Could not stop the capture stream: {exc}", flush=True)

    def _drain_audio(self):
        """Collect audio the callback delivered before the stream stopped."""
        while True:
            try:
                self._handle_audio_block(self.audio.get_nowait())
            except queue.Empty:
                break
        if self._send_buffer.size and self._session_started:
            try:
                self._send_to_server(self._send_buffer)
            except Exception as exc:
                print(f"[!] Could not send the last audio block: {exc}", flush=True)
        self._send_buffer = np.empty(0, dtype=np.int16)

    def _close_session(self):
        """Tell the server the audio ended and wait (bounded) for its flush."""
        if not self.tc.client.recording:
            return
        try:
            self.tc.client.send_packet_to_server(Client.END_OF_AUDIO.encode("utf-8"))
        except Exception as exc:
            print(f"[!] Could not send END_OF_AUDIO: {exc}", flush=True)
            return
        deadline = time.monotonic() + SERVER_FLUSH_TIMEOUT
        while self.tc.client.recording and time.monotonic() < deadline:
            time.sleep(0.05)

    def _write_recording(self):
        if self.output_wav is None:
            return
        chunk_paths = [
            os.path.join(self.chunk_dir, f"{index}.wav")
            for index in range(self._chunks_written)
            if os.path.exists(os.path.join(self.chunk_dir, f"{index}.wav"))
        ]
        if not chunk_paths:
            print("[!] No audio was captured; no WAV file written.", flush=True)
            shutil.rmtree(self.chunk_dir, ignore_errors=True)
            return
        _merge_wavs(chunk_paths, self.output_wav, self.args.channels, self.rate)
        shutil.rmtree(self.chunk_dir, ignore_errors=True)
        print(f"[*] Recording written to {self.output_wav}", flush=True)

    def _report_level(self):
        """Report the capture level and why the server may have stayed silent."""
        if self._level_samples:
            peak_db = 20 * np.log10(max(self._level_peak, 1) / 32768.0)
            rms = (self._level_sumsq / self._level_samples) ** 0.5
            rms_db = 20 * np.log10(max(rms, 1e-9) / 32768.0)
            print(f"[*] Capture level: peak {peak_db:.1f} dBFS, RMS {rms_db:.1f} dBFS")
            if peak_db < -40.0:
                print(
                    "[!] That is very quiet - raise the source level (MR18 AUX gain) "
                    "or use --gain to boost it."
                )

        if not self._session_started:
            return
        client = self.tc.client
        if len(client.transcript) == 0 and client.last_segment is None:
            print(
                "[!] The server sent no segments. WhisperLive discards every segment "
                "whose no_speech_prob is above the client's no_speech_thresh, so quiet "
                "or sparse audio yields nothing. Check the level above, and see "
                "--no-speech-thresh."
            )

    def _close_stream(self):
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception as exc:
                print(f"[!] Error closing the capture stream: {exc}", flush=True)
            self.stream = None
        if self.pa is not None:
            try:
                self.pa.terminate()
            except Exception as exc:
                print(f"[!] Error terminating PortAudio: {exc}", flush=True)
            self.pa = None

    # -- entry point -------------------------------------------------------

    def run(self):
        try:
            self._connect()
            if not self._wait_for_server():
                return 0
            print("[*] Listening. Press Ctrl+C to stop and finalize.\n", flush=True)
            self._capture()
        except _Aborted:
            return 0
        except Exception as exc:
            print(f"[!] {exc}", flush=True)
            self.exit_code = 1
        finally:
            self._finalize()
        return self.exit_code


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="WhisperLive capture PoC with a Ctrl+C-proof shutdown path."
    )
    parser.add_argument("--host", default=HOST, help=f"Server host (default: {HOST}).")
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"Server port (default: {PORT})."
    )
    parser.add_argument("--lang", default=LANG, help=f"Source language (default: {LANG}).")
    parser.add_argument(
        "--model", default=MODEL, help=f"Whisper model size (default: {MODEL})."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Input device index or name substring (default: PortAudio's default "
        "input, which is often the unusable ALSA 'default').",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="Capture sample rate in Hz (default: the device's default rate). "
        "Audio is resampled to 16 kHz for the server.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=CHANNELS,
        help=f"Capture channels, downmixed to mono for the server (default: {CHANNELS}).",
    )
    parser.add_argument(
        "--output-recording",
        default="./output_recording.wav",
        help="WAV file for the captured audio, written at the capture rate "
        "(default: ./output_recording.wav).",
    )
    parser.add_argument(
        "--recording",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write the captured audio to disk as a WAV, staged in chunks while "
        "capturing. Off by default: no WAV and no staging directory are "
        "created. Transcription is unaffected either way, so this only matters "
        "if you want a recording of the session.",
    )
    parser.add_argument(
        "--output-srt",
        default="./output.srt",
        help="SRT file for the transcript (default: ./output.srt).",
    )
    parser.add_argument(
        "--n-display-segments",
        "--n_display_segments",
        type=int,
        default=DISPLAY_SEGMENTS,
        help="Number of transcript segments to keep on screen (default: "
        f"{DISPLAY_SEGMENTS}). The terminal is cleared on every update, so only "
        "this many lines stay visible.",
    )
    parser.add_argument(
        "--enable-timestamps",
        action="store_true",
        help="Show each transcript segment with its [start -> end] timestamps. "
        "Off by default, which prints plain text.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=GAIN,
        help="Linear gain applied to captured audio before it is sent and written "
        "to the WAV (default: 1.0). Use e.g. 8.0 for a quiet source.",
    )
    parser.add_argument(
        "--no-speech-thresh",
        type=float,
        default=NO_SPEECH_THRESH,
        help="no_speech_thresh sent to the server: segments with a higher no-speech "
        "probability are discarded (default: 0.45, the library default). Raise it "
        "towards 1.0 to keep quiet or ambiguous segments.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=CHUNK_SECONDS,
        help="Audio buffered in memory before a chunk is flushed to disk (default: 60).",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=READY_TIMEOUT,
        help="Seconds to wait for the server to report ready (default: 30).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List PortAudio input devices and exit.",
    )
    args = parser.parse_args(argv)
    if args.recording and not args.output_recording.endswith(".wav"):
        parser.error("--output-recording must end with '.wav'")
    if not args.output_srt.endswith(".srt"):
        parser.error("--output-srt must end with '.srt'")
    if args.channels < 1:
        parser.error("--channels must be at least 1")
    return args


def main():
    args = _parse_args()
    if args.list_devices:
        _list_input_devices()
        return 0
    session = CaptureSession(args)
    session.install_signal_handlers()
    return session.run()


if __name__ == "__main__":
    sys.exit(main())
