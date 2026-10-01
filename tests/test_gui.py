import io
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from gui.config import AudioConfig, LlmConfig, ServerConfig, load_config
from gui.pipewire import PipeWireRouter
from gui.workers import LlmWorker, SttWorker, TranscriptBuffer


class PipeWireRouterTests(unittest.TestCase):
    def test_relink_matches_configured_sources_to_node_inputs(self):
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            if command == ["pw-link", "-i"]:
                return type("Result", (), {"stdout": "whisperlive-capture:input_0\n"})()
            if command == ["pw-link", "-o"]:
                return type(
                    "Result", (), {"stdout": "capture_AUX0\nmonitor_AUX0\n"}
                )()
            return type("Result", (), {})()

        router = PipeWireRouter(
            "whisperlive-capture", ["capture_AUX0", "monitor_AUX0"], runner=runner
        )
        linked, missing = router.relink()

        self.assertEqual(linked, ["capture_AUX0", "monitor_AUX0"])
        self.assertEqual(missing, [])
        self.assertIn(["pw-link", "capture_AUX0", "whisperlive-capture:input_0"], calls)

    def test_existing_link_does_not_block_later_sources(self):
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            if command == ["pw-link", "-i"]:
                return type("Result", (), {"stdout": "whisperlive-capture:input_0\n"})()
            if command == ["pw-link", "-o"]:
                return type(
                    "Result", (), {"stdout": "capture_AUX0\nmonitor_AUX0\n"}
                )()
            if command[1] == "capture_AUX0":
                raise subprocess.CalledProcessError(
                    1, command, stderr="failed to link ports: File exists"
                )
            return type("Result", (), {})()

        router = PipeWireRouter(
            "whisperlive-capture", ["capture_AUX0", "monitor_AUX0"], runner=runner
        )
        linked, missing = router.relink()

        self.assertEqual(linked, ["capture_AUX0", "monitor_AUX0"])
        self.assertEqual(missing, [])
        self.assertIn(
            ["pw-link", "monitor_AUX0", "whisperlive-capture:input_0"], calls
        )


class GuiConfigTests(unittest.TestCase):
    def test_loads_root_config(self):
        config = load_config(Path(__file__).parents[1] / "config.toml")
        self.assertEqual(config.server.port, 9090)
        self.assertEqual(config.audio.sources, ["capture_AUX0", "monitor_AUX0"])

    def test_rejects_unknown_filename_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                '[export]\nfilename_template = "{date}_{unknown}.md"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                load_config(path)


class WorkerSupportTests(unittest.TestCase):
    def test_clear_transcript_ignores_stale_callback_until_reconnect(self):
        buffer = TranscriptBuffer()
        worker = SttWorker(AudioConfig(), ServerConfig(), buffer)
        worker._segments = [{"start": 0.0, "end": 1.0, "text": "old"}]
        buffer.set("old")

        worker.clear_transcript()
        worker._on_transcription("old", [{"start": 0.0, "end": 1.0, "text": "old"}])
        self.assertEqual(buffer.snapshot(), "")
        self.assertEqual(worker._segments, [])
        self.assertTrue(worker._reconnect.is_set())

        worker._ignore_transcripts.clear()
        worker._on_transcription("new", [{"start": 0.0, "end": 1.0, "text": "new"}])
        self.assertEqual(buffer.snapshot(), "[0.0 -> 1.0] new")

    def test_transcript_buffer_is_replaceable_and_clearable(self):
        buffer = TranscriptBuffer()
        buffer.set("hello")
        self.assertEqual(buffer.snapshot(), "hello")
        buffer.clear()
        self.assertEqual(buffer.snapshot(), "")

    def test_llm_worker_reads_streamed_ndjson(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter(
                    [
                        json.dumps({"response": "first "}).encode() + b"\n",
                        json.dumps({"response": "summary", "done": True}).encode()
                        + b"\n",
                    ]
                )

        worker = LlmWorker(LlmConfig(), TranscriptBuffer())
        with patch("gui.workers.urllib.request.urlopen", return_value=Response()):
            result = worker._request_summary("transcript")
        self.assertEqual(result, "first summary")

    def test_llm_prompt_includes_meeting_duration(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([b'{"response":"summary","done":true}\n'])

        worker = LlmWorker(LlmConfig(), TranscriptBuffer())
        worker.set_meeting_duration(3661)
        requests = []

        def open_request(request, timeout):
            requests.append(request)
            return Response()

        with patch("gui.workers.urllib.request.urlopen", side_effect=open_request):
            worker._request_summary("transcript")
        payload = json.loads(requests[0].data.decode("utf-8"))
        self.assertIn("The meeting had been in progress for 01:01.", payload["prompt"])

    def test_autosummary_countdown_is_none_when_disabled(self):
        worker = LlmWorker(LlmConfig(summary_interval_seconds=60), TranscriptBuffer())
        self.assertIsNotNone(worker.seconds_until_next_summary())
        worker.set_auto_enabled(False)
        self.assertIsNone(worker.seconds_until_next_summary())
        worker.set_auto_enabled(True)
        self.assertIsNotNone(worker.seconds_until_next_summary())

    def test_llm_worker_emits_final_summary_for_pending_text(self):
        worker = LlmWorker(LlmConfig(), TranscriptBuffer())
        worker.buffer.set("pending transcript")
        summaries = []
        worker.final_summary_ready.connect(summaries.append)
        with patch.object(worker, "_request_summary", return_value="final summary"):
            worker._summarize_current(final=True)
        self.assertEqual(summaries, ["final summary"])

    def test_manual_request_wakes_worker_before_interval(self):
        worker = LlmWorker(
            LlmConfig(summary_interval_seconds=60), TranscriptBuffer()
        )
        worker.buffer.set("pending transcript")
        completed = threading.Event()

        def summarize(_text):
            completed.set()
            return "manual summary"

        with patch.object(worker, "_request_summary", side_effect=summarize):
            worker.start()
            try:
                worker.request_summary()
                self.assertTrue(completed.wait(2))
            finally:
                worker.stop()
                worker.wait(2000)

    def test_manual_request_resets_the_autosummary_interval(self):
        worker = LlmWorker(
            LlmConfig(summary_interval_seconds=0.5), TranscriptBuffer()
        )
        worker.buffer.set("pending transcript")
        completed = threading.Event()
        calls = []

        def summarize(_text):
            calls.append(len(calls) + 1)
            completed.set()
            return "summary"

        with patch.object(worker, "_request_summary", side_effect=summarize):
            worker.start()
            try:
                worker.request_summary()
                self.assertTrue(completed.wait(2))
                completed.clear()
                self.assertFalse(completed.wait(0.1))
            finally:
                worker.stop()
                worker.wait(2000)
        self.assertEqual(calls, [1])

    def test_auto_disabled_skips_interval_but_manual_still_runs(self):
        worker = LlmWorker(
            LlmConfig(summary_interval_seconds=0.2), TranscriptBuffer()
        )
        worker.buffer.set("pending transcript")
        calls = []

        def summarize(_text):
            calls.append("summary")
            return "summary"

        with patch.object(worker, "_request_summary", side_effect=summarize):
            worker.start()
            try:
                worker.set_auto_enabled(False)
                time.sleep(0.5)
                self.assertEqual(calls, [])
                worker.request_summary()
                deadline = time.time() + 2
                while not calls and time.time() < deadline:
                    time.sleep(0.01)
                self.assertEqual(calls, ["summary"])
            finally:
                worker.stop()
                worker.wait(2000)