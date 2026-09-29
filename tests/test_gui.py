import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gui.config import load_config
from gui.pipewire import PipeWireRouter
from gui.workers import LlmWorker, TranscriptBuffer


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

        from gui.config import LlmConfig

        worker = LlmWorker(LlmConfig(), TranscriptBuffer())
        with patch("gui.workers.urllib.request.urlopen", return_value=Response()):
            result = worker._request_summary("transcript")
        self.assertEqual(result, "first summary")