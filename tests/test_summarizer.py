import json
from unittest.mock import patch

import pytest

from whisper_live.summarizer import (
    AutoSummarizer,
    build_system_prompt,
    discover_summary_templates,
    resolve_summary_template,
)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter([b'{"response":"summary","done":true}\n'])


def test_default_template_is_used_as_complete_ollama_system_prompt(tmp_path):
    summarizer = AutoSummarizer(
        client=None,
        interval_minutes=1,
        output_dir=str(tmp_path),
    )

    with patch(
        "whisper_live.summarizer.urllib.request.urlopen", return_value=_Response()
    ) as urlopen:
        assert summarizer._request_summary("hello", 0, 60) == "summary"

    payload = json.loads(urlopen.call_args.args[0].data)
    assert payload["system"] == build_system_prompt()
    assert payload["prompt"] == "Transcript excerpt, 00:00:00 to 00:01:00:\n\nhello\n"


def test_custom_prompt_replaces_selected_template(tmp_path):
    summarizer = AutoSummarizer(
        client=None,
        interval_minutes=1,
        output_dir=str(tmp_path),
        prompt="Use a custom role.",
    )

    with patch(
        "whisper_live.summarizer.urllib.request.urlopen", return_value=_Response()
    ) as urlopen:
        summarizer._request_summary("hello", 0, 60)

    payload = json.loads(urlopen.call_args.args[0].data)
    assert payload["system"] == build_system_prompt("Use a custom role.")


def test_template_discovery_and_default_selection(tmp_path):
    (tmp_path / "SUMMARIZER_TEMPLATE-Zebra.md").write_text("zebra", encoding="utf-8")
    (tmp_path / "SUMMARIZER_TEMPLATE-alpha-default.md").write_text(
        "default", encoding="utf-8"
    )
    (tmp_path / "SUMMARIZER_TEMPLATE-Beta.md").write_text("beta", encoding="utf-8")
    (tmp_path / "SUMMARIZER_TEMPLATE-MyDefault.md").write_text(
        "another default", encoding="utf-8"
    )
    (tmp_path / "STYLE_GUIDE.md").write_text("ignored", encoding="utf-8")
    (tmp_path / "SUMMARIZER_TEMPLATE-.md").write_text("ignored", encoding="utf-8")

    filenames = discover_summary_templates(tmp_path)
    assert filenames == [
        "SUMMARIZER_TEMPLATE-alpha-default.md",
        "SUMMARIZER_TEMPLATE-Beta.md",
        "SUMMARIZER_TEMPLATE-MyDefault.md",
        "SUMMARIZER_TEMPLATE-Zebra.md",
    ]
    assert resolve_summary_template(template_directory=tmp_path) == filenames[0]
    assert resolve_summary_template("SUMMARIZER_TEMPLATE-Beta.md", tmp_path) == filenames[1]


def test_explicit_template_is_the_complete_system_prompt(tmp_path):
    selected = "SUMMARIZER_TEMPLATE-Engineering.md"
    unselected = "SUMMARIZER_TEMPLATE-Sales.md"
    (tmp_path / selected).write_text("Use engineering terminology.", encoding="utf-8")
    (tmp_path / unselected).write_text("Use sales terminology.", encoding="utf-8")

    system_prompt = build_system_prompt(
        template=selected, template_directory=tmp_path
    )

    assert system_prompt == "Use engineering terminology."


def test_no_default_or_explicit_none_uses_no_system_prompt(tmp_path):
    template_name = "SUMMARIZER_TEMPLATE-group.md"
    (tmp_path / template_name).write_text("group instructions", encoding="utf-8")

    assert resolve_summary_template(template_directory=tmp_path) is None
    assert build_system_prompt(template="none", template_directory=tmp_path) == ""


def test_comment_only_template_produces_no_system_prompt(tmp_path):
    template_name = "SUMMARIZER_TEMPLATE-default.md"
    (tmp_path / template_name).write_text("<!-- placeholder -->\n", encoding="utf-8")

    assert build_system_prompt(template_directory=tmp_path) == ""


def test_custom_prompt_bypasses_missing_template_selection(tmp_path):
    assert (
        build_system_prompt(
            prompt="Custom instructions.",
            template="SUMMARIZER_TEMPLATE-unavailable.md",
            template_directory=tmp_path,
        )
        == "Custom instructions."
    )


def test_empty_custom_prompt_is_an_empty_system_prompt(tmp_path):
    assert build_system_prompt(prompt="", template_directory=tmp_path) == ""


def test_explicit_unknown_template_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown summarizer template"):
        resolve_summary_template("../arbitrary.md", tmp_path)


def test_missing_template_directory_is_allowed(tmp_path):
    missing = tmp_path / "missing"
    assert discover_summary_templates(missing) == []
    assert build_system_prompt(template_directory=missing) == ""


def test_summarize_srt_can_list_templates_without_transcripts(capsys):
    import summarize_srt

    assert summarize_srt.main(["--list-summary-templates"]) == 0
    output = capsys.readouterr().out
    assert "Automatic default: SUMMARIZER_TEMPLATE-default.md" in output
    assert "SUMMARIZER_TEMPLATE-default.md" in output


def test_summarize_srt_accepts_template_selection():
    from summarize_srt import _parse_args

    args = _parse_args(["meeting.srt", "--summary-template", "none"])
    assert args.summary_template == "none"


def test_summarize_srt_prompt_file_replaces_template_selection(tmp_path):
    from summarize_srt import _parse_args

    prompt_file = tmp_path / "custom-system.md"
    prompt_file.write_text("Complete custom instructions.", encoding="utf-8")

    args = _parse_args(
        [
            "meeting.srt",
            "--prompt-file",
            str(prompt_file),
            "--summary-template",
            "SUMMARIZER_TEMPLATE-not-installed.md",
        ]
    )

    assert args.prompt == "Complete custom instructions."