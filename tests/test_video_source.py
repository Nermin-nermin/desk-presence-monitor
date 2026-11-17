"""Credential handling: env expansion, .env loading, redaction."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_source import load_env_file, redact, resolve_source  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("rtsp://admin:S3cr3t!x@192.168.1.64:554/Streaming/Channels/102", "rtsp://***@192.168.1.64:554/Streaming/Channels/102"),
    ("rtsp://admin@10.0.0.5/stream", "rtsp://***@10.0.0.5/stream"),
    ("Could not open rtsp://u:p@cam/1 and http://x:y@h/z", "Could not open rtsp://***@cam/1 and http://***@h/z"),
    ("rtsp://10.0.0.5/stream", "rtsp://10.0.0.5/stream"),
    ("office.mp4", "office.mp4"),
])
def test_redact(raw, expected):
    assert redact(raw) == expected


def test_env_file_and_resolve(tmp_path, monkeypatch):
    monkeypatch.delenv("CAM_TEST_URL", raising=False)
    env = tmp_path / ".env"
    env.write_text('# cameras\nexport CAM_TEST_URL="rtsp://admin:pw@10.1.1.1/live"\n\n')
    load_env_file(env)
    assert resolve_source("${CAM_TEST_URL}") == "rtsp://admin:pw@10.1.1.1/live"
    assert resolve_source("0") == 0


def test_existing_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_TEST_URL", "from-environment")
    (tmp_path / ".env").write_text("CAM_TEST_URL=from-file\n")
    load_env_file(tmp_path / ".env")
    assert os.environ["CAM_TEST_URL"] == "from-environment"


def test_undefined_variable_exits_without_leaking(monkeypatch):
    monkeypatch.delenv("NOPE_URL", raising=False)
    with pytest.raises(SystemExit) as e:
        resolve_source("${NOPE_URL}")
    assert "NOPE_URL" in str(e.value)
