"""Assert-based unit tests for STT LLM post-processing and context helpers.

Run directly:  python tests/test_stt.py
Or via pytest: python -m pytest tests/test_stt.py -q
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stt import (
    DEFAULT_LLM_SYSTEM_PROMPT,
    _get_ark_api_key,
    _get_hermes_chat_context,
    _llm_postprocess,
)


def test_get_ark_api_key_from_config():
    assert _get_ark_api_key({"ark_api_key": "test-key-123"}) == "test-key-123"


def test_get_ark_api_key_from_env(monkeypatch=None):
    orig = os.environ.get("VOLCENGINE_ARK_API_KEY")
    try:
        os.environ["VOLCENGINE_ARK_API_KEY"] = "env-key-456"
        assert _get_ark_api_key({}) == "env-key-456"
    finally:
        if orig is None:
            os.environ.pop("VOLCENGINE_ARK_API_KEY", None)
        else:
            os.environ["VOLCENGINE_ARK_API_KEY"] = orig


def test_default_prompt_integrity():
    assert "数字" in DEFAULT_LLM_SYSTEM_PROMPT
    assert "口吃" in DEFAULT_LLM_SYSTEM_PROMPT or "重复词" in DEFAULT_LLM_SYSTEM_PROMPT


def test_llm_postprocess_empty():
    assert _llm_postprocess("", {}, "fake-key") == ""
    assert _llm_postprocess("   ", {}, "fake-key") == "   "


def test_llm_postprocess_fallback_on_invalid_endpoint():
    # When endpoint is unreachable, it should gracefully fall back to raw transcript
    raw = "今天天气真好"
    cfg = {"ark_endpoint": "http://127.0.0.1:59999/invalid", "llm_timeout": 0.2}
    assert _llm_postprocess(raw, cfg, "fake-key") == raw


def test_chat_context_safely_handles_missing_db():
    # Should return empty string without raising exceptions
    orig = os.environ.get("HERMES_HOME")
    try:
        os.environ["HERMES_HOME"] = "/nonexistent/directory"
        res = _get_hermes_chat_context()
        assert isinstance(res, str)
    finally:
        if orig is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = orig


if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and isinstance(fn, types.FunctionType)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed.")
