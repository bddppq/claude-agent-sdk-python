"""End-to-end integration tests for the interactive PTY transport.

These spawn the **real** :class:`PtyCLITransport` (PTY allocation, keystroke
injection, transcript tailing, message translation) against a *fake* ``claude``
binary, then drive it through the public ``query()`` / ``ClaudeSDKClient`` APIs.
No authentication or real model is required.

The fake CLI:
  * reads keystrokes from its PTY stdin (the prompt the transport types),
  * appends ``user`` / ``assistant`` / ``turn_duration`` records to the session
    transcript at the path the transport tails (passed via ``FAKE_TRANSCRIPT``),
  * exits on stdin EOF.

This proves the new transport is a drop-in replacement: the same public API
surfaces the same ``AssistantMessage`` / ``ResultMessage`` / ``UserMessage``
objects it always has, sourced from the interactive transcript instead of the
former ``stream-json`` pipe.
"""

from __future__ import annotations

import os
import stat
import uuid
from dataclasses import replace
from pathlib import Path

import anyio
import pytest

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    UserMessage,
    query,
)
from claude_agent_sdk._internal.sessions import (
    _canonicalize_path,
    _get_projects_dir,
    _sanitize_path,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="PtyCLITransport requires a POSIX platform"
)

# A self-contained fake ``claude``: reads typed prompts from the PTY and writes
# the transcript records the transport tails. Behavior is steered by env vars so
# one script covers several scenarios.
_FAKE_CLI = r"""#!/usr/bin/env python3
import os, sys, json

argv = sys.argv[1:]
sid = argv[argv.index("--session-id") + 1] if "--session-id" in argv else "sess"
mode = os.environ.get("FAKE_MODE", "echo")

if mode == "exit-error":
    sys.stderr.write("fatal: simulated startup failure\n")
    sys.stderr.flush()
    sys.exit(3)

if mode == "stderr":
    # Write a couple of diagnostic lines to fd 2 before behaving like echo, so
    # the transport's dedicated stderr pipe (H3) can deliver them to the
    # options.stderr callback.
    os.write(2, b"stderr line one\n")
    os.write(2, b"stderr line two\n")

path = os.environ["FAKE_TRANSCRIPT"]
os.makedirs(os.path.dirname(path), exist_ok=True)


def append(entry):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


buf = b""
turn = 0
while True:
    chunk = os.read(0, 1)
    if not chunk:
        break
    if chunk == b"\r":
        # Only a carriage return submits. Newlines inside a bracketed paste are
        # literal content, and the guards are consumed like a real terminal.
        line = buf.decode("utf-8", "replace")
        line = line.replace("\x1b[200~", "").replace("\x1b[201~", "")
        buf = b""
        if not line.strip():  # the empty warmup-dismiss Enter
            continue
        turn += 1
        append({"type": "user", "sessionId": sid, "uuid": "u%d" % turn,
                "message": {"role": "user", "content": line}})
        if mode == "tool":
            append({"type": "user", "sessionId": sid, "uuid": "tr%d" % turn,
                    "message": {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "t1",
                         "content": "tool-output"}]}})
        append({"type": "assistant", "sessionId": sid, "uuid": "a%d" % turn,
                "message": {"role": "assistant", "model": "fake-model",
                            "content": [{"type": "text", "text": "echo:" + line}],
                            "stop_reason": "end_turn"}})
        append({"type": "system", "subtype": "turn_duration", "sessionId": sid,
                "uuid": "s%d" % turn, "durationMs": 5, "messageCount": 2})
    else:
        buf += chunk
"""


def _setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, mode: str = "echo"
) -> ClaudeAgentOptions:
    """Create the fake CLI + dirs and return wired ``ClaudeAgentOptions``."""
    fake = tmp_path / "fake_claude"
    fake.write_text(_FAKE_CLI)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    cwd = tmp_path / "work"
    cwd.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    session_id = str(uuid.uuid4())

    env = {"CLAUDE_CONFIG_DIR": str(config_dir)}
    options = ClaudeAgentOptions(
        cli_path=str(fake), cwd=str(cwd), env=env, session_id=session_id
    )

    # The path the transport will tail; the fake writes there.
    transcript = (
        _get_projects_dir(env_override=env)
        / _sanitize_path(_canonicalize_path(str(cwd)))
        / f"{session_id}.jsonl"
    )
    monkeypatch.setenv("FAKE_TRANSCRIPT", str(transcript))
    monkeypatch.setenv("FAKE_MODE", mode)
    # Skip the version-check subprocess and the multi-second TUI warmup.
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    monkeypatch.setattr(
        "claude_agent_sdk._internal.transport.pty_cli._WARMUP_SECONDS", 0.0
    )
    return options


class TestQueryOneShot:
    @pytest.mark.anyio
    async def test_query_surfaces_assistant_and_result(self, monkeypatch, tmp_path):
        options = _setup(monkeypatch, tmp_path)

        messages = []
        with anyio.fail_after(30):
            async for msg in query(prompt="hello world", options=options):
                messages.append(msg)

        assistants = [m for m in messages if isinstance(m, AssistantMessage)]
        results = [m for m in messages if isinstance(m, ResultMessage)]

        assert len(assistants) == 1
        assert assistants[0].model == "fake-model"
        assert isinstance(assistants[0].content[0], TextBlock)
        assert assistants[0].content[0].text == "echo:hello world"

        assert len(results) == 1
        assert results[0].subtype == "success"
        assert results[0].is_error is False

    @pytest.mark.anyio
    async def test_plain_text_user_prompt_is_not_echoed(self, monkeypatch, tmp_path):
        options = _setup(monkeypatch, tmp_path)

        with anyio.fail_after(30):
            messages = [m async for m in query(prompt="just text", options=options)]

        # The transcript's plain-text user record (our own typed prompt) must not
        # surface as a UserMessage; only assistant + result do.
        assert not any(isinstance(m, UserMessage) for m in messages)
        assert any(isinstance(m, AssistantMessage) for m in messages)
        assert any(isinstance(m, ResultMessage) for m in messages)

    @pytest.mark.anyio
    async def test_tool_result_user_record_is_surfaced(self, monkeypatch, tmp_path):
        options = _setup(monkeypatch, tmp_path, mode="tool")

        with anyio.fail_after(30):
            messages = [m async for m in query(prompt="use a tool", options=options)]

        users = [m for m in messages if isinstance(m, UserMessage)]
        assert len(users) == 1
        assert isinstance(users[0].content[0], ToolResultBlock)
        assert users[0].content[0].content == "tool-output"

    @pytest.mark.anyio
    async def test_early_exit_surfaces_error_result(self, monkeypatch, tmp_path):
        options = _setup(monkeypatch, tmp_path, mode="exit-error")

        with anyio.fail_after(30):
            messages = [m async for m in query(prompt="will fail", options=options)]

        results = [m for m in messages if isinstance(m, ResultMessage)]
        assert len(results) == 1
        assert results[0].is_error is True


class TestStderrCallback:
    @pytest.mark.anyio
    async def test_stderr_lines_delivered_to_callback(self, monkeypatch, tmp_path):
        """The child's stderr (on its own pipe, H3) reaches options.stderr per
        line, while stdout stays on the PTY and the turn still completes."""
        options = _setup(monkeypatch, tmp_path, mode="stderr")
        lines: list[str] = []
        options = replace(options, stderr=lines.append)

        results = []
        with anyio.fail_after(30):
            async for msg in query(prompt="hi", options=options):
                if isinstance(msg, ResultMessage):
                    results.append(msg)

        # The turn completed (stdout/transcript path still works)...
        assert len(results) == 1
        # ...and the child's stderr lines were delivered, in order, stripped.
        assert "stderr line one" in lines
        assert "stderr line two" in lines
        assert lines.index("stderr line one") < lines.index("stderr line two")


class TestStreamingClientMultiTurn:
    @pytest.mark.anyio
    async def test_multi_turn_session(self, monkeypatch, tmp_path):
        options = _setup(monkeypatch, tmp_path)

        replies: list[str] = []
        with anyio.fail_after(45):
            async with ClaudeSDKClient(options=options) as client:
                for prompt in ("first prompt", "second prompt"):
                    await client.query(prompt)
                    async for msg in client.receive_response():
                        if isinstance(msg, AssistantMessage):
                            for block in msg.content:
                                if isinstance(block, TextBlock):
                                    replies.append(block.text)

        assert replies == ["echo:first prompt", "echo:second prompt"]


class TestPromptFidelity:
    @pytest.mark.anyio
    async def test_newline_and_leading_slash_round_trip(self, monkeypatch, tmp_path):
        """A multi-line prompt with a leading '/' reaches the CLI with newlines
        preserved (only a single leading space is added so the TUI does not
        enter command mode)."""
        options = _setup(monkeypatch, tmp_path)
        prompt = "/keep this line one\nline two"

        captured: list[str] = []
        with anyio.fail_after(30):
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            captured.append(block.text)

        # The fake echoes the exact line it received.
        assert captured == ["echo: /keep this line one\nline two"]


class TestOptionValidation:
    @pytest.mark.anyio
    async def test_hooks_option_is_rejected(self, monkeypatch, tmp_path):
        """Unsupported options fail loudly through the public query() API rather
        than hanging or silently no-op-ing."""
        from claude_agent_sdk import CLIConnectionError

        options = _setup(monkeypatch, tmp_path)
        options = replace(options, hooks={"PreToolUse": [{"hooks": [lambda *a: None]}]})

        with pytest.raises(CLIConnectionError, match="hooks"), anyio.fail_after(30):
            async for _ in query(prompt="hi", options=options):
                pass
