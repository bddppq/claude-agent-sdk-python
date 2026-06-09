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


def _run_shim(event):
    # Locate the synthesized shim command from --settings and run it with the
    # hook-event JSON on stdin, returning the parsed decision (win #3 chain).
    import subprocess
    settings_raw = argv[argv.index("--settings") + 1] if "--settings" in argv else "{}"
    settings = json.loads(settings_raw)
    hooks = settings.get("hooks", {})
    matchers = hooks.get(event.get("hook_event_name"), [])
    if not matchers:
        return {}
    command = matchers[0]["hooks"][0]["command"]
    proc = subprocess.run(command, shell=True, input=json.dumps(event),
                          capture_output=True, text=True, env=os.environ.copy())
    try:
        return json.loads(proc.stdout) if proc.stdout.strip() else {}
    except Exception:
        return {}


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
        if mode == "hooks":
            # Simulate the CLI firing PreToolUse + PostToolUse for a Write, run
            # the shim for each, and record the decisions where the test reads
            # them. Honors the PreToolUse permissionDecision: deny -> skip the
            # tool_result; allow -> use updatedInput if present.
            pre = _run_shim({"hook_event_name": "PreToolUse", "tool_name": "Write",
                             "tool_input": {"file_path": "/tmp/x.txt", "content": "ORIG"},
                             "tool_use_id": "tu_%d" % turn})
            with open(os.environ["FAKE_HOOK_OUT"], "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "PreToolUse", "decision": pre}) + "\n")
                f.flush()
            hso = pre.get("hookSpecificOutput", {}) if isinstance(pre, dict) else {}
            decision = hso.get("permissionDecision")
            eff_input = hso.get("updatedInput", {"file_path": "/tmp/x.txt", "content": "ORIG"})
            if decision != "deny":
                post = _run_shim({"hook_event_name": "PostToolUse", "tool_name": "Write",
                                  "tool_input": eff_input, "tool_response": {"ok": True},
                                  "tool_use_id": "tu_%d" % turn})
                with open(os.environ["FAKE_HOOK_OUT"], "a", encoding="utf-8") as f:
                    f.write(json.dumps({"event": "PostToolUse", "decision": post}) + "\n")
                    f.flush()
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
    # Where the "hooks" fake-CLI mode records the shim decisions for assertion.
    monkeypatch.setenv("FAKE_HOOK_OUT", str(tmp_path / "hook_decisions.jsonl"))
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
    async def test_hooks_option_is_accepted(self, monkeypatch, tmp_path):
        """Programmatic ``hooks`` are now SUPPORTED (win #3): the turn completes
        and the IPC bridge is wired (no CLIConnectionError)."""
        from claude_agent_sdk import AssistantMessage, HookMatcher

        async def _noop_hook(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {}

        options = _setup(monkeypatch, tmp_path)
        options = replace(
            options,
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_noop_hook])]},
        )

        messages = []
        with anyio.fail_after(30):
            async for msg in query(prompt="hi", options=options):
                messages.append(msg)

        # The turn completed normally (hooks no longer reject); the fake CLI does
        # not run the shim, so the callback is simply never invoked here.
        assert any(isinstance(m, AssistantMessage) for m in messages)

    @pytest.mark.anyio
    async def test_unsupported_option_still_rejected(self, monkeypatch, tmp_path):
        """A genuinely unsupported option still fails loudly through query()."""
        from claude_agent_sdk import CLIConnectionError

        options = _setup(monkeypatch, tmp_path)
        options = replace(options, permission_prompt_tool_name="my_tool")

        with (
            pytest.raises(CLIConnectionError, match="permission_prompt_tool_name"),
            anyio.fail_after(30),
        ):
            async for _ in query(prompt="hi", options=options):
                pass


def _read_hook_decisions(tmp_path: Path) -> list[dict]:
    import json as _json

    out = tmp_path / "hook_decisions.jsonl"
    if not out.exists():
        return []
    return [
        _json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestHookIpcBridge:
    """win #3: programmatic hooks + can_use_tool routed through the settings-hook
    IPC bridge end-to-end (transport -> synthesized --settings -> shim subprocess
    -> SDK IPC server -> callbacks -> decision back to the CLI).

    The fake CLI (``hooks`` mode) parses the synthesized ``--settings``, runs the
    shim for PreToolUse (and PostToolUse) with the injected
    ``CLAUDE_AGENT_SDK_HOOK_IPC`` env, and records the decisions for assertion.
    """

    @pytest.mark.anyio
    async def test_pretooluse_hook_fires_with_correct_shape(
        self, monkeypatch, tmp_path
    ):
        from claude_agent_sdk import HookMatcher

        seen: list[tuple] = []

        async def _hook(inp, tuid, ctx):  # noqa: ANN001, ANN202
            seen.append(
                (
                    inp.get("hook_event_name"),
                    inp.get("tool_name"),
                    inp.get("tool_input"),
                    tuid,
                )
            )
            return {}

        options = _setup(monkeypatch, tmp_path, mode="hooks")
        options = replace(
            options,
            hooks={"PreToolUse": [HookMatcher(matcher="Write", hooks=[_hook])]},
        )

        with anyio.fail_after(30):
            async for _ in query(prompt="go", options=options):
                pass

        assert len(seen) == 1
        event, tool, tinput, tuid = seen[0]
        assert event == "PreToolUse"
        assert tool == "Write"
        assert tinput == {"file_path": "/tmp/x.txt", "content": "ORIG"}
        assert tuid == "tu_1"

    @pytest.mark.anyio
    async def test_pretooluse_hook_can_deny(self, monkeypatch, tmp_path):
        from claude_agent_sdk import HookMatcher

        async def _deny(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "nope",
                }
            }

        options = _setup(monkeypatch, tmp_path, mode="hooks")
        options = replace(
            options, hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_deny])]}
        )

        with anyio.fail_after(30):
            async for _ in query(prompt="go", options=options):
                pass

        decisions = _read_hook_decisions(tmp_path)
        pre = [d for d in decisions if d["event"] == "PreToolUse"]
        post = [d for d in decisions if d["event"] == "PostToolUse"]
        assert len(pre) == 1
        assert pre[0]["decision"]["hookSpecificOutput"]["permissionDecision"] == "deny"
        # deny short-circuits the tool -> PostToolUse never runs.
        assert post == []

    @pytest.mark.anyio
    async def test_can_use_tool_deny_via_hook(self, monkeypatch, tmp_path):
        from claude_agent_sdk import PermissionResultDeny

        called: list[tuple] = []

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            called.append((name, inp, ctx.tool_use_id))
            return PermissionResultDeny(message="blocked by callback")

        options = _setup(monkeypatch, tmp_path, mode="hooks")
        options = replace(options, can_use_tool=_can_use)

        with anyio.fail_after(30):
            async with ClaudeSDKClient(options=options) as client:
                await client.query("go")
                async for _ in client.receive_response():
                    pass

        assert len(called) == 1
        name, inp, tuid = called[0]
        assert name == "Write"
        assert inp == {"file_path": "/tmp/x.txt", "content": "ORIG"}
        assert tuid == "tu_1"
        decisions = _read_hook_decisions(tmp_path)
        pre = [d for d in decisions if d["event"] == "PreToolUse"][0]
        hso = pre["decision"]["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert hso["permissionDecisionReason"] == "blocked by callback"

    @pytest.mark.anyio
    async def test_can_use_tool_allow_with_updated_input(self, monkeypatch, tmp_path):
        from claude_agent_sdk import PermissionResultAllow

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            new = dict(inp)
            new["content"] = "REWRITTEN"
            return PermissionResultAllow(updated_input=new)

        options = _setup(monkeypatch, tmp_path, mode="hooks")
        options = replace(options, can_use_tool=_can_use)

        with anyio.fail_after(30):
            async with ClaudeSDKClient(options=options) as client:
                await client.query("go")
                async for _ in client.receive_response():
                    pass

        decisions = _read_hook_decisions(tmp_path)
        pre = [d for d in decisions if d["event"] == "PreToolUse"][0]
        hso = pre["decision"]["hookSpecificOutput"]
        assert hso["permissionDecision"] == "allow"
        # The headline updated_input fidelity: the executed input is rewritten.
        assert hso["updatedInput"]["content"] == "REWRITTEN"
        # PostToolUse then runs with the REWRITTEN input (the CLI applies it).
        post = [d for d in decisions if d["event"] == "PostToolUse"]
        assert post  # tool was allowed -> PostToolUse fired

    @pytest.mark.anyio
    async def test_posttooluse_hook_fires_with_response(self, monkeypatch, tmp_path):
        from claude_agent_sdk import HookMatcher

        seen: list[tuple] = []

        async def _post(inp, tuid, ctx):  # noqa: ANN001, ANN202
            seen.append(
                (
                    inp.get("hook_event_name"),
                    inp.get("tool_name"),
                    inp.get("tool_response"),
                )
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "noted",
                }
            }

        options = _setup(monkeypatch, tmp_path, mode="hooks")
        options = replace(
            options, hooks={"PostToolUse": [HookMatcher(matcher=None, hooks=[_post])]}
        )

        with anyio.fail_after(30):
            async for _ in query(prompt="go", options=options):
                pass

        assert len(seen) == 1
        event, tool, response = seen[0]
        assert event == "PostToolUse"
        assert tool == "Write"
        assert response == {"ok": True}
