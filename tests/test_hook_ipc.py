"""Unit tests for the settings-hook IPC bridge (win #3).

Cover the IPC server's dispatch logic, the settings synthesis, matcher
semantics, and the shim's fail-open behavior -- without spawning the real CLI.
``test_pty_integration.py::TestHookIpcBridge`` exercises the full transport
chain end-to-end against a fake CLI that actually runs the shim.
"""

from __future__ import annotations

import json
import os
import subprocess

import anyio
import pytest

from claude_agent_sdk._internal.transport._hook_ipc import (
    ENV_HOOK_IPC,
    HookIpcServer,
    build_hooks_settings,
    shim_command,
)
from claude_agent_sdk.types import (
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
)

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------------- #
# Matcher semantics (mirrors the CLI's matchesPattern)
# --------------------------------------------------------------------------- #


class TestMatcherMatches:
    @pytest.mark.parametrize(
        ("pattern", "tool", "expected"),
        [
            (None, "Write", True),
            ("", "Write", True),
            ("*", "Write", True),
            ("Write", "Write", True),
            ("Write", "Read", False),
            ("Write|Edit", "Edit", True),
            ("Write|Edit", "Bash", False),
            ("^Write$", "Write", True),
            (".*Tool", "MyTool", True),
            ("Write", None, False),  # non-tool event, named matcher -> no match
        ],
    )
    def test_matches(self, pattern, tool, expected):
        assert HookIpcServer._matcher_matches(pattern, tool) is expected

    def test_invalid_regex_falls_back_to_exact(self):
        # An unbalanced group is not valid regex -> exact compare.
        assert HookIpcServer._matcher_matches("(unclosed", "(unclosed") is True
        assert HookIpcServer._matcher_matches("(unclosed", "Write") is False


# --------------------------------------------------------------------------- #
# Settings synthesis
# --------------------------------------------------------------------------- #


class TestBuildHooksSettings:
    async def test_preserves_user_matchers_and_wires_shim(self):
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {}

        server = HookIpcServer(
            hooks={
                "PreToolUse": [HookMatcher(matcher="Write|Edit", hooks=[_h])],
                "PostToolUse": [HookMatcher(matcher=None, hooks=[_h])],
            },
            can_use_tool=None,
        )
        block = build_hooks_settings(server)
        assert set(block) == {"PreToolUse", "PostToolUse"}
        assert block["PreToolUse"][0]["matcher"] == "Write|Edit"
        # PostToolUse matcher=None -> no matcher key (catch-all).
        assert "matcher" not in block["PostToolUse"][0]
        cmd = block["PreToolUse"][0]["hooks"][0]
        assert cmd["type"] == "command"
        assert "_hook_shim" in cmd["command"]

    async def test_can_use_tool_adds_catchall_pretooluse(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultAllow()

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        block = build_hooks_settings(server)
        assert list(block) == ["PreToolUse"]
        assert "matcher" not in block["PreToolUse"][0]

    async def test_can_use_tool_reuses_existing_pretooluse_entry(self):
        # If the user configured PreToolUse hooks, the shim is already wired for
        # that event (one shim call dispatches BOTH user hooks AND can_use_tool),
        # so no extra catch-all entry is added.
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {}

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultAllow()

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher="Write", hooks=[_h])]},
            can_use_tool=_can_use,
        )
        block = build_hooks_settings(server)
        assert len(block["PreToolUse"]) == 1
        assert block["PreToolUse"][0]["matcher"] == "Write"

    async def test_timeout_seconds_emitted(self):
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {}

        server = HookIpcServer(
            hooks={"Stop": [HookMatcher(matcher=None, hooks=[_h])]}, can_use_tool=None
        )
        block = build_hooks_settings(server, timeout_seconds=30)
        assert block["Stop"][0]["hooks"][0]["timeout"] == 30


# --------------------------------------------------------------------------- #
# Dispatch (in-process, no shim subprocess)
# --------------------------------------------------------------------------- #


class TestDispatch:
    async def test_user_hook_fires_with_shape(self):
        seen = []

        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            seen.append((inp, tuid, ctx))
            return {}

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher="Write", hooks=[_h])]},
            can_use_tool=None,
        )
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/a", "content": "x"},
                "tool_use_id": "tu_1",
            }
        )
        assert out == {}
        assert len(seen) == 1
        inp, tuid, ctx = seen[0]
        assert inp["tool_name"] == "Write"
        assert tuid == "tu_1"
        assert ctx == {"signal": None}

    async def test_user_hook_matcher_filters(self):
        seen = []

        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            seen.append(inp["tool_name"])
            return {}

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher="Edit", hooks=[_h])]},
            can_use_tool=None,
        )
        await server._dispatch(
            {"hook_event_name": "PreToolUse", "tool_name": "Write", "tool_input": {}}
        )
        assert seen == []  # Write does not match "Edit"

    async def test_async_continue_field_conversion(self):
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {"continue_": False, "stopReason": "halt"}

        server = HookIpcServer(
            hooks={"Stop": [HookMatcher(matcher=None, hooks=[_h])]}, can_use_tool=None
        )
        out = await server._dispatch({"hook_event_name": "Stop"})
        assert out["continue"] is False
        assert "continue_" not in out
        assert out["stopReason"] == "halt"

    async def test_can_use_tool_allow_with_updated_input(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            new = dict(inp)
            new["content"] = "REWRITTEN"
            return PermissionResultAllow(updated_input=new)

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/a", "content": "ORIG"},
                "tool_use_id": "tu_1",
            }
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "allow"
        assert hso["updatedInput"]["content"] == "REWRITTEN"

    async def test_can_use_tool_deny_with_message(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultDeny(message="nope")

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        out = await server._dispatch(
            {"hook_event_name": "PreToolUse", "tool_name": "Write", "tool_input": {}}
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert hso["permissionDecisionReason"] == "nope"

    async def test_can_use_tool_raise_denies(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            raise RuntimeError("boom")

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        out = await server._dispatch(
            {"hook_event_name": "PreToolUse", "tool_name": "Write", "tool_input": {}}
        )
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_permission_decision_sink_fires_on_deny(self):
        decisions = []

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultDeny(message="x")

        server = HookIpcServer(
            hooks=None,
            can_use_tool=_can_use,
            on_permission_decision=lambda n, d, t: decisions.append((n, d, t)),
        )
        await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {},
                "tool_use_id": "tu_9",
            }
        )
        assert decisions == [("Write", "deny", "tu_9")]

    async def test_user_hook_and_can_use_tool_both_dispatch(self):
        # A user PreToolUse hook + can_use_tool both run on the same event; the
        # can_use_tool permission decision lands in hookSpecificOutput while the
        # user hook's non-permission fields are preserved.
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {"systemMessage": "from-user-hook"}

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultDeny(message="denied")

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_h])]},
            can_use_tool=_can_use,
        )
        out = await server._dispatch(
            {"hook_event_name": "PreToolUse", "tool_name": "Write", "tool_input": {}}
        )
        assert out["systemMessage"] == "from-user-hook"
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


# --------------------------------------------------------------------------- #
# Server lifecycle + real round-trip via the shim subprocess
# --------------------------------------------------------------------------- #


def _run_shim(spec: str, event: dict) -> str:
    env = dict(os.environ)
    env[ENV_HOOK_IPC] = spec
    proc = subprocess.run(
        shim_command(),
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return proc.stdout


class TestServerLifecycleAndShim:
    async def test_start_exposes_spec_and_stop_is_clean(self):
        server = HookIpcServer(hooks=None, can_use_tool=None)
        # Nothing to wire -> the transport would not start it, but the server
        # itself still binds when asked (defensive).
        await server.start()
        try:
            assert server.spec  # tcp:... or unix:...
        finally:
            await server.stop()
        # Double-stop is safe.
        await server.stop()

    async def test_real_shim_round_trip(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            new = dict(inp)
            new["content"] = "REWRITTEN"
            return PermissionResultAllow(updated_input=new)

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        await server.start()
        try:
            out = await anyio.to_thread.run_sync(
                _run_shim,
                server.spec,
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Write",
                    "tool_input": {"file_path": "/a", "content": "ORIG"},
                    "tool_use_id": "tu_1",
                },
            )
        finally:
            await server.stop()
        parsed = json.loads(out)
        assert parsed["hookSpecificOutput"]["updatedInput"]["content"] == "REWRITTEN"

    async def test_shim_bad_token_returns_noop(self):
        server = HookIpcServer(hooks=None, can_use_tool=lambda *a: None)
        await server.start()
        try:
            bad = server.spec.rsplit(":", 1)[0] + ":deadbeef"
            out = await anyio.to_thread.run_sync(
                _run_shim,
                bad,
                {"hook_event_name": "PreToolUse", "tool_name": "Write"},
            )
        finally:
            await server.stop()
        assert json.loads(out) == {}

    async def test_shim_no_endpoint_fails_open(self):
        # No CLAUDE_AGENT_SDK_HOOK_IPC set -> shim emits `{}` and exits 0.
        env = dict(os.environ)
        env.pop(ENV_HOOK_IPC, None)
        proc = await anyio.to_thread.run_sync(
            lambda: subprocess.run(  # noqa: S603
                shim_command(),
                input="{}",
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
        )
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {}
