"""Tests for the PTY-based interactive CLI transport.

These cover the transport's logic without a real CLI: transcript translation,
command construction, env handling, keystroke encoding, the transcript tail
loop, and the control-request -> interactive-keystroke mappings. A full
end-to-end run against a fake ``claude`` lives in ``test_pty_integration.py``.
"""

import json
from unittest.mock import patch

import anyio
import pytest

from claude_agent_sdk._internal.transport import pty_cli
from claude_agent_sdk._internal.transport.pty_cli import (
    _SHIFT_TAB,
    PtyCLITransport,
    _sanitize_assistant_message,
    _translate_transcript_entry,
)
from claude_agent_sdk.types import ClaudeAgentOptions

DEFAULT_CLI = "/usr/bin/claude"


def make_transport(**kwargs: object) -> PtyCLITransport:
    cli_path = kwargs.pop("cli_path", DEFAULT_CLI)
    options = ClaudeAgentOptions(cli_path=cli_path, **kwargs)
    return PtyCLITransport(prompt="hi", options=options)


# --------------------------------------------------------------------------- #
# Transcript translation
# --------------------------------------------------------------------------- #


class TestTranslateTranscriptEntry:
    def test_assistant_entry_passes_message_through(self):
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "pong"}],
            },
            "sessionId": "s-1",
            "uuid": "u-1",
        }
        out = _translate_transcript_entry(entry, "fallback")
        assert out is not None
        assert out["type"] == "assistant"
        assert out["session_id"] == "s-1"
        assert out["uuid"] == "u-1"
        assert out["message"]["content"][0]["text"] == "pong"

    def test_user_plain_text_prompt_is_skipped(self):
        # Plain-text user records are the echo of what we typed; not surfaced.
        entry = {
            "type": "user",
            "message": {"role": "user", "content": "reply with pong"},
            "sessionId": "s-1",
        }
        assert _translate_transcript_entry(entry, "fallback") is None

    def test_user_tool_result_is_surfaced(self):
        entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ],
            },
            "sessionId": "s-1",
            "uuid": "u-2",
        }
        out = _translate_transcript_entry(entry, "fallback")
        assert out is not None
        assert out["type"] == "user"
        assert out["message"]["content"][0]["type"] == "tool_result"

    def test_user_text_block_list_without_tool_result_is_skipped(self):
        entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            },
        }
        assert _translate_transcript_entry(entry, "fallback") is None

    def test_turn_duration_synthesizes_result(self):
        entry = {
            "type": "system",
            "subtype": "turn_duration",
            "durationMs": 1234,
            "messageCount": 4,
            "sessionId": "s-1",
            "uuid": "u-3",
        }
        out = _translate_transcript_entry(entry, "fallback")
        assert out is not None
        assert out["type"] == "result"
        assert out["subtype"] == "success"
        assert out["is_error"] is False
        assert out["duration_ms"] == 1234
        assert out["num_turns"] == 4
        assert out["session_id"] == "s-1"

    @pytest.mark.parametrize(
        "entry_type",
        ["queue-operation", "last-prompt", "ai-title", "attachment", "mode"],
    )
    def test_bookkeeping_types_translate_to_none(self, entry_type):
        assert _translate_transcript_entry({"type": entry_type}, "fallback") is None

    def test_falls_back_to_default_session_id(self):
        entry = {
            "type": "assistant",
            "message": {"role": "assistant", "model": "m", "content": []},
        }
        out = _translate_transcript_entry(entry, "fallback-sid")
        assert out is not None
        assert out["session_id"] == "fallback-sid"


class TestSanitizeAssistantMessage:
    def test_backfills_missing_thinking_signature(self):
        message = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "hmm"}],
        }
        out = _sanitize_assistant_message(message)
        assert out["content"][0]["signature"] == ""

    def test_preserves_existing_signature(self):
        message = {
            "content": [{"type": "thinking", "thinking": "x", "signature": "sig"}],
        }
        out = _sanitize_assistant_message(message)
        assert out["content"][0]["signature"] == "sig"


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #


class TestBuildCommand:
    def test_strips_stream_json_io_flags(self):
        t = make_transport()
        t._cli_path = DEFAULT_CLI
        cmd = t._build_command()
        # Interactive mode must not carry the headless stream-json protocol.
        assert "--input-format" not in cmd
        assert "--output-format" not in cmd
        assert "--verbose" not in cmd
        assert "stream-json" not in cmd
        assert "--print" not in cmd
        assert "-p" not in cmd

    def test_includes_session_id_for_transcript_discovery(self):
        t = make_transport()
        t._cli_path = DEFAULT_CLI
        cmd = t._build_command()
        assert "--session-id" in cmd
        assert cmd[cmd.index("--session-id") + 1] == t._session_id

    def test_forwards_common_options(self):
        t = make_transport(model="claude-sonnet-4-6", permission_mode="acceptEdits")
        t._cli_path = DEFAULT_CLI
        cmd = t._build_command()
        assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-6"
        assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"


class TestComputeTranscriptPath:
    def test_path_ends_with_session_jsonl(self):
        t = make_transport(session_id="11111111-1111-1111-1111-111111111111")
        path = t._compute_transcript_path()
        assert path.name == "11111111-1111-1111-1111-111111111111.jsonl"
        assert "projects" in path.parts


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


class TestBuildEnv:
    def test_sets_pty_entrypoint_and_filters_claudecode(self):
        with patch.dict("os.environ", {"CLAUDECODE": "1"}):
            env = make_transport()._build_env()
        assert env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py-pty"
        assert "CLAUDECODE" not in env
        assert "CLAUDE_AGENT_SDK_VERSION" in env

    def test_normalizes_is_sandbox_to_one_when_root(self):
        with (
            patch("os.geteuid", return_value=0),
            patch.dict("os.environ", {"IS_SANDBOX": "yes"}),
        ):
            env = make_transport()._build_env()
        # The CLI only accepts the exact value "1".
        assert env["IS_SANDBOX"] == "1"

    def test_caller_can_override_is_sandbox(self):
        with patch("os.geteuid", return_value=0):
            env = make_transport(env={"IS_SANDBOX": "custom"})._build_env()
        assert env["IS_SANDBOX"] == "custom"


# --------------------------------------------------------------------------- #
# Transport interface behavior (no real subprocess)
# --------------------------------------------------------------------------- #


class TestWriteRouting:
    def test_control_request_enqueues_success_response(self):
        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t.write(
                json.dumps(
                    {
                        "type": "control_request",
                        "request_id": "req-7",
                        "request": {"subtype": "initialize"},
                    }
                )
                + "\n"
            )
            msg = t._out_recv.receive_nowait()
            assert msg["type"] == "control_response"
            assert msg["response"]["subtype"] == "success"
            assert msg["response"]["request_id"] == "req-7"

        anyio.run(_test)

    def test_user_message_is_typed_into_pty(self):
        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            typed: list[str] = []

            async def fake_type(text: str) -> None:
                typed.append(text)

            t._type_prompt = fake_type  # type: ignore[method-assign]
            await t.write(
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": "hello there"},
                    }
                )
                + "\n"
            )
            assert typed == ["hello there"]

        anyio.run(_test)

    def test_interrupt_writes_escape(self):
        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            written: list[bytes] = []

            async def fake_pty_write(data: bytes) -> None:
                written.append(data)

            t._pty_write = fake_pty_write  # type: ignore[method-assign]
            await t.write(
                json.dumps(
                    {
                        "type": "control_request",
                        "request_id": "r",
                        "request": {"subtype": "interrupt"},
                    }
                )
                + "\n"
            )
            assert b"\x1b" in written

        anyio.run(_test)


class TestEndInput:
    def test_end_input_sets_flag(self):
        async def _test():
            t = make_transport()
            assert t._input_ended is False
            await t.end_input()
            assert t._input_ended is True

        anyio.run(_test)


class TestTypePromptCollapsesNewlines:
    def test_newlines_become_spaces_then_submit(self):
        async def _test():
            t = make_transport()
            t._warmed_up = True  # skip the startup-toast warmup delay
            writes: list[bytes] = []

            async def fake_pty_write(data: bytes) -> None:
                writes.append(data)

            t._pty_write = fake_pty_write  # type: ignore[method-assign]
            await t._type_prompt("line one\nline two")
            assert writes[0] == b"line one line two"
            assert writes[-1] == b"\r"

        anyio.run(_test)


# --------------------------------------------------------------------------- #
# Transcript tail loop (no subprocess) — feed a file, collect SDK messages
# --------------------------------------------------------------------------- #


def _drain(transport: PtyCLITransport) -> list[dict]:
    out: list[dict] = []
    while True:
        try:
            out.append(transport._out_recv.receive_nowait())
        except anyio.WouldBlock:
            break
        except anyio.EndOfStream:
            break
    return out


class TestTailLoop:
    def _run_with_transcript(self, tmp_path, lines: list[dict]) -> list[dict]:
        async def _test() -> list[dict]:
            t = make_transport()
            path = tmp_path / "sess.jsonl"
            path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            # One-shot: stop once a result is emitted.
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        return anyio.run(_test)

    def test_assistant_and_result_emitted(self, tmp_path):
        msgs = self._run_with_transcript(
            tmp_path,
            [
                {
                    "type": "user",
                    "sessionId": "s",
                    "message": {"role": "user", "content": "hi"},
                },
                {
                    "type": "assistant",
                    "sessionId": "s",
                    "uuid": "a1",
                    "message": {
                        "role": "assistant",
                        "model": "m",
                        "content": [{"type": "text", "text": "yo"}],
                    },
                },
                {
                    "type": "system",
                    "subtype": "turn_duration",
                    "sessionId": "s",
                    "durationMs": 3,
                },
            ],
        )
        types = [m["type"] for m in msgs]
        # plain-text user record is dropped; assistant + synthesized result remain
        assert types == ["assistant", "result"]
        assert msgs[0]["message"]["content"][0]["text"] == "yo"
        assert msgs[1]["subtype"] == "success"

    def test_skips_bookkeeping_records(self, tmp_path):
        msgs = self._run_with_transcript(
            tmp_path,
            [
                {"type": "queue-operation", "sessionId": "s"},
                {"type": "file-history-snapshot", "sessionId": "s"},
                {
                    "type": "assistant",
                    "sessionId": "s",
                    "uuid": "a1",
                    "message": {
                        "role": "assistant",
                        "model": "m",
                        "content": [{"type": "text", "text": "ok"}],
                    },
                },
                {"type": "system", "subtype": "turn_duration", "sessionId": "s"},
            ],
        )
        assert [m["type"] for m in msgs] == ["assistant", "result"]

    def test_tool_result_user_record_surfaces(self, tmp_path):
        msgs = self._run_with_transcript(
            tmp_path,
            [
                {
                    "type": "user",
                    "sessionId": "s",
                    "uuid": "u1",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "out",
                            }
                        ],
                    },
                },
                {"type": "system", "subtype": "turn_duration", "sessionId": "s"},
            ],
        )
        assert msgs[0]["type"] == "user"
        assert msgs[0]["message"]["content"][0]["type"] == "tool_result"

    def test_ignores_malformed_lines(self, tmp_path):
        async def _test():
            t = make_transport()
            path = tmp_path / "sess.jsonl"
            path.write_text(
                "not json\n"
                + json.dumps(
                    {
                        "type": "assistant",
                        "sessionId": "s",
                        "uuid": "a",
                        "message": {
                            "role": "assistant",
                            "model": "m",
                            "content": [{"type": "text", "text": "x"}],
                        },
                    }
                )
                + "\n"
                + json.dumps({"type": "system", "subtype": "turn_duration"})
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        assert [m["type"] for m in msgs] == ["assistant", "result"]


class TestEarlyExit:
    def test_nonzero_exit_emits_error_result(self):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            t._recent_output = b"\x1b[31mfatal: boom\x1b[0m\n"
            await t._handle_early_exit(2)
            return _drain(t)

        msgs = anyio.run(_test)
        assert len(msgs) == 1
        assert msgs[0]["type"] == "result"
        assert msgs[0]["is_error"] is True
        assert "boom" in msgs[0]["result"]

    def test_clean_exit_emits_nothing(self):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t._handle_early_exit(0)
            return _drain(t)

        assert anyio.run(_test) == []


# --------------------------------------------------------------------------- #
# Control-request -> interactive keystroke/slash-command mappings
# --------------------------------------------------------------------------- #


def _capture_pty_writes(t: PtyCLITransport) -> list[bytes]:
    writes: list[bytes] = []

    async def fake_pty_write(data: bytes) -> None:
        writes.append(data)

    t._pty_write = fake_pty_write  # type: ignore[method-assign]
    t._warmed_up = True
    return writes


class TestControlMappings:
    @pytest.mark.parametrize(
        ("start", "target", "expected_shift_tabs"),
        [
            ("default", "acceptEdits", 1),
            ("default", "plan", 2),
            ("plan", "default", 1),  # wraps around the 3-cycle
            ("acceptEdits", "default", 2),
        ],
    )
    def test_set_permission_mode_cycles(self, start, target, expected_shift_tabs):
        async def _test():
            t = make_transport()
            t._permission_mode = start
            writes = _capture_pty_writes(t)
            await t._set_permission_mode(target)
            assert writes.count(_SHIFT_TAB) == expected_shift_tabs
            assert t._permission_mode == target

        anyio.run(_test)

    def test_set_permission_mode_noop_when_same(self):
        async def _test():
            t = make_transport()
            t._permission_mode = "plan"
            writes = _capture_pty_writes(t)
            await t._set_permission_mode("plan")
            assert writes == []

        anyio.run(_test)

    def test_set_permission_mode_bypass_not_cyclable(self):
        async def _test():
            t = make_transport()
            t._permission_mode = "default"
            writes = _capture_pty_writes(t)
            await t._set_permission_mode("bypassPermissions")
            assert _SHIFT_TAB not in writes  # cannot reach by cycling
            assert t._permission_mode == "default"

        anyio.run(_test)

    def test_set_model_types_slash_model(self):
        async def _test():
            t = make_transport()
            writes = _capture_pty_writes(t)
            await t._set_model("opus")
            joined = b"".join(writes)
            assert b"/model opus" in joined
            assert writes[-1] == b"\r"

        anyio.run(_test)

    def test_set_model_none_is_noop(self):
        async def _test():
            t = make_transport()
            writes = _capture_pty_writes(t)
            await t._set_model(None)
            assert writes == []

        anyio.run(_test)

    @pytest.mark.parametrize(
        ("subtype", "command"),
        [
            ("mcp_status", b"/mcp"),
            ("get_context_usage", b"/context"),
            ("rewind_files", b"/rewind"),
        ],
    )
    def test_slash_command_controls(self, subtype, command):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            writes = _capture_pty_writes(t)
            await t._handle_control_request(
                {
                    "type": "control_request",
                    "request_id": "r",
                    "request": {"subtype": subtype},
                }
            )
            assert command in b"".join(writes)
            # still acknowledges the request
            resp = t._out_recv.receive_nowait()
            assert resp["type"] == "control_response"
            assert resp["response"]["subtype"] == "success"

        anyio.run(_test)


class TestReadMessageOrdering:
    def test_init_then_control_response(self):
        """connect() enqueues the init message; the initialize handshake then
        gets a control_response, so consumers see init first."""

        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            # Simulate what connect() emits, then the initialize handshake.
            await t._out_send.send(
                {"type": "system", "subtype": "init", "session_id": t._session_id}
            )
            await t.write(
                json.dumps(
                    {
                        "type": "control_request",
                        "request_id": "init-1",
                        "request": {"subtype": "initialize"},
                    }
                )
                + "\n"
            )
            first = t._out_recv.receive_nowait()
            second = t._out_recv.receive_nowait()
            assert first["type"] == "system" and first["subtype"] == "init"
            assert second["type"] == "control_response"
            assert second["response"]["request_id"] == "init-1"

        anyio.run(_test)


class TestWarmupConfigurable:
    def test_warmup_seconds_is_patchable(self, monkeypatch):
        """Integration tests rely on shrinking the warmup; guard the knob."""
        monkeypatch.setattr(pty_cli, "_WARMUP_SECONDS", 0.0)
        assert pty_cli._WARMUP_SECONDS == 0.0
