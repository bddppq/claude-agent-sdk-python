"""Tests for the shared CLI command/env/discovery helpers (``_cli_command``).

These were previously exercised through ``SubprocessCLITransport._build_command``
/ ``connect``; that pipe transport has been replaced by the interactive PTY
transport, and the option-to-flag logic now lives in ``_cli_command``. The
command produced here is the *interactive* command, so it never contains the
headless ``stream-json`` I/O flags or ``--print``.
"""

import json
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from claude_agent_sdk._errors import CLINotFoundError
from claude_agent_sdk._internal.transport import _cli_command
from claude_agent_sdk.types import ClaudeAgentOptions

DEFAULT_CLI_PATH = "/usr/bin/claude"
SESSION = "11111111-1111-1111-1111-111111111111"


def build(**kwargs: object) -> list[str]:
    """Build an interactive command from options for the given kwargs."""
    session_id = kwargs.pop("session_id", SESSION)
    options = ClaudeAgentOptions(session_id=session_id, **kwargs)  # type: ignore[arg-type]
    return _cli_command.build_command(DEFAULT_CLI_PATH, options, SESSION)


# --------------------------------------------------------------------------- #
# CLI discovery
# --------------------------------------------------------------------------- #


class TestFindCli:
    def test_find_cli_not_found_raises(self):
        with (
            patch(
                "claude_agent_sdk._internal.transport._cli_command.find_bundled_cli",
                return_value=None,
            ),
            patch(
                "claude_agent_sdk._internal.transport._cli_command.shutil.which",
                return_value=None,
            ),
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(CLINotFoundError) as exc_info,
        ):
            _cli_command.find_cli()
        assert "Claude Code not found" in str(exc_info.value)

    def test_find_cli_prefers_which(self):
        with (
            patch(
                "claude_agent_sdk._internal.transport._cli_command.find_bundled_cli",
                return_value=None,
            ),
            patch(
                "claude_agent_sdk._internal.transport._cli_command.shutil.which",
                return_value="/somewhere/claude",
            ),
        ):
            assert _cli_command.find_cli() == "/somewhere/claude"


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #


class TestBuildCommand:
    def test_basic_has_no_stream_json_or_print(self):
        cmd = build()
        assert cmd[0] == DEFAULT_CLI_PATH
        assert "--output-format" not in cmd
        assert "--input-format" not in cmd
        assert "--verbose" not in cmd
        assert "stream-json" not in cmd
        assert "--print" not in cmd
        assert "-p" not in cmd
        # default empty system prompt
        assert cmd[cmd.index("--system-prompt") + 1] == ""

    def test_always_includes_session_id(self):
        cmd = build(session_id=None)
        assert "--session-id" in cmd
        assert cmd[cmd.index("--session-id") + 1] == SESSION

    def test_explicit_session_id_used(self):
        cmd = build(session_id="550e8400-e29b-41d4-a716-446655440000")
        assert cmd[cmd.index("--session-id") + 1] == (
            "550e8400-e29b-41d4-a716-446655440000"
        )

    def test_system_prompt_string(self):
        cmd = build(system_prompt="Be helpful")
        assert cmd[cmd.index("--system-prompt") + 1] == "Be helpful"

    def test_system_prompt_preset_plain(self):
        cmd = build(system_prompt={"type": "preset", "preset": "claude_code"})
        assert "--system-prompt" not in cmd
        assert "--append-system-prompt" not in cmd

    def test_system_prompt_preset_append(self):
        cmd = build(
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": "Be concise.",
            }
        )
        assert "--append-system-prompt" in cmd
        assert "Be concise." in cmd

    def test_system_prompt_file(self):
        cmd = build(system_prompt={"type": "file", "path": "/p/prompt.md"})
        assert "--system-prompt-file" in cmd
        assert "/p/prompt.md" in cmd

    def test_common_options(self):
        cmd = build(
            allowed_tools=["Read", "Write"],
            disallowed_tools=["Bash"],
            model="claude-sonnet-4-5",
            permission_mode="acceptEdits",
            max_turns=5,
        )
        assert cmd[cmd.index("--allowedTools") + 1] == "Read,Write"
        assert cmd[cmd.index("--disallowedTools") + 1] == "Bash"
        assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-5"
        assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
        assert cmd[cmd.index("--max-turns") + 1] == "5"

    def test_fallback_model_and_betas(self):
        cmd = build(model="opus", fallback_model="sonnet", betas=["b1", "b2"])
        assert cmd[cmd.index("--fallback-model") + 1] == "sonnet"
        assert cmd[cmd.index("--betas") + 1] == "b1,b2"

    def test_task_budget(self):
        cmd = build(task_budget={"total": 100000})
        assert cmd[cmd.index("--task-budget") + 1] == "100000"

    def test_max_budget_usd(self):
        cmd = build(max_budget_usd=12.5)
        assert cmd[cmd.index("--max-budget-usd") + 1] == "12.5"

    def test_session_continuation(self):
        cmd = build(continue_conversation=True, resume="session-123")
        assert "--continue" in cmd
        assert cmd[cmd.index("--resume") + 1] == "session-123"

    def test_add_dirs(self):
        cmd = build(add_dirs=["/a", Path("/b")])
        idxs = [i for i, x in enumerate(cmd) if x == "--add-dir"]
        assert len(idxs) == 2
        dirs = {cmd[i + 1] for i in idxs}
        assert dirs == {"/a", "/b"}

    def test_extra_args(self):
        cmd = build(extra_args={"new-flag": "value", "boolean-flag": None})
        assert "--new-flag" in cmd
        assert cmd[cmd.index("--new-flag") + 1] == "value"
        assert "--boolean-flag" in cmd

    def test_fork_session_and_strict_mcp(self):
        cmd = build(fork_session=True, strict_mcp_config=True)
        assert "--fork-session" in cmd
        assert "--strict-mcp-config" in cmd

    def test_effort(self):
        cmd = build(effort="xhigh")
        assert cmd[cmd.index("--effort") + 1] == "xhigh"

    @pytest.mark.parametrize(
        ("thinking", "expected", "absent"),
        [
            ({"type": "adaptive"}, ["--thinking", "adaptive"], "--max-thinking-tokens"),
            (
                {"type": "enabled", "budget_tokens": 5000},
                ["--max-thinking-tokens", "5000"],
                "--thinking",
            ),
            ({"type": "disabled"}, ["--thinking", "disabled"], "--max-thinking-tokens"),
        ],
    )
    def test_thinking(self, thinking, expected, absent):
        cmd = build(thinking=thinking)
        idx = cmd.index(expected[0])
        assert cmd[idx : idx + 2] == expected
        assert absent not in cmd

    def test_thinking_precedence_over_max_thinking_tokens(self):
        cmd = build(thinking={"type": "adaptive"}, max_thinking_tokens=9999)
        assert cmd[cmd.index("--thinking") + 1] == "adaptive"
        assert "--max-thinking-tokens" not in cmd

    def test_tools_array_and_empty_and_preset(self):
        assert (
            build(tools=["Read", "Edit"])[
                build(tools=["Read", "Edit"]).index("--tools") + 1
            ]
            == "Read,Edit"
        )
        cmd_empty = build(tools=[])
        assert cmd_empty[cmd_empty.index("--tools") + 1] == ""
        cmd_preset = build(tools={"type": "preset", "preset": "claude_code"})
        assert cmd_preset[cmd_preset.index("--tools") + 1] == "default"

    def test_output_format_json_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        cmd = build(output_format={"type": "json_schema", "schema": schema})
        assert cmd[cmd.index("--json-schema") + 1] == json.dumps(schema)

    def test_plugins_local(self):
        cmd = build(plugins=[{"type": "local", "path": "/plug"}])
        assert cmd[cmd.index("--plugin-dir") + 1] == "/plug"


class TestMcpServers:
    def test_dict_config(self):
        servers = {
            "test-server": {"type": "stdio", "command": "/srv", "args": ["--x"]},
        }
        cmd = build(mcp_servers=servers)
        config = json.loads(cmd[cmd.index("--mcp-config") + 1])
        assert config == {"mcpServers": servers}

    def test_string_path(self):
        cmd = build(mcp_servers="/path/to/mcp-config.json")
        assert cmd[cmd.index("--mcp-config") + 1] == "/path/to/mcp-config.json"

    def test_sdk_server_strips_instance(self):
        servers = {"s": {"type": "sdk", "name": "s", "instance": object()}}
        cmd = build(mcp_servers=servers)
        config = json.loads(cmd[cmd.index("--mcp-config") + 1])
        assert "instance" not in config["mcpServers"]["s"]


class TestSettingsAndSandbox:
    def test_settings_file_passthrough(self):
        cmd = build(settings="/path/to/settings.json")
        assert cmd[cmd.index("--settings") + 1] == "/path/to/settings.json"

    def test_settings_json_passthrough(self):
        settings = '{"permissions": {"allow": ["Bash(ls:*)"]}}'
        cmd = build(settings=settings)
        assert cmd[cmd.index("--settings") + 1] == settings

    def test_sandbox_only_merges(self):
        cmd = build(sandbox={"enabled": True})
        parsed = json.loads(cmd[cmd.index("--settings") + 1])
        assert parsed == {"sandbox": {"enabled": True}}

    def test_sandbox_merged_into_settings_json(self):
        cmd = build(
            settings='{"verbose": true}',
            sandbox={"enabled": True, "excludedCommands": ["git"]},
        )
        parsed = json.loads(cmd[cmd.index("--settings") + 1])
        assert parsed["verbose"] is True
        assert parsed["sandbox"]["excludedCommands"] == ["git"]


class TestSkillsMatrix:
    @pytest.mark.parametrize(
        ("skills", "extra", "want_tools", "want_sources"),
        [
            (None, {}, None, None),
            ("all", {}, "Skill", "user,project"),
            (["pdf", "docx"], {}, "Skill(pdf),Skill(docx)", "user,project"),
            (["pdf"], {"setting_sources": ["project"]}, "Skill(pdf)", "project"),
            (
                ["pdf"],
                {"allowed_tools": ["Read", "Bash"]},
                "Read,Bash,Skill(pdf)",
                "user,project",
            ),
            ([], {}, None, "user,project"),
        ],
    )
    def test_skills(self, skills, extra, want_tools, want_sources):
        cmd = build(skills=skills, **extra)
        if want_tools is None:
            assert "--allowedTools" not in cmd
        else:
            assert cmd[cmd.index("--allowedTools") + 1] == want_tools
        if want_sources is None:
            assert not any(a.startswith("--setting-sources") for a in cmd)
        else:
            assert f"--setting-sources={want_sources}" in cmd

    def test_skills_does_not_mutate_options(self):
        options = ClaudeAgentOptions(allowed_tools=["Read"], skills=["pdf"])
        _cli_command.build_command(DEFAULT_CLI_PATH, options, SESSION)
        assert options.allowed_tools == ["Read"]
        assert options.setting_sources is None

    def test_skills_idempotent(self):
        cmd = build(allowed_tools=["Skill(pdf)"], skills=["pdf"])
        assert cmd[cmd.index("--allowedTools") + 1] == "Skill(pdf)"


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def env_for(**kwargs: object) -> dict[str, str]:
    options = ClaudeAgentOptions(**kwargs)  # type: ignore[arg-type]
    return _cli_command.build_env(options, cwd=None, entrypoint="sdk-py-pty")


class TestBuildEnv:
    def test_custom_env_passed(self):
        value = f"test-{uuid.uuid4().hex[:8]}"
        env = env_for(env={"MY_TEST_VAR": value})
        assert env["MY_TEST_VAR"] == value

    def test_sdk_managed_vars(self):
        env = env_for()
        assert env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py-pty"
        assert "CLAUDE_AGENT_SDK_VERSION" in env

    def test_options_env_cannot_override_sdk_version(self):
        from claude_agent_sdk._version import __version__

        env = env_for(env={"CLAUDE_AGENT_SDK_VERSION": "0.0.0"})
        assert env["CLAUDE_AGENT_SDK_VERSION"] == __version__

    def test_claudecode_filtered(self):
        with patch.dict(os.environ, {"CLAUDECODE": "1", "OTHER": "kept"}):
            env = env_for()
        assert "CLAUDECODE" not in env
        assert env["OTHER"] == "kept"

    def test_caller_can_override_entrypoint(self):
        env = env_for(env={"CLAUDE_CODE_ENTRYPOINT": "custom"})
        assert env["CLAUDE_CODE_ENTRYPOINT"] == "custom"

    def test_enable_file_checkpointing(self):
        env = env_for(enable_file_checkpointing=True)
        assert env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] == "true"

    def test_pwd_set_when_cwd_given(self):
        options = ClaudeAgentOptions()
        env = _cli_command.build_env(options, cwd="/work/dir", entrypoint="sdk-py-pty")
        assert env["PWD"] == "/work/dir"


class TestOtelPropagation:
    def test_active_span_injected(self):
        def fake_inject(carrier):
            carrier["traceparent"] = (
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
            )
            carrier["tracestate"] = "vendor=value"

        fake_propagate = MagicMock()
        fake_propagate.inject = fake_inject
        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(propagate=fake_propagate),
                "opentelemetry.propagate": fake_propagate,
            },
        ):
            env = env_for()
        assert (
            env["TRACEPARENT"]
            == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        )
        assert env["TRACESTATE"] == "vendor=value"

    def test_options_env_wins_over_propagator(self):
        def fake_inject(carrier):
            carrier["traceparent"] = "00-aaaa-bbbb-01"

        fake_propagate = MagicMock()
        fake_propagate.inject = fake_inject
        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(propagate=fake_propagate),
                "opentelemetry.propagate": fake_propagate,
            },
        ):
            env = env_for(env={"TRACEPARENT": "custom"})
        assert env["TRACEPARENT"] == "custom"

    def test_propagator_error_does_not_break(self):
        fake_propagate = MagicMock()
        fake_propagate.inject = MagicMock(side_effect=RuntimeError("boom"))
        with (
            patch.dict(
                "sys.modules",
                {
                    "opentelemetry": MagicMock(propagate=fake_propagate),
                    "opentelemetry.propagate": fake_propagate,
                },
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("TRACEPARENT", None)
            env = env_for()  # must not raise
        assert "TRACEPARENT" not in env


class TestIsSandbox:
    def test_normalized_to_one_when_root(self):
        with (
            patch("os.geteuid", return_value=0),
            patch.dict(os.environ, {"IS_SANDBOX": "yes"}),
        ):
            env = env_for()
        assert env["IS_SANDBOX"] == "1"

    def test_caller_override_respected_when_root(self):
        with patch("os.geteuid", return_value=0):
            env = env_for(env={"IS_SANDBOX": "custom"})
        assert env["IS_SANDBOX"] == "custom"


# --------------------------------------------------------------------------- #
# Version check
# --------------------------------------------------------------------------- #


class TestVersionCheck:
    def test_old_version_warns(self, caplog):
        async def _run():
            proc = MagicMock()
            proc.stdout = MagicMock()
            proc.stdout.receive = AsyncMock(return_value=b"1.0.0 (Claude Code)")
            proc.terminate = MagicMock()
            proc.wait = AsyncMock()
            with (
                patch(
                    "claude_agent_sdk._internal.transport._cli_command.anyio.open_process",
                    new_callable=AsyncMock,
                    return_value=proc,
                ),
                caplog.at_level("WARNING"),
            ):
                await _cli_command.check_claude_version("/usr/bin/claude")

        anyio.run(_run)
        assert any("unsupported" in r.message for r in caplog.records)

    def test_current_version_no_warning(self, caplog):
        async def _run():
            proc = MagicMock()
            proc.stdout = MagicMock()
            proc.stdout.receive = AsyncMock(return_value=b"2.5.0 (Claude Code)")
            proc.terminate = MagicMock()
            proc.wait = AsyncMock()
            with (
                patch(
                    "claude_agent_sdk._internal.transport._cli_command.anyio.open_process",
                    new_callable=AsyncMock,
                    return_value=proc,
                ),
                caplog.at_level("WARNING"),
            ):
                await _cli_command.check_claude_version("/usr/bin/claude")

        anyio.run(_run)
        assert not any("unsupported" in r.message for r in caplog.records)
