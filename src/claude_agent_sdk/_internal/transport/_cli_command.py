"""Shared helpers for locating the Claude Code CLI and building its invocation.

This logic is transport-agnostic: it resolves the CLI binary, assembles the
command-line flags from :class:`ClaudeAgentOptions`, builds the subprocess
environment (including OpenTelemetry trace propagation), and performs the
version check. The interactive PTY transport builds on these.

The command produced here is for the *interactive* CLI -- it deliberately does
not include the headless ``stream-json`` I/O flags (``--output-format`` /
``--input-format`` / ``--verbose``) or ``--print``.
"""

import json
import logging
import os
import platform
import re
import shutil
from pathlib import Path
from subprocess import PIPE
from typing import Any, cast

import anyio

from ..._errors import CLINotFoundError
from ..._version import __version__
from ...types import ClaudeAgentOptions, SystemPromptFile, SystemPromptPreset

logger = logging.getLogger(__name__)

MINIMUM_CLAUDE_CODE_VERSION = "2.0.0"


# --------------------------------------------------------------------------- #
# CLI discovery
# --------------------------------------------------------------------------- #


def find_bundled_cli() -> str | None:
    """Find the bundled CLI binary if it exists."""
    cli_name = "claude.exe" if platform.system() == "Windows" else "claude"
    # _bundled lives at the package root (../../_bundled relative to this file's
    # package, i.e. claude_agent_sdk/_bundled).
    bundled_path = Path(__file__).parent.parent.parent / "_bundled" / cli_name
    if bundled_path.exists() and bundled_path.is_file():
        logger.info(f"Using bundled Claude Code CLI: {bundled_path}")
        return str(bundled_path)
    return None


def find_cli() -> str:
    """Find the Claude Code CLI binary, raising if it cannot be located."""
    if bundled := find_bundled_cli():
        return bundled

    if cli := shutil.which("claude"):
        return cli

    locations = [
        Path.home() / ".npm-global/bin/claude",
        Path("/usr/local/bin/claude"),
        Path.home() / ".local/bin/claude",
        Path.home() / "node_modules/.bin/claude",
        Path.home() / ".yarn/bin/claude",
        Path.home() / ".claude/local/claude",
    ]
    for path in locations:
        if path.exists() and path.is_file():
            return str(path)

    raise CLINotFoundError(
        "Claude Code not found. Install with:\n"
        "  npm install -g @anthropic-ai/claude-code\n"
        "\nIf already installed locally, try:\n"
        '  export PATH="$HOME/node_modules/.bin:$PATH"\n'
        "\nOr provide the path via ClaudeAgentOptions:\n"
        "  ClaudeAgentOptions(cli_path='/path/to/claude')"
    )


# --------------------------------------------------------------------------- #
# Settings / skills
# --------------------------------------------------------------------------- #


def build_settings_value(options: ClaudeAgentOptions) -> str | None:
    """Build the ``--settings`` value, merging sandbox settings if provided."""
    has_settings = options.settings is not None
    has_sandbox = options.sandbox is not None

    if not has_settings and not has_sandbox:
        return None

    if has_settings and not has_sandbox:
        return options.settings

    settings_obj: dict[str, Any] = {}

    if has_settings:
        assert options.settings is not None
        settings_str = options.settings.strip()
        if settings_str.startswith("{") and settings_str.endswith("}"):
            try:
                settings_obj = json.loads(settings_str)
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse settings as JSON, treating as file path: {settings_str}"
                )
                settings_path = Path(settings_str)
                if settings_path.exists():
                    with settings_path.open(encoding="utf-8") as f:
                        settings_obj = json.load(f)
        else:
            settings_path = Path(settings_str)
            if settings_path.exists():
                with settings_path.open(encoding="utf-8") as f:
                    settings_obj = json.load(f)
            else:
                logger.warning(f"Settings file not found: {settings_path}")

    if has_sandbox:
        settings_obj["sandbox"] = options.sandbox

    return json.dumps(settings_obj)


def apply_skills_defaults(
    options: ClaudeAgentOptions,
) -> tuple[list[str], list[str] | None]:
    """Compute effective allowed_tools and setting_sources for skills.

    Does not mutate the original options object.
    """
    allowed_tools: list[str] = list(options.allowed_tools)
    setting_sources: list[str] | None = (
        list(options.setting_sources) if options.setting_sources is not None else None
    )

    skills = options.skills
    if skills is None:
        return allowed_tools, setting_sources

    if skills == "all":
        if "Skill" not in allowed_tools:
            allowed_tools.append("Skill")
    else:
        for name in skills:
            pattern = f"Skill({name})"
            if pattern not in allowed_tools:
                allowed_tools.append(pattern)

    if setting_sources is None:
        setting_sources = ["user", "project"]

    return allowed_tools, setting_sources


# --------------------------------------------------------------------------- #
# Command construction (interactive)
# --------------------------------------------------------------------------- #


def build_command(
    cli_path: str, options: ClaudeAgentOptions, session_id: str
) -> list[str]:
    """Build the interactive CLI command from ``options``.

    ``session_id`` is always passed via ``--session-id`` so the transport can
    locate the transcript file the CLI writes.
    """
    cmd = [cli_path]

    if options.system_prompt is None:
        cmd.extend(["--system-prompt", ""])
    elif isinstance(options.system_prompt, str):
        cmd.extend(["--system-prompt", options.system_prompt])
    else:
        sp = options.system_prompt
        if sp.get("type") == "file":
            cmd.extend(["--system-prompt-file", cast(SystemPromptFile, sp)["path"]])
        elif sp.get("type") == "preset" and "append" in sp:
            cmd.extend(
                ["--append-system-prompt", cast(SystemPromptPreset, sp)["append"]]
            )

    if options.tools is not None:
        tools = options.tools
        if isinstance(tools, list):
            cmd.extend(["--tools", ",".join(tools) if tools else ""])
        else:
            cmd.extend(["--tools", "default"])

    effective_allowed_tools, effective_setting_sources = apply_skills_defaults(options)

    if effective_allowed_tools:
        cmd.extend(["--allowedTools", ",".join(effective_allowed_tools)])

    if options.max_turns:
        cmd.extend(["--max-turns", str(options.max_turns)])

    if options.max_budget_usd is not None:
        cmd.extend(["--max-budget-usd", str(options.max_budget_usd)])

    if options.disallowed_tools:
        cmd.extend(["--disallowedTools", ",".join(options.disallowed_tools)])

    if options.task_budget is not None:
        cmd.extend(["--task-budget", str(options.task_budget["total"])])

    if options.model:
        cmd.extend(["--model", options.model])

    if options.fallback_model:
        cmd.extend(["--fallback-model", options.fallback_model])

    if options.betas:
        cmd.extend(["--betas", ",".join(options.betas)])

    # "stdio" is the SDK-internal sentinel set when can_use_tool is provided; it
    # selects the stream-json control-protocol permission channel, which the
    # interactive CLI does not understand. The PTY transport answers prompts via
    # the TUI detector instead, so do not pass the flag for that sentinel.
    if options.permission_prompt_tool_name and options.permission_prompt_tool_name != (
        "stdio"
    ):
        cmd.extend(["--permission-prompt-tool", options.permission_prompt_tool_name])

    if options.permission_mode:
        cmd.extend(["--permission-mode", options.permission_mode])

    if options.continue_conversation:
        cmd.append("--continue")

    if options.resume:
        cmd.extend(["--resume", options.resume])

    # Always pass a session id so the transcript file can be located.
    cmd.extend(["--session-id", options.session_id or session_id])

    settings_value = build_settings_value(options)
    if settings_value:
        cmd.extend(["--settings", settings_value])

    if options.add_dirs:
        for directory in options.add_dirs:
            cmd.extend(["--add-dir", str(directory)])

    if options.mcp_servers:
        if isinstance(options.mcp_servers, dict):
            servers_for_cli: dict[str, Any] = {}
            for name, config in options.mcp_servers.items():
                if isinstance(config, dict) and config.get("type") == "sdk":
                    servers_for_cli[name] = {
                        k: v for k, v in config.items() if k != "instance"
                    }
                else:
                    servers_for_cli[name] = config
            if servers_for_cli:
                cmd.extend(
                    ["--mcp-config", json.dumps({"mcpServers": servers_for_cli})]
                )
        else:
            cmd.extend(["--mcp-config", str(options.mcp_servers)])

    if options.strict_mcp_config:
        cmd.append("--strict-mcp-config")

    # Honor --include-hook-events at the CLI level like the stream-json baseline
    # did, rather than silently dropping it (R8). NOTE: empirically the
    # interactive CLI does NOT write hook lifecycle records to the transcript
    # even with this flag (it emits them only on the stream-json stdout channel
    # the PTY cannot read), so consumers still won't receive HookEventMessage
    # objects -- pty_cli._validate_options warns about that. Passing the flag is
    # still the faithful drop-in: it is accepted by the CLI and keeps behavior
    # closest to the baseline command line.
    if options.include_hook_events:
        cmd.append("--include-hook-events")

    if options.fork_session:
        cmd.append("--fork-session")

    if effective_setting_sources is not None:
        cmd.append(f"--setting-sources={','.join(effective_setting_sources)}")

    if options.plugins:
        for plugin in options.plugins:
            if plugin["type"] == "local":
                cmd.extend(["--plugin-dir", plugin["path"]])
            else:
                raise ValueError(f"Unsupported plugin type: {plugin['type']}")

    for flag, value in options.extra_args.items():
        if value is None:
            cmd.append(f"--{flag}")
        else:
            cmd.extend([f"--{flag}", str(value)])

    if options.thinking is not None:
        t = options.thinking
        if t["type"] == "adaptive":
            cmd.extend(["--thinking", "adaptive"])
        elif t["type"] == "enabled":
            cmd.extend(["--max-thinking-tokens", str(t["budget_tokens"])])
        elif t["type"] == "disabled":
            cmd.extend(["--thinking", "disabled"])

        if t["type"] != "disabled" and "display" in t:
            cmd.extend(["--thinking-display", t["display"]])
    elif options.max_thinking_tokens is not None:
        cmd.extend(["--max-thinking-tokens", str(options.max_thinking_tokens)])

    if options.effort is not None:
        cmd.extend(["--effort", options.effort])

    if (
        options.output_format is not None
        and isinstance(options.output_format, dict)
        and options.output_format.get("type") == "json_schema"
    ):
        schema = options.output_format.get("schema")
        if schema is not None:
            cmd.extend(["--json-schema", json.dumps(schema)])

    return cmd


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def build_env(
    options: ClaudeAgentOptions, cwd: str | None, *, entrypoint: str
) -> dict[str, str]:
    """Build the subprocess environment.

    Mirrors the historical behavior: filter ``CLAUDECODE`` so the child does not
    think it is nested in a parent Claude Code, set the SDK entrypoint/version,
    propagate active OTEL trace context, honor ``enable_file_checkpointing``,
    and normalize ``IS_SANDBOX`` when running as root.
    """
    inherited_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    process_env = {
        **inherited_env,
        "CLAUDE_CODE_ENTRYPOINT": entrypoint,
        **options.env,
        "CLAUDE_AGENT_SDK_VERSION": __version__,
    }

    # Propagate active OTEL trace context. Best-effort; never break startup.
    try:
        from opentelemetry import propagate

        carrier: dict[str, str] = {}
        propagate.inject(carrier)
        if "traceparent" in carrier:
            for key in ("TRACEPARENT", "TRACESTATE"):
                if key not in options.env:
                    process_env.pop(key, None)
            for k, v in carrier.items():
                key = k.upper()
                if key not in options.env:
                    process_env[key] = v
    except Exception:  # noqa: BLE001 - tracing must never break startup
        logger.debug("OTEL trace context injection failed", exc_info=True)

    if options.enable_file_checkpointing:
        process_env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"

    if cwd:
        process_env["PWD"] = cwd

    # Running as root, the CLI refuses bypassPermissions /
    # --dangerously-skip-permissions unless IS_SANDBOX marks a contained
    # environment. The CLI only accepts the exact value "1", so normalize any
    # inherited value (e.g. "yes") unless the caller set one explicitly.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        process_env["IS_SANDBOX"] = options.env.get("IS_SANDBOX", "1")

    return process_env


# --------------------------------------------------------------------------- #
# Version check
# --------------------------------------------------------------------------- #


async def check_claude_version(cli_path: str) -> None:
    """Check the Claude Code version and warn if below the minimum."""
    version_process = None
    try:
        with anyio.fail_after(2):
            version_process = await anyio.open_process(
                [cli_path, "-v"], stdout=PIPE, stderr=PIPE
            )
            if version_process.stdout:
                stdout_bytes = await version_process.stdout.receive()
                version_output = stdout_bytes.decode().strip()
                match = re.match(r"([0-9]+\.[0-9]+\.[0-9]+)", version_output)
                if match:
                    version = match.group(1)
                    version_parts = [int(x) for x in version.split(".")]
                    min_parts = [int(x) for x in MINIMUM_CLAUDE_CODE_VERSION.split(".")]
                    if version_parts < min_parts:
                        logger.warning(
                            "Claude Code version %s at %s is unsupported in the Agent "
                            "SDK. Minimum required version is %s. Some features may not "
                            "work correctly.",
                            version,
                            cli_path,
                            MINIMUM_CLAUDE_CODE_VERSION,
                        )
    except Exception:
        pass
    finally:
        if version_process:
            from contextlib import suppress

            with suppress(Exception):
                version_process.terminate()
            with suppress(Exception):
                await version_process.wait()
