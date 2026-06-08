"""PTY-based transport that drives the Claude Code CLI in interactive mode.

This is the SDK's transport. Rather than speaking the headless ``stream-json``
protocol over pipes (the former ``--print`` mode), it launches the
*interactive* CLI attached to a pseudo-terminal (PTY). It then:

* types user prompts into the PTY (as a real terminal would), and
* reads the model's responses by **tailing the session transcript** the CLI
  writes to ``<config>/projects/<sanitized-cwd>/<session-id>.jsonl``.

The transcript records are translated back into the same message dictionaries
the rest of the SDK already understands (``assistant`` / ``user`` / ``result``
/ ``system``), so consumers and ``message_parser`` are unaffected.

POSIX only -- PTYs are not available on Windows.

``can_use_tool`` IS supported: the interactive CLI renders tool-permission
prompts as on-screen dialogs, which a background watcher detects (via the
``pty_question`` screen parser) and answers by keystroke -- routing the decision
through the ``can_use_tool`` callback when provided, or a safe default otherwise
so turns never hang on an unanswered prompt.

Unsupported options (interactive mode has no equivalent SDK channel). These are
rejected up front by :meth:`PtyCLITransport._validate_options` with an
actionable error rather than failing silently or hanging:

* ``hooks`` -- programmatic hook callbacks require the bidirectional protocol;
* in-process ``mcp_servers`` of ``type="sdk"`` -- reachable only over the
  control protocol (external stdio/http/sse MCP servers still work);
* ``session_store`` -- relied on ``transcript_mirror`` stdout frames;
* a caller-supplied ``permission_prompt_tool_name`` (the SDK-internal ``"stdio"``
  sentinel set alongside ``can_use_tool`` is accepted and handled via the TUI).

``include_partial_messages`` and ``include_hook_events`` are accepted but warned
about (no partial/hook-event records exist in the transcript).
"""

import atexit
import contextlib
import fcntl
import json
import logging
import os
import pty
import re
import signal
import struct
import tempfile
import termios
import time
import tty
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from subprocess import Popen
from typing import Any, Literal

import anyio

from ..._errors import CLIConnectionError, CLINotFoundError
from ...types import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    ToolPermissionContext,
)
from .._task_compat import TaskHandle, spawn_detached
from ..sessions import _canonicalize_path, _get_projects_dir, _sanitize_path
from . import Transport, _cli_command
from ._usage import TurnUsageAccumulator
from .pty_question import (
    SCREEN_COLS,
    SCREEN_ROWS,
    DetectedQuestion,
    QuestionOption,
    choose_option,
    parse_question,
)

logger = logging.getLogger(__name__)

# Track live child process groups + master fds so we can terminate them when the
# parent Python process exits, mirroring the old subprocess transport's
# parent-exit cleanup. Prevents orphaned interactive ``claude`` processes when a
# caller crashes or exits before awaiting close().
_ACTIVE_CHILDREN: "set[PtyCLITransport]" = set()


def _kill_active_children() -> None:
    for transport in list(_ACTIVE_CHILDREN):
        with contextlib.suppress(Exception):
            transport._terminate_process()
    _ACTIVE_CHILDREN.clear()


atexit.register(_kill_active_children)

# Transcript record ``type`` values that carry no SDK-visible message. These are
# internal bookkeeping entries the CLI writes alongside the conversation.
_SKIP_TRANSCRIPT_TYPES = frozenset(
    {
        "queue-operation",
        "last-prompt",
        "ai-title",
        "attachment",
        "mode",
        "file-history-snapshot",
        "summary",
    }
)

# Keystrokes sent into the PTY. A bare carriage return submits the current
# prompt in the TUI; ESC interrupts an in-progress turn (app:interrupt);
# shift+tab (CSI Z, "back-tab") cycles the permission mode (chat:cycleMode).
_SUBMIT = b"\r"
_INTERRUPT = b"\x1b"
_SHIFT_TAB = b"\x1b[Z"

# Bracketed-paste guards. Wrapping the prompt in these makes the TUI insert the
# text verbatim into the editor -- preserving newlines and NOT interpreting a
# leading "/", "!", or "#" as a slash/bash/memory command -- after which a
# single carriage return submits it. This is critical for prompt fidelity.
_PASTE_START = b"\x1b[200~"
_PASTE_END = b"\x1b[201~"

# Seconds to let the interactive TUI render before the first prompt is typed.
# The CLI shows a transient startup toast that swallows the first Enter; we wait
# this long, then send one dismissal Enter. Module-level so tests can shrink it.
# Tightened from 3.0 -> 1.5 (L1): live-verified that the first prompt still
# submits reliably at 1.5s across repeated runs (and even at 1.0s), so the
# previous 3.0s was conservative headroom; 1.5s keeps margin over the observed
# floor while halving startup latency.
_WARMUP_SECONDS = 1.5

# Upper bound on the per-record uuid dedup set (R9). Re-reads only revisit the
# transcript tail after a compaction/rewrite, so retaining a generous recent
# window is sufficient to prevent double-emits while keeping memory bounded for
# a long-lived multi-turn client.
_SEEN_UUIDS_MAX = 2_048

# Order the TUI cycles through on shift+tab. bypassPermissions is not part of
# the cycle (it is only reachable via launch flag), so it cannot be set live.
_PERMISSION_CYCLE = ("default", "acceptEdits", "plan")


def _translate_transcript_entry(
    entry: dict[str, Any], session_id: str
) -> dict[str, Any] | None:
    """Translate a transcript ``.jsonl`` record into an SDK message dict.

    Returns ``None`` for records that should not surface to consumers (internal
    bookkeeping, or the plain-text user prompt we typed ourselves).
    """
    entry_type = entry.get("type")
    sid = entry.get("sessionId") or session_id

    # parent_tool_use_id carries sub-agent attribution; the transcript records
    # it under a few possible keys.
    parent = entry.get("parent_tool_use_id") or entry.get("parentToolUseId")

    if entry_type == "assistant":
        message = entry.get("message")
        if not isinstance(message, dict):
            return None
        return {
            "type": "assistant",
            "message": _sanitize_assistant_message(message),
            "session_id": sid,
            "uuid": entry.get("uuid"),
            "parent_tool_use_id": parent,
        }

    if entry_type == "user":
        message = entry.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        # Skip the echo of the plain-text prompt we typed (the CLI records the
        # typed prompt as string content). All *structured* (list) user records
        # are surfaced -- tool results AND non-tool-result blocks such as image /
        # document content (M4): previously list records without a tool_result
        # were dropped, suppressing legitimate non-tool-result user content.
        if isinstance(content, str):
            return None
        return {
            "type": "user",
            "message": message,
            "session_id": sid,
            "uuid": entry.get("uuid"),
            "parent_tool_use_id": parent,
        }

    return None


def _sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Coerce a transcript assistant message into a parser-safe shape.

    The transcript is an internal format with weaker guarantees than the old
    ``--print`` stream, so defend against records the parser would reject:

    * ``content`` as a string -> wrap in a single text block;
    * ``content`` missing/non-list -> empty list;
    * ``thinking`` blocks missing ``signature`` -> default it;
    * ``model`` missing -> default to ``"unknown"`` (parser requires it).
    """
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        message["content"] = []
    else:
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                block.setdefault("signature", "")
    message.setdefault("model", "unknown")
    return message


def _has_tool_result(message: Any) -> bool:
    """True if a user message carries at least one tool_result block."""
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _extract_text(message: dict[str, Any]) -> str:
    """Concatenate the text blocks of an assistant message."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


class PtyCLITransport(Transport):
    """Transport that runs the interactive CLI over a PTY and tails its transcript."""

    def __init__(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        options: ClaudeAgentOptions,
    ):
        if os.name != "posix":
            raise CLIConnectionError(
                "PtyCLITransport requires a POSIX platform (PTYs are unavailable "
                "on Windows)."
            )
        self._prompt = prompt
        self._options = options
        self._cli_path: str | None = (
            str(options.cli_path) if options.cli_path is not None else None
        )
        # ``_cwd`` is the directory we *compute the transcript path* from and is
        # always concrete (defaults to the current process cwd, which is what the
        # child inherits). ``_spawn_cwd`` is what we hand to Popen: None when the
        # caller did not set cwd, so the child inherits the parent's working
        # directory exactly as the old transport did (L2), rather than us pinning
        # it to a snapshot of Path.cwd().
        self._cwd = str(options.cwd) if options.cwd else str(Path.cwd())
        self._spawn_cwd = str(options.cwd) if options.cwd else None
        self._session_id = options.session_id or str(uuid.uuid4())
        # Track the live permission mode so set_permission_mode can compute how
        # many shift+tab cycles are needed to reach a target.
        self._permission_mode: str = options.permission_mode or "default"

        self._proc: Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._transcript_path: Path | None = None

        self._out_send: Any = None
        self._out_recv: Any = None
        self._drain_task: TaskHandle | None = None
        self._tail_task: TaskHandle | None = None
        self._question_task: TaskHandle | None = None
        # Fingerprints of questions already answered, so the watcher does not
        # re-answer the same on-screen dialog while it lingers before redraw.
        self._answered_questions: set[str] = set()

        self._ready = False
        self._closed = False
        self._input_ended = False
        self._result_emitted = False
        self._warmed_up = False
        self._spawn_time = 0.0
        # Tail of recent PTY output, kept for diagnostics if the CLI exits
        # before producing any transcript output.
        self._recent_output = b""

        # Optional terminal-screen emulator (pyte) for introspecting blocking
        # questions rendered in the TUI. Stays None when pyte is unavailable, in
        # which case detect_question() simply returns None.
        self._question_screen: Any = None
        self._question_stream: Any = None

        # Serializes keystroke sequences so concurrent prompts / control actions
        # don't interleave bytes into the PTY.
        self._write_lock = anyio.Lock()
        # Per-turn state used to synthesize a faithful ``result`` message.
        self._turn_count = 0
        self._turn_text = ""
        self._turn_usage = TurnUsageAccumulator()
        self._turn_is_error = False
        self._turn_error_text: str | None = None
        self._turn_subtype: str | None = None
        self._turn_stop_reason: str | None = None
        self._turn_model: str | None = None
        # Count of tool-result user records in the current turn. num_turns is
        # (tool_result_count + 1): each tool result is one API round-trip back
        # to the model, plus the final answer turn. Verified against live
        # stream-json num_turns over many prompts (PONG=1/0 results, 2-write+
        # 2-read=8/7 results, etc.) -- the transcript's turn_duration
        # ``messageCount`` counts streamed snapshots, NOT API turns, so it must
        # not be used here.
        self._turn_tool_results = 0
        # Permission denials observed in the current turn (answered "deny" via
        # the TUI question detector). Surfaced on the result like stream-json.
        self._turn_permission_denials: list[dict[str, Any]] = []
        # Real session id discovered from transcript records (resume/fork can
        # make the CLI use an id different from the one we generated).
        self._observed_session_id: str | None = None
        # Cached server-info payload for get_server_info() / initialize ack.
        self._server_info: dict[str, Any] | None = None
        # Dedup transcript records by uuid so a mid-session compaction/rewrite
        # (which resets the read offset) cannot re-emit already-seen messages.
        # Bounded (R9): a long-lived multi-turn client would otherwise retain
        # every record's uuid for the transport's whole life. We keep the most
        # recent ``_SEEN_UUIDS_MAX`` in insertion order and drop the oldest;
        # re-reads only ever revisit the *tail* of the transcript (after a
        # compaction/rewrite), so an old uuid evicted from the front cannot
        # reappear and be wrongly re-emitted.
        self._seen_uuids: OrderedDict[str, None] = OrderedDict()
        # Monotonic time the current turn's prompt was submitted, used to derive
        # a faithful ``duration_ms`` if the turn ends without a turn_duration
        # record (e.g. an interrupt). ``None`` when no turn is in flight.
        self._turn_start_time: float | None = None
        # Dedup assistant messages by message.id: the transcript writes several
        # streaming snapshots per assistant message (same id, distinct uuid), so
        # uuid dedup alone would emit the same assistant message multiple times.
        self._seen_assistant_ids: set[str] = set()

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        if self._proc is not None:
            return

        self._validate_options()

        if self._cli_path is None:
            self._cli_path = await anyio.to_thread.run_sync(_cli_command.find_cli)

        if not os.environ.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
            await _cli_command.check_claude_version(self._cli_path)

        # Interactive mode shows first-run onboarding (theme/login) screens that
        # block programmatic input. Mark onboarding complete so the CLI drops
        # straight into the prompt. Best-effort and non-destructive.
        await anyio.to_thread.run_sync(self._ensure_onboarding_complete)

        cmd = self._build_command()

        master_fd, slave_fd = pty.openpty()
        # Give the TUI a sane window and put the line discipline in raw mode so
        # our writes are not echoed back into the transcript-tailing path.
        with contextlib.suppress(OSError):
            fcntl.ioctl(
                slave_fd,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", SCREEN_ROWS, SCREEN_COLS, 0, 0),
            )
        with contextlib.suppress(OSError):
            tty.setraw(slave_fd)

        try:
            self._proc = Popen(  # noqa: S603 - cmd is built from vetted options
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                # None => inherit parent's cwd (L2); only pin when caller set it.
                cwd=self._spawn_cwd,
                env=self._build_env(),
                close_fds=True,
                # Run as the configured OS user, matching the old transport (L3).
                user=self._options.user,
                preexec_fn=os.setsid,  # own process group so we can signal it
            )
        except FileNotFoundError as e:
            os.close(master_fd)
            os.close(slave_fd)
            if not Path(self._cwd).exists():
                raise CLIConnectionError(
                    f"Working directory does not exist: {self._cwd}"
                ) from e
            raise CLINotFoundError(f"Claude Code not found at: {self._cli_path}") from e
        except Exception as e:
            os.close(master_fd)
            os.close(slave_fd)
            raise CLIConnectionError(f"Failed to start Claude Code: {e}") from e

        os.close(slave_fd)  # parent keeps only the master end
        self._master_fd = master_fd
        self._spawn_time = time.monotonic()
        _ACTIVE_CHILDREN.add(self)

        self._transcript_path = self._compute_transcript_path()

        # Honor options.max_buffer_size for the message buffer (M8). It bounded a
        # byte pipe in the old transport; here it bounds the count of buffered
        # message dicts, which is the closest interactive equivalent. Default to
        # 1000 when unset.
        buffer_size = self._options.max_buffer_size or 1000
        self._out_send, self._out_recv = anyio.create_memory_object_stream[
            dict[str, Any]
        ](max_buffer_size=buffer_size)

        self._init_question_screen()

        self._drain_task = spawn_detached(self._drain_loop())
        self._tail_task = spawn_detached(self._tail_loop())
        # Watch for blocking TUI permission/plan dialogs and answer them (C5/C6)
        # so turns complete. Only needed when there is something to decide with:
        # a can_use_tool callback, or default-mode prompts that would otherwise
        # hang. Always running it is cheap (it polls the emulated screen).
        if self._question_screen is not None:
            self._question_task = spawn_detached(self._question_watch_loop())

        self._ready = True

        # Surface an init message so consumers can read the session id and
        # server capabilities before the first turn lands in the transcript.
        # Populated to mirror the stream-json system/init shape (M1).
        init = self._build_init_data()
        init.update(
            {
                "type": "system",
                "subtype": "init",
                "uuid": str(uuid.uuid4()),
            }
        )
        await self._out_send.send(init)

    def _build_init_data(self) -> dict[str, Any]:
        """Build the init/server-info payload from the configured options.

        The interactive transcript carries no init record (verified: the only
        ``system`` subtypes written are ``turn_duration`` and
        ``stop_hook_summary``), so we reconstruct the stream-json ``system/init``
        fields from options and faithful defaults. This keeps ``get_server_info``
        non-empty and gives consumers the documented ``tools`` / ``mcp_servers``
        / ``model`` / ``permissionMode`` / ``slash_commands`` / ``output_style``
        keys instead of a 5-field stub.
        """
        o = self._options
        tools: list[str] = []
        if isinstance(o.tools, list):
            tools = list(o.tools)
        tools.extend(t for t in o.allowed_tools if t not in tools)

        mcp_servers: list[dict[str, Any]] = []
        if isinstance(o.mcp_servers, dict):
            for name, cfg in o.mcp_servers.items():
                entry: dict[str, Any] = {"name": name}
                if isinstance(cfg, dict) and "type" in cfg:
                    entry["type"] = cfg["type"]
                mcp_servers.append(entry)

        agents: list[dict[str, Any]] = []
        if isinstance(o.agents, dict):
            agents = [{"name": name} for name in o.agents]

        # Resolved model: prefer the model observed on the first assistant
        # record of the session (the CLI's actually-resolved model, R6); fall
        # back to the caller's requested model. The init message is emitted at
        # connect before any transcript record, so on the first connect this is
        # ``options.model``; later reconstructions pick up the observed model.
        model = self._turn_model or o.model or ""

        return {
            "session_id": self._observed_session_id or self._session_id,
            "cwd": self._cwd,
            "tools": tools,
            "mcp_servers": mcp_servers,
            "model": model,
            "permissionMode": self._permission_mode,
            "apiKeySource": "none",
            "slash_commands": [],
            "output_style": "default",
            # Additional baseline system/init keys, surfaced with safe defaults
            # so consumers reading them don't see missing keys (R6). The
            # transcript carries no init record, so the catalogs (plugins,
            # skills, full command/tool lists) and toggles are not
            # PTY-observable; agents is backfilled from options.
            "agents": agents,
            "plugins": [],
            "skills": [],
        }

    def _build_server_info(self) -> dict[str, Any]:
        """Build the ``initialize`` control-response / ``get_server_info`` payload.

        This is the shape ``client.get_server_info()`` returns -- the real
        ``initialize`` CONTROL RESPONSE, which has a DIFFERENT key set than the
        ``system/init`` MESSAGE (R5). The baseline top keys are ``account``,
        ``agents``, ``available_output_styles``, ``commands``, ``models``,
        ``output_style`` and ``pid``. Consumers do ``info.get('commands', [])``
        / ``info.get('output_style')`` etc., so every baseline key is present
        with a list/dict default even where the value is not PTY-observable.

        Observable values are populated: ``pid`` is the live CLI subprocess pid;
        ``agents`` is backfilled from ``options.agents`` when the caller defined
        any. The slash-command catalog, model catalog and account identity are
        only available over the deleted stream-json ``initialize`` channel (the
        transcript has no init record), so those default to empty rather than
        being fabricated.
        """
        agents: list[dict[str, Any]] = []
        if isinstance(self._options.agents, dict):
            agents = [{"name": name} for name in self._options.agents]

        return {
            "commands": [],
            "available_output_styles": ["default"],
            "output_style": "default",
            "models": [],
            "account": {},
            "agents": agents,
            "pid": self._proc.pid if self._proc is not None else None,
        }

    def _validate_options(self) -> None:
        """Reject options the interactive transport cannot honor.

        Fails loudly up front instead of silently no-op-ing or hanging mid-turn.
        """
        o = self._options
        unsupported: list[str] = []
        if o.hooks:
            unsupported.append(
                "hooks (programmatic hook callbacks require the bidirectional "
                "control protocol; use command hooks in settings instead)"
            )
        # ``permission_prompt_tool_name == "stdio"`` is the SDK-internal sentinel
        # the client sets when can_use_tool is provided; the interactive
        # transport answers permission prompts via the TUI detector instead, so
        # that sentinel is expected and not an error. A *caller-supplied* tool
        # name has no interactive equivalent.
        if o.permission_prompt_tool_name and o.permission_prompt_tool_name != "stdio":
            unsupported.append("permission_prompt_tool_name")
        if o.session_store is not None:
            unsupported.append(
                "session_store (transcript mirroring relied on stream-json frames)"
            )
        if isinstance(o.mcp_servers, dict):
            sdk_servers = [
                name
                for name, cfg in o.mcp_servers.items()
                if isinstance(cfg, dict) and cfg.get("type") == "sdk"
            ]
            if sdk_servers:
                unsupported.append(
                    "in-process SDK MCP servers "
                    f"({', '.join(sdk_servers)}) -- external stdio/http/sse MCP "
                    "servers are still supported"
                )
        if unsupported:
            raise CLIConnectionError(
                "These options are not supported by the interactive transport:\n  - "
                + "\n  - ".join(unsupported)
            )

        # Accepted-but-inert observability flags: warn rather than fail.
        if o.include_partial_messages:
            logger.warning(
                "include_partial_messages has no effect with the interactive "
                "transport; no partial/stream_event records exist in the transcript."
            )
        if o.include_hook_events:
            logger.warning(
                "include_hook_events is passed to the CLI but yields no "
                "HookEventMessage objects with the interactive transport: the "
                "CLI emits hook lifecycle events only on the stream-json stdout "
                "channel, not into the transcript the PTY tails (verified "
                "empirically -- no hook records appear in the transcript)."
            )
        if o.stderr is not None:
            logger.warning(
                "The stderr callback is not invoked by the interactive transport "
                "(the CLI's stderr is multiplexed onto the PTY)."
            )
        if o.max_buffer_size is not None:
            logger.debug(
                "max_buffer_size bounds the count of buffered message dicts in "
                "the interactive transport (the closest equivalent to the old "
                "byte-pipe buffer)."
            )

    def _build_env(self) -> dict[str, str]:
        # Use the same entrypoint tag as the stream-json baseline so telemetry
        # is not keyed differently for drop-in consumers (E1).
        return _cli_command.build_env(self._options, self._cwd, entrypoint="sdk-py")

    def _ensure_onboarding_complete(self) -> None:
        """Clear interactive gates that would block programmatic input.

        The interactive CLI blocks the prompt behind two screens that a real
        user dismisses by hand:

        * first-run onboarding (theme/login) -- gated by
          ``hasCompletedOnboarding``;
        * a per-folder trust dialog -- gated by
          ``projects[<cwd>].hasTrustDialogAccepted``.

        We merge both flags into the CLI config, preserving all existing data.
        Pre-seeding the config is more robust than timing blind keystrokes
        against a full-screen TUI.
        """
        config_dir = self._options.env.get("CLAUDE_CONFIG_DIR") or os.environ.get(
            "CLAUDE_CONFIG_DIR"
        )
        config_path = (
            Path(config_dir) / ".claude.json"
            if config_dir
            else Path.home() / ".claude.json"
        )
        try:
            data: dict[str, Any] = {}
            if config_path.exists():
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    data = json.loads(config_path.read_text(encoding="utf-8"))

            changed = False
            if data.get("hasCompletedOnboarding") is not True:
                data["hasCompletedOnboarding"] = True
                changed = True

            projects = data.setdefault("projects", {})
            # The CLI keys projects by both the literal cwd and its realpath in
            # different code paths; trust both so the dialog never appears.
            for key in {self._cwd, os.path.realpath(self._cwd)}:
                project = projects.setdefault(key, {})
                if project.get("hasTrustDialogAccepted") is not True:
                    project["hasTrustDialogAccepted"] = True
                    project.setdefault("projectOnboardingSeenCount", 1)
                    changed = True

            if not changed:
                return
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # Write atomically (temp file + os.replace) so a concurrent CLI or
            # another SDK client racing this write cannot read or leave behind a
            # half-written / clobbered ~/.claude.json.
            fd, tmp_name = tempfile.mkstemp(
                dir=str(config_path.parent), prefix=".claude.json.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                Path(tmp_name).replace(config_path)
            except OSError:
                with contextlib.suppress(OSError):
                    Path(tmp_name).unlink()
                raise
            logger.debug("Pre-seeded onboarding/trust flags in %s", config_path)
        except OSError:
            logger.debug("Could not update CLI config flags", exc_info=True)

    def _build_command(self) -> list[str]:
        """Build the interactive CLI command from the configured options."""
        if self._cli_path is None:
            raise CLINotFoundError("CLI path not resolved. Call connect() first.")
        return _cli_command.build_command(
            self._cli_path, self._options, self._session_id
        )

    def _compute_transcript_path(self) -> Path:
        project_dir = _get_projects_dir(
            env_override=self._options.env
        ) / _sanitize_path(_canonicalize_path(self._cwd))
        return project_dir / f"{self._session_id}.jsonl"

    def _resolve_transcript_path(self) -> Path | None:
        """Locate the transcript the CLI is actually writing, or ``None``.

        Resolution is deliberately conservative -- returning ``None`` (keep
        waiting) is always safer than guessing another session's transcript:

        1. the exact computed path (the normal case);
        2. any ``<our-session-id>.jsonl`` anywhere under the projects dir
           (handles a long-cwd directory-hash mismatch; still keyed to *our*
           session id, so it can't match a different conversation);
        3. only when the CLI may have chosen a different id
           (``--fork-session`` / ``--resume`` / ``--continue``): the newest
           ``*.jsonl`` in *our own project dir* modified at/after spawn.

        Step 3 is scoped to our project directory and gated on those flags so a
        concurrent unrelated ``claude`` session in another directory is never
        picked up.
        """
        if self._transcript_path and self._transcript_path.exists():
            return self._transcript_path

        projects = _get_projects_dir(env_override=self._options.env)
        fname = f"{self._session_id}.jsonl"
        with contextlib.suppress(OSError):
            for p in projects.rglob(fname):
                return p

        o = self._options
        may_fork = bool(o.fork_session or o.resume or o.continue_conversation)
        project_dir = self._transcript_path.parent if self._transcript_path else None
        if not may_fork or project_dir is None or not project_dir.is_dir():
            return None

        best: Path | None = None
        best_mtime = self._spawn_time  # only files touched at/after spawn qualify
        with contextlib.suppress(OSError):
            for p in project_dir.glob("*.jsonl"):
                mtime = p.stat().st_mtime
                if mtime >= best_mtime:
                    best, best_mtime = p, mtime
        return best

    # ------------------------------------------------------------------ #
    # Background loops
    # ------------------------------------------------------------------ #

    async def _drain_loop(self) -> None:
        """Continuously read and discard PTY output.

        The TUI keeps rendering to the terminal; if we never drain the master
        end the pty buffer fills and the CLI blocks. We do not parse this
        output -- the transcript file is the source of truth.
        """
        assert self._master_fd is not None
        fd = self._master_fd
        while not self._closed:
            try:
                data = await anyio.to_thread.run_sync(
                    self._blocking_read, fd, abandon_on_cancel=True
                )
            except anyio.get_cancelled_exc_class():
                raise
            except Exception:
                break
            if not data:
                break
            # Keep a bounded tail for diagnostics (e.g. early-exit error text).
            self._recent_output = (self._recent_output + data)[-4096:]
            # Feed the terminal emulator so detect_question() can read the
            # current screen. Defensive suppress: a malformed escape must never
            # take down the drain loop.
            if self._question_stream is not None:
                with contextlib.suppress(Exception):
                    self._question_stream.feed(data)

    def _init_question_screen(self) -> None:
        """Create the pyte screen used by detect_question(), if pyte is present.

        Geometry matches the PTY winsize so the emulated screen mirrors what the
        CLI draws. Absence of pyte is non-fatal: detect_question() returns None.
        """
        try:
            import pyte
        except ImportError:
            self._question_screen = None
            self._question_stream = None
            return
        self._question_screen = pyte.Screen(SCREEN_COLS, SCREEN_ROWS)
        self._question_stream = pyte.ByteStream(self._question_screen)

    def detect_question(self) -> DetectedQuestion | None:
        """Return the question currently blocking the TUI, or None.

        Reconstructs the rendered screen and parses any tool-permission prompt,
        plan-approval, AskUserQuestion form, or app-level confirmation into a
        structured :class:`DetectedQuestion`. Requires the optional ``pyte``
        dependency (the ``pty-introspect`` extra); returns None without it.
        """
        screen = self._question_screen
        if screen is None:
            return None
        return parse_question([line.rstrip() for line in screen.display])

    @staticmethod
    def _blocking_read(fd: int) -> bytes:
        try:
            return os.read(fd, 65536)
        except OSError:
            return b""

    async def _question_watch_loop(self) -> None:
        """Poll for a blocking TUI dialog and answer it so the turn completes.

        Permission/plan prompts render in the TUI and block the CLI; with no one
        to answer them the turn hangs forever (C6). When a ``can_use_tool``
        callback is configured we route the decision through it (C5); otherwise
        we apply a safe default driven by the permission mode.
        """
        try:
            while not self._closed:
                await anyio.sleep(0.25)
                question = self.detect_question()
                if question is None:
                    continue
                fp = self._question_fingerprint(question)
                if fp in self._answered_questions:
                    continue
                with contextlib.suppress(Exception):
                    answered = await self._answer_question(question)
                    if answered:
                        self._answered_questions.add(fp)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            logger.debug("Question watch loop failed", exc_info=True)

    @staticmethod
    def _question_fingerprint(question: DetectedQuestion) -> str:
        """Stable key for a detected dialog (so we answer it at most once)."""
        opts = "|".join(f"{o.index}:{o.label}" for o in question.options)
        return f"{question.kind}::{question.tool}::{question.target}::{opts}"

    async def _answer_question(self, question: DetectedQuestion) -> bool:
        """Decide and send the answer for a blocking dialog. Returns success.

        Only permission and plan dialogs are auto-answered. AskUserQuestion and
        app-level dialogs need real user/content input that the SDK consumer must
        supply, so they are left for the caller (and would otherwise have been
        un-answerable over stream-json too).
        """
        if question.kind not in ("permission", "plan"):
            return False

        want = await self._decide_permission(question)
        option = choose_option(question, want)
        if option is None:
            return False
        await self._send_option_choice(option)
        # Record a denial for the result's permission_denials, mirroring the
        # stream-json field shape (tool + input target).
        if want == "deny":
            self._turn_permission_denials.append(
                {
                    "tool_name": question.tool,
                    "tool_input": {"target": question.target}
                    if question.target
                    else {},
                }
            )
        return True

    async def _decide_permission(
        self, question: DetectedQuestion
    ) -> Literal["allow", "deny"]:
        """Return "allow" or "deny" for a permission/plan dialog.

        Routes through ``can_use_tool`` when configured (C5); otherwise uses a
        safe default keyed to the permission mode (C6): permissive modes allow so
        turns complete; ``default``/``plan`` allow-once as well (the dialog only
        appears for actions the CLI would otherwise gate, and hanging is worse
        for a drop-in consumer than completing). Callers wanting denial should
        provide ``can_use_tool``.
        """
        callback = self._options.can_use_tool
        if callback is not None and question.tool:
            context = ToolPermissionContext(
                tool_use_id=None,
                title=question.question,
                display_name=question.tool,
            )
            tool_input: dict[str, Any] = (
                {"target": question.target} if question.target else {}
            )
            try:
                result = await callback(question.tool, tool_input, context)
            except Exception:
                logger.debug("can_use_tool callback raised; denying", exc_info=True)
                return "deny"
            # Defensive: treat anything that is not an explicit Allow as deny
            # (covers a callback that returns a malformed value at runtime).
            return "allow" if isinstance(result, PermissionResultAllow) else "deny"
        # No callback: allow so the turn completes (C6). The prompt only appears
        # in modes that gate; consumers that need gating should pass can_use_tool
        # or use disallowed_tools / a restrictive permission mode.
        return "allow"

    async def _send_option_choice(self, option: QuestionOption) -> None:
        """Answer a numbered dialog by typing the option digit then Enter."""
        async with self._write_lock:
            await self._pty_write(str(option.index).encode("ascii"))
            await anyio.sleep(0.1)
            await self._pty_write(_SUBMIT)

    async def _tail_loop(self) -> None:
        """Tail the transcript file and translate new records into messages."""
        path: Path | None = None
        offset = 0
        buffer = b""
        try:
            while not self._closed:
                if path is None:
                    path = self._resolve_transcript_path()
                    if path is not None:
                        self._transcript_path = path

                size = 0
                if path is not None:
                    try:
                        size = path.stat().st_size
                    except OSError:
                        size = 0

                    # The transcript was truncated/compacted (rewritten shorter):
                    # re-read from the start. The uuid dedup in _emit_line keeps
                    # already-seen records from being emitted twice.
                    if size < offset:
                        offset = 0
                        buffer = b""

                    if size > offset:
                        with path.open("rb") as f:
                            f.seek(offset)
                            chunk = f.read()
                            offset = f.tell()
                        buffer += chunk
                        *lines, buffer = buffer.split(b"\n")
                        for raw_line in lines:
                            await self._emit_line(raw_line)

                # One-shot termination: input has been closed and the turn
                # finished, so there is nothing more to wait for.
                if self._input_ended and self._result_emitted:
                    break

                # Process exited; drain any remaining content, then stop.
                if self._proc is not None and self._proc.poll() is not None:
                    if path is not None and size > offset:
                        await anyio.sleep(0.05)
                        continue
                    await self._handle_early_exit(self._proc.returncode)
                    break

                await anyio.sleep(0.1)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            logger.debug("Transcript tail loop failed", exc_info=True)
        finally:
            if self._out_send is not None:
                with contextlib.suppress(Exception):
                    self._out_send.close()

    async def _handle_early_exit(self, returncode: int | None) -> None:
        """If the CLI exited non-zero without producing a turn, report it.

        Surfaces a result message with ``is_error`` so consumers see a real
        failure (e.g. an unusable flag) instead of an empty, silent stream.
        """
        if returncode in (None, 0) or self._result_emitted:
            return
        text = re.sub(
            r"\x1b\[[0-9;?]*[A-Za-z]",
            "",
            self._recent_output.decode("utf-8", "replace"),
        )
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())[
            -500:
        ]
        self._result_emitted = True
        await self._send(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "duration_ms": 0,
                "duration_api_ms": 0,
                "is_error": True,
                "num_turns": self._turn_count,
                "session_id": self._session_id,
                "result": text or f"Claude Code exited with code {returncode}",
                "errors": [text] if text else [f"exit code {returncode}"],
                "uuid": str(uuid.uuid4()),
            }
        )

    async def _emit_line(self, raw_line: bytes) -> None:
        line = raw_line.strip()
        if not line:
            return
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(entry, dict):
            return
        entry_type = entry.get("type")

        # The CLI writes a permission-mode record whenever the live mode changes
        # (e.g. after a shift+tab cycle or a launch flag takes effect). Use it as
        # the source of truth for the current mode so set_permission_mode can
        # confirm the change and does not drift (H1, L5). It is otherwise not an
        # SDK-visible message.
        if entry_type == "permission-mode":
            mode = entry.get("permissionMode")
            if isinstance(mode, str) and mode:
                self._permission_mode = mode
            return

        if entry_type in _SKIP_TRANSCRIPT_TYPES:
            return

        # Track the real session id the CLI is using (resume/fork may differ
        # from our generated one). Used to keep session_id consistent on every
        # emitted message and on the result.
        sid = entry.get("sessionId")
        if isinstance(sid, str) and sid:
            self._observed_session_id = sid

        # Dedup by uuid so a compaction-triggered re-read can't double-emit.
        # Bounded LRU-by-insertion (R9): cap the set so a long interactive
        # session doesn't retain every record forever.
        uid = entry.get("uuid")
        if isinstance(uid, str):
            if uid in self._seen_uuids:
                return
            self._seen_uuids[uid] = None
            while len(self._seen_uuids) > _SEEN_UUIDS_MAX:
                self._seen_uuids.popitem(last=False)

        # turn_duration is the turn-complete signal -> synthesize a result that
        # carries the turn's final text, accumulated usage, and error state.
        if entry_type == "system" and entry.get("subtype") == "turn_duration":
            await self._emit_result(entry)
            return

        message = _translate_transcript_entry(entry, self._session_id)
        if message is None:
            return

        if entry_type == "assistant":
            msg = message["message"]
            text = _extract_text(msg)
            if text:
                self._turn_text = text
            model = msg.get("model")
            if isinstance(model, str) and model and model != "unknown":
                self._turn_model = model
            stop_reason = msg.get("stop_reason")
            if isinstance(stop_reason, str):
                self._turn_stop_reason = stop_reason
            # Dedup by message.id: the interactive transcript writes MULTIPLE
            # snapshots of the same assistant message as it streams (same
            # message.id, distinct top-level uuid), so the uuid dedup above does
            # not collapse them. add() keeps only the final snapshot per id, and
            # we suppress re-emitting an already-seen assistant id to consumers
            # so the same message is not delivered several times.
            msg_id = msg.get("id")
            self._turn_usage.add(msg_id, model, msg.get("usage"))
            err = msg.get("error")
            if err or stop_reason == "refusal":
                self._turn_is_error = True
                self._turn_subtype = self._error_subtype(err, stop_reason)
            if isinstance(msg_id, str) and msg_id:
                if msg_id in self._seen_assistant_ids:
                    return  # already emitted this assistant message; drop snapshot
                self._seen_assistant_ids.add(msg_id)
        elif entry_type == "user":
            # num_turns counts API round-trips: each tool_result user record is
            # the model being called again with the tool output. Counted here;
            # num_turns = tool_results + 1 (the final answer turn).
            if _has_tool_result(message.get("message")):
                self._turn_tool_results += 1

        await self._send(message)

    @staticmethod
    def _error_subtype(error: Any, stop_reason: str | None) -> str:
        """Map an assistant error/stop_reason to a result subtype.

        Mirrors the stream-json result subtypes so consumers branching on
        ``subtype`` (refusal, max-turns, budget) keep working.
        """
        if stop_reason == "refusal":
            return "error_during_execution"
        if isinstance(error, str):
            low = error.lower()
            if "max_turns" in low or "max turns" in low:
                return "error_max_turns"
            if "budget" in low:
                return "error_max_budget_usd"
        return "error_during_execution"

    async def _emit_result(self, entry: dict[str, Any]) -> None:
        """Synthesize and emit a result from a ``turn_duration`` record."""
        self._turn_count += 1
        duration = entry.get("durationMs", 0)
        # num_turns = (tool_result records this turn) + 1. Each tool_result is
        # one API round-trip back to the model; the +1 is the final answer turn.
        # Verified against live stream-json num_turns across many prompts. The
        # turn_duration record's own ``messageCount`` counts streamed snapshots
        # (NOT API turns) and over-counts badly, so it is deliberately not used.
        num_turns = self._turn_tool_results + 1

        subtype = (
            (self._turn_subtype or "error_during_execution")
            if self._turn_is_error
            else "success"
        )
        result: dict[str, Any] = {
            "type": "result",
            "subtype": subtype,
            "duration_ms": duration,
            # No per-API duration is recorded in the transcript; the turn's
            # wall-clock duration is the closest faithful value.
            "duration_api_ms": duration,
            "is_error": self._turn_is_error,
            "num_turns": num_turns,
            "session_id": self._observed_session_id
            or entry.get("sessionId")
            or self._session_id,
            "result": self._turn_text or None,
            "stop_reason": self._turn_stop_reason,
            "uuid": entry.get("uuid"),
        }
        if self._turn_usage.has_data():
            result["usage"] = self._turn_usage.aggregate_usage()
            cost = self._turn_usage.total_cost()
            if cost is not None:
                result["total_cost_usd"] = cost
            model_usage = self._turn_usage.model_usage()
            if model_usage is not None:
                # Emit under the camelCase ``modelUsage`` wire key: the real CLI
                # uses that key and ``message_parser.parse_message`` reads
                # ``data.get("modelUsage")`` (R1). Writing snake_case
                # ``model_usage`` here made ``ResultMessage.model_usage`` always
                # None at the consumer.
                result["modelUsage"] = model_usage
        # permission_denials: empty list (faithful default; stream-json always
        # sent a list, never None, so formatting that iterates it works).
        result["permission_denials"] = list(self._turn_permission_denials)
        # structured_output: when a json_schema output_format was requested, the
        # final assistant text is the structured JSON; parse it so the result
        # carries structured_output like the stream-json baseline (H6).
        structured = self._extract_structured_output()
        if structured is not None:
            result["structured_output"] = structured
        self._result_emitted = True
        await self._send(result)
        self._reset_turn_state()

    def _reset_turn_state(self) -> None:
        """Reset per-turn accumulators for the next turn."""
        self._turn_text = ""
        self._turn_usage = TurnUsageAccumulator()
        self._turn_is_error = False
        self._turn_subtype = None
        self._turn_stop_reason = None
        self._turn_tool_results = 0
        self._seen_assistant_ids = set()
        self._turn_permission_denials = []
        self._turn_start_time = None

    async def _emit_interrupt_result(self) -> None:
        """Synthesize a terminating result after an ``interrupt`` (R4).

        ESC aborts the in-progress turn, so the CLI never writes a
        ``turn_duration`` record and ``_emit_result`` never fires. The
        stream-json baseline produced an ``error_during_execution`` result on
        interrupt so ``receive_response()`` terminates; without one the
        documented interrupt pattern hangs forever. We mirror the baseline:
        ``subtype=error_during_execution``, ``is_error=True``, ``result=None``,
        carrying whatever usage/turn count accumulated before the abort.
        """
        if self._result_emitted:
            return
        self._turn_count += 1
        duration = 0
        if self._turn_start_time is not None:
            duration = int((time.monotonic() - self._turn_start_time) * 1000)
        result: dict[str, Any] = {
            "type": "result",
            "subtype": "error_during_execution",
            "duration_ms": duration,
            "duration_api_ms": duration,
            "is_error": True,
            # Each tool round-trip already happened; +1 for the aborted turn,
            # matching the baseline's interrupt num_turns shape.
            "num_turns": self._turn_tool_results + 1,
            "session_id": self._observed_session_id or self._session_id,
            # Baseline interrupt result carries no text and no stop_reason.
            "result": None,
            "stop_reason": None,
            "uuid": str(uuid.uuid4()),
        }
        if self._turn_usage.has_data():
            result["usage"] = self._turn_usage.aggregate_usage()
            cost = self._turn_usage.total_cost()
            if cost is not None:
                result["total_cost_usd"] = cost
            model_usage = self._turn_usage.model_usage()
            if model_usage is not None:
                result["modelUsage"] = model_usage
        result["permission_denials"] = list(self._turn_permission_denials)
        self._result_emitted = True
        await self._send(result)
        self._reset_turn_state()

    def _extract_structured_output(self) -> Any | None:
        """Parse the turn's final text as structured output, if requested.

        Only when ``options.output_format`` is a json_schema format -- in that
        mode the CLI constrains the final assistant text to the schema, so the
        text is valid JSON. Returns the parsed value, or ``None`` if no schema
        was requested or the text is not parseable JSON.
        """
        of = self._options.output_format
        if not (isinstance(of, dict) and of.get("type") == "json_schema"):
            return None
        text = self._turn_text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Some CLIs wrap the JSON in a fenced code block; try to recover.
            stripped = text.strip("`").strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                logger.debug("structured_output text was not valid JSON")
                return None

    async def _send(self, message: dict[str, Any]) -> None:
        if self._out_send is not None:
            with contextlib.suppress(
                anyio.BrokenResourceError, anyio.ClosedResourceError
            ):
                await self._out_send.send(message)

    # ------------------------------------------------------------------ #
    # Transport interface
    # ------------------------------------------------------------------ #

    async def write(self, data: str) -> None:
        if not self._ready:
            raise CLIConnectionError("PtyCLITransport is not ready for writing")
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            msg_type = obj.get("type")
            if msg_type == "control_request":
                await self._handle_control_request(obj)
            elif msg_type == "user":
                await self._handle_user_message(obj)
            # Other frame types have no interactive equivalent; ignore them.

    # Control subtypes with a faithful interactive equivalent. Anything else is
    # answered with an error control_response (rather than a fake success) so
    # callers get a clear failure instead of a silent no-op.
    _SUPPORTED_CONTROLS = frozenset(
        {
            "initialize",
            "interrupt",
            "set_permission_mode",
            "set_model",
            "mcp_status",
        }
    )

    async def _handle_control_request(self, obj: dict[str, Any]) -> None:
        """Map an SDK control request to its interactive-TUI equivalent.

        Supported: ``initialize`` (local ack), ``interrupt`` (ESC),
        ``set_permission_mode`` (shift+tab cycling), ``set_model`` (``/model``).
        Everything else (mcp_status, get_context_usage, mcp_reconnect,
        mcp_toggle, stop_task, rewind_files, ...) has no interactive channel
        that returns data, so it gets an explicit error control_response.
        """
        request = obj.get("request", {})
        subtype = request.get("subtype")
        request_id = obj.get("request_id")
        error: str | None = None
        payload: dict[str, Any] = {}

        try:
            if subtype == "initialize":
                # Return the initialize CONTROL-RESPONSE shape (R5) -- the key
                # set get_server_info() consumers expect (commands,
                # available_output_styles, models, account, pid, agents,
                # output_style) -- NOT the system/init MESSAGE shape.
                payload = self._build_server_info()
            elif subtype == "interrupt":
                async with self._write_lock:
                    await self._pty_write(_INTERRUPT)
                # ESC aborts the turn with no turn_duration record, so synthesize
                # a terminating result (R4) or receive_response() hangs forever.
                await self._emit_interrupt_result()
            elif subtype == "set_permission_mode":
                error = await self._set_permission_mode(request.get("mode"))
            elif subtype == "set_model":
                error = await self._set_model(request.get("model"))
            elif subtype == "mcp_status":
                payload = self._mcp_status_payload()
            elif subtype not in self._SUPPORTED_CONTROLS:
                error = (
                    f"control request '{subtype}' is not supported by the "
                    "interactive transport"
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("Interactive control action failed", exc_info=True)
            error = f"interactive control action failed: {e}"

        if error is not None:
            await self._send(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "error",
                        "request_id": request_id,
                        "error": error,
                    },
                }
            )
        else:
            await self._send(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": request_id,
                        "response": payload,
                    },
                }
            )

    def _mcp_status_payload(self) -> dict[str, Any]:
        """Best-effort MCP status from the configured servers (C4).

        The interactive transcript exposes no live MCP connection state, so we
        report each configured server with a ``pending`` status (the CLI's
        "configured but state unknown" value) and its config. This is a faithful
        drop-in shape (``{"mcpServers": [...]}``) rather than raising, while not
        fabricating a ``connected`` status we cannot verify.
        """
        servers: list[dict[str, Any]] = []
        o = self._options
        if isinstance(o.mcp_servers, dict):
            for name, cfg in o.mcp_servers.items():
                entry: dict[str, Any] = {"name": name, "status": "pending"}
                if isinstance(cfg, dict):
                    entry["config"] = {k: v for k, v in cfg.items() if k != "instance"}
                servers.append(entry)
        return {"mcpServers": servers}

    async def _set_permission_mode(self, mode: str | None) -> str | None:
        """Cycle the TUI permission mode to ``mode`` via shift+tab.

        Returns an error string if the mode can't be reached by cycling.
        """
        if mode is None or mode == self._permission_mode:
            return None
        if (
            mode not in _PERMISSION_CYCLE
            or self._permission_mode not in _PERMISSION_CYCLE
        ):
            # bypassPermissions / dontAsk / auto are not in the shift+tab cycle;
            # they can only be set via the launch flag, not live.
            return (
                f"permission mode {mode!r} cannot be set live over the "
                f"interactive transport (only {', '.join(_PERMISSION_CYCLE)} "
                "are reachable; set others via options.permission_mode)"
            )
        await self._warmup()
        async with self._write_lock:
            # Compute steps from the live mode (kept current by the
            # permission-mode transcript records), so we don't drift from a
            # stale assumption about the starting mode (L5).
            start = self._permission_mode
            current = (
                _PERMISSION_CYCLE.index(start) if start in _PERMISSION_CYCLE else 0
            )
            target = _PERMISSION_CYCLE.index(mode)
            steps = (target - current) % len(_PERMISSION_CYCLE)
            for _ in range(steps):
                await self._pty_write(_SHIFT_TAB)
                await anyio.sleep(0.1)
        # Optimistically record the requested mode. The CLI then writes a
        # permission-mode transcript record with the actual mode, which the tail
        # loop reads into self._permission_mode -- so any drift (L5) or failure
        # to apply (H1) is corrected automatically from the CLI's own state
        # without blocking this call on a transcript round-trip.
        self._permission_mode = mode
        return None

    async def _set_model(self, model: str | None) -> str | None:
        """Switch the model via the ``/model`` slash command.

        The interactive ``/model`` command applies the change; subsequent
        assistant messages in the transcript carry the new ``model`` id, which
        the tail loop records as ``self._turn_model``. We optimistically track
        the requested model so a follow-up set_model computes from it.
        """
        if not model:
            return None
        await self._run_slash_command(f"/model {model}")
        self._turn_model = model
        return None

    async def _run_slash_command(self, command: str) -> None:
        await self._warmup()
        async with self._write_lock:
            await self._pty_write(command.encode("utf-8"))
            await anyio.sleep(0.2)
            await self._pty_write(_SUBMIT)

    async def _handle_user_message(self, obj: dict[str, Any]) -> None:
        message = obj.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            # Warn (once) about structured blocks that can't be typed into a TUI.
            dropped = {
                b.get("type")
                for b in content
                if isinstance(b, dict) and b.get("type") not in ("text",)
            }
            if dropped:
                logger.warning(
                    "Dropping non-text content blocks the interactive transport "
                    "cannot send: %s",
                    ", ".join(sorted(str(d) for d in dropped)),
                )
        else:
            text = ""
        if text.strip():
            await self._type_prompt(text)

    async def _type_prompt(self, text: str) -> None:
        await self._warmup()
        # Bracketed paste makes the TUI insert the text verbatim, preserving
        # newlines and most special characters, after which a single CR submits.
        # Exception: a leading "/", "!" or "#" still triggers the TUI's
        # slash/bash/memory mode even when pasted (and Enter then fails to
        # submit), so prepend a single space in that one case. The model treats
        # leading whitespace as insignificant.
        if text[:1] in ("/", "!", "#"):
            text = " " + text
        payload = _PASTE_START + text.encode("utf-8") + _PASTE_END
        # A new turn is starting: mark it in flight and reset the
        # turn-complete latch so an interrupt or turn_duration for this turn can
        # synthesize a result.
        self._turn_start_time = time.monotonic()
        self._result_emitted = False
        async with self._write_lock:
            await self._pty_write(payload)
            await anyio.sleep(0.3)
            await self._pty_write(_SUBMIT)

    async def _warmup(self) -> None:
        """Wait for the TUI to render and dismiss the startup notification.

        The interactive CLI shows a transient startup toast that swallows the
        first Enter. Letting the UI settle and sending one dismissal Enter
        (an empty prompt is a no-op) ensures the subsequent submit registers.
        """
        if self._warmed_up:
            return
        self._warmed_up = True
        elapsed = time.monotonic() - self._spawn_time
        if elapsed < _WARMUP_SECONDS:
            await anyio.sleep(_WARMUP_SECONDS - elapsed)
        await self._pty_write(_SUBMIT)  # dismiss startup toast
        await anyio.sleep(0.5)

    async def _pty_write(self, data: bytes) -> None:
        if self._master_fd is None:
            return
        fd = self._master_fd

        def _write_all() -> None:
            view = data
            while view:
                written = os.write(fd, view)
                view = view[written:]

        with contextlib.suppress(OSError):
            await anyio.to_thread.run_sync(_write_all)

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        return self._read_messages_impl()

    async def _read_messages_impl(self) -> AsyncIterator[dict[str, Any]]:
        if self._out_recv is None:
            raise CLIConnectionError("Not connected")
        async for message in self._out_recv:
            yield message

    async def end_input(self) -> None:
        # Interactive mode has no stdin EOF semantics; record that the caller is
        # done so the tail loop can terminate after the current turn (one-shot).
        self._input_ended = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready = False
        _ACTIVE_CHILDREN.discard(self)

        for task in (self._tail_task, self._drain_task, self._question_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(Exception):
                    await task.wait()
        self._tail_task = None
        self._drain_task = None
        self._question_task = None

        if self._out_send is not None:
            with contextlib.suppress(Exception):
                self._out_send.close()

        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            with contextlib.suppress(Exception):
                await anyio.to_thread.run_sync(self._wait_proc, abandon_on_cancel=True)

        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._master_fd)
            self._master_fd = None

        self._proc = None

    def _wait_proc(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=5)

    def _terminate_process(self) -> None:
        """Synchronously kill the child process group and close the master fd.

        Used by the ``atexit`` cleanup so a caller that crashes without awaiting
        ``close()`` does not leak the interactive ``claude`` process or its fd.
        """
        proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._master_fd)
            self._master_fd = None
        _ACTIVE_CHILDREN.discard(self)

    def is_ready(self) -> bool:
        return self._ready
