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

``hooks`` and ``can_use_tool`` ARE supported via the settings-hook IPC bridge
(see :mod:`._hook_ipc` / :mod:`._hook_shim`): connect() starts a localhost IPC
server and synthesizes a ``--settings`` ``hooks`` block wiring the relevant hook
events to a tiny shim command. The CLI runs the shim, which forwards the
hook-event JSON to the SDK, which dispatches to the user's programmatic
``options.hooks`` callbacks and -- for ``PreToolUse`` when ``can_use_tool`` is
set -- routes the tool permission through it DETERMINISTICALLY (the hook's
``permissionDecision``/``updatedInput`` short-circuits the CLI's permission
flow, so ``PermissionResultAllow(updated_input=...)`` actually changes the
executed tool input -- something the TUI dialog cannot express). If the bridge
cannot start, the transport falls back to the screen-scrape watcher: the
interactive CLI renders tool-permission prompts as on-screen dialogs, which a
background watcher detects (via the ``pty_question`` screen parser) and answers
by keystroke, routing through ``can_use_tool`` when provided or a safe default
otherwise so turns never hang. The watcher always handles plan-approval /
AskUserQuestion / app dialogs (those are not PreToolUse hooks).

Unsupported options (interactive mode has no equivalent SDK channel). These are
rejected up front by :meth:`PtyCLITransport._validate_options` with an
actionable error rather than failing silently or hanging:

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
import shutil
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
    PermissionUpdate,
    ToolPermissionContext,
)
from .._task_compat import TaskHandle, spawn_detached
from ..sessions import _canonicalize_path, _get_projects_dir, _sanitize_path
from . import Transport, _cli_command
from ._api_monitor import ApiMonitor
from ._hook_ipc import ENV_HOOK_IPC, HookIpcServer, build_hooks_settings
from ._usage import TurnUsageAccumulator, _limits_for
from .pty_question import (
    SCREEN_COLS,
    SCREEN_ROWS,
    DetectedQuestion,
    QuestionOption,
    TaskDialogRow,
    choose_option,
    parse_question,
    parse_tasks_dialog,
    task_label_key,
    task_row_matches,
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
# Down-arrow moves the selection in list dialogs; ``x`` kills the selected task
# in the /tasks BackgroundTasksDialog (no confirmation). ESC closes the dialog
# (it is also the turn-interrupt key, so it is only sent to dismiss a dialog we
# opened).
_ARROW_DOWN = b"\x1b[B"
_KILL_TASK = b"x"

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

# Background-task tracking. Tools that spawn work which outlives the turn that
# started it. ``Bash`` only spawns a background task when its
# ``run_in_background`` input is set, so it is matched separately. ``Agent``
# /``Task`` cover the sub-agent tool under either name the CLI version uses.
# Shared by the turn-hold tracker AND the stop_task spawn detector
# (``_is_task_spawn``).
_BACKGROUND_TASK_TOOLS = frozenset({"Monitor", "Agent", "Task", "Workflow"})

# Terminal <status> values in a <task-notification>: once a task reports one of
# these it is done and stops holding the turn open. ``pending``/``running`` are
# non-terminal and keep the turn alive.
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "killed"})

# A <task-notification> is injected as a user transcript record when a background
# task changes state. We pull the tool-use id and status out of its XML to drain
# the pending set. Tags can appear in any order, so each is matched on its own.
_TASK_NOTIFICATION_RE = re.compile(
    r"<task-notification>(.*?)</task-notification>", re.DOTALL
)
_TASK_TOOL_USE_ID_RE = re.compile(r"<tool-use-id>\s*([^<\s]+)\s*</tool-use-id>")
_TASK_STATUS_RE = re.compile(r"<status>\s*([^<\s]+)\s*</status>")

# Safety cap on how long the turn is held open waiting for background tasks. The
# CLI bounds its own tasks (e.g. Monitor's max watch window), so this is a
# backstop against a task that never reports terminal: once exceeded, the turn
# finishes with whatever has accumulated. Matches the CLI's 1h Monitor ceiling.
_BACKGROUND_TASK_WAIT_CAP_SECS = 3600.0

# Order the TUI cycles through on shift+tab. bypassPermissions is not part of
# the cycle (it is only reachable via launch flag), so it cannot be set live.
_PERMISSION_CYCLE = ("default", "acceptEdits", "plan")

# The short task-id the CLI assigns (used by stop_task to identify a task) is
# reported in the spawning tool's RESULT text (not its input), with tool-specific
# phrasing -- e.g. "Task ID: w2tsm5ui9" or "Monitor task started with ID: ...".
# This matches the id token after either phrasing (lower/upper, optional
# backticks). The <task-notification> regex + terminal-status set are shared with
# the turn-hold tracker above.
_TASK_ID_RE = re.compile(
    r"(?:task\s+id|started\s+with\s+id)\b\s*[:=]?\s*`?([A-Za-z0-9][A-Za-z0-9_-]{2,})`?",
    re.IGNORECASE,
)

# Upper bound on tracked tasks / pending spawns so a long session cannot grow
# either dict without limit (a backgrounded tool whose result never yields a
# task-id, or a task that completes without a terminal notification, would
# otherwise leak). Oldest entries are evicted first.
_MAX_TRACKED_TASKS = 256

# Upper bound on keystroke iterations while navigating the tasks dialog, so a
# misparsed/never-matching dialog can never loop unbounded.
_MAX_TASK_NAV_STEPS = 64


def _xml_tag(block: str, tag: str) -> str | None:
    """Return the trimmed text of ``<tag>...</tag>`` within ``block``, or None."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL)
    return m.group(1).strip() if m else None


def _bound_dict(d: dict[str, Any], max_size: int) -> None:
    """Evict oldest insertion-ordered entries until ``d`` fits ``max_size``."""
    while len(d) > max_size:
        d.pop(next(iter(d)))


def _is_task_spawn(name: str, tool_input: dict[str, Any]) -> bool:
    """True if a tool_use of ``name`` with ``tool_input`` spawns a background task."""
    if name in _BACKGROUND_TASK_TOOLS:
        return True
    # Bash is only a background task when explicitly backgrounded.
    return name == "Bash" and bool(tool_input.get("run_in_background"))


def _task_description(name: str, tool_input: dict[str, Any]) -> str:
    """Best-effort human description for a spawned task, for dialog row matching.

    Mirrors what the TUI renders per task type: a Bash/Monitor task shows its
    command; an Agent/Workflow task shows its description/prompt. Falls back to
    the first string-valued input so a row is still matchable.
    """
    if name in ("Bash", "Monitor"):
        candidate = tool_input.get("command") or tool_input.get("description")
    elif name in ("Agent", "Task"):
        candidate = tool_input.get("description") or tool_input.get("prompt")
    elif name == "Workflow":
        candidate = (
            tool_input.get("description")
            or tool_input.get("name")
            or tool_input.get("workflow")
        )
    else:
        candidate = None
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    for value in tool_input.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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

        # Effective env override consulted by env construction AND transcript
        # path resolution (L4). It starts as the caller's options.env and may be
        # augmented in connect() with an SDK-owned ``CLAUDE_CONFIG_DIR`` default
        # so the child never writes the user's real ~/.claude.json. Both the
        # child env (_build_env) and every transcript-path lookup read THIS dict,
        # so the config dir stays internally consistent (transcript tailing finds
        # the session .jsonl under the same dir the child writes to). We copy so
        # we never mutate the caller's ClaudeAgentOptions.env in place.
        self._effective_env: dict[str, str] = dict(options.env)

        self._proc: Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._transcript_path: Path | None = None
        # Byte offset to start tailing at. For a brand-new session this is 0; for
        # a --resume/--continue against an existing transcript it is the file's
        # size at spawn, so the pre-existing prior turn's records (especially the
        # old turn_duration) are NOT replayed as the current turn's result (RW3).
        self._initial_tail_offset = 0

        self._out_send: Any = None
        self._out_recv: Any = None
        self._drain_task: TaskHandle | None = None
        self._tail_task: TaskHandle | None = None
        self._question_task: TaskHandle | None = None
        # Read end of the child's dedicated stderr pipe (H3). stdout stays on the
        # PTY (the TUI needs a tty), but stderr is given a separate pipe so the
        # ``options.stderr`` callback can receive the CLI's real stderr lines --
        # matching the old stream-json transport's per-line behavior. ``None``
        # when no stderr callback is configured (we then leave stderr on the PTY).
        self._stderr_read_fd: int | None = None
        self._stderr_task: TaskHandle | None = None
        # Fingerprints of questions already answered, so the watcher does not
        # re-answer the same on-screen dialog while it lingers before redraw.
        self._answered_questions: set[str] = set()

        self._ready = False
        self._closed = False
        self._input_ended = False
        self._result_emitted = False
        # Background-task hold: tool-use ids of tasks the agent spawned that have
        # not yet reported a terminal <task-notification>. While non-empty, the
        # turn is held open past turn_duration so the work is not orphaned by
        # closing the session -- matching ``claude -p``, which blocks on
        # background tasks before exiting.
        self._pending_tasks: set[str] = set()
        # Monotonic deadline for the whole hold, set on the first deferral and
        # cleared when a real result is emitted. ``None`` means "not waiting".
        self._task_wait_deadline: float | None = None
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
        # Number of prompts submitted so far. The init message for turn 1 is
        # emitted at connect(); a fresh system/init is emitted before each
        # subsequent prompt to match the baseline's per-turn ordering (RR4).
        self._submitted_turns = 0
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
        # Tool names denied via can_use_tool this turn but not yet correlated to
        # the rejected tool_result transcript record. The TUI dialog only exposes
        # a target string, so we recover the real tool_use_id and full input by
        # matching the rejected ``tool_result`` (is_error) back to its
        # ``tool_use`` block (RR2). FIFO so repeated denials of the same tool keep
        # order.
        self._pending_denied_tools: list[str] = []
        # tool_use blocks seen this turn, keyed by tool_use id -> {name, input}.
        # Used to recover the full original input + id for a denial (RR2) and to
        # identify which rejected tool_results came from a permission deny so they
        # are excluded from num_turns (RR5).
        self._turn_tool_uses: dict[str, dict[str, Any]] = {}
        # tool_use_ids whose rejected tool_result came from a permission deny, so
        # _has_tool_result excludes them from the num_turns count (RR5).
        self._denied_tool_use_ids: set[str] = set()
        # Set when can_use_tool denied a tool this turn: the interactive CLI then
        # goes idle (no turn_duration record), so the tail loop synthesizes a
        # terminating result once the rejected tool_result lands (RR1).
        self._deny_terminated = False
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
        # Track message.ids whose message-level usage has already been counted.
        # The interactive transcript writes each content block of one assistant
        # message as a SEPARATE record sharing the same ``message.id`` (distinct
        # uuids), and EACH record repeats the SAME message-level ``usage``. We
        # emit every block-record as its own AssistantMessage (matching the
        # stream-json baseline's per-block granularity -- live-verified: the
        # baseline emits ``A[Thinking] | A[Text] | A[ToolUse] | U | A[Text]``,
        # one AssistantMessage per block, NOT a merged message), but the repeated
        # usage must be counted only ONCE per id (C2). This set records ids whose
        # usage has been counted (the TurnUsageAccumulator dedups too; this is
        # the belt-and-suspenders guard).
        self._seen_assistant_ids: set[str] = set()

        # ----- API monitor (always-on pure-relay proxy) ----------------- #
        # The transport interposes a loopback proxy between the CLI and the real
        # Anthropic API (see _api_monitor). It forwards bytes UNCHANGED and tees
        # a copy of each /v1/messages call so we can enrich the result with data
        # the transcript cannot show: per-call usage for ALL calls including the
        # auxiliary helper-model (e.g. haiku title) call the transcript never
        # records (C1/R3), real per-call API durations (R7), HTTP error statuses
        # (C1), and the FULL tool_use.input while a permission dialog blocks
        # (RV2). ALWAYS ON -- no config/env toggle; it is simply how connect()
        # launches the CLI. The monitor is None only if start() failed (a
        # non-fatal degradation: turns still run, just without the enrichment).
        self._api_monitor: ApiMonitor | None = None
        # /v1/messages call records observed in the current turn (one dict per
        # call, in arrival order). Consumed when synthesizing the result.
        self._turn_api_calls: list[dict[str, Any]] = []
        # Full tool inputs recovered from intercepted responses this turn, keyed
        # by tool_use_id -> {name, input}. Used to supply the FULL input to a
        # blocking can_use_tool dialog (RV2), correlating by tool_use_id or, when
        # the TUI only exposes a tool name, by the latest input seen for it.
        self._turn_tool_inputs: dict[str, dict[str, Any]] = {}
        # Most-recently-seen full input per tool NAME (RV2 correlation fallback:
        # while a permission dialog blocks the TUI exposes only the tool name,
        # not its id, but the tool_use is already in the intercepted response).
        self._turn_tool_input_by_name: dict[str, dict[str, Any]] = {}
        # Tool catalog (ordered tool NAMES) the CLI actually sent on a
        # /v1/messages request, captured from the request body the relay tees
        # (RL10). Unlike the thin options-derived defaults, this is the CLI's
        # real resolved tool list (built-ins + MCP + the options' allowed set),
        # so init/get_server_info report what the model was actually offered.
        # Persists across turns (the catalog is session-stable); the latest
        # request wins. ``None`` until the first /v1/messages request is seen.
        self._observed_tools: list[str] | None = None
        # Model id the CLI actually sent on a /v1/messages request (RL10): the
        # resolved main model, used to populate init/get_server_info ``model``.
        self._observed_request_model: str | None = None
        # Latest per-call usage seen on a successful /v1/messages response, used
        # by get_context_usage to report the live context size (RL11). The main
        # model's ``input_tokens`` (+ cache fields) of the most recent real call
        # approximate the current context window occupancy.
        self._latest_context_usage: dict[str, Any] | None = None
        self._latest_context_model: str | None = None

        # ----- Background-task registry (stop_task) --------------------- #
        # The interactive transport has NO list-tasks control channel, so the
        # set of live background tasks and their ids is reconstructed from the
        # transcript (see _track_tasks): a spawning tool_use, then the
        # tool_result that carries the assigned task-id, then a
        # ``<task-notification>`` whose terminal status retires the task. This
        # registry is what stop_task(task_id) consults to identify which row to
        # kill in the /tasks dialog. Keyed by task_id ->
        # {"tool_use_id", "tool", "description", "status", "output_file"}.
        self._tasks: dict[str, dict[str, Any]] = {}
        # Spawn tool_use blocks awaiting their task-id (the id arrives later, in
        # the tool_result TEXT, not the tool_use input). tool_use_id ->
        # {"tool", "description"}.
        self._pending_task_spawns: dict[str, dict[str, Any]] = {}

        # ----- Hook IPC bridge (win #3) --------------------------------- #
        # When the caller configures programmatic ``options.hooks`` and/or
        # ``options.can_use_tool``, connect() starts a localhost IPC server and
        # wires the relevant hook events to a shim command via a synthesized
        # ``--settings`` hooks block. The CLI runs the shim, which forwards the
        # hook-event JSON to this server; the server dispatches to the user's hook
        # callbacks and (for PreToolUse) routes tool permissions through
        # can_use_tool, returning the CLI's permissionDecision/updatedInput. This
        # is the DETERMINISTIC channel that the TUI-scraping watcher cannot match
        # (notably: applying ``updated_input``). ``None`` when there is nothing to
        # wire OR when the server failed to start (then we fall back to the
        # watcher for permissions; programmatic hooks simply do not fire).
        self._hook_ipc: HookIpcServer | None = None
        # True once can_use_tool is being routed through the hook channel above;
        # the watcher then must NOT also answer the (now-suppressed) PreToolUse
        # permission dialog -- the hook decision short-circuits it server-side.
        self._can_use_tool_via_hook = False

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        if self._proc is not None:
            return

        self._validate_options()

        # L4: isolate config writes to an SDK-owned dir (seeded from the user's
        # real config so auth/login survive) so the child never mutates the
        # user's ~/.claude.json. Must run before onboarding (which writes the
        # trust/onboarding flags) and before _build_env / transcript resolution.
        await anyio.to_thread.run_sync(self._resolve_config_dir)

        if self._cli_path is None:
            self._cli_path = await anyio.to_thread.run_sync(_cli_command.find_cli)

        if not os.environ.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
            await _cli_command.check_claude_version(self._cli_path)

        # Interactive mode shows first-run onboarding (theme/login) screens that
        # block programmatic input. Mark onboarding complete so the CLI drops
        # straight into the prompt. Best-effort and non-destructive.
        await anyio.to_thread.run_sync(self._ensure_onboarding_complete)

        # Start the always-on API monitor BEFORE building the env so the child's
        # ANTHROPIC_BASE_URL can be pointed at the loopback proxy (RV2/C1/R3/R7).
        # Must happen before Popen consumes _build_env().
        await self._start_api_monitor()

        # Start the hook IPC bridge (win #3) BEFORE building the command, so the
        # synthesized ``--settings`` hooks block (which references the shim) can be
        # merged into the CLI invocation and the shim's endpoint injected into the
        # child env. Non-fatal: on failure we fall back to the TUI watcher.
        await self._start_hook_ipc()

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

        # H3: give the child a SEPARATE stderr pipe when a stderr callback is
        # registered, so its stderr lines reach ``options.stderr`` (the old
        # transport's behavior) instead of being muxed onto the PTY and lost.
        # stdin/stdout stay on the slave PTY because the interactive TUI requires
        # a tty. With no callback, leave stderr on the PTY (unchanged behavior).
        stderr_write_fd: int | None = None
        if self._options.stderr is not None:
            r, w = os.pipe()
            self._stderr_read_fd = r
            stderr_write_fd = w
        stderr_dest = stderr_write_fd if stderr_write_fd is not None else slave_fd

        try:
            self._proc = Popen(  # noqa: S603 - cmd is built from vetted options
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=stderr_dest,
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
            self._close_stderr_pipe(stderr_write_fd)
            if not Path(self._cwd).exists():
                raise CLIConnectionError(
                    f"Working directory does not exist: {self._cwd}"
                ) from e
            raise CLINotFoundError(f"Claude Code not found at: {self._cli_path}") from e
        except Exception as e:
            os.close(master_fd)
            os.close(slave_fd)
            self._close_stderr_pipe(stderr_write_fd)
            raise CLIConnectionError(f"Failed to start Claude Code: {e}") from e

        os.close(slave_fd)  # parent keeps only the master end
        # The child owns the write end now; the parent only reads. Closing it
        # here means the reader sees EOF when the child exits (no fd leak).
        if stderr_write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(stderr_write_fd)
        self._master_fd = master_fd
        self._spawn_time = time.monotonic()
        _ACTIVE_CHILDREN.add(self)

        self._transcript_path = self._compute_transcript_path()

        # When resuming/continuing an existing transcript, start tailing past the
        # records already on disk so the prior turn's trailing turn_duration is
        # not replayed as a stale result for the NEW turn (RW3). For a brand-new
        # session the file does not exist yet, so the offset stays 0.
        if self._options.resume or self._options.continue_conversation:
            resume_path = self._resume_transcript_path()
            if resume_path is not None:
                self._transcript_path = resume_path
                with contextlib.suppress(OSError):
                    self._initial_tail_offset = resume_path.stat().st_size

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
        # H3: read the child's dedicated stderr pipe and invoke options.stderr
        # per line (only spawned when the pipe was created, i.e. a callback set).
        if self._stderr_read_fd is not None:
            self._stderr_task = spawn_detached(self._stderr_loop())
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
        # RL10: prefer the CLI's real resolved tool catalog observed on a teed
        # /v1/messages request (built-ins + MCP + the options' allowed set) over
        # the thin options-derived list. Falls back to the options when no
        # request has been seen yet (e.g. before the first turn).
        if self._observed_tools:
            tools: list[str] = list(self._observed_tools)
        else:
            tools = []
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
        # RL10: the model the CLI actually sent on a /v1/messages request is the
        # most authoritative resolved id; prefer it when available.
        model = self._observed_request_model or self._turn_model or o.model or ""

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

        # RL10: surface the CLI's real resolved tool catalog + model observed on
        # a teed /v1/messages request. The slash-command/model catalogs and
        # account identity are only available over the deleted stream-json
        # ``initialize`` channel, so those stay empty rather than fabricated; the
        # tool list and model, however, ARE recoverable from the traffic and are
        # exactly what the model was offered.
        tools: list[str] = list(self._observed_tools) if self._observed_tools else []
        model = (
            self._observed_request_model
            or self._turn_model
            or (self._options.model or "")
        )
        models = [model] if model else []

        return {
            "commands": [],
            "available_output_styles": ["default"],
            "output_style": "default",
            "models": models,
            "model": model,
            "tools": tools,
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
        # NOTE: programmatic ``hooks`` ARE now supported (win #3): connect() wires
        # them to the CLI's settings.json command-hook mechanism via a localhost
        # IPC shim (see _hook_ipc / _hook_shim and _start_hook_ipc below), so they
        # are no longer rejected here.
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
        # NOTE: include_partial_messages IS honored (RL9): the relay tees the
        # /v1/messages SSE stream and reconstructs stream_event/StreamEvent records,
        # so no warning is emitted for it.
        if o.include_hook_events:
            logger.warning(
                "include_hook_events is passed to the CLI but yields no "
                "HookEventMessage objects with the interactive transport: the "
                "CLI emits hook lifecycle events only on the stream-json stdout "
                "channel, not into the transcript the PTY tails (verified "
                "empirically -- no hook records appear in the transcript)."
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
        env = _cli_command.build_env(self._options, self._cwd, entrypoint="sdk-py")
        # L4: route the child at the SDK-owned config dir resolved in connect()
        # (only set when the caller/env did NOT pin CLAUDE_CONFIG_DIR), so the
        # child writes its onboarding/trust/login state there, never into the
        # user's real ~/.claude.json. build_env already merged options.env, so an
        # explicit caller value is preserved (we only add the SDK default).
        sdk_config_dir = self._effective_env.get("CLAUDE_CONFIG_DIR")
        if sdk_config_dir and "CLAUDE_CONFIG_DIR" not in self._options.env:
            env["CLAUDE_CONFIG_DIR"] = sdk_config_dir
        # Always-on API monitor: point the CHILD CLI at the loopback proxy so its
        # /v1/messages traffic is observed (and forwarded UNCHANGED to the real
        # upstream the proxy captured from the original ANTHROPIC_BASE_URL). The
        # user never sees this. If the monitor failed to start, leave the env as
        # is so the CLI talks to the real API directly (non-fatal degradation).
        if self._api_monitor is not None:
            env["ANTHROPIC_BASE_URL"] = self._api_monitor.base_url
        # Hook IPC bridge (win #3): tell the shim where to reach the SDK's IPC
        # endpoint (+ the auth token). Only set when the bridge actually started.
        if self._hook_ipc is not None:
            env[ENV_HOOK_IPC] = self._hook_ipc.spec
        return env

    def _upstream_base_url(self) -> str:
        """The real upstream the monitor forwards to.

        Read from the CURRENT ANTHROPIC_BASE_URL (options.env wins over the
        process env), defaulting to the public API. This is captured BEFORE we
        overwrite the child's value with the loopback proxy, so the proxy always
        forwards to wherever the CLI would have gone unmonitored.
        """
        return (
            self._options.env.get("ANTHROPIC_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        )

    async def _start_api_monitor(self) -> None:
        """Start the loopback pure-relay API monitor (always on; non-fatal)."""
        try:
            monitor = ApiMonitor(self._upstream_base_url(), self._on_api_call)
            await monitor.start()
            self._api_monitor = monitor
        except Exception:
            # Never let monitor startup break the turn -- fall back to the CLI
            # talking to the real API directly (just without the enrichment).
            logger.debug("API monitor failed to start; continuing", exc_info=True)
            self._api_monitor = None

    async def _start_hook_ipc(self) -> None:
        """Start the settings-hook IPC bridge when there is something to wire.

        Wires the user's programmatic ``options.hooks`` callbacks and -- when
        ``options.can_use_tool`` is set -- a deterministic PreToolUse permission
        route through the CLI's own settings.json command-hook mechanism (see
        :mod:`._hook_ipc`). Non-fatal: any failure leaves ``_hook_ipc`` None, so
        connect() falls back to the existing TUI watcher for permissions and
        programmatic hooks simply do not fire (the prior behavior).
        """
        o = self._options
        # ``permission_prompt_tool_name == "stdio"`` is the SDK-internal sentinel
        # the client sets when can_use_tool is provided; a caller-supplied tool
        # name was already rejected in _validate_options, so any truthy value here
        # means "route can_use_tool".
        route_can_use_tool = o.can_use_tool is not None
        hooks = self._normalized_hooks()
        if not hooks and not route_can_use_tool:
            return
        try:
            server = HookIpcServer(
                hooks=hooks,
                can_use_tool=o.can_use_tool if route_can_use_tool else None,
                permission_mode=o.permission_mode,
                on_permission_decision=self._on_hook_permission_decision,
            )
            await server.start()
            self._hook_ipc = server
            self._can_use_tool_via_hook = route_can_use_tool
        except Exception:
            logger.debug("hook IPC bridge failed to start; continuing", exc_info=True)
            self._hook_ipc = None
            self._can_use_tool_via_hook = False

    def _normalized_hooks(self) -> dict[str, list[Any]] | None:
        """Return ``options.hooks`` as ``{event: [HookMatcher, ...]}`` or None.

        ``ClaudeAgentOptions.hooks`` is typed ``dict[HookEvent, list[HookMatcher]]``
        but tolerated loosely; we only keep events that actually carry matchers.
        """
        raw = self._options.hooks
        if not raw:
            return None
        normalized: dict[str, list[Any]] = {}
        for event, matchers in raw.items():
            if matchers:
                normalized[str(event)] = list(matchers)
        return normalized or None

    def _on_hook_permission_decision(
        self, tool_name: str, decision: str, tool_use_id: str | None
    ) -> None:
        """Record a hook-channel ``can_use_tool`` deny for result correlation.

        Called from the IPC serve task right after a PreToolUse permission is
        decided. On a deny, queue the tool name so _correlate_denials can build
        the baseline-shaped ``permission_denials`` entry from the rejected
        ``tool_result`` that lands in the transcript -- the watcher path (which
        normally populates this) is bypassed when the hook short-circuits the
        dialog. Allows are not recorded (no denial entry needed).
        """
        if decision == "deny" and tool_name:
            self._pending_denied_tools.append(tool_name)

    def _on_api_call(self, record: dict[str, Any]) -> None:
        """Receive one observed /v1/messages call from the monitor (best-effort).

        Runs on the monitor's serve task after the bytes were already forwarded,
        so it can never affect the relay. Everything here is wrapped so a bad
        record cannot crash the monitor. We accumulate per-turn so the result can
        compute exact cost/usage across ALL calls (incl. helper-model calls),
        real api duration, and error statuses, and recover full tool inputs.
        """
        try:
            self._turn_api_calls.append(record)
            for block in record.get("content_blocks") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tid = block.get("id")
                name = block.get("name")
                inp = block.get("input")
                entry = {"name": name, "input": inp}
                if isinstance(tid, str) and tid:
                    self._turn_tool_inputs[tid] = entry
                if isinstance(name, str) and name:
                    self._turn_tool_input_by_name[name] = entry
            # RL10: capture the CLI's real resolved tool catalog + model from the
            # request body the relay teed, so init/get_server_info report what the
            # model was actually offered (not the thin options-derived defaults).
            self._capture_tool_catalog(record.get("request"))
            # RL11: remember the latest successful per-call usage as the live
            # context-size signal for get_context_usage.
            self._capture_context_usage(record)
            # RL9: when include_partial_messages is set, reconstruct and emit
            # stream_event messages from the SSE events the relay already saw.
            if self._options.include_partial_messages:
                self._emit_stream_events(record.get("sse_events"))
        except Exception:
            logger.debug("API monitor record handling failed", exc_info=True)

    def _capture_tool_catalog(self, request: Any) -> None:
        """Record the tool names + model from a teed /v1/messages request (RL10).

        The CLI's request body carries the actual resolved ``tools`` list it
        offered the model (built-ins + MCP + the options' allowed set) and the
        resolved ``model`` id -- neither of which the transcript exposes. The
        catalog is session-stable, so the latest request wins and the value
        persists across turns.
        """
        if not isinstance(request, dict):
            return
        raw_tools = request.get("tools")
        if isinstance(raw_tools, list):
            names: list[str] = []
            for tool in raw_tools:
                if isinstance(tool, dict):
                    name = tool.get("name")
                    if isinstance(name, str) and name and name not in names:
                        names.append(name)
            if names:
                self._observed_tools = names
        model = request.get("model")
        if isinstance(model, str) and model:
            self._observed_request_model = model

    def _capture_context_usage(self, record: dict[str, Any]) -> None:
        """Track the latest real per-call usage for get_context_usage (RL11).

        The most recent successful /v1/messages call's ``usage`` (input +
        cache_read + cache_creation tokens) approximates the current context
        window occupancy. Skip non-2xx / usage-less calls so an error response
        doesn't blank the live figure.
        """
        status = record.get("status")
        if not (isinstance(status, int) and 200 <= status < 300):
            return
        usage = record.get("usage")
        if not isinstance(usage, dict) or not usage:
            return
        self._latest_context_usage = usage
        model = record.get("model") or self._observed_request_model
        if isinstance(model, str) and model:
            self._latest_context_model = model

    def _emit_stream_events(self, sse_events: Any) -> None:
        """Emit one ``stream_event`` message per raw SSE event (RL9).

        Reconstructs the stream-json baseline's ``StreamEvent`` shape
        (``{type:"stream_event", uuid, session_id, event, parent_tool_use_id}``)
        from the Anthropic SSE events the relay teed. Runs on the monitor's serve
        task (same event loop), so it uses the non-blocking ``send_nowait`` to
        hand the messages to the output stream; a full buffer just drops the
        partial event (best-effort, never blocks the relay/turn). Emits nothing
        when the response was not streaming (empty ``sse_events``).

        N5 (necessary residual): unlike the stream-json baseline, which
        backpressures partial delivery through the consumer-paced stream, this
        cannot block. It runs on the monitor's serve task; blocking here would
        stall relay forwarding -- a transparency violation that would affect the
        actual upstream turn, which is strictly worse than dropping an
        observability-only partial. The drop only affects partial/stream_event
        messages (never result correctness), and the output buffer is already
        generous (default 1000 dicts) and user-tunable via
        ``ClaudeAgentOptions.max_buffer_size`` for slow consumers on long,
        ping-heavy turns. So the non-blocking drop is retained deliberately.
        """
        if not isinstance(sse_events, list) or not sse_events:
            return
        if self._out_send is None:
            return
        session_id = self._observed_session_id or self._session_id
        for event in sse_events:
            if not isinstance(event, dict) or not event.get("type"):
                continue
            message = {
                "type": "stream_event",
                "uuid": str(uuid.uuid4()),
                "session_id": session_id,
                "event": event,
                "parent_tool_use_id": None,
            }
            with contextlib.suppress(
                anyio.WouldBlock,
                anyio.BrokenResourceError,
                anyio.ClosedResourceError,
            ):
                self._out_send.send_nowait(message)

    def _resolve_config_dir(self) -> None:
        """Default ``CLAUDE_CONFIG_DIR`` to an SDK-owned dir, seeded from the user's
        real config, so connecting never writes the user's ~/.claude.json (L4).

        The CLI persists OAuth/login state and settings in ``~/.claude.json`` and
        writes onboarding/trust flags there. The old transport ran the CLI
        directly against the user's home, so a drop-in must not silently corrupt
        or churn that file. We instead point the child at a **persistent**,
        per-user SDK cache dir (not a throwaway-per-connect dir, so login state
        persists across runs) and seed it once from the user's real config so
        auth + settings carry over. ``_ensure_onboarding_complete`` then writes
        the flags into the SDK copy, leaving the user's file untouched.

        No-op when the caller or environment already set ``CLAUDE_CONFIG_DIR`` --
        that is an explicit choice we respect unchanged.
        """
        if self._effective_env.get("CLAUDE_CONFIG_DIR") or os.environ.get(
            "CLAUDE_CONFIG_DIR"
        ):
            return  # caller/env pinned it; respect their choice

        # Persistent per-user SDK cache dir. $XDG_CACHE_HOME wins (per the XDG
        # base-dir spec), else ~/.cache. Stable path => login persists across
        # runs (NOT a throwaway tempdir).
        cache_root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        sdk_config_dir = Path(cache_root) / "claude-agent-sdk" / "config"
        try:
            sdk_config_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Cannot create the isolated dir -- fall back to the prior behavior
            # (child uses the user's home). Better a working turn than a crash.
            logger.debug("Could not create SDK config dir; using default home")
            return

        # Seed auth + settings from the user's real ~/.claude.json the FIRST time
        # only (best-effort). Once seeded, the SDK copy is the source of truth and
        # the user's file is never read/written again, so subsequent runs keep
        # whatever login state the CLI refreshed into the SDK copy.
        sdk_config_file = sdk_config_dir / ".claude.json"
        if not sdk_config_file.exists():
            user_config = Path.home() / ".claude.json"
            if user_config.exists():
                with contextlib.suppress(OSError):
                    shutil.copy2(user_config, sdk_config_file)

        # Route the child + transcript resolution + onboarding at the SDK dir.
        self._effective_env["CLAUDE_CONFIG_DIR"] = str(sdk_config_dir)

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
        # Use the resolved effective config dir (the SDK-owned dir when we
        # defaulted it in _resolve_config_dir; the caller's value otherwise), so
        # the flags land in the SAME file the child reads -- and never in the
        # user's real ~/.claude.json once isolation is active (L4).
        config_dir = self._effective_env.get("CLAUDE_CONFIG_DIR") or os.environ.get(
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
        cmd = _cli_command.build_command(
            self._cli_path, self._options, self._session_id
        )
        # win #3: merge the synthesized hooks block (wiring hook events to the
        # IPC shim) into the ``--settings`` JSON the CLI parses, so the CLI runs
        # our shim for the relevant events. Done here (not in _cli_command) so the
        # transport-owned IPC server's address/matchers stay encapsulated.
        if self._hook_ipc is not None:
            cmd = self._inject_hook_settings(cmd)
        return cmd

    def _inject_hook_settings(self, cmd: list[str]) -> list[str]:
        """Merge the IPC hooks block into the command's ``--settings`` value.

        If a ``--settings`` flag is already present (caller settings and/or the
        sandbox merge from :func:`_cli_command.build_settings_value`), parse its
        JSON object and add/extend the ``hooks`` key; otherwise append a fresh
        ``--settings`` carrying only ``hooks``. A non-object existing settings
        value (e.g. a bare file path the SDK left as-is) is left untouched and the
        hooks are appended as a SECOND ``--settings`` (the CLI merges multiple).
        Best-effort: on any error the original command is returned unchanged.
        """
        assert self._hook_ipc is not None
        try:
            hooks_block = build_hooks_settings(self._hook_ipc)
        except Exception:
            logger.debug("failed to build hook settings; skipping", exc_info=True)
            return cmd
        if not hooks_block:
            return cmd

        new_cmd = list(cmd)
        # Find an existing --settings flag.
        idx = None
        for i, arg in enumerate(new_cmd):
            if arg == "--settings" and i + 1 < len(new_cmd):
                idx = i + 1
                break

        if idx is None:
            new_cmd.extend(["--settings", json.dumps({"hooks": hooks_block})])
            return new_cmd

        existing = new_cmd[idx]
        try:
            obj = json.loads(existing)
        except (json.JSONDecodeError, TypeError):
            obj = None
        if not isinstance(obj, dict):
            # Existing value is a path / non-object: add hooks as a second
            # --settings (the CLI accepts and merges repeated --settings).
            new_cmd.extend(["--settings", json.dumps({"hooks": hooks_block})])
            return new_cmd

        merged_hooks = obj.get("hooks")
        if not isinstance(merged_hooks, dict):
            merged_hooks = {}
        # Our synthesized entries take precedence for the events we wire (the
        # caller's own command hooks for those events still run -- the shim
        # dispatches the user's programmatic hooks too, and any settings-file
        # command hooks the CLI loads separately are unaffected by --settings).
        for event, entries in hooks_block.items():
            merged_hooks.setdefault(event, [])
            merged_hooks[event] = list(merged_hooks[event]) + entries
        obj["hooks"] = merged_hooks
        new_cmd[idx] = json.dumps(obj)
        return new_cmd

    def _compute_transcript_path(self) -> Path:
        project_dir = _get_projects_dir(
            env_override=self._effective_env
        ) / _sanitize_path(_canonicalize_path(self._cwd))
        # The CLI writes/extends the transcript under the id it actually uses.
        # For an explicit/auto session id that is ``self._session_id``; for a
        # --resume (without an explicit session id) the CLI appends to the
        # resumed transcript ``<resume>.jsonl``, so we must tail THAT file, not a
        # fresh id, or resume context is lost (RW3).
        transcript_id = self._session_id
        if self._options.resume and not self._options.session_id:
            transcript_id = self._options.resume
        return project_dir / f"{transcript_id}.jsonl"

    def _resume_transcript_path(self) -> Path | None:
        """Resolve the EXISTING transcript a --resume/--continue targets (RW3).

        Returns the file the CLI will append the resumed turn to, so we can seed
        the tail offset past the pre-existing records. Returns ``None`` if no
        such file exists yet (then the offset stays 0 and the normal resolution
        path applies).

        * ``--resume <id>`` (no explicit session id): ``<id>.jsonl`` -- the id is
          known deterministically.
        * ``--continue``: the CLI continues the most recently modified session in
          the cwd's project dir, so pick the newest pre-existing ``*.jsonl``.
        """
        o = self._options
        if o.session_id:
            # Caller pinned the id; the CLI writes/extends <session_id>.jsonl.
            candidate = self._compute_transcript_path()
            return candidate if candidate.exists() else None
        if o.resume:
            candidate = self._compute_transcript_path()
            return candidate if candidate.exists() else None
        if o.continue_conversation:
            project_dir = (
                self._transcript_path.parent if self._transcript_path else None
            )
            if project_dir is None or not project_dir.is_dir():
                return None
            newest: Path | None = None
            newest_mtime = -1.0
            with contextlib.suppress(OSError):
                for p in project_dir.glob("*.jsonl"):
                    mtime = p.stat().st_mtime
                    if mtime > newest_mtime:
                        newest, newest_mtime = p, mtime
            return newest
        return None

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

        projects = _get_projects_dir(env_override=self._effective_env)
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

    def _close_stderr_pipe(self, write_fd: int | None) -> None:
        """Close the stderr pipe ends after a failed spawn (H3/W3).

        Closes BOTH the half-opened write end and the parent's read end (and
        clears ``_stderr_read_fd``), so a failed ``Popen`` never leaks the read
        fd even when a direct ``PtyCLITransport`` caller skips ``close()`` after
        a failed ``connect()``.
        """
        if write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(write_fd)
        if self._stderr_read_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._stderr_read_fd)
            self._stderr_read_fd = None

    async def _stderr_loop(self) -> None:
        """Read the child's dedicated stderr pipe and call options.stderr per line.

        Mirrors the old stream-json transport's ``_handle_stderr``: split the
        stream into lines, strip the trailing newline, skip blank lines, and
        invoke ``options.stderr(line)`` for each, isolating a raising callback so
        one bad line does not drop the rest of the session. Non-fatal: any read
        error just ends the loop (the turn is unaffected -- stderr is purely
        observability). The pipe fd is closed in ``close()``.
        """
        fd = self._stderr_read_fd
        callback = self._options.stderr
        if fd is None or callback is None:
            return
        buffer = b""
        try:
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
                    break  # EOF: child closed its stderr / exited
                buffer += data
                *lines, buffer = buffer.split(b"\n")
                for raw in lines:
                    self._invoke_stderr(callback, raw)
            # Flush any trailing partial line (no terminating newline).
            if buffer.strip():
                self._invoke_stderr(callback, buffer)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            logger.debug("stderr reader loop failed", exc_info=True)

    @staticmethod
    def _invoke_stderr(callback: Any, raw: bytes) -> None:
        """Decode one stderr line and hand it to the user callback (isolated)."""
        line = raw.decode("utf-8", "replace").rstrip()
        if not line:
            return
        try:
            callback(line)
        except Exception:
            logger.debug("stderr callback raised; continuing", exc_info=True)

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

        Exception (#10): the "Bypass Permissions mode" startup confirmation that
        ``permission_mode="bypassPermissions"`` triggers IS auto-accepted -- it is
        a fixed yes/no acknowledgement the caller already implied by choosing that
        mode, and leaving it unanswered hangs the session forever. (When running
        as root it never appears, suppressed by IS_SANDBOX=1; this covers the
        non-root case.)
        """
        if question.is_bypass:
            return await self._accept_bypass_dialog(question)

        if question.kind not in ("permission", "plan"):
            return False

        want = await self._decide_permission(question)
        option = choose_option(question, want)
        if option is None:
            return False
        await self._send_option_choice(option)
        if want == "deny":
            # Defer building the permission_denials entry: the TUI dialog only
            # exposes a target string, but the baseline shape carries the real
            # ``tool_use_id`` and the FULL original tool input. Both are
            # recoverable from the transcript -- the rejected ``tool_result``
            # record carries ``tool_use_id`` and the preceding ``tool_use`` block
            # carries the real ``input``/``id`` -- so we record the denied tool
            # name here and correlate it to the transcript in _emit_line (RR2).
            if question.tool:
                self._pending_denied_tools.append(question.tool)
            # A denied tool leaves the interactive CLI idle with no
            # ``turn_duration`` record, so the turn would hang forever (RR1).
            # Mark the turn as deny-terminated so _emit_line synthesizes a
            # terminating result once the rejected tool_result lands.
            self._deny_terminated = True
        return True

    async def _accept_bypass_dialog(self, question: DetectedQuestion) -> bool:
        """Press the "Yes, I accept" option on the bypass-permissions dialog (#10).

        The dialog renders two numbered options: "No, exit" (action ``deny``) and
        "Yes, I accept" (action ``allow_once``). We pick the accept option so the
        ``bypassPermissions`` session proceeds. Returns True if an accept option
        was found and sent.
        """
        accept = next(
            (o for o in question.options if o.action == "allow_once"),
            None,
        )
        if accept is None:
            return False
        await self._send_option_choice(accept)
        return True

    async def _decide_permission(
        self, question: DetectedQuestion
    ) -> Literal["allow", "allow_persist", "deny"]:
        """Return the decision for a permission/plan dialog.

        Routes through ``can_use_tool`` when configured (C5); otherwise uses a
        safe default keyed to the permission mode (C6): permissive modes allow so
        turns complete; ``default``/``plan`` allow-once as well (the dialog only
        appears for actions the CLI would otherwise gate, and hanging is worse
        for a drop-in consumer than completing). Callers wanting denial should
        provide ``can_use_tool``.

        RL12: when the callback's ``PermissionResultAllow`` carries a
        session-broad ``updated_permissions`` (see
        :meth:`_should_persist_allow`), return ``"allow_persist"`` so the
        watcher presses the TUI's "allow all edits during this session" option
        and the grant survives for the rest of the session. Narrow rules that
        the coarse TUI option would over-grant fall back to plain ``"allow"``.

        win #3: when ``can_use_tool`` is being routed through the deterministic
        hook channel (``_can_use_tool_via_hook``), DO NOT invoke the callback
        again here -- the PreToolUse hook already decided server-side and the CLI
        short-circuits the dialog accordingly (allow skips the prompt; deny never
        prompts). A permission dialog reaching the watcher in that mode is an
        edge case (e.g. a settings ``ask`` rule that fires despite a hook allow,
        or an unexpected non-PreToolUse permission prompt); we allow-once so the
        turn completes rather than re-running the callback (which would
        double-invoke it and could disagree with the authoritative hook decision).
        """
        if self._can_use_tool_via_hook:
            return "allow"

        callback = self._options.can_use_tool
        if callback is not None and question.tool:
            # RV2: prefer the FULL tool_use.input recovered from the intercepted
            # API response over the TUI's scraped {target}. While the dialog
            # blocks, the transcript has no tool_use record yet -- but the API
            # monitor has already seen the assistant's tool_use in the /v1/
            # messages response, so we correlate by tool name to supply the real,
            # complete input (and tool_use_id) the baseline passes.
            #
            # Deterministic ordering (fixes the RV2 race): the CLI renders this
            # dialog only AFTER receiving the full /v1/messages response the relay
            # also fully received, so the tee callback (_on_api_call, which fills
            # the recovery maps) is guaranteed to run within a small bounded
            # window. The tee runs on the monitor's serve task in this same event
            # loop, so we yield to it via a bounded await rather than racing it.
            full_input, tool_use_id = await self._await_recovered_tool_input(
                question.tool
            )
            context = ToolPermissionContext(
                tool_use_id=tool_use_id,
                title=question.question,
                display_name=question.tool,
            )
            if full_input is not None:
                tool_input: dict[str, Any] = full_input
            else:
                # Fall back to the scraped target when the monitor saw nothing
                # (e.g. it failed to start, or the call hasn't landed yet).
                tool_input = {"target": question.target} if question.target else {}
            try:
                result = await callback(question.tool, tool_input, context)
            except Exception:
                logger.debug("can_use_tool callback raised; denying", exc_info=True)
                return "deny"
            # Defensive: treat anything that is not an explicit Allow as deny
            # (covers a callback that returns a malformed value at runtime).
            if not isinstance(result, PermissionResultAllow):
                return "deny"
            # RL12: a non-empty, session-broad updated_permissions maps onto the
            # TUI's session-allow option; otherwise apply this call only.
            if self._should_persist_allow(question, result.updated_permissions):
                return "allow_persist"
            return "allow"
        # No callback: allow so the turn completes (C6). The prompt only appears
        # in modes that gate; consumers that need gating should pass can_use_tool
        # or use disallowed_tools / a restrictive permission mode.
        return "allow"

    @staticmethod
    def _is_session_broad(upd: PermissionUpdate) -> bool:
        """Is a SINGLE :class:`PermissionUpdate` at-least-as-broad as the TUI's
        "allow all edits this session" grant? (RL15 helper.)

        STRICT POSITIVE ALLOWLIST -- default ``False``. Returns ``True`` ONLY for:
          (a) a ``setMode`` update to a BROADENING mode
              (``acceptEdits`` / ``bypassPermissions``); a narrowing/re-tightening
              ``plan``/``default`` mode, or any other / ``None`` value -> ``False``.
          (b) an ``addRules``/``replaceRules`` update with an explicit
              ``behavior == "allow"``, a ``destination`` in ``{None, "session"}``
              (disk destinations the TUI session press cannot express -> ``False``),
              a NON-EMPTY rule list (a no-op grant of zero rules is not broad), and
              NO narrowing ``rule_content`` on any rule (a path glob is finer than
              "all edits this session").
        Everything else (deny/ask/None behavior, narrowing rule_content, disk
        destination, empty/None rules, addDirectories/removeDirectories/
        removeRules, any unknown/None type) -> ``False``.
        """
        # (a) setMode -- only the broadening modes map onto "allow all edits
        # this session". plan/default/None/other are narrowing or non-broad.
        if upd.type == "setMode":
            return upd.mode in ("acceptEdits", "bypassPermissions")
        # (b) addRules/replaceRules -- session-broad ONLY when ALL hold.
        if upd.type in ("addRules", "replaceRules"):
            # behavior must be an explicit "allow" (deny/ask, and an absent
            # behavior, cannot be expressed by pressing "allow all").
            if upd.behavior != "allow":
                return False
            # destination must be the session (the only scope the TUI's
            # session-allow option honors); disk destinations or any unknown
            # value would over-claim what the keystroke actually does. A None
            # destination defaults to session, so it is accepted.
            if upd.destination not in (None, "session"):
                return False
            # rules must be NON-EMPTY: an update granting zero rules (rules=[]
            # or None) is a no-op, not session-broad.
            rules = upd.rules or []
            if not rules:
                return False
            # rules must be tool-category broad: any narrowing rule_content
            # (e.g. a path glob) is finer than "all edits this session".
            return not any(r.rule_content for r in rules)
        # Everything else (addDirectories/removeDirectories/removeRules and any
        # unknown/None type) does NOT correspond to the "allow all edits this
        # session" press.
        return False

    @staticmethod
    def _should_persist_allow(
        question: DetectedQuestion,
        updated_permissions: list[PermissionUpdate] | None,
    ) -> bool:
        """Decide whether to press the TUI session-allow option (RL12).

        The interactive CLI only exposes a COARSE "allow all edits during this
        session" affordance: it persists by tool-category for the session, not
        an arbitrary :class:`PermissionUpdate` rule. Pressing it to satisfy a
        narrow rule (e.g. "allow Write to /tmp" only) would OVER-GRANT. So we
        require BOTH:

        1. A persist option actually exists in the rendered dialog (otherwise
           there is nothing to press -- e.g. Bash shows "always allow access to
           tmp/" which is a different, path-scoped action; we degrade to
           allow_once via :func:`choose_option`).
        2. The requested update is genuinely SESSION-BROAD, i.e. it would not be
           surprising for the user to see "all edits this session" granted:
             - a ``setMode`` update to a BROADENING mode (``acceptEdits`` /
               ``bypassPermissions``) -- a narrowing ``plan``/``default`` mode
               must NOT trigger the session accept-edits press, OR
             - an ``addRules``/``replaceRules`` *allow* update scoped to the
               ``"session"`` destination whose rules are tool-category-broad
               (no narrowing ``rule_content``).

        A narrow rule (``rule_content`` set), a non-session destination
        (userSettings/projectSettings/localSettings -- the TUI session option
        cannot express persistence to disk), a deny/ask behavior, an EMPTY/None
        rule list (a no-op grant of zero rules cannot justify the broadest
        session press), or any other update shape all fail the test, so we fall
        back to one-shot allow rather than silently granting broader-than-
        requested. The exact persisted rule then simply cannot be expressed via
        the TUI; we apply this call faithfully.

        This is a STRICT POSITIVE ALLOWLIST: the default is to NOT persist.
        ``True`` is returned ONLY for an update shape that is affirmatively
        at-least-as-broad as the TUI's "allow all edits this session" grant.
        Every other shape -- other update types, narrowing/unknown/None modes,
        deny/ask behavior, disk destinations, empty/None rules, narrowing
        ``rule_content`` -- falls through to ``False`` (allow_once). The
        function must NEVER press allow_persist for anything narrower than the
        session grant.
        """
        if not updated_permissions:
            return False
        has_persist_option = any(o.action == "allow_persist" for o in question.options)
        if not has_persist_option:
            return False
        # UNANIMITY (RL15): press allow_persist ONLY when EVERY element of the
        # requested update is itself session-broad. A multi-element list whose
        # COMBINED intent is narrower -- e.g. a broadening setMode paired with a
        # later deny/narrowing rule, or a broad allow followed by a path-scoped
        # rule -- must NOT short-circuit to the broadest "allow all edits this
        # session" press, dropping the narrowing elements. If ANY element is not
        # session-broad we fall back to allow_once (apply this call faithfully;
        # the narrower combined rule simply cannot be expressed via the coarse
        # TUI affordance).
        return all(
            PtyCLITransport._is_session_broad(upd) for upd in updated_permissions
        )

    async def _await_recovered_tool_input(
        self, tool_name: str, timeout_s: float = 2.0, poll_s: float = 0.02
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Bounded-await the recovered full tool input for a blocking dialog (RV2).

        Resolves the RV2 race: the question watcher can detect the permission
        dialog a few ms before the relay's tee callback has parsed and
        correlated the assistant's ``tool_use`` from the (already fully relayed)
        /v1/messages response. Because the CLI only shows the dialog AFTER that
        full response -- which the relay also fully received -- the tee callback
        is guaranteed to land within a small bounded window (observed <200ms), so
        we poll the recovery maps (yielding to the monitor's serve task on the
        same event loop) up to ``timeout_s`` before giving up. Only when the
        monitor never started / saw nothing does this fall through to (None, None)
        and the caller's scraped-``{target}`` fallback.
        """
        # Nothing to await for when the monitor is not running -- the recovery
        # maps will never be populated, so don't burn the timeout.
        if self._api_monitor is None:
            return self._recover_tool_input(tool_name)
        deadline = time.monotonic() + timeout_s
        while True:
            full_input, tool_use_id = self._recover_tool_input(tool_name)
            if full_input is not None:
                return full_input, tool_use_id
            if time.monotonic() >= deadline:
                return full_input, tool_use_id
            await anyio.sleep(poll_s)

    def _recover_tool_input(
        self, tool_name: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Recover the FULL tool input + tool_use_id for a blocking dialog (RV2).

        The intercepted /v1/messages response carried the assistant's ``tool_use``
        block (id, name, full input) before the TUI dialog finished blocking, so
        we look it up by the dialog's tool name. Returns ``(input, tool_use_id)``
        -- either may be ``None`` if nothing matched or the input was not a dict.
        Correlation is by name (the dialog exposes only the name, not the id);
        the latest tool_use seen for that name wins, which is correct because the
        dialog blocks on the most recent tool call.
        """
        # Prefer the by-name index (latest input per tool name).
        entry = self._turn_tool_input_by_name.get(tool_name)
        tool_use_id: str | None = None
        if entry is None:
            # Fall back to scanning the id-keyed index for a matching name.
            for tid, e in self._turn_tool_inputs.items():
                if e.get("name") == tool_name:
                    entry = e
                    tool_use_id = tid
            if entry is None:
                return None, None
        else:
            for tid, e in self._turn_tool_inputs.items():
                if e is entry:
                    tool_use_id = tid
                    break
        inp = entry.get("input")
        return (inp if isinstance(inp, dict) else None), tool_use_id

    async def _send_option_choice(self, option: QuestionOption) -> None:
        """Answer a numbered dialog by typing the option digit then Enter."""
        async with self._write_lock:
            await self._pty_write(str(option.index).encode("ascii"))
            await anyio.sleep(0.1)
            await self._pty_write(_SUBMIT)

    async def _tail_loop(self) -> None:
        """Tail the transcript file and translate new records into messages."""
        path: Path | None = None
        # Skip pre-existing records when resuming an existing transcript (RW3);
        # 0 for a brand-new session.
        offset = self._initial_tail_offset
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
                # finished, so there is nothing more to wait for. While a
                # background task holds the turn, _emit_result defers (leaving
                # _result_emitted False), so this naturally does not fire until
                # the tasks drain and the terminal result is emitted.
                if self._input_ended and self._result_emitted:
                    break

                # Background-task hold timed out: a spawned task never reported
                # terminal within the cap. Stop waiting, finish the turn with
                # whatever accumulated, and terminate -- without this a hung task
                # would hold the turn open forever.
                if (
                    self._input_ended
                    and self._pending_tasks
                    and self._task_wait_deadline is not None
                    and time.monotonic() >= self._task_wait_deadline
                ):
                    logger.warning(
                        "Background task hold: %d task(s) still pending after "
                        "%.0fs; finishing turn.",
                        len(self._pending_tasks),
                        _BACKGROUND_TASK_WAIT_CAP_SECS,
                    )
                    self._pending_tasks.clear()
                    await self._emit_result(
                        {"durationMs": 0, "uuid": str(uuid.uuid4())}
                    )
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

        # Track background tasks (spawn tool_use / terminal task-notification)
        # so the turn can be held open past turn_duration until they finish
        # (the hold), AND maintain the stop_task registry (task_id ->
        # description/status) so stop_task can identify which task to kill.
        # Best-effort: a malformed record must never break tailing.
        self._track_background_tasks(entry)
        with contextlib.suppress(Exception):
            self._track_tasks(entry)

        # turn_duration is the turn-complete signal -> synthesize a result that
        # carries the turn's final text, accumulated usage, and error state.
        if entry_type == "system" and entry.get("subtype") == "turn_duration":
            await self._emit_result(entry)
            return

        message = _translate_transcript_entry(entry, self._session_id)
        if message is None:
            return

        if entry_type == "assistant":
            await self._handle_assistant(message)
            return

        if entry_type == "user":
            user_msg = message.get("message")
            # Correlate any rejected tool_result with a pending deny so the
            # result's permission_denials carries the baseline shape (RR2) and
            # the denied tool_result is excluded from num_turns (RR5).
            self._correlate_denials(user_msg)
            # num_turns counts API round-trips: each tool_result user record is
            # the model being called again with the tool output. Counted here;
            # num_turns = tool_results + 1 (the final answer turn). A rejected
            # tool_result from a permission deny is NOT a real round-trip back to
            # the model, so it is excluded (RR5).
            if self._counts_as_round_trip(user_msg):
                self._turn_tool_results += 1

        await self._send(message)

        # A denied tool leaves the interactive CLI idle (no turn_duration
        # record), so synthesize a terminating result once the rejected
        # tool_result has been processed (RR1). Done after _send so the
        # tool_result message reaches consumers before the result.
        if (
            entry_type == "user"
            and self._deny_terminated
            and not self._result_emitted
            and _has_tool_result(message.get("message"))
        ):
            await self._emit_deny_result(entry)

    # ------------------------------------------------------------------ #
    # Background-task registry (feeds stop_task)
    # ------------------------------------------------------------------ #

    def _track_tasks(self, entry: dict[str, Any]) -> None:
        """Update the live background-task registry from one transcript record.

        Three record shapes drive it (all verified against real transcripts):

        * an assistant ``tool_use`` for a spawning tool -> remember it pending
          (the assigned task-id is not in the input yet);
        * a user ``tool_result`` for that tool_use whose TEXT carries the
          task-id -> promote it to a live task keyed by task-id;
        * a user ``<task-notification>`` -> update status, and retire the task
          on a terminal status (completed/failed/killed).
        """
        etype = entry.get("type")
        message = entry.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")

        if etype == "assistant":
            if not isinstance(content, list):
                return
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name")
                tool_use_id = block.get("id")
                inp = block.get("input")
                inp = inp if isinstance(inp, dict) else {}
                if not isinstance(name, str) or not isinstance(tool_use_id, str):
                    continue
                if not _is_task_spawn(name, inp):
                    continue
                self._pending_task_spawns[tool_use_id] = {
                    "tool": name,
                    "description": _task_description(name, inp),
                }
                _bound_dict(self._pending_task_spawns, _MAX_TRACKED_TASKS)
            return

        if etype != "user":
            return

        # A <task-notification> record (string- or text-block content) reports
        # lifecycle for an existing task.
        text = self._user_record_text(content)
        if "<task-notification>" in text:
            self._apply_task_notifications(text)

        # A spawning tool's result text carries the freshly assigned task-id.
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id")
                if tool_use_id not in self._pending_task_spawns:
                    continue
                m = _TASK_ID_RE.search(self._block_text(block.get("content")))
                if not m:
                    continue
                task_id = m.group(1)
                meta = self._pending_task_spawns.pop(tool_use_id)
                self._tasks[task_id] = {
                    "tool_use_id": tool_use_id,
                    "tool": meta["tool"],
                    "description": meta["description"],
                    "status": "running",
                    "output_file": None,
                }
                _bound_dict(self._tasks, _MAX_TRACKED_TASKS)

    def _apply_task_notifications(self, text: str) -> None:
        """Update/retire tasks from one or more ``<task-notification>`` blocks."""
        for block in _TASK_NOTIFICATION_RE.findall(text):
            task_id = _xml_tag(block, "task-id")
            if not task_id:
                continue
            status = (_xml_tag(block, "status") or "").lower()
            if status in _TERMINAL_TASK_STATUSES:
                self._tasks.pop(task_id, None)
                continue
            task = self._tasks.get(task_id)
            if task is None:
                # A notification for a task whose spawn we missed (e.g. a
                # resumed session): register it so it is still stoppable. The
                # <summary> is a status sentence, not the command the dialog
                # renders, so such a task may not be locatable by description --
                # we still register it so the registry reflects it.
                task = self._tasks.setdefault(
                    task_id,
                    {
                        "tool_use_id": _xml_tag(block, "tool-use-id"),
                        "tool": None,
                        "description": _xml_tag(block, "summary") or "",
                        "status": status or "running",
                        "output_file": None,
                    },
                )
                _bound_dict(self._tasks, _MAX_TRACKED_TASKS)
            if status:
                task["status"] = status
            output_file = _xml_tag(block, "output-file")
            if output_file:
                task["output_file"] = output_file

    @staticmethod
    def _user_record_text(content: Any) -> str:
        """Flatten a user record's content to text (string or text blocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "\n".join(parts)
        return ""

    @staticmethod
    def _block_text(content: Any) -> str:
        """Flatten a tool_result's content (string or list of text blocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return ""

    async def _handle_assistant(self, message: dict[str, Any]) -> None:
        """Emit one assistant block-record as its own AssistantMessage (RV1).

        The interactive transcript writes EACH content block of a single
        assistant message as a SEPARATE record sharing the same ``message.id``
        (with distinct top-level uuids) -- e.g. ``[thinking]``, ``[text]``,
        ``[tool_use]`` -- and same-id ``tool_use`` records are INTERLEAVED with
        the ``user``/``tool_result`` records they trigger. The stream-json
        baseline emits ONE ``AssistantMessage`` per such block-record (live A/B:
        ``A[Thinking] | A[Text] | A[ToolUse] | U | A[Text]`` and, for 3 Write
        calls, ``A[Text] | A[ToolUse] | U | A[ToolUse] | U | A[ToolUse] | U |
        A[Text]``), NOT a merged message. So we emit every distinct block-record
        directly; the per-uuid dedup in ``_emit_line`` prevents a compaction
        re-read from double-emitting. The earlier emit-suppression by
        ``message.id`` (which dropped every block after the first) was the RV1
        bug; merging by ``message.id`` would also drop the interleaved later
        ``tool_use`` blocks. Each record repeats the SAME message-level
        ``usage``, so usage is counted only ONCE per id (C2).
        """
        msg = message["message"]
        msg_id = msg.get("id")

        # Turn bookkeeping (cost/usage/error/tool_use index).
        text = _extract_text(msg)
        if text:
            self._turn_text = text
        model = msg.get("model")
        if isinstance(model, str) and model and model != "unknown":
            self._turn_model = model
        stop_reason = msg.get("stop_reason")
        if isinstance(stop_reason, str):
            self._turn_stop_reason = stop_reason
        # Usage is deduped by message.id (the accumulator keeps the last snapshot
        # per id); the repeated per-block usage is counted once (C2/RR1).
        self._turn_usage.add(msg_id, model, msg.get("usage"))
        err = msg.get("error")
        if err or stop_reason == "refusal":
            self._turn_is_error = True
            self._turn_subtype = self._error_subtype(err, stop_reason)
        # Record this turn's tool_use blocks (id -> name/input) so a later deny
        # can recover the real tool_use_id + full input (RR2).
        self._record_tool_uses(msg)
        if isinstance(msg_id, str) and msg_id:
            self._seen_assistant_ids.add(msg_id)

        # Emit this block-record as its own AssistantMessage (per-block
        # granularity, matching the baseline). uuid dedup (in _emit_line)
        # already guards against re-read duplicates.
        await self._send(message)

    def _record_tool_uses(self, message: dict[str, Any]) -> None:
        """Index this assistant message's tool_use blocks by id (RR2)."""
        content = message.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tid = block.get("id")
            if isinstance(tid, str) and tid:
                self._turn_tool_uses[tid] = {
                    "name": block.get("name"),
                    "input": block.get("input"),
                }

    def _correlate_denials(self, message: Any) -> None:
        """Build baseline-shaped permission_denials from rejected tool_results.

        The TUI deny only exposed a target string; the baseline entry is
        ``{tool_name, tool_use_id, tool_input(full)}`` (RR2). We match a rejected
        ``tool_result`` (``is_error``) back to its ``tool_use`` block (by id) and,
        if its tool name is in the pending-deny queue, emit the full entry and
        record the id so it is excluded from num_turns (RR5).
        """
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            if not block.get("is_error"):
                continue
            tuid = block.get("tool_use_id")
            tool_use = self._turn_tool_uses.get(tuid) if isinstance(tuid, str) else None
            tool_name = tool_use.get("name") if tool_use else None
            # Only treat it as a permission denial if we denied this tool.
            if tool_name in self._pending_denied_tools:
                self._pending_denied_tools.remove(tool_name)
            elif self._pending_denied_tools and tool_name is None:
                # Fall back to FIFO order when the tool name is not recoverable.
                tool_name = self._pending_denied_tools.pop(0)
            else:
                continue
            tool_input = tool_use.get("input") if tool_use else None
            entry: dict[str, Any] = {
                "tool_name": tool_name,
                "tool_use_id": tuid,
                "tool_input": tool_input if isinstance(tool_input, dict) else {},
            }
            self._turn_permission_denials.append(entry)
            if isinstance(tuid, str):
                self._denied_tool_use_ids.add(tuid)

    def _counts_as_round_trip(self, message: Any) -> bool:
        """True if a user record is a real API round-trip for num_turns (RR5).

        A normal tool_result counts (the model is called again with the output).
        A tool_result rejected by a permission deny does NOT -- the model is not
        re-invoked with it -- so it is excluded so num_turns matches the baseline
        for a denied turn.
        """
        if not isinstance(message, dict):
            return False
        content = message.get("content")
        if not isinstance(content, list):
            return False
        has_real = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            if block.get("tool_use_id") in self._denied_tool_use_ids:
                continue
            has_real = True
        return has_real

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

    def _traffic_usage(self) -> TurnUsageAccumulator | None:
        """Build a usage accumulator from the turn's intercepted /v1/messages.

        This is the AUTHORITATIVE usage/cost source (C1/R3): unlike the
        transcript -- which records only the primary assistant messages -- the
        captured traffic includes EVERY API call in the turn, including the
        auxiliary helper-model call (e.g. haiku title generation) the transcript
        never shows. So summing per-call usage here closes the helper-line gap
        that made the PTY total under-report vs the baseline.

        Each intercepted call is one COMPLETE request/response (the proxy reads
        each connection once -- there are no streamed snapshots to dedup the way
        the transcript has), so every call gets a distinct synthetic key and is
        counted exactly once. Returns ``None`` when no call carried usage (the
        caller then falls back to the transcript-based accumulator).
        """
        if not self._turn_api_calls:
            return None
        acc = TurnUsageAccumulator()
        found = False
        for i, call in enumerate(self._turn_api_calls):
            usage = call.get("usage")
            if not isinstance(usage, dict) or not usage:
                continue
            model = call.get("model")
            acc.add(f"_call_{i}", model if isinstance(model, str) else None, usage)
            found = True
        return acc if found else None

    def _duration_api_ms(self) -> int | None:
        """Sum the real per-call request->response durations (R7).

        Returns ``None`` when no traffic was observed (the caller then falls back
        to the wall-clock duration, as before).
        """
        if not self._turn_api_calls:
            return None
        total = 0
        any_timed = False
        for call in self._turn_api_calls:
            d = call.get("duration_ms")
            if isinstance(d, (int, float)):
                total += int(d)
                any_timed = True
        return total if any_timed else None

    def _terminal_api_error_status(self) -> int | None:
        """The HTTP status if the turn's LAST API call was a non-2xx error (C1).

        Mirrors the baseline's ``api_error_status``: the status of the API error
        the turn ENDED on. A transient non-2xx (e.g. a 429 the CLI then retried
        successfully) is deliberately NOT surfaced -- it is followed by a 2xx, so
        the last call is 2xx and the turn did not fail on it. Returns ``None``
        when the final observed call succeeded or no calls were observed.
        """
        if not self._turn_api_calls:
            return None
        last_status = self._turn_api_calls[-1].get("status")
        if (
            isinstance(last_status, int)
            and last_status
            and not (200 <= last_status < 300)
        ):
            return last_status
        return None

    def _apply_usage_fields(self, result: dict[str, Any]) -> None:
        """Populate usage/cost/model_usage + duration_api_ms + api error status.

        Prefers the intercepted-traffic usage (authoritative, includes the
        helper-model call -- C1/R3) and falls back to the transcript-derived
        accumulator when no traffic was observed (e.g. the monitor failed to
        start). Also overrides ``duration_api_ms`` with the real summed per-call
        API time when available (R7) and surfaces any non-2xx statuses (C1).
        """
        usage_acc = self._traffic_usage() or (
            self._turn_usage if self._turn_usage.has_data() else None
        )
        if usage_acc is not None and usage_acc.has_data():
            result["usage"] = usage_acc.aggregate_usage()
            cost = usage_acc.total_cost()
            if cost is not None:
                result["total_cost_usd"] = cost
            model_usage = usage_acc.model_usage()
            if model_usage is not None:
                # camelCase ``modelUsage`` wire key so message_parser picks it up
                # (R1); snake_case made ResultMessage.model_usage always None.
                result["modelUsage"] = model_usage

        api_ms = self._duration_api_ms()
        if api_ms is not None:
            result["duration_api_ms"] = api_ms

        # api_error_status: only when the turn ended on a non-2xx API call (a
        # retried-then-recovered transient is not a turn failure) -- matches the
        # baseline field message_parser reads (C1).
        error_status = self._terminal_api_error_status()
        if error_status is not None:
            result["api_error_status"] = error_status

    def _track_background_tasks(self, entry: dict[str, Any]) -> None:
        """Track background tasks the agent spawns and the records that end them.

        The turn is held open while any spawned task is still running. An
        ``assistant`` ``tool_use`` for a background-spawning tool
        (``Monitor``/``Agent``/``Workflow``, or ``Bash`` with
        ``run_in_background``) adds that task keyed by its tool-use id; a
        ``user`` record carrying a ``<task-notification>`` with a terminal
        ``<status>`` drains it. tool-use id is the uniform correlation key
        between the two. Both add and discard are idempotent, so a re-read of
        the same record (streamed snapshot, compaction) is harmless.
        """
        entry_type = entry.get("type")
        message = entry.get("message")
        if not isinstance(message, dict):
            return
        content = message.get("content")
        if entry_type == "assistant" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name")
                tool_input = block.get("input")
                is_background = name in _BACKGROUND_TASK_TOOLS or (
                    name == "Bash"
                    and isinstance(tool_input, dict)
                    and bool(tool_input.get("run_in_background"))
                )
                tool_use_id = block.get("id")
                if is_background and isinstance(tool_use_id, str):
                    self._pending_tasks.add(tool_use_id)
            return
        if entry_type == "user" and self._pending_tasks:
            text = content if isinstance(content, str) else json.dumps(content)
            if "<task-notification>" not in text:
                return
            for body in _TASK_NOTIFICATION_RE.findall(text):
                status_match = _TASK_STATUS_RE.search(body)
                id_match = _TASK_TOOL_USE_ID_RE.search(body)
                if (
                    status_match
                    and id_match
                    and status_match.group(1) in _TERMINAL_TASK_STATUSES
                ):
                    self._pending_tasks.discard(id_match.group(1))

    async def _emit_result(self, entry: dict[str, Any]) -> None:
        """Synthesize and emit a result from a ``turn_duration`` record."""
        # A deny (RR1) or interrupt (R4) may have already emitted a terminating
        # result for this turn; a stray turn_duration must not double-emit.
        if self._result_emitted:
            return
        # Hold the turn open while the agent has live background tasks: this
        # turn_duration is not terminal -- the agent will continue once a task
        # reports back. Defer the result and reset per-turn state so the
        # continuation turn accumulates cleanly, arming the wait deadline on the
        # first deferral. The terminal result is emitted on the turn_duration
        # that fires once _pending_tasks has drained (see _tail_loop cap).
        if self._pending_tasks:
            if self._task_wait_deadline is None:
                self._task_wait_deadline = (
                    time.monotonic() + _BACKGROUND_TASK_WAIT_CAP_SECS
                )
            self._reset_turn_state()
            return
        # Reached a terminal turn_duration with no tasks pending: disarm the hold.
        self._task_wait_deadline = None
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
            # duration_api_ms defaults to the wall-clock duration; _apply_usage_
            # fields overrides it with the real summed per-call API time when the
            # monitor observed the traffic (R7).
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
        # usage/cost/model_usage (C1/R3), real duration_api_ms (R7), and any
        # non-2xx api status (C1) -- preferring the intercepted traffic.
        self._apply_usage_fields(result)
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
        self._pending_denied_tools = []
        self._turn_tool_uses = {}
        self._denied_tool_use_ids = set()
        self._deny_terminated = False
        self._turn_start_time = None
        # API-monitor per-turn accumulators (traffic-derived enrichment).
        self._turn_api_calls = []
        self._turn_tool_inputs = {}
        self._turn_tool_input_by_name = {}

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
        # An interrupt while no turn is in flight is a no-op (RW2). Without this
        # guard an idle interrupt() synthesizes a spurious error_during_execution
        # result into the buffer, which the NEXT real turn's receive_response()
        # reads first and returns immediately -- corrupting that turn. The
        # baseline treats an idle interrupt as harmless. ``_turn_start_time`` is
        # set in ``_type_prompt`` for the duration of an active turn and cleared
        # by ``_reset_turn_state``, so it is the "no active turn" sentinel.
        if self._turn_start_time is None:
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
        # usage/cost/model_usage (C1/R3) + real duration_api_ms (R7) + api error
        # status (C1) from whatever traffic completed before the abort.
        self._apply_usage_fields(result)
        result["permission_denials"] = list(self._turn_permission_denials)
        self._result_emitted = True
        await self._send(result)
        self._reset_turn_state()

    async def _emit_deny_result(self, entry: dict[str, Any]) -> None:
        """Synthesize a terminating result after a permission deny (RR1).

        When ``can_use_tool`` denies a tool, the interactive CLI lands the deny
        (writing a rejected ``tool_result``) and then goes IDLE awaiting further
        user input -- it never writes a ``turn_duration`` record, so
        ``_emit_result`` never fires and ``receive_response()`` would deadlock.
        The stream-json baseline returned a terminating ``ResultMessage`` for a
        denied turn (live-verified ``subtype=success``, ``is_error=False``,
        carrying ``permission_denials``), so we mirror that here.

        Guarded against double-emit: if a real ``turn_duration`` does arrive
        later (e.g. the CLI resumes and finishes the turn), ``_result_emitted``
        suppresses a second result.
        """
        if self._result_emitted:
            return
        self._turn_count += 1
        duration = 0
        if self._turn_start_time is not None:
            duration = int((time.monotonic() - self._turn_start_time) * 1000)
        # num_turns excludes the rejected (denied) tool_result -- it is not a
        # real round-trip back to the model (RR5).
        num_turns = self._turn_tool_results + 1
        result: dict[str, Any] = {
            "type": "result",
            # Baseline denied turn was subtype=success / is_error=False.
            "subtype": "success",
            "duration_ms": duration,
            "duration_api_ms": duration,
            "is_error": False,
            "num_turns": num_turns,
            "session_id": self._observed_session_id
            or entry.get("sessionId")
            or self._session_id,
            # The interactive CLI produced no final assistant text after the deny
            # (it idles), so result text is whatever accumulated before, if any.
            "result": self._turn_text or None,
            "stop_reason": self._turn_stop_reason,
            "uuid": str(uuid.uuid4()),
        }
        # usage/cost/model_usage (C1/R3) + real duration_api_ms (R7) + api error
        # status (C1), preferring the intercepted traffic.
        self._apply_usage_fields(result)
        result["permission_denials"] = list(self._turn_permission_denials)
        structured = self._extract_structured_output()
        if structured is not None:
            result["structured_output"] = structured
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
            "get_context_usage",
            "stop_task",
        }
    )

    async def _handle_control_request(self, obj: dict[str, Any]) -> None:
        """Map an SDK control request to its interactive-TUI equivalent.

        Supported: ``initialize`` (local ack), ``interrupt`` (ESC),
        ``set_permission_mode`` (shift+tab cycling), ``set_model`` (``/model``),
        ``mcp_status`` / ``get_context_usage`` (from observed state), and
        ``stop_task`` (drives the ``/tasks`` dialog). Everything else
        (mcp_reconnect, mcp_toggle, rewind_files, ...) has no interactive
        channel, so it gets an explicit error control_response.
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
            elif subtype == "get_context_usage":
                payload = self._context_usage_payload()
            elif subtype == "stop_task":
                error = await self._stop_task(request.get("task_id"))
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

    def _context_usage_payload(self) -> dict[str, Any]:
        """Build a get_context_usage payload from observed traffic (RL11).

        The interactive transcript has no ``/context`` data, but the relay tees
        the real per-call ``usage`` of every /v1/messages call. The latest
        successful call's input side (uncached ``input_tokens`` + cache reads +
        cache writes) is the model's current context-window occupancy, so we
        report it as ``totalTokens`` and derive ``percentage`` against the
        model's context window.

        Only observable fields are populated. The category breakdown, memory
        files, MCP tool token costs, etc. are NOT recoverable from the wire, so
        they are returned empty rather than fabricated -- a faithful, drop-in
        shape (every ``ContextUsageResponse`` key present with a real or empty
        value) that does not invent numbers.
        """
        usage = self._latest_context_usage
        model = (
            self._latest_context_model
            or self._observed_request_model
            or (self._turn_model or self._options.model or "")
        )
        total = 0
        if isinstance(usage, dict):
            for key in (
                "input_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            ):
                value = usage.get(key)
                if isinstance(value, int):
                    total += value
        limits = _limits_for(model)
        raw_max = limits["contextWindow"] if limits else 0
        percentage = (total / raw_max * 100.0) if raw_max else 0.0
        return {
            "categories": [],
            "totalTokens": total,
            "maxTokens": raw_max,
            "rawMaxTokens": raw_max,
            "percentage": percentage,
            "model": model,
            "isAutoCompactEnabled": False,
            "memoryFiles": [],
            "mcpTools": [],
            "agents": [],
            "gridRows": [],
        }

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

    # ------------------------------------------------------------------ #
    # stop_task (drive the /tasks BackgroundTasksDialog)
    # ------------------------------------------------------------------ #

    async def _stop_task(self, task_id: Any) -> str | None:
        """Stop the background task ``task_id`` via the ``/tasks`` dialog.

        The interactive transport has no machine channel to stop a task, so it
        opens the TUI's BackgroundTasksDialog, navigates to the row matching the
        task's description, and presses ``x``. Returns ``None`` on success or an
        error string. The kill keystroke is only sent once the parsed
        *selected* row positively matches the target description, so a layout
        drift fails safe ("could not locate ...") rather than killing the wrong
        task.
        """
        if not isinstance(task_id, str) or not task_id:
            return "stop_task requires a string 'task_id'"
        task = self._tasks.get(task_id)
        if task is None:
            known = ", ".join(sorted(self._tasks)) or "none"
            return (
                f"no known running background task with id {task_id!r} "
                f"(known running tasks: {known}). The interactive transport can "
                "only stop tasks observed spawning in the session transcript."
            )
        if self._question_screen is None:
            return (
                "stop_task needs the optional 'pyte' dependency to read the "
                "tasks dialog; install claude-agent-sdk[pty-introspect]."
            )

        description = task.get("description") or ""
        await self._warmup()
        async with self._write_lock:
            # Open the dialog.
            await self._pty_write(b"/tasks")
            await anyio.sleep(0.2)
            await self._pty_write(_SUBMIT)
            rows = await self._await_tasks_dialog()
            if rows is None:
                # The dialog did not render. Do NOT send ESC here: if a turn is
                # in flight, ESC would interrupt it -- and with no confirmed
                # dialog there is nothing to dismiss.
                return f"could not open or parse the /tasks dialog to stop {task_id!r}"
            outcome = await self._navigate_and_kill(description)
            # A dialog was confirmed open, so ESC dismisses it (not a turn).
            await self._pty_write(_INTERRUPT)

        if outcome == "ambiguous":
            return (
                f"multiple background tasks in the /tasks dialog match "
                f"{description!r}; refusing to stop {task_id!r} to avoid killing "
                "the wrong task"
            )
        if outcome != "killed":
            return (
                f"could not locate task {task_id!r} ({description!r}) in the "
                "tasks dialog to stop it"
            )
        # Confirm the kill: the killed <task-notification> retires the task from
        # the registry (processed by the concurrent tail loop). The write lock is
        # released here so tailing is not starved. If we never observe it, report
        # an error rather than a false success.
        if not await self._await_task_retired(task_id):
            return (
                f"sent the stop keystroke for task {task_id!r} but did not "
                "observe it terminate; it may still be running"
            )
        return None

    def _read_screen_lines(self) -> list[str]:
        """Snapshot the emulated TUI screen as rstripped lines (or [] w/o pyte)."""
        screen = self._question_screen
        if screen is None:
            return []
        return [line.rstrip() for line in screen.display]

    async def _await_tasks_dialog(
        self, timeout: float = 2.0
    ) -> list[TaskDialogRow] | None:
        """Poll the screen until the tasks dialog renders. Returns rows or None."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rows = parse_tasks_dialog(self._read_screen_lines())
            if rows is not None:
                return rows
            await anyio.sleep(0.1)
        return parse_tasks_dialog(self._read_screen_lines())

    async def _navigate_and_kill(self, description: str) -> str:
        """Navigate to the uniquely-matching row and press ``x``.

        Returns ``"killed"``, ``"ambiguous"`` (more than one row matches -- never
        kill), or ``"not_found"``. The kill keystroke is sent only when the
        *selected* row matches AND exactly one row in the current frame matches,
        so neither a layout drift nor sibling commands cause the wrong task to be
        killed. The wrap-around guard hashes a status/age-stripped row key
        (``task_label_key``) so a live-updating age column cannot defeat it.
        """
        seen: set[tuple[tuple[bool, str], ...]] = set()
        for _ in range(_MAX_TASK_NAV_STEPS):
            rows = parse_tasks_dialog(self._read_screen_lines())
            if not rows:
                return "not_found"
            match_count = sum(task_row_matches(r.label, description) for r in rows)
            if match_count == 0:
                return "not_found"
            if match_count > 1:
                return "ambiguous"
            selected = next((r for r in rows if r.selected), None)
            if selected is None:
                return "not_found"
            if task_row_matches(selected.label, description):
                await self._pty_write(_KILL_TASK)
                await anyio.sleep(0.1)
                return "killed"
            signature = tuple((r.selected, task_label_key(r.label)) for r in rows)
            if signature in seen:
                return "not_found"  # selection wrapped without reaching the row
            seen.add(signature)
            await self._pty_write(_ARROW_DOWN)
            await anyio.sleep(0.1)
        return "not_found"

    async def _await_task_retired(self, task_id: str, timeout: float = 3.0) -> bool:
        """Return True once the killed task leaves the live registry, else False."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if task_id not in self._tasks:
                return True
            await anyio.sleep(0.1)
        return task_id not in self._tasks

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
        else:
            # Empty / whitespace-only prompt (RW1). A truly empty turn cannot be
            # submitted over the TUI (bracketed-paste of "" + Enter is a no-op),
            # so no transcript record or turn_duration is ever written and
            # receive_response() would hang forever. The baseline returns a
            # terminating ResultMessage for an empty prompt, so we synthesize one
            # (mirroring the deny/interrupt synthesis) rather than dropping the
            # turn -- the documented contract is that receive_response() always
            # terminates.
            await self._emit_empty_prompt_result()

    async def _emit_empty_prompt_result(self) -> None:
        """Synthesize a terminating result for an empty/whitespace prompt (RW1).

        The TUI cannot submit an empty turn, so the CLI never runs one and never
        writes a result. The stream-json baseline returned ``subtype=success``
        for an empty prompt, so we mirror that subtype/fields here so
        ``receive_response()`` terminates instead of deadlocking.
        """
        # Per-turn init still leads every turn after the first, matching the
        # baseline ordering (RR4).
        if self._submitted_turns > 0:
            await self._emit_init_message()
        self._submitted_turns += 1
        self._turn_count += 1
        result: dict[str, Any] = {
            "type": "result",
            "subtype": "success",
            "duration_ms": 0,
            "duration_api_ms": 0,
            "is_error": False,
            "num_turns": 1,
            "session_id": self._observed_session_id or self._session_id,
            "result": "",
            "stop_reason": None,
            "uuid": str(uuid.uuid4()),
        }
        result["permission_denials"] = list(self._turn_permission_denials)
        self._result_emitted = True
        await self._send(result)
        self._reset_turn_state()

    async def _emit_init_message(self) -> None:
        """Emit a fresh ``system/init`` message (RR4).

        The baseline emits a ``system/init`` at the START OF EVERY TURN; the PTY
        previously emitted it once at connect. Re-emitting it at each turn start
        matches the baseline's per-turn message ordering so a consumer that reads
        session id / capabilities at each turn boundary sees it every turn. By
        now ``_build_init_data`` can carry the observed model / real session id.
        """
        init = self._build_init_data()
        init.update(
            {
                "type": "system",
                "subtype": "init",
                "uuid": str(uuid.uuid4()),
            }
        )
        await self._send(init)

    async def _type_prompt(self, text: str) -> None:
        await self._warmup()
        # The init message for the first turn was emitted at connect(); emit a
        # fresh one for every subsequent turn to match the baseline ordering
        # (system/init leads every turn) (RR4).
        if self._submitted_turns > 0:
            await self._emit_init_message()
        self._submitted_turns += 1
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

        for task in (
            self._tail_task,
            self._drain_task,
            self._question_task,
            self._stderr_task,
        ):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(Exception):
                    await task.wait()
        self._tail_task = None
        self._drain_task = None
        self._question_task = None
        self._stderr_task = None

        # H3: close the read end of the child's stderr pipe (no fd leak).
        if self._stderr_read_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._stderr_read_fd)
            self._stderr_read_fd = None

        # Stop the always-on API monitor (safe from any task -- it uses a
        # detached serve handle, not a task-affine cancel scope).
        if self._api_monitor is not None:
            with contextlib.suppress(Exception):
                await self._api_monitor.stop()
            self._api_monitor = None

        # Stop the hook IPC bridge (win #3): cancel its serve task and unlink the
        # socket file (no leak). Same detached-handle safety as the API monitor.
        if self._hook_ipc is not None:
            with contextlib.suppress(Exception):
                await self._hook_ipc.stop()
            self._hook_ipc = None

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
