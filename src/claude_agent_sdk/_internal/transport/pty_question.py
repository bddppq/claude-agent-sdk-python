"""Detect and extract blocking questions from the interactive TUI.

The interactive CLI renders questions (tool-permission prompts, plan approval,
``AskUserQuestion`` forms, and app-level confirmations) as on-screen dialogs and
then blocks on a keystroke. Unlike the stream-json transport there is no
machine-readable control message for these -- the only signal is what is drawn
to the terminal. This module reconstructs the rendered screen and parses the
dialog into a structured :class:`DetectedQuestion`.

Reconstruction needs a terminal emulator because the TUI positions text with
cursor moves (naive ANSI stripping runs words together). We use ``pyte`` for
this; it is an optional dependency (the ``pty-introspect`` extra). When it is
unavailable, :func:`reconstruct` raises and the transport degrades to no
introspection.

The parser (:func:`parse_question`) operates on already-reconstructed lines, so
it is dependency-free and unit-testable without a live terminal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# Terminal geometry must match the PTY winsize the transport sets
# (see PtyCLITransport.connect: TIOCSWINSZ rows=40, cols=120).
SCREEN_ROWS = 40
SCREEN_COLS = 120

QuestionKind = Literal["permission", "ask", "plan", "app_dialog"]
OptionAction = Literal[
    "allow_once", "allow_persist", "deny", "amend", "select", "other"
]

# A numbered option line, optionally preceded by a selection caret.
_OPTION_RE = re.compile(r"^([❯>])?\s*(\d+)\.\s+(.*)$")
# A rendered tool-use header, e.g. "● Write(note.txt)" / "● Update(foo.py)". The
# leading bullet and the parenthesized target are required so prose / box labels
# like "Edit file" are not mistaken for the header.
_TOOL_RE = re.compile(
    r"●\s*(Write|Edit|Update|MultiEdit|Bash|Read|Glob|Grep|WebFetch|NotebookEdit)"
    r"\(([^)]+)\)"
)
# A shell command line as rendered in a Bash permission preview ("$ <cmd>").
_CMD_RE = re.compile(r"^\s*(?:⎿\s*)?\$\s+(.*)$")
# Box-header labels that introduce the affected target on the next line.
_BOX_HEADERS = {"Create file": "Write", "Edit file": "Edit", "Bash command": "Bash"}

# Footer / navigation chrome the assistant's own text never emits. The presence
# of one of these is what distinguishes a *blocking dialog* from ordinary
# generation output (which may itself contain numbered lists).
_CHROME_ANCHORS = (
    "esctocancel",
    "esctoconfirm",
    "entertoselect",
    "entertoconfirm",
    "shift+tabtoapprove",
    "tabtoamend",
    "tab/arrowkeystonavigate",
)


@dataclass
class QuestionOption:
    """A single selectable answer in a dialog."""

    index: int
    label: str
    description: str | None = None
    action: OptionAction = "select"
    selected: bool = False


@dataclass
class DetectedQuestion:
    """A structured view of the question currently blocking the TUI."""

    kind: QuestionKind
    question: str | None
    options: list[QuestionOption]
    tool: str | None = None
    target: str | None = None
    preview: list[str] = field(default_factory=list)
    # ``AskUserQuestion`` renders a tabbed multi-question form; ``headers`` holds
    # the per-question tab labels (only the active tab's options are on screen).
    headers: list[str] = field(default_factory=list)
    # True only for the "Bypass Permissions mode" startup confirmation
    # (``kind == "app_dialog"``). The transport auto-accepts this one so a
    # ``permission_mode="bypassPermissions"`` session does not hang on it (#10);
    # other app_dialogs (e.g. folder-trust) are left alone -- trust is handled by
    # pre-seeding the config, and unknown app dialogs need real user input.
    is_bypass: bool = False


def reconstruct(
    raw: bytes, cols: int = SCREEN_COLS, rows: int = SCREEN_ROWS
) -> list[str]:
    """Reconstruct the visible screen from raw PTY bytes as a list of lines.

    Requires the optional ``pyte`` dependency. Trailing whitespace is stripped
    from each line; blank lines are preserved so vertical structure is intact.
    """
    try:
        import pyte
    except ImportError as exc:  # pragma: no cover - exercised via the extra
        raise RuntimeError(
            "PTY question extraction requires the optional 'pyte' dependency. "
            "Install it with: pip install 'claude-agent-sdk[pty-introspect]'"
        ) from exc

    screen = pyte.Screen(cols, rows)
    stream = pyte.ByteStream(screen)
    stream.feed(raw)
    return [line.rstrip() for line in screen.display]


def _normalize(lines: list[str]) -> str:
    """Whitespace-stripped, lowercased join used for chrome/keyword matching.

    The TUI positions characters individually, so inter-word spacing is
    unreliable; removing all whitespace makes anchor matching robust.
    """
    return re.sub(r"\s+", "", "\n".join(lines)).lower()


def _classify(normalized: str) -> QuestionKind:
    if "wouldyouliketoproceed" in normalized or "readytoexecute" in normalized:
        return "plan"
    if "tab/arrowkeystonavigate" in normalized or (
        "submit" in normalized and "entertoselect" in normalized
    ):
        return "ask"
    if (
        "bypasspermissionsmode" in normalized
        or "doyoutrust" in normalized
        or "trustthefiles" in normalized
    ):
        return "app_dialog"
    return "permission"


def _option_action(label: str, kind: QuestionKind) -> OptionAction:
    low = label.lower()
    if kind == "ask":
        if "type something" in low:
            return "amend"
        if "chat about this" in low:
            return "other"
        return "select"
    if re.search(r"\bno\b|exit|keep planning|don't|reject|cancel", low):
        return "deny"
    if "tell claude" in low or "refine" in low or "amend" in low:
        return "amend"
    if re.search(
        r"always|all edits|this session|auto mode|always allow|accept all", low
    ):
        return "allow_persist"
    if re.search(r"\byes\b|proceed|approve|accept|allow", low):
        return "allow_once"
    return "select"


def _nonempty_indexed(lines: list[str]) -> list[tuple[int, str]]:
    return [(i, line) for i, line in enumerate(lines) if line.strip()]


def _parse_options(
    indexed: list[tuple[int, str]], kind: QuestionKind
) -> list[QuestionOption]:
    options: list[QuestionOption] = []
    for pos, (_, line) in enumerate(indexed):
        m = _OPTION_RE.match(line.strip())
        if not m:
            continue
        label = m.group(3).strip()
        description = None
        if kind == "ask":
            # The description is the indented, non-numbered line(s) that follow.
            for _, follow in indexed[pos + 1 :]:
                stripped = follow.strip()
                if not stripped or _OPTION_RE.match(stripped):
                    break
                if set(stripped) <= set("─╌│ "):  # separators
                    break
                description = stripped
                break
        options.append(
            QuestionOption(
                index=int(m.group(2)),
                label=label,
                description=description,
                action=_option_action(label, kind),
                selected=m.group(1) in ("❯", ">"),
            )
        )
    return options


def _find_question(indexed: list[tuple[int, str]], first_option_pos: int) -> str | None:
    for pos in range(first_option_pos - 1, -1, -1):
        text = indexed[pos][1].strip()
        if text.endswith("?"):
            return text
    return None


def _parse_tool_target(indexed: list[tuple[int, str]]) -> tuple[str | None, str | None]:
    tool: str | None = None
    target: str | None = None
    for _, line in indexed:
        m = _TOOL_RE.search(line)
        if m:
            tool = m.group(1)
            target = m.group(2).strip() or None
    if tool == "Update":  # the TUI relabels Edit as "Update"
        tool = "Edit"
    command: str | None = None
    for _, line in indexed:
        m = _CMD_RE.match(line)
        if m:
            command = m.group(1).strip()
    if tool is None:
        for pos, (_, line) in enumerate(indexed):
            header = line.strip()
            if header in _BOX_HEADERS and pos + 1 < len(indexed):
                tool = _BOX_HEADERS[header]
                target = indexed[pos + 1][1].strip()
                break
    if tool == "Bash" and command:
        target = command
    elif command and tool is None:
        tool, target = "Bash", command
    return tool, target


def _parse_headers(indexed: list[tuple[int, str]]) -> list[str]:
    """Parse the AskUserQuestion tab bar (e.g. "← [ ] A [ ] B √ Submit →")."""
    for _, line in indexed:
        if "Submit" in line and ("←" in line or "→" in line or "[" in line):
            tokens = re.sub(r"[←→\[\]√]", " ", line).split()
            return [t for t in tokens if t != "Submit"]
    return []


def parse_question(lines: list[str]) -> DetectedQuestion | None:
    """Parse a reconstructed screen into a :class:`DetectedQuestion`.

    Returns ``None`` when the screen is not showing a blocking dialog. Detection
    is gated on dialog chrome so that numbered lists in ordinary model output do
    not produce false positives.
    """
    normalized = _normalize(lines)
    if not any(anchor in normalized for anchor in _CHROME_ANCHORS):
        return None

    indexed = _nonempty_indexed(lines)
    option_positions = [
        pos for pos, (_, line) in enumerate(indexed) if _OPTION_RE.match(line.strip())
    ]
    if not option_positions:
        return None

    kind = _classify(normalized)
    options = _parse_options(indexed, kind)
    question = _find_question(indexed, option_positions[0])

    tool = target = None
    preview: list[str] = []
    headers: list[str] = []

    if kind in ("permission",):
        tool, target = _parse_tool_target(indexed)
        # Preview = content between the box top and the question line.
        q_pos = option_positions[0]
        if question is not None:
            for pos, (_, line) in enumerate(indexed):
                if line.strip() == question:
                    q_pos = pos
                    break
        for _, line in indexed[:q_pos]:
            stripped = line.strip()
            if stripped and not set(stripped) <= set("─╌│ "):
                preview.append(stripped)
    elif kind == "plan":
        q_pos = option_positions[0]
        if question is not None:
            for pos, (_, line) in enumerate(indexed):
                if line.strip() == question:
                    q_pos = pos
                    break
        for _, line in indexed[:q_pos]:
            stripped = line.strip()
            if stripped and not set(stripped) <= set("─╌│ "):
                preview.append(stripped)
    elif kind == "ask":
        headers = _parse_headers(indexed)

    return DetectedQuestion(
        kind=kind,
        question=question,
        options=options,
        tool=tool,
        target=target,
        preview=preview,
        headers=headers,
        is_bypass=(kind == "app_dialog" and "bypasspermissionsmode" in normalized),
    )


@dataclass
class TaskDialogRow:
    """One task row rendered in the ``/tasks`` BackgroundTasksDialog.

    ``label`` is the visible description/command text (tool prefix and trailing
    status/age columns included, normalized only by whitespace collapse).
    ``selected`` is True for the row currently bearing the navigation caret --
    the row a subsequent ``x`` keypress would kill.
    """

    label: str
    selected: bool


# Footer/navigation chrome lines that are NOT task rows. Matched as substrings
# of a whitespace-stripped, lowercased line. These are dialog-specific phrases
# (not bare words like "esc"), so a task row whose command merely CONTAINS
# "esc"/"kill" (e.g. ``./escape.sh``) is not wrongly dropped. NOTE: the dialog
# layout is an undocumented TUI contract that can shift between CLI versions --
# re-verify against the bundled CLI and update the fixtures in
# test_pty_question.py if row identification regresses.
_TASKS_ROW_CHROME = (
    "escto",  # "esc to close" / "esc to cancel" / "esc to exit"
    "xtokill",
    "xtostop",
    "tab/arrow",
    "arrowkeys",
    "enterfordetails",
    "entertoselect",
    "entertoview",
)

# The kill/stop affordance is the dialog-specific footer that gates detection.
# Ordinary model output does not render "x to kill" / "x to stop" as chrome, so
# gating on it (rather than loose words like "kill"+"esc") prevents misdetecting
# generation output -- including markdown blockquotes -- as the tasks dialog.
_TASKS_DIALOG_ANCHORS = ("xtokill", "xtostop")

# Trailing status/age columns rendered after a task's description. They update
# live (e.g. the age ticks "30s" -> "31s"), so they are stripped before matching
# and before computing a navigation signature -- otherwise a changing age would
# make every screen read look "new" and defeat the wrap-around loop guard.
_TASK_STATUS_TOKENS = frozenset(
    {
        "running",
        "completed",
        "complete",
        "failed",
        "failure",
        "killed",
        "pending",
        "queued",
        "done",
        "stopped",
        "success",
        "succeeded",
        "error",
    }
)
_AGE_RE = re.compile(r"^\d+(?:\.\d+)?[smhdw]$")
_TASK_TOOL_PREFIXES = ("bash ", "agent ", "task ", "workflow ", "monitor ")


def parse_tasks_dialog(lines: list[str]) -> list[TaskDialogRow] | None:
    """Parse the ``/tasks`` BackgroundTasksDialog into its task rows, or None.

    Returns ``None`` when the screen is not showing the tasks dialog. Detection
    is gated on the dialog's kill/stop FOOTER affordance (``x to kill`` /
    ``x to stop``), which ordinary model output never renders, so generation
    output -- even text mentioning "kill" with an "esc to interrupt" footer, or
    a markdown ``>`` blockquote -- is not misdetected as the dialog (and its
    ``>`` lines are not mistaken for a selection caret).

    Best-effort by design: callers must NOT act (press ``x``) unless a single
    clearly-selected row positively matches the intended task, so a layout drift
    degrades to "could not locate the task" rather than killing the wrong one.
    """
    normalized = _normalize(lines)
    if not any(anchor in normalized for anchor in _TASKS_DIALOG_ANCHORS):
        return None

    rows: list[TaskDialogRow] = []
    title_skipped = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        low = re.sub(r"\s+", "", stripped).lower()
        # The dialog title ("Background tasks") leads the list; skip it once.
        if not title_skipped and "background" in low and "task" in low:
            title_skipped = True
            continue
        # Skip box-drawing separators and footer/navigation chrome.
        if set(stripped) <= set("─╌│ ·—-"):
            continue
        if any(anchor in low for anchor in _TASKS_ROW_CHROME):
            continue
        selected = stripped[0] in ("❯", ">")
        label = stripped[1:].strip() if selected else stripped
        # A leading "1." / "1)" index, when present, is navigation chrome.
        label = re.sub(r"^\d+[.)]\s*", "", label).strip()
        if label:
            rows.append(TaskDialogRow(label=label, selected=selected))
    return rows or None


def _normalize_task_text(text: str | None) -> str:
    """Whitespace-collapsed, lowercased form for task description matching."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _strip_status_columns(label: str) -> str:
    """Drop trailing status/age tokens (e.g. ``running 2m``) from a row label."""
    tokens = label.split()
    while tokens:
        last = tokens[-1].lower().strip("·.")
        if last in _TASK_STATUS_TOKENS or _AGE_RE.match(last):
            tokens.pop()
        else:
            break
    return " ".join(tokens)


def task_label_key(label: str) -> str:
    """Stable identity for a dialog row: normalized, status/age columns removed.

    Used both for matching and for the navigation-loop wrap-around signature, so
    a live-updating age column cannot make a static row look like it changed.
    """
    return _strip_status_columns(_normalize_task_text(label))


def _strip_leading_tool(text: str) -> str:
    """Strip a single leading tool label (``bash ``/``agent ``/...) if present."""
    for prefix in _TASK_TOOL_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def task_row_matches(row_label: str, target_description: str) -> bool:
    """True if a dialog row refers to the task with ``target_description``.

    Matching is deliberately strict so it discriminates *sibling* commands
    (e.g. ``npm run build:prod`` vs ``npm run build:staging``) and never kills
    the wrong task:

    * the FULL target must appear verbatim in the row's description (after
      stripping the live status/age columns) -- so a longer sibling that merely
      shares a prefix does not match; OR
    * for a row truncated with an ellipsis, the visible stem (minus an optional
      leading tool label) must be a long-enough PREFIX of the target.

    Ambiguity (two rows matching) is rejected by the caller, not here.
    """
    target = _normalize_task_text(target_description)
    if not target:
        return False
    stripped = task_label_key(row_label)
    if not stripped:
        return False
    # Verbatim containment of the full target -> unambiguous match.
    if target in stripped:
        return True
    # Truncated row: require a long shared prefix (after the tool label) so a
    # short shared prefix between siblings cannot match.
    truncated = stripped.endswith("…") or stripped.endswith("...")
    stem = _strip_leading_tool(stripped.rstrip("… ."))
    return truncated and len(stem) >= 16 and target.startswith(stem)


def choose_option(
    question: DetectedQuestion, want: Literal["allow", "allow_persist", "deny"]
) -> QuestionOption | None:
    """Pick the option that best expresses an allow/deny decision.

    Used to answer a blocking permission/plan dialog by keystroke. ``want`` is
    the decision derived from ``can_use_tool`` (or the safe default). Returns the
    chosen :class:`QuestionOption` (whose ``index`` is the digit to type), or
    ``None`` if no suitable option exists.

    ``"allow"`` prefers a one-shot allow over a persist-all option (least
    surprising: we do not silently widen permissions for the whole session).
    ``"allow_persist"`` (RL12) maps a session-broad ``updated_permissions``
    request onto the TUI's "allow all edits during this session" option; when no
    such option exists it degrades to one-shot allow (the persisted rule cannot
    be expressed by keystroke, so we apply this call only and never over-grant).
    ``"deny"`` prefers an explicit deny/no option.
    """
    if not question.options:
        return None
    if want == "deny":
        deny = [o for o in question.options if o.action == "deny"]
        if deny:
            return deny[0]
        # Plan dialogs phrase rejection as "keep planning"/"no"; fall back to the
        # last option, which is conventionally the negative choice.
        return question.options[-1]
    once = [o for o in question.options if o.action == "allow_once"]
    persist = [o for o in question.options if o.action == "allow_persist"]
    if want == "allow_persist":
        # RL12: honor the persist intent via the session-allow option when it
        # exists; otherwise fall back to one-shot allow (apply this call). The
        # TUI has no narrower persist affordance, so this never over-grants
        # beyond what the dialog itself offers.
        if persist:
            return persist[0]
        if once:
            return once[0]
        return question.options[0]
    # want == "allow"
    if once:
        return once[0]
    if persist:
        return persist[0]
    # No clearly-allow option (e.g. an ask form); fall back to the first option.
    return question.options[0]
