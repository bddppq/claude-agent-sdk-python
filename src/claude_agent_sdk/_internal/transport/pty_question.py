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
    )
