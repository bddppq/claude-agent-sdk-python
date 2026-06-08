"""Tests for the interactive-TUI question parser.

These exercise ``parse_question`` against golden screen frames captured from the
real CLI (reconstructed with pyte at 40x120). They are dependency-free: no live
model and no terminal -- the fixtures are the already-reconstructed lines, so the
parser is validated in isolation.

The fixture content mirrors what stream-json emits as the ``can_use_tool``
control_request for the same actions (tool name, target, options), which is the
property the live comparison harness asserts end-to-end.
"""

from claude_agent_sdk._internal.transport.pty_question import (
    DetectedQuestion,
    QuestionOption,
    choose_option,
    parse_question,
)

WRITE_FRAME = [
    "● Write(note.txt)",
    "─" * 60,
    "Create file",
    "note.txt",
    "╌" * 60,
    " 1 hello world",
    "╌" * 60,
    "Do you want to create note.txt?",
    "> 1. Yes",
    "  2. Yes, allow all edits during this session (shift+tab)",
    "  3. No",
    "Esc to cancel · Tab to amend",
]

BASH_FRAME = [
    "⎿  $ touch /tmp/probe.txt",
    "─" * 60,
    "Bash command",
    "  touch /tmp/probe.txt",
    "  Create empty probe file",
    "Do you want to proceed?",
    "> 1. Yes",
    "  2. Yes, and always allow access to tmp/ from this project",
    "  3. No",
    "Esc to cancel · Tab to amend · ctrl+e to explain",
]

EDIT_FRAME = [
    "● Update(animals.txt)",
    "─" * 60,
    "Edit file",
    "animals.txt",
    "╌" * 60,
    " 1 -the cat sat",
    " 1 +the dog sat",
    "╌" * 60,
    "Do you want to make this edit to animals.txt?",
    "> 1. Yes",
    "  2. Yes, allow all edits during this session (shift+tab)",
    "  3. No",
    "Esc to cancel · Tab to amend",
]

PLAN_FRAME = [
    "Plan: Create greet.py",
    "Create greet.py with a hello() function.",
    "─" * 60,
    "Claude has written up a plan and is ready to execute. Would you like to proceed?",
    "> 1. Yes, and use auto mode",
    "  2. Yes, manually approve edits",
    "  3. No, refine with Ultraplan on Claude Code on the web",
    "  4. Tell Claude what to change",
    "   shift+tab to approve with this feedback",
    "ctrl-g to edit in  Vim  · ~/.claude/plans/x.md",
]

ASK_FRAME = [
    "←  [ ] Language  [ ] License  √ Submit  →",
    "Which programming language should I use?",
    "> 1. Python",
    "     Use Python for the implementation.",
    "  2. Rust",
    "     Use Rust for the implementation.",
    "  3. Go",
    "     Use Go for the implementation.",
    "  4. Type something.",
    "  5. Chat about this",
    "Enter to select · Tab/Arrow keys to navigate · Esc to cancel",
]

APP_DIALOG_FRAME = [
    "WARNING: Claude Code running in Bypass Permissions mode",
    "In Bypass Permissions mode, Claude Code will not ask for your approval",
    "before running potentially dangerous commands.",
    "> 1. No, exit",
    "  2. Yes, I accept",
    "Enter to confirm · Esc to cancel",
]

# Negative: ordinary generation output (no dialog chrome) must not be detected,
# even when it contains a numbered list.
GENERATION_FRAME = [
    "● I'll outline the steps for you.",
    "Here is the plan:",
    "1. First, read the file",
    "2. Then, edit it",
    "3. Finally, run the tests",
    "esc to interrupt",
]


def test_write_permission() -> None:
    q = parse_question(WRITE_FRAME)
    assert q is not None
    assert q.kind == "permission"
    assert q.tool == "Write"
    assert q.target == "note.txt"
    assert q.question == "Do you want to create note.txt?"
    assert [o.action for o in q.options] == ["allow_once", "allow_persist", "deny"]
    assert "hello world" in " ".join(q.preview)


def test_bash_permission() -> None:
    q = parse_question(BASH_FRAME)
    assert q is not None
    assert q.kind == "permission"
    assert q.tool == "Bash"
    assert q.target == "touch /tmp/probe.txt"
    assert [o.action for o in q.options] == ["allow_once", "allow_persist", "deny"]


def test_edit_permission() -> None:
    q = parse_question(EDIT_FRAME)
    assert q is not None
    assert q.kind == "permission"
    assert q.tool == "Edit"  # the TUI label "Update" is mapped back to Edit
    assert q.target == "animals.txt"
    preview = " ".join(q.preview)
    assert "the cat sat" in preview and "the dog sat" in preview


def test_plan_approval() -> None:
    q = parse_question(PLAN_FRAME)
    assert q is not None
    assert q.kind == "plan"
    assert len(q.options) == 4
    assert [o.action for o in q.options] == [
        "allow_persist",
        "allow_once",
        "deny",
        "amend",
    ]
    assert "greet.py" in " ".join(q.preview)


def test_ask_user_question() -> None:
    q = parse_question(ASK_FRAME)
    assert q is not None
    assert q.kind == "ask"
    assert q.headers == ["Language", "License"]
    selectable = [o for o in q.options if o.action == "select"]
    assert [o.label for o in selectable] == ["Python", "Rust", "Go"]
    assert selectable[0].description == "Use Python for the implementation."


def test_app_dialog() -> None:
    q = parse_question(APP_DIALOG_FRAME)
    assert q is not None
    assert q.kind == "app_dialog"
    assert q.options[0].action == "deny"  # "No, exit"
    assert q.options[1].action == "allow_once"  # "Yes, I accept"


def test_generation_is_not_a_question() -> None:
    # No dialog chrome -> not detected, despite the numbered list.
    assert parse_question(GENERATION_FRAME) is None


def test_empty_screen() -> None:
    assert parse_question([]) is None
    assert parse_question(["", "  ", ""]) is None


def _q(*options: QuestionOption) -> DetectedQuestion:
    return DetectedQuestion(kind="permission", question="?", options=list(options))


def test_choose_option_allow_prefers_allow_once() -> None:
    q = _q(
        QuestionOption(index=1, label="Yes", action="allow_once"),
        QuestionOption(index=2, label="Yes, always", action="allow_persist"),
        QuestionOption(index=3, label="No", action="deny"),
    )
    chosen = choose_option(q, "allow")
    assert chosen is not None and chosen.index == 1


def test_choose_option_allow_falls_back_to_persist() -> None:
    q = _q(
        QuestionOption(index=1, label="Always allow", action="allow_persist"),
        QuestionOption(index=2, label="No", action="deny"),
    )
    chosen = choose_option(q, "allow")
    assert chosen is not None and chosen.index == 1


def test_choose_option_deny_prefers_deny() -> None:
    q = _q(
        QuestionOption(index=1, label="Yes", action="allow_once"),
        QuestionOption(index=2, label="No", action="deny"),
    )
    chosen = choose_option(q, "deny")
    assert chosen is not None and chosen.index == 2


def test_choose_option_deny_falls_back_to_last() -> None:
    # No explicit deny action (e.g. a plan dialog "keep planning" last option).
    q = _q(
        QuestionOption(index=1, label="Proceed", action="allow_once"),
        QuestionOption(index=2, label="Keep planning", action="select"),
    )
    chosen = choose_option(q, "deny")
    assert chosen is not None and chosen.index == 2


def test_choose_option_no_options() -> None:
    assert choose_option(_q(), "allow") is None


def test_choose_option_allow_persist_prefers_persist() -> None:
    # RL12: a session-broad updated_permissions maps onto the persist option.
    q = _q(
        QuestionOption(index=1, label="Yes", action="allow_once"),
        QuestionOption(
            index=2,
            label="Yes, allow all edits during this session",
            action="allow_persist",
        ),
        QuestionOption(index=3, label="No", action="deny"),
    )
    chosen = choose_option(q, "allow_persist")
    assert chosen is not None and chosen.index == 2


def test_choose_option_allow_persist_falls_back_to_once() -> None:
    # No persist option in the dialog (e.g. Bash "always allow access to tmp/")
    # -> degrade to one-shot allow rather than over-granting.
    q = _q(
        QuestionOption(index=1, label="Yes", action="allow_once"),
        QuestionOption(index=2, label="No", action="deny"),
    )
    chosen = choose_option(q, "allow_persist")
    assert chosen is not None and chosen.index == 1
