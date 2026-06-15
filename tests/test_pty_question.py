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
    parse_tasks_dialog,
    task_label_key,
    task_row_matches,
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


TRUST_DIALOG_FRAME = [
    "Do you trust the files in this folder?",
    "/home/user/project",
    "> 1. Yes, proceed",
    "  2. No, exit",
    "Enter to confirm · Esc to cancel",
]


def test_app_dialog() -> None:
    q = parse_question(APP_DIALOG_FRAME)
    assert q is not None
    assert q.kind == "app_dialog"
    assert q.options[0].action == "deny"  # "No, exit"
    assert q.options[1].action == "allow_once"  # "Yes, I accept"
    # #10: the bypass-permissions startup dialog is flagged so the transport
    # auto-accepts it.
    assert q.is_bypass is True


def test_trust_app_dialog_is_not_flagged_bypass() -> None:
    # The folder-trust dialog is app_dialog too, but is NOT a bypass dialog
    # (it is handled by config pre-seeding, not auto-accept).
    q = parse_question(TRUST_DIALOG_FRAME)
    assert q is not None
    assert q.kind == "app_dialog"
    assert q.is_bypass is False


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


# --------------------------------------------------------------------------- #
# /tasks BackgroundTasksDialog parsing (drives stop_task)
# --------------------------------------------------------------------------- #

# A plausible reconstruction of the dialog. The exact layout is an undocumented
# TUI contract -- if row identification regresses against the bundled CLI,
# recapture this fixture from a live frame and update parse_tasks_dialog.
TASKS_FRAME = [
    "Background tasks",
    "─" * 60,
    "❯ Bash  npm run dev                       running   2m",
    "  Agent  investigate flaky test           running   1m",
    "  Monitor  watch build output             running   30s",
    "─" * 60,
    "x to kill · enter for details · esc to close",
]


def test_parse_tasks_dialog_returns_rows_with_selection() -> None:
    rows = parse_tasks_dialog(TASKS_FRAME)
    assert rows is not None
    assert [r.selected for r in rows] == [True, False, False]
    assert rows[0].label.startswith("Bash")
    assert "investigate flaky test" in rows[1].label
    # Title, separators and footer chrome are not rows.
    assert len(rows) == 3


def test_parse_tasks_dialog_skips_non_dialog_screens() -> None:
    # An ordinary numbered list with no tasks chrome is not the dialog.
    assert parse_tasks_dialog(["Here are steps:", "1. do a", "2. do b"]) is None
    # A permission dialog (different chrome) is also not the tasks dialog.
    assert parse_tasks_dialog(WRITE_FRAME) is None


def test_parse_tasks_dialog_ignores_generation_output_mentioning_kill() -> None:
    # Regression (C1): ordinary model output that mentions "kill" and ends with
    # the turn's "esc to interrupt" footer -- including a markdown `>` quote --
    # must NOT be detected as the tasks dialog (detection is gated on the
    # dialog-specific "x to kill"/"x to stop" footer affordance).
    frame = [
        "● I'll now kill the stuck server and restart it.",
        "> kill the dev server gracefully",
        "1. stop process",
        "esc to interrupt",
    ]
    assert parse_tasks_dialog(frame) is None


def test_parse_tasks_dialog_strips_leading_index() -> None:
    frame = [
        "Background tasks",
        "❯ 1. Bash  long running job",
        "  2. Agent  some research",
        "esc to close · x to kill",
    ]
    rows = parse_tasks_dialog(frame)
    assert rows is not None
    assert rows[0].label.startswith("Bash")
    assert rows[1].label.startswith("Agent")


def test_parse_tasks_dialog_keeps_rows_whose_command_contains_esc() -> None:
    # Regression (H1): a task row whose command merely contains "esc"
    # (escape/describe/rescue) must not be dropped as footer chrome.
    frame = [
        "Background tasks",
        "❯ Bash  ./escape.sh --force        running",
        "  Agent  Describe the architecture  running",
        "x to stop · esc to close",
    ]
    rows = parse_tasks_dialog(frame)
    assert rows is not None
    assert len(rows) == 2
    assert rows[0].label.startswith("Bash  ./escape.sh")
    assert "Describe the architecture" in rows[1].label


def test_task_row_matches_tolerates_prefix_and_suffix() -> None:
    # The row carries a tool prefix + trailing status columns; the full command
    # appears verbatim -> match.
    assert task_row_matches("Bash  npm run dev   running 2m", "npm run dev")
    # Truncated long command -> long shared-prefix match (after tool label).
    assert task_row_matches(
        "Bash  python train.py --epochs…", "python train.py --epochs 100 --lr 0.01"
    )
    # Unrelated row -> no match (so stop_task fails safe instead of mis-killing).
    assert not task_row_matches("Agent  unrelated research", "npm run dev")
    # Empty target never matches.
    assert not task_row_matches("Bash  something", "")


def test_task_row_matches_discriminates_sibling_commands() -> None:
    # Regression (C2): sibling commands sharing a long prefix must NOT match the
    # wrong one -- the full distinguishing target must appear in the row.
    assert not task_row_matches(
        "Bash  npm run build:staging  running", "npm run build:prod"
    )
    assert task_row_matches("Bash  npm run build:prod  running", "npm run build:prod")
    # A short shared prefix on a truncated sibling must not match either.
    assert not task_row_matches("Bash  npm run build:…", "npm run build:prod")
    # Near-identical descriptions differing only at the tail.
    assert not task_row_matches(
        "Agent  Refactor the authentication flow",
        "Refactor the authentication module",
    )


def test_task_label_key_strips_status_and_age() -> None:
    # Regression (H2): the volatile status/age column is stripped, so a ticking
    # age cannot make a static row look like it changed.
    assert task_label_key("Bash  npm run dev   running 2m") == "bash npm run dev"
    assert task_label_key("Bash  npm run dev   running 31s") == "bash npm run dev"
    assert task_label_key("Agent  do a thing  killed") == "agent do a thing"
