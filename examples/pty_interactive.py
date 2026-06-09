#!/usr/bin/env python3
"""Demonstrates the SDK driving the interactive Claude Code CLI over a PTY.

Since v-next, the SDK launches the *interactive* CLI attached to a
pseudo-terminal instead of the headless ``stream-json`` pipe. This is an
internal implementation detail -- the public ``query()`` / ``ClaudeSDKClient``
API is unchanged -- so this example looks like any other.

Run it on a machine where you are already logged in to Claude Code::

    python examples/pty_interactive.py
"""

import anyio

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    query,
)


async def one_shot() -> None:
    """A single prompt via the one-shot ``query()`` helper."""
    print("=== one-shot query() ===")
    async for message in query(prompt="Reply with a single word: pong"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text.strip()}")
        elif isinstance(message, ResultMessage):
            print(f"(done in {message.duration_ms} ms)")
    print()


async def multi_turn() -> None:
    """A persistent interactive session with several turns."""
    print("=== multi-turn ClaudeSDKClient ===")
    options = ClaudeAgentOptions()
    async with ClaudeSDKClient(options=options) as client:
        for prompt in (
            "Say the word 'apple' and nothing else.",
            "Now say the word 'banana' and nothing else.",
        ):
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"Claude: {block.text.strip()}")
    print()


async def main() -> None:
    await one_shot()
    await multi_turn()


if __name__ == "__main__":
    anyio.run(main)
