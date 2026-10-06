#!/usr/bin/env python3
"""Compare stock vs patched encoder output for a tool-calling conversation.

This is a diagnostic, not a unit test. It loads the live-chain encoder fixture,
applies the issue #237 patch to a copy, and encodes a realistic multi-turn
tool-calling conversation (the shape Copilot sends) with both. It prints the
encoded prompts side by side so you can see whether the patch actually drops
historical reasoning_content.

Run from the repo root:
    python3 scripts/diag-issue237-encode-compare.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts" / "fixtures" / "issue144-effort-align" / "deepseek_v4_encoding-48095b345-live-chain.py"
PATCHER = ROOT / "patches" / "hotfix-dsv4-drop-history-reasoning.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def apply_patch(source: str) -> str:
    spec = importlib.util.spec_from_file_location("p", PATCHER)
    p = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(p)
    updated, status = p.patch_text(source)
    assert status == "applied", f"patch status: {status}"
    return updated


def build_conversation():
    """A realistic multi-turn tool-calling conversation (Copilot-like).

    Mirrors the real vLLM layer: tools are placed on a system message at index 0
    (see deepseek_v4.py apply_chat_template), and tool results are merged into
    user messages by merge_tool_messages.
    """
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    return [
        {"role": "system", "content": "You are a coding agent.", "tools": tools},
        {"role": "user", "content": "Read the file src/main.py and fix the bug."},
        {
            "role": "assistant",
            "content": "I'll read the file first.",
            "reasoning_content": "The user wants me to read src/main.py. Let me call the read_file tool.",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/main.py\"}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "def main():\n    print('hello')\n"},
        {
            "role": "assistant",
            "content": "I see the bug. Let me fix it.",
            "reasoning_content": "The file prints hello. The bug is that it doesn't return a value. Let me apply a patch.",
            "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "apply_patch", "arguments": "{\"patch\": \"...\"}"}}],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "Patch applied."},
        {
            "role": "assistant",
            "content": "The bug is fixed.",
            "reasoning_content": "The patch was applied successfully. The bug is now fixed.",
        },
    ]


def main() -> int:
    stock_src = FIXTURE.read_text(encoding="utf-8")
    patched_src = apply_patch(stock_src)

    stock = load_module(FIXTURE, "stock_enc")

    # Write patched source to a temp file and load it
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="i237-diag-"))
    patched_path = tmp / "patched_enc.py"
    patched_path.write_text(patched_src, encoding="utf-8")
    patched = load_module(patched_path, "patched_enc")

    msgs = build_conversation()

    # Inspect preprocessing: does reasoning_content survive merge/sort?
    merged = stock.merge_tool_messages(msgs)
    print("=== after merge_tool_messages ===")
    for i, m in enumerate(merged):
        print(f"  [{i}] role={m.get('role')} has_reasoning={'reasoning_content' in m} tools={'tools' in m} tool_calls={'tool_calls' in m}")

    # Encode with tools present (the trigger condition)
    stock_out = stock.encode_messages(msgs, "thinking")
    patched_out = patched.encode_messages(msgs, "thinking")

    print("=" * 70)
    print("STOCK (tools-override present) — historical reasoning REPLAYED")
    print("=" * 70)
    print(stock_out)
    print()
    print("=" * 70)
    print("PATCHED (tools-override removed) — historical reasoning DROPPED?")
    print("=" * 70)
    print(patched_out)
    print()

    # Count how many times reasoning_content appears in each
    stock_reasoning = stock_out.count("The user wants me to read")
    patched_reasoning = patched_out.count("The user wants me to read")
    print(f"Historical reasoning occurrences (turn 1): stock={stock_reasoning}, patched={patched_reasoning}")
    print(f"Historical reasoning occurrences (turn 2): stock={stock_out.count('The file prints hello')}, patched={patched_out.count('The file prints hello')}")

    if stock_out == patched_out:
        print("\nRESULT: IDENTICAL — the patch does NOT change the encoded prompt for this tool conversation.")
        print("        The loop is likely NOT caused by historical reasoning replay, OR drop_thinking is False at the call site.")
    else:
        print("\nRESULT: DIFFERENT — the patch DOES change the encoded prompt.")
        print("        Inspect the diff above to confirm historical reasoning is dropped.")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
