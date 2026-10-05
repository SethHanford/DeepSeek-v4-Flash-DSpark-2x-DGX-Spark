#!/usr/bin/env python3
"""Hotfix Issue #237: reasoning self-contamination loop in agent mode.

Upstream: deepseek-ai/DeepSeek-V4-Flash-0731 encoding/encoding_dsv4.py
(installed into vLLM as tokenizers/deepseek_v4_encoding.py).

`_encode_messages_text` force-disables reasoning dropping whenever the request
carries any tool definitions:

    # Resolve drop_thinking: if any message has tools defined, don't drop thinking
    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

Agent harnesses (Copilot CLI, etc.) send tool definitions on every turn, so
`_drop_thinking_messages` never runs and every earlier assistant turn's
``reasoning_content`` is re-encoded into the prompt. The replayed reasoning
self-contaminates the context and the thinking channel degenerates into a
repetition loop that runs until ``max_tokens`` (issue #237; mirrors the premature
`finish_reason=stop` face reported against the hosted DeepSeek API). The official
DeepSeek API does not replay historical reasoning.

This patch removes the tools override so ``drop_thinking`` is honored even when
tools are present: earlier-turn reasoning is dropped, the live turn (messages at
or after the last user index) still reasons. Live-turn behavior, EOS, tool
rendering, and the non-thinking path are unchanged.

Opt-in / default-off: the compose entrypoint only runs this patcher when
``DSPARK_ENABLE_DROP_HISTORY_REASONING=1``. Unset/0 keeps the stock encoder.

Usage (inside container, after the encoder copy):
  python3 hotfix-dsv4-drop-history-reasoning.py
  python3 hotfix-dsv4-drop-history-reasoning.py --status
  python3 hotfix-dsv4-drop-history-reasoning.py /path/to/deepseek_v4_encoding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py"
)

MARK = "# [drop-history-reasoning]"

OLD = (
    "    # Resolve drop_thinking: if any message has tools defined, don't drop thinking\n"
    "    effective_drop_thinking = drop_thinking\n"
    "    if any(m.get(\"tools\") for m in full_messages):\n"
    "        effective_drop_thinking = False\n"
)

NEW = (
    "    # Resolve drop_thinking: if any message has tools defined, don't drop thinking\n"
    "    # [drop-history-reasoning] The tools override is removed so earlier-turn\n"
    "    # reasoning_content is dropped even when tool definitions are present,\n"
    "    # matching the official DeepSeek API (which does not replay historical\n"
    "    # reasoning). This breaks the reasoning self-contamination loop seen in\n"
    "    # agent mode (issue #237). The live turn (index >= last user) still reasons.\n"
    "    effective_drop_thinking = drop_thinking\n"
)


def patch_text(source: str) -> tuple[str, str]:
    """Return (updated_text, status) where status is applied|skipped|missing."""
    if MARK in source:
        return source, "skipped"
    if OLD not in source:
        return source, "missing"
    return source.replace(OLD, NEW, 1), "applied"


def patch_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    updated, status = patch_text(text)
    if status == "applied":
        path.write_text(updated, encoding="utf-8")
    return status


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "--status":
        target = Path(argv[2]) if len(argv) > 2 else DEFAULT_TARGET
        if not target.is_file():
            print("issue237 drop-history-reasoning: NOT APPLIED (encoder file missing)")
            return 0
        _, st = patch_text(target.read_text(encoding="utf-8"))
        print(
            "issue237 drop-history-reasoning:",
            "APPLIED" if st != "missing" else "NOT APPLIED",
        )
        return 0
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_TARGET
    if not target.is_file():
        print(f"[FAIL] encoding file not found: {target}", file=sys.stderr)
        return 1
    status = patch_file(target)
    if status == "applied":
        print(f"[OK] Issue #237 drop-history-reasoning applied: {target}")
        return 0
    if status == "skipped":
        print(f"[OK] Issue #237 drop-history-reasoning already present: {target}")
        return 0
    print(
        "[FAIL] effective_drop_thinking tools-override pattern not found; "
        f"Issue #237 patch not applied: {target}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
