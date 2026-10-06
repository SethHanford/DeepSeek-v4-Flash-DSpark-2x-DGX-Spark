#!/usr/bin/env python3
"""Instrument the live encoder to log what _encode_messages_text receives.

Run inside the vllm-dspark container. It monkeypatches _encode_messages_text to
print the messages, thinking_mode, drop_thinking, and the resolved
effective_drop_thinking for every call, so you can see exactly what the loop
request looks like and whether historical reasoning is being dropped.

Usage (inside container):
    python3 /opt/instrument-issue237.py

Then send a tool-calling request that triggers the loop and watch the log.
"""
from __future__ import annotations

import json
import sys
from vllm.tokenizers import deepseek_v4_encoding as enc

_orig = enc._encode_messages_text


def _patched(messages, thinking_mode, context=None, drop_thinking=True,
             add_default_bos_token=True, reasoning_effort=None):
    # Replicate the patch's effective_drop_thinking resolution for reporting.
    full = (context or []) + messages
    effective = drop_thinking
    if any(m.get("tools") for m in full):
        effective = False

    print("=" * 70, file=sys.stderr)
    print(f"[issue237-instrument] thinking_mode={thinking_mode!r} "
          f"drop_thinking={drop_thinking!r} effective_drop_thinking={effective!r} "
          f"n_messages={len(messages)} n_context={len(context or [])}", file=sys.stderr)
    for i, m in enumerate(messages):
        role = m.get("role")
        has_rc = "reasoning_content" in m
        rc = m.get("reasoning_content")
        rc_preview = (rc[:80] + "...") if isinstance(rc, str) and len(rc) > 80 else rc
        tools = "tools" in m
        tcs = m.get("tool_calls")
        print(f"  [{i}] role={role!r} has_reasoning={has_rc} tools={tools} "
              f"tool_calls={len(tcs) if tcs else 0} "
              f"reasoning={rc_preview!r}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    return _orig(messages, thinking_mode, context=context,
                 drop_thinking=drop_thinking,
                 add_default_bos_token=add_default_bos_token,
                 reasoning_effort=reasoning_effort)


enc._encode_messages_text = _patched
print("[issue237-instrument] installed. Send a tool-calling request and watch stderr.", file=sys.stderr)
