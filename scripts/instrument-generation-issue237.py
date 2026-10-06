#!/usr/bin/env python3
"""Generation-phase instrumentation for issue #237 chat-mode loop.

Monkeypatches vllm.v1.request.Request.append_output_token_ids to detect token
repetition (a fixed-length window repeating) and log the request id, token
sequence, and repetition window to /tmp/issue237_gen_capture.jsonl.

This is diagnostic only. It does not change model behavior. Deploy it as a sitecustomize.py on the Python path so it auto-loads in the
engine process (a docker exec of a separate process will NOT work).

Usage (inside container, before the engine starts):
    python3 /opt/instrument-generation-issue237.py
"""
from __future__ import annotations

import json
import os
import sys

CAPTURE = "/tmp/issue237_gen_capture.jsonl"
WINDOW = 10  # number of tokens to compare for repetition


def _detect_repeat(token_ids: list[int]) -> bool:
    """Return True if the last WINDOW tokens repeat any earlier WINDOW-sized run.

    Checks the tail against every prior WINDOW-sized window (not just the
    immediately preceding one) to catch partial/overlapping repeats that a
    strict adjacent-window comparison misses.
    """
    if len(token_ids) < 2 * WINDOW:
        return False
    tail = token_ids[-WINDOW:]
    for start in range(0, len(token_ids) - WINDOW):
        if token_ids[start:start + WINDOW] == tail:
            return True
    return False


def _install() -> None:
    """Monkeypatch Request.append_output_token_ids to log every generated token.

    Deferred until called so it can run after vllm is imported (sitecustomize
    loads before vllm, so importing vllm at module import time would fail).
    """
    try:
        from vllm.v1 import request as request_mod
    except Exception as e:
        print(f"[issue237-gen] vllm import failed, will retry: {e}", file=sys.stderr)
        return

    orig = request_mod.Request.append_output_token_ids

    def patched(self, token_ids):
        orig(self, token_ids)
        try:
            seq = list(self._output_token_ids)
            rec = {
                "request_id": self.request_id,
                "n_output_tokens": len(seq),
                "last_token": seq[-1] if seq else None,
                "repeat": _detect_repeat(seq),
                "max_tokens": getattr(self, "max_tokens", None),
            }
            with open(CAPTURE, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    request_mod.Request.append_output_token_ids = patched
    print(f"[issue237-gen] installed token logger (window={WINDOW}) -> {CAPTURE}", file=sys.stderr)


# Auto-install on import so it loads in the engine process via sitecustomize.
# The vllm import is deferred inside _install() so this does not crash at
# sitecustomize import time (vllm is not yet imported then).
_install()
