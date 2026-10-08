#!/usr/bin/env python3
"""Issue #237: circuit breaker for the agentic "Let me [action]" narration loop.

Symptom
-------
The model enters a token-repetition loop in ``thinking_mode: "chat"`` with
large conversations. The model narrates an action it intends to take ("Let me
check the API key", "Wait, let me...", "Actually, let me...") but never emits
the tool-call DSML structure, so it loops on the narration. Prompt directives
(e.g., a tool-call directive) are weak — the model ignores them.

Fix
---
Add a deterministic circuit breaker in the streaming output path
(``vllm/entrypoints/openai/chat_completion/serving.py``). It tracks the
accumulated output text for each request and detects near-duplicate recent
paragraphs (a sliding-window signature comparison). When the count of
near-duplicate paragraph pairs exceeds a threshold, it forces the generation
to stop (sets ``finish_reason`` to "stop") so the loop is broken
deterministically, rather than relying on the model to follow an instruction.

The paragraph and line detectors are the sole trip triggers. On a trip, a
single log line reports the observed count against the threshold (e.g.
``paragraphs=12/10 lines=3/10``).

The detection is fail-open (a detection error does not break the normal
streaming path), but the intervention is deterministic.

Gating and fail-closed operation
--------------------------------
The compose entrypoint invokes this script only when
``DSPARK_ENABLE_ISSUE237_CIRCUIT_BREAKER`` is exactly ``1`` (default ``0`` =
stock behavior, this script never runs) and chains it with ``|| exit 1``. The
anchored region (the ``previous_texts[i] += delta_text`` line) must appear
exactly once; the region constant is sha256-pinned. After writing, a self-check
must pass or the original bytes are restored and the boot fails.

Usage (inside container, after vLLM source is on disk):
  python3 hotfix-vllm-issue237-circuit-breaker.py            # apply (or verify if already applied)
  python3 hotfix-vllm-issue237-circuit-breaker.py --status   # classify the served serving.py
  python3 hotfix-vllm-issue237-circuit-breaker.py --check    # preflight: classify the bytes the next boot will copy
"""
from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
import tempfile
from pathlib import Path

PRODUCTION_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"
)
MARK = "[dspark-issue237-circuit-breaker]"

HELPER_ANCHOR = "import asyncio\n"
HELPER_NEW = (
    "import asyncio\n"
    "# "
    + MARK
    + "\n"
    "def _issue237_circuit_breaker_enabled() -> bool:\n"
    "    import os as _os\n"
    "    return _os.environ.get(\"DSPARK_ENABLE_ISSUE237_CIRCUIT_BREAKER\", \"0\") == \"1\"\n"
    "\n"
    "def _issue237_circuit_breaker_log_enabled() -> bool:\n"
    "    import os as _os\n"
    "    return _os.environ.get(\"DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG\", \"0\") == \"1\"\n"
    "\n"
    "def _issue237_circuit_breaker_log(msg: str) -> None:\n"
    "    \"\"\"Emit a single circuit-breaker log line when logging is enabled.\"\"\"\n"
    "    if _issue237_circuit_breaker_log_enabled():\n"
    "        import sys as _sys\n"
    "        print(f\"[dspark-issue237-circuit-breaker] {msg}\", file=_sys.stderr, flush=True)\n"
    "\n"
    "def _issue237_loop_score(text: str, key: object, state: dict, split, normalize, is_hit, min_len: int) -> int:\n"
    "    \"\"\"Generic sliding-window loop detector (stateful per request).\n"
    "\n"
    "    Split the accumulated text into units with ``split``, normalize each\n"
    "    completed unit with ``normalize``, and flag a unit as a repeat when\n"
    "    ``is_hit`` matches it against the recent window. A hit increments a\n"
    "    counter that resets after 6 consecutive misses (the loop broke).\n"
    "    Returns the current hit count. State is keyed by ``key`` (the request\n"
    "    id) so concurrent requests do not interfere.\n"
    "    \"\"\"\n"
    "    from collections import deque\n"
    "    if not text:\n"
    "        return 0\n"
    "    st = state.setdefault(\n"
    "        key, {\"window\": deque(maxlen=6), \"hits\": 0, \"misses\": 0, \"pos\": 0}\n"
    "    )\n"
    "    # Only process text appended since the last call; ``pos`` is an absolute\n"
    "    # offset into the monotonically-growing accumulated text.\n"
    "    new = text[st[\"pos\"]:]\n"
    "    if not new:\n"
    "        return st[\"hits\"]\n"
    "    parts = split(new)\n"
    "    # The last element may be a partial unit (no trailing delimiter yet).\n"
    "    complete = parts[:-1] if len(parts) > 1 else []\n"
    "    if not complete:\n"
    "        return st[\"hits\"]\n"
    "    st[\"pos\"] += len(new) - len(parts[-1])\n"
    "    for _unit in complete:\n"
    "        _norm = normalize(_unit)\n"
    "        if len(_norm) < min_len:\n"
    "            continue\n"
    "        _hit = is_hit(_norm, st[\"window\"])\n"
    "        if _hit:\n"
    "            st[\"hits\"] += 1\n"
    "            st[\"misses\"] = 0\n"
    "        else:\n"
    "            st[\"misses\"] += 1\n"
    "            if st[\"misses\"] >= 6:\n"
    "                st[\"hits\"] = 0\n"
    "        st[\"window\"].append(_norm)\n"
    "    return st[\"hits\"]\n"
    "\n"
    "def _issue237_paragraph_loop_score(text: str, key: object) -> int:\n"
    "    \"\"\"Sliding-window paragraph loop detector (stateful per request).\n"
    "\n"
    "    Splits on blank lines and compares each completed paragraph's token-set\n"
    "    against the recent window; a near-identical match (small symmetric\n"
    "    difference) counts as a repeat.\n"
    "    \"\"\"\n"
    "    import re as _re\n"
    "    def _split(t):\n"
    "        return _re.split(r\"\\n\\s*\\n\", t)\n"
    "    def _normalize(p):\n"
    "        return set(_re.findall(r\"[a-z0-9]+\", p.lower()))\n"
    "    def _is_hit(toks, window):\n"
    "        for sig in window:\n"
    "            union = toks | sig\n"
    "            if not union:\n"
    "                continue\n"
    "            inter = toks & sig\n"
    "            # Near-identical if the symmetric difference is a small fraction\n"
    "            # of the union (<= 35%).\n"
    "            if (len(union) - len(inter)) / len(union) <= 0.35:\n"
    "                return True\n"
    "        return False\n"
    "    return _issue237_loop_score(text, key, _issue237_paragraph_state, _split, _normalize, _is_hit, 8)\n"
    "\n"
    "def _issue237_paragraph_loop_threshold() -> int:\n"
    "    import os as _os\n"
    "    try:\n"
    "        return int(_os.environ.get(\"DSPARK_ISSUE237_PARAGRAPH_LOOP_THRESHOLD\", \"10\"))\n"
    "    except ValueError:\n"
    "        return 10\n"
    "\n"
    "def _issue237_line_loop_score(text: str, key: object) -> int:\n"
    "    \"\"\"Token-level loop detector: repeated lines (stateful per request).\n"
    "\n"
    "    The paragraph detector splits on blank lines, so a loop that repeats a\n"
    "    single line separated by newlines (e.g. the model echoing one source\n"
    "    line over and over) never forms a complete paragraph and is missed.\n"
    "    This detector splits on newlines and flags a line that repeats a recent\n"
    "    line in the window.\n"
    "    \"\"\"\n"
    "    def _split(t):\n"
    "        return t.split(\"\\n\")\n"
    "    def _normalize(line):\n"
    "        return line.strip()\n"
    "    def _is_hit(s, window):\n"
    "        return s in window\n"
    "    return _issue237_loop_score(text, key, _issue237_line_state, _split, _normalize, _is_hit, 8)\n"
    "\n"
    "def _issue237_line_loop_threshold() -> int:\n"
    "    import os as _os\n"
    "    try:\n"
    "        return int(_os.environ.get(\"DSPARK_ISSUE237_LINE_LOOP_THRESHOLD\", \"10\"))\n"
    "    except ValueError:\n"
    "        return 10\n"
    "\n"
    "_issue237_paragraph_state = {}\n"
    "_issue237_line_state = {}\n"
)

REGION_OLD = (
    "                    previous_texts[i] += delta_text\n"
)
REGION_NEW = (
    "                    previous_texts[i] += delta_text\n"
    "                    # [dspark-issue237-circuit-breaker] Deterministic circuit\n"
    "                    # breaker for the agentic narration loop. The paragraph\n"
    "                    # and line detectors are the sole trip triggers: they fire\n"
    "                    # only on near-duplicate recent paragraphs or repeated\n"
    "                    # lines, which is the reliable loop signal. Log a single\n"
    "                    # line per trip (gated by\n"
    "                    # DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG) reporting the\n"
    "                    # observed count against the threshold. Fail-open on\n"
    "                    # error.\n"
    "                    if _issue237_circuit_breaker_enabled():\n"
    "                        try:\n"
    "                            _txt = previous_texts[i]\n"
    "                            _para = _issue237_paragraph_loop_score(_txt, request_id)\n"
    "                            _pthr = _issue237_paragraph_loop_threshold()\n"
    "                            _line = _issue237_line_loop_score(_txt, request_id)\n"
    "                            _lthr = _issue237_line_loop_threshold()\n"
    "                            if _para >= _pthr or _line >= _lthr:\n"
    "                                if not _issue237_paragraph_state[request_id].get(\"tripped\"):\n"
    "                                    _issue237_paragraph_state[request_id][\"tripped\"] = True\n"
    "                                    _issue237_circuit_breaker_log(\n"
    "                                        f\"trip paragraphs={_para}/{_pthr} \"\n"
    "                                        f\"lines={_line}/{_lthr}\"\n"
    "                                    )\n"
    "                                    # Abort the request at the engine so\n"
    "                                    # generation stops immediately instead of\n"
    "                                    # continuing to burn tokens in the\n"
    "                                    # background. The finish_reason below is\n"
    "                                    # what the client sees; the abort is what\n"
    "                                    # actually halts the engine. Safe to call\n"
    "                                    # mid-generator: vLLM treats an aborted\n"
    "                                    # in-flight request as finished.\n"
    "                                    await self.engine_client.abort(request_id)\n"
    "                                output.finish_reason = \"stop\"\n"
    "                        except Exception:\n"
    "                            pass\n"
)

# Self-pins: the region constant must not drift inside this file.
REGION_OLD_SHA256 = "94de98ec8d8311897767cd54de6b1119a5944eae3b9c0328639e66ae1630913d"
REGION_NEW_SHA256 = "b85e2583e710ce74ed8d4799daf8e9ec2857bb8f363c0c3f03f6e9e44a55e48e"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_self() -> None:
    if _sha256(REGION_OLD.encode()) != REGION_OLD_SHA256:
        raise HotfixError("REGION_OLD does not match its pinned sha256")
    if REGION_NEW_SHA256 and _sha256(REGION_NEW.encode()) != REGION_NEW_SHA256:
        raise HotfixError("REGION_NEW does not match its pinned sha256")


class HotfixError(RuntimeError):
    pass


def inspect_bytes(data: bytes) -> str:
    """Classify serving.py bytes; anything but exactly-one anchor fails closed."""
    _verify_self()
    old_n = data.count(REGION_OLD.encode())
    new_n = data.count(REGION_NEW.encode())
    # REGION_NEW begins with the exact REGION_OLD line, so a patched file
    # always contains one embedded REGION_OLD. Subtract those to get the
    # count of standalone (unpatched) region sites.
    standalone_old = old_n - new_n
    marked = MARK.encode() in data
    if new_n == 1 and standalone_old == 0 and marked:
        return "patched"
    if new_n == 0 and standalone_old == 1 and not marked:
        return "stock"
    raise HotfixError(
        f"unsupported serving.py bytes (region_old x{old_n}, region_new x{new_n}, "
        f"mark={marked}, sha256={_sha256(data)}); expected exactly one of either"
    )


def inspect(target: Path) -> tuple[str, bytes]:
    try:
        st = target.lstat()
    except FileNotFoundError:
        raise HotfixError(f"target is missing: {target}")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise HotfixError(f"target is not a regular file: {target}")
    data = target.read_bytes()
    return inspect_bytes(data), data


def transform(stock: bytes) -> bytes:
    """Stock bytes -> patched bytes; refuses anything but exactly one site."""
    if inspect_bytes(stock) != "stock":
        raise HotfixError("transform requires stock bytes")
    patched = stock.replace(HELPER_ANCHOR.encode(), HELPER_NEW.encode(), 1)
    patched = patched.replace(REGION_OLD.encode(), REGION_NEW.encode(), 1)
    compile(patched, "serving.py", "exec")
    return patched


def _issue237_verify_detectors(
    para, line, para_threshold, line_threshold
) -> tuple[bool, str]:
    """Behavioral proof that the loop detectors trip on a loop and stay quiet
    on distinct prose. Returns (ok, why)."""
    def _feed(score, units, key):
        # Grow the accumulated text one unit at a time, as the streaming path
        # does, so the stateful pos/window advance and the detector sees the
        # full sequence of units.
        text = ""
        for u in units:
            text += u
            score(text, key)
        return score(text, key)

    repeated_para = "The API key rotation failed and the token loop continued indefinitely.\n\n"
    if _feed(para, [repeated_para] * 12, "sc-para-repeat") < 10:
        return False, "paragraph detector did not trip on repeated paragraphs"

    distinct_para = [
        p + "\n\n"
        for p in [
            "Quantum computing leverages superposition and entanglement for computation.",
            "The weather forecast predicts rain showers across the coastal region tomorrow.",
            "Baking sourdough bread requires patience, hydration, and a warm kitchen.",
            "Mountain biking trails wind through pine forests and rocky switchbacks.",
        ]
    ]
    if _feed(para, distinct_para, "sc-para-distinct") != 0:
        return False, "paragraph detector false-positived on distinct paragraphs"

    repeated_line = '    _stub("vllm.entrypoints.openai.tool_parsers.tool_parsers_utils")\n'
    if _feed(line, [repeated_line] * 12, "sc-line-repeat") < 10:
        return False, "line detector did not trip on repeated lines"

    distinct_line = [
        f"This is distinct line number {i} about a different topic entirely.\n"
        for i in range(5)
    ]
    if _feed(line, distinct_line, "sc-line-distinct") != 0:
        return False, "line detector false-positived on distinct lines"

    # Thresholds must resolve to sane defaults.
    if para_threshold() < 1:
        return False, "paragraph loop threshold not sane"
    if line_threshold() < 1:
        return False, "line loop threshold not sane"

    return True, "detectors trip on loops and ignore distinct prose"


def _self_check(patched_src: bytes) -> tuple[bool, str]:
    """Behavioral proof: the circuit breaker helpers are present, the module
    compiles, and the loop detectors actually trip on repeated input while
    staying quiet on distinct prose."""
    import importlib.util

    stock_src = patched_src.replace(REGION_NEW.encode(), REGION_OLD.encode(), 1)
    stock_src = stock_src.replace(HELPER_NEW.encode(), HELPER_ANCHOR.encode(), 1)
    with tempfile.TemporaryDirectory(prefix="dspark-issue237-cb-") as tmpdir:
        try:
            def _load(src: bytes, name: str):
                path = Path(tmpdir) / f"{name}.py"
                path.write_bytes(src)
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise HotfixError(f"cannot load module spec for {name}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            patched = _load(patched_src, "serving_patched_cb")
            stock = _load(stock_src, "serving_stock_cb")

            # The circuit breaker helpers must be present in the patched module
            if not hasattr(patched, "_issue237_circuit_breaker_enabled"):
                return False, "circuit breaker helper missing in patched module"
            if not hasattr(patched, "_issue237_paragraph_loop_score"):
                return False, "paragraph loop scorer missing in patched module"
            if not hasattr(patched, "_issue237_paragraph_loop_threshold"):
                return False, "paragraph loop threshold helper missing in patched module"
            if not hasattr(patched, "_issue237_line_loop_score"):
                return False, "line loop scorer missing in patched module"
            if not hasattr(patched, "_issue237_line_loop_threshold"):
                return False, "line loop threshold helper missing in patched module"
            if not hasattr(patched, "_issue237_circuit_breaker_log_enabled"):
                return False, "circuit breaker log helper missing in patched module"
            if not hasattr(patched, "_issue237_circuit_breaker_log"):
                return False, "circuit breaker log emitter missing in patched module"
            # The stock module must NOT have the helpers
            if hasattr(stock, "_issue237_circuit_breaker_enabled"):
                return False, "circuit breaker helper unexpectedly present in stock module"
            if hasattr(stock, "_issue237_paragraph_loop_score"):
                return False, "paragraph loop scorer unexpectedly present in stock module"

            # Behavioral proof: the detectors must actually trip on a loop and
            # stay quiet on distinct prose.
            ok, why = _issue237_verify_detectors(
                patched._issue237_paragraph_loop_score,
                patched._issue237_line_loop_score,
                patched._issue237_paragraph_loop_threshold,
                patched._issue237_line_loop_threshold,
            )
            if not ok:
                return False, why
        except HotfixError:
            raise
        except Exception as err:  # broken/unimportable patch must fail closed
            return False, f"self-check raised {type(err).__name__}: {err}"
    return True, "circuit breaker helpers verified; detectors trip on loops and ignore distinct prose"


def apply(target: Path) -> str:
    state, data = inspect(target)
    if state == "patched":
        ok, why = _self_check(data)
        if not ok:
            raise HotfixError(f"already patched but self-check failed: {why}")
        return "already-patched"
    patched = transform(data)
    ok, why = _self_check(patched)
    if not ok:
        raise HotfixError(f"self-check failed before write: {why}")
    fd, tmp_name = tempfile.mkstemp(prefix=".dspark-issue237-cb-", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(patched)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    verify_state, verify_data = inspect(target)
    if verify_state != "patched":
        raise HotfixError("post-apply verification failed")
    ok, why = _self_check(verify_data)
    if not ok:
        # Fail closed: never leave a written-but-unverified file behind.
        target.write_bytes(data)
        raise HotfixError(f"post-apply self-check failed, original restored: {why}")
    return "applied"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="preflight: classify the bytes the next boot will copy")
    parser.add_argument("--status", action="store_true",
                        help="classify the served serving.py copy")
    parser.add_argument("--target", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        target = args.target or PRODUCTION_TARGET
        if args.check or args.status:
            state, data = inspect(target)
            ok, why = _self_check(data if state == "patched" else transform(data))
            if not ok:
                raise HotfixError(f"self-check failed: {why}")
            print(f"dspark-issue237-circuit-breaker: {state} ({target})")
            return 0
        outcome = apply(target)
        print(f"dspark-issue237-circuit-breaker: {outcome} ({target})")
        return 0
    except HotfixError as error:
        print(f"dspark-issue237-circuit-breaker: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
