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

The paragraph detector is the sole trip trigger. The "Let me" phrase counter
and the stuck/restart phrase counter are computed and logged as metrics only
(they are too noisy to trip on, since "let me" is common in legitimate prose).

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
    "def _issue237_count_loop_phrases(text: str) -> int:\n"
    "    \"\"\"Count occurrences of the agentic narration loop phrases.\"\"\"\n"
    "    import re as _re\n"
    "    if not text:\n"
    "        return 0\n"
    "    # Match \"Let me\", \"Wait, let me\", \"Actually, let me\", and variants.\n"
    "    # Word boundaries avoid matching \"let me\" inside words like \"outlet\"\n"
    "    # or \"tablet\". Case-insensitive; count each occurrence.\n"
    "    return len(_re.findall(r\"(?i)(?:\\blet me\\b|\\bwait[,.]?\\s+let me\\b|\\bactually[,.]?\\s+let me\\b)\", text))\n"
    "\n"
    "def _issue237_count_stuck_phrases(text: str) -> int:\n"
    "    \"\"\"Count self-acknowledged stuck/restart phrases.\n"
    "\n"
    "    Catches the response-restart loop style where the model admits it is\n"
    "    looping and restarts (\"I'm looping again. Let me stop and...\") but the\n"
    "    paragraphs are otherwise distinct, so the paragraph detector misses it.\n"
    "    \"\"\"\n"
    "    import re as _re\n"
    "    if not text:\n"
    "        return 0\n"
    "    _t = text.lower()\n"
    "    _patterns = [\n"
    "        r\"\\bi['’]?m looping\\b\", r\"\\bi am looping\\b\", r\"\\bi keep looping\\b\",\n"
    "        r\"\\bi['’]?m stuck\\b\", r\"\\bi am stuck\\b\", r\"\\bi am going in circles\\b\",\n"
    "        r\"\\bi keep repeating\\b\", r\"\\bi am repeating myself\\b\",\n"
    "        r\"\\blet me stop\\b\", r\"\\blet me take a step back\\b\", r\"\\blet me step back\\b\",\n"
    "        r\"\\blet me restart\\b\", r\"\\blet me start over\\b\", r\"\\blet me begin again\\b\",\n"
    "        r\"\\blet me try again\\b\", r\"\\blet me redo\\b\", r\"\\blet me re-examine\\b\",\n"
    "        r\"\\blet me revisit\\b\", r\"\\blet me make a plan\\b\", r\"\\blet me outline a plan\\b\",\n"
    "        r\"\\blet me check in\\b\", r\"\\blet me report back\\b\", r\"\\blet me pause\\b\",\n"
    "        r\"\\blet me regroup\\b\", r\"\\blet me reset\\b\", r\"\\blet me hold on\\b\",\n"
    "    ]\n"
    "    return sum(len(_re.findall(_p, _t)) for _p in _patterns)\n"
    "\n"
    "def _issue237_paragraph_loop_score(text: str, key: object) -> int:\n"
    "    \"\"\"Sliding-window paragraph loop detector (stateful per request).\n"
    "\n"
    "    Split the accumulated text into paragraphs on blank lines. Keep a\n"
    "    backward window of the last K paragraph token-sets. When a new\n"
    "    paragraph completes, compare it against the window; a near-identical\n"
    "    match (small symmetric difference) increments a hit counter. The\n"
    "    counter resets after K consecutive misses (the loop broke). Returns\n"
    "    the current hit count. State is keyed by ``key`` (the request id) so\n"
    "    concurrent requests do not interfere.\n"
    "    \"\"\"\n"
    "    import re as _re\n"
    "    from collections import deque\n"
    "    if not text:\n"
    "        return 0\n"
    "    st = _issue237_para_state.setdefault(\n"
    "        key, {\"window\": deque(maxlen=6), \"hits\": 0, \"misses\": 0, \"pos\": 0, \"tripped\": False}\n"
    "    )\n"
    "    # Only process text appended since the last call; ``pos`` is an absolute\n"
    "    # offset into the monotonically-growing accumulated text.\n"
    "    new = text[st[\"pos\"]:]\n"
    "    if not new:\n"
    "        return st[\"hits\"]\n"
    "    parts = _re.split(r\"\\n\\s*\\n\", new)\n"
    "    # The last element may be a partial paragraph (no trailing blank line yet).\n"
    "    complete = parts[:-1] if len(parts) > 1 else []\n"
    "    if not complete:\n"
    "        return st[\"hits\"]\n"
    "    st[\"pos\"] += len(new) - len(parts[-1])\n"
    "    for _p in complete:\n"
    "        _toks = set(_re.findall(r\"[a-z0-9]+\", _p.lower()))\n"
    "        if len(_toks) < 8:\n"
    "            continue\n"
    "        _hit = False\n"
    "        for _sig in st[\"window\"]:\n"
    "            _union = _toks | _sig\n"
    "            if not _union:\n"
    "                continue\n"
    "            _inter = _toks & _sig\n"
    "            # Near-identical if the symmetric difference is a small fraction\n"
    "            # of the union (<= 35%).\n"
    "            if (len(_union) - len(_inter)) / len(_union) <= 0.35:\n"
    "                _hit = True\n"
    "                break\n"
    "        if _hit:\n"
    "            st[\"hits\"] += 1\n"
    "            st[\"misses\"] = 0\n"
    "        else:\n"
    "            st[\"misses\"] += 1\n"
    "            if st[\"misses\"] >= 6:\n"
    "                st[\"hits\"] = 0\n"
    "        st[\"window\"].append(_toks)\n"
    "    return st[\"hits\"]\n"
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
    "    line in the window. State is keyed by ``key`` (the request id) so\n"
    "    concurrent requests do not interfere.\n"
    "    \"\"\"\n"
    "    from collections import deque\n"
    "    if not text:\n"
    "        return 0\n"
    "    st = _issue237_line_state.setdefault(\n"
    "        key, {\"window\": deque(maxlen=6), \"hits\": 0, \"misses\": 0, \"pos\": 0}\n"
    "    )\n"
    "    new = text[st[\"pos\"]:]\n"
    "    if not new:\n"
    "        return st[\"hits\"]\n"
    "    parts = new.split(\"\\n\")\n"
    "    # The last element may be a partial line (no trailing newline yet).\n"
    "    complete = parts[:-1] if len(parts) > 1 else []\n"
    "    if not complete:\n"
    "        return st[\"hits\"]\n"
    "    st[\"pos\"] += len(new) - len(parts[-1])\n"
    "    for _line in complete:\n"
    "        _s = _line.strip()\n"
    "        if len(_s) < 8:\n"
    "            continue\n"
    "        _hit = _s in st[\"window\"]\n"
    "        if _hit:\n"
    "            st[\"hits\"] += 1\n"
    "            st[\"misses\"] = 0\n"
    "        else:\n"
    "            st[\"misses\"] += 1\n"
    "            if st[\"misses\"] >= 6:\n"
    "                st[\"hits\"] = 0\n"
    "        st[\"window\"].append(_s)\n"
    "    return st[\"hits\"]\n"
    "\n"
    "def _issue237_line_loop_threshold() -> int:\n"
    "    import os as _os\n"
    "    try:\n"
    "        return int(_os.environ.get(\"DSPARK_ISSUE237_LINE_LOOP_THRESHOLD\", \"10\"))\n"
    "    except ValueError:\n"
    "        return 10\n"
    "\n"
    "_issue237_para_state = {}\n"
    "_issue237_line_state = {}\n"
)

REGION_OLD = (
    "                    previous_texts[i] += delta_text\n"
)
REGION_NEW = (
    "                    previous_texts[i] += delta_text\n"
    "                    # [dspark-issue237-circuit-breaker] Deterministic circuit\n"
    "                    # breaker for the agentic narration loop. The paragraph\n"
    "                    # detector is the sole trip trigger: it fires only on\n"
    "                    # near-duplicate recent paragraphs, which is the reliable\n"
    "                    # loop signal. The 'Let me' phrase counter and the\n"
    "                    # stuck/restart phrase counter are logged as metrics only\n"
    "                    # (they are too noisy to trip on, since 'let me' is common\n"
    "                    # in legitimate prose). Log a single line per trip (gated\n"
    "                    # by DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG). Fail-open on\n"
    "                    # error.\n"
    "                    if _issue237_circuit_breaker_enabled():\n"
    "                        try:\n"
    "                            _txt = previous_texts[i]\n"
    "                            _count = _issue237_count_loop_phrases(_txt)\n"
    "                            _stuck = _issue237_count_stuck_phrases(_txt)\n"
    "                            _para = _issue237_paragraph_loop_score(_txt, id(request))\n"
    "                            _pthr = _issue237_paragraph_loop_threshold()\n"
    "                            _line = _issue237_line_loop_score(_txt, id(request))\n"
    "                            _lthr = _issue237_line_loop_threshold()\n"
    "                            if _para >= _pthr or _line >= _lthr:\n"
    "                                if not _issue237_para_state[id(request)].get(\"tripped\"):\n"
    "                                    _issue237_para_state[id(request)][\"tripped\"] = True\n"
    "                                    _issue237_circuit_breaker_log(\n"
    "                                        f\"trip paragraphs={_para}/{_pthr} \"\n"
    "                                        f\"lines={_line}/{_lthr} \"\n"
    "                                        f\"(phrases={_count} stuck={_stuck})\"\n"
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
REGION_NEW_SHA256 = "29f2bcd7ef6221b8e795977ffaa1228181259cd2ed91a8cb7a05742c8b1caac7"


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


def _self_check(patched_src: bytes) -> tuple[bool, str]:
    """Behavioral proof: the circuit breaker helpers are present and the module compiles."""
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
            if not hasattr(patched, "_issue237_count_loop_phrases"):
                return False, "loop phrase counter missing in patched module"
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
            if not hasattr(patched, "_issue237_count_stuck_phrases"):
                return False, "stuck phrase counter missing in patched module"
            # The stock module must NOT have the helpers
            if hasattr(stock, "_issue237_circuit_breaker_enabled"):
                return False, "circuit breaker helper unexpectedly present in stock module"
            if hasattr(stock, "_issue237_paragraph_loop_score"):
                return False, "paragraph loop scorer unexpectedly present in stock module"
            if hasattr(stock, "_issue237_count_stuck_phrases"):
                return False, "stuck phrase counter unexpectedly present in stock module"
        except HotfixError:
            raise
        except Exception as err:  # broken/unimportable patch must fail closed
            return False, f"self-check raised {type(err).__name__}: {err}"
    return True, "circuit breaker helpers verified; stock module unchanged"


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
