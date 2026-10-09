#!/usr/bin/env python3
"""Issue #237: emit a tool call on the "Let me [action]" narration loop.

Symptom
-------
The model enters a token-repetition loop in ``thinking_mode: "chat"`` with
large conversations. The model narrates an action it intends to take ("Let me
check the API key") but never emits the tool-call DSML structure, so it loops
on the narration. The circuit-breaker hotfix
(``hotfix-vllm-issue237-circuit-breaker.py``) breaks the loop deterministically
by aborting the request and forcing ``finish_reason`` to "stop".

This hotfix explores a different intervention: instead of merely stopping, it
*completes the skipped step*. The model has already reasoned about which tool
it wants (that is why it narrates "Let me check the API key"), but it failed to
emit the DSML ``<invoke name="...">`` structure. This hotfix recognizes the
narrated intent, matches it against the tools declared in ``request.tools``
(the same set the model was reasoning over), and synthesizes a real tool call
so the agent lane receives a dispatchable tool call instead of dead prose.

The intervention is layered on the same loop detection as the circuit breaker
(near-duplicate recent paragraphs or repeated lines). On a trip it:

1. Extracts the most recent "Let me [action]" / "I need to [action]" /
   "I'll [action]" / "I should [action]" / "Let's [action]" narration.
2. Scores each declared tool (name + description) against the narrated action
   by token overlap and picks the best match above a confidence threshold.
3. If a tool matches, synthesizes a ``DeltaMessage`` carrying that tool call,
   marks ``tools_streamed`` so the finish logic reports
   ``finish_reason="tool_calls"``, and aborts the request at the engine so the
   loop stops burning tokens.
4. If no tool matches (or no narration is found), it falls back to the
   circuit-breaker behavior: abort and force ``finish_reason`` to "stop".

Detection is fail-open (a detection error does not break the normal streaming
path); the intervention is deterministic.

Gating and fail-closed operation
--------------------------------
The compose entrypoint invokes this script only when
``DSPARK_ENABLE_ISSUE237_TOOLCALL_ON_LOOP`` is exactly ``1`` (default ``0`` =
stock behavior, this script never runs) and chains it with ``|| exit 1``. The
anchored region (the ``previous_texts[i] += delta_text`` line) must appear
exactly once; the region constant is sha256-pinned. After writing, a self-check
must pass or the original bytes are restored and the boot fails.

Usage (inside container, after vLLM source is on disk):
  python3 hotfix-vllm-issue237-toolcall-on-loop.py            # apply (or verify if already applied)
  python3 hotfix-vllm-issue237-toolcall-on-loop.py --status   # classify the served serving.py
  python3 hotfix-vllm-issue237-toolcall-on-loop.py --check    # preflight: classify the bytes the next boot will copy
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
MARK = "[dspark-issue237-toolcall-on-loop]"

HELPER_ANCHOR = "import asyncio\n"
HELPER_NEW = (
    "import asyncio\n"
    "# "
    + MARK
    + "\n"
    "def _issue237_toolcall_on_loop_enabled() -> bool:\n"
    "    import os as _os\n"
    "    return _os.environ.get(\"DSPARK_ENABLE_ISSUE237_TOOLCALL_ON_LOOP\", \"0\") == \"1\"\n"
    "\n"
    "def _issue237_toolcall_on_loop_log_enabled() -> bool:\n"
    "    import os as _os\n"
    "    return _os.environ.get(\"DSPARK_ISSUE237_TOOLCALL_ON_LOOP_LOG\", \"0\") == \"1\"\n"
    "\n"
    "def _issue237_toolcall_on_loop_log(msg: str) -> None:\n"
    "    \"\"\"Emit a single toolcall-on-loop log line when logging is enabled.\"\"\"\n"
    "    if _issue237_toolcall_on_loop_log_enabled():\n"
    "        import sys as _sys\n"
    "        print(f\"[dspark-issue237-toolcall-on-loop] {msg}\", file=_sys.stderr, flush=True)\n"
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
    "    new = text[st[\"pos\"]:]\n"
    "    if not new:\n"
    "        return st[\"hits\"]\n"
    "    parts = split(new)\n"
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
    "    \"\"\"Sliding-window paragraph loop detector (stateful per request).\"\"\"\n"
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
    "    \"\"\"Token-level loop detector: repeated lines (stateful per request).\"\"\"\n"
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
    "def _issue237_extract_intent(text: str) -> str | None:\n"
    "    \"\"\"Extract the most recent narrated action from a 'Let me [action]' loop.\n"
    "\n"
    "    The model narrates an action it intends to take but never emits the\n"
    "    tool-call DSML structure. This pulls the action phrase out of the\n"
    "    narration so it can be matched against the declared tools. Returns the\n"
    "    last (most recent) match, or None if no narration is present.\n"
    "    \"\"\"\n"
    "    import re as _re\n"
    "    patterns = (\n"
    "        r\"\\bLet me\\s+(.+?)(?:[.!?]|\\n|$)\",\n"
    "        r\"\\bI need to\\s+(.+?)(?:[.!?]|\\n|$)\",\n"
    "        r\"\\bI'?ll\\s+(.+?)(?:[.!?]|\\n|$)\",\n"
    "        r\"\\bI should\\s+(.+?)(?:[.!?]|\\n|$)\",\n"
    "        r\"\\bLet'?s\\s+(.+?)(?:[.!?]|\\n|$)\",\n"
    "    )\n"
    "    best = None\n"
    "    for pat in patterns:\n"
    "        for m in _re.finditer(pat, text, _re.IGNORECASE):\n"
    "            best = m.group(1).strip()\n"
    "    return best or None\n"
    "\n"
    "def _issue237_tool_tokens(tool: object) -> tuple[str, set]:\n"
    "    \"\"\"Return (tool_name, token_set) for a declared tool.\n"
    "\n"
    "    Handles both ``FunctionTool`` (``.name``/``.description``) and\n"
    "    ``ChatCompletionToolsParam`` (``.function.name``/``.function.description``)\n"
    "    shapes. Tokens are lowercased and split on non-alphanumerics and\n"
    "    camelCase boundaries so ``get_api_key`` and ``getApiKey`` both yield\n"
    "    ``{\"get\", \"api\", \"key\"}``.\n"
    "    \"\"\"\n"
    "    import re as _re\n"
    "    fn = getattr(tool, \"function\", None)\n"
    "    if fn is not None:\n"
    "        name = getattr(fn, \"name\", None)\n"
    "        desc = getattr(fn, \"description\", None)\n"
    "    else:\n"
    "        name = getattr(tool, \"name\", None)\n"
    "        desc = getattr(tool, \"description\", None)\n"
    "    if not name:\n"
    "        return \"\", set()\n"
    "    def _tokens(s):\n"
    "        s = _re.sub(r\"([a-z0-9])([A-Z])\", r\"\\1 \\2\", s or \"\")\n"
    "        return set(w.lower() for w in _re.findall(r\"[a-zA-Z0-9]+\", s))\n"
    "    return name, _tokens(name) | _tokens(desc)\n"
    "\n"
    "def _issue237_match_tool(intent: str, tools: object) -> str | None:\n"
    "    \"\"\"Match a narrated action against the declared tools.\n"
    "\n"
    "    Scores each tool by token overlap between the narrated action and the\n"
    "    tool's name (weighted double) plus its description. Returns the best\n"
    "    tool name when the score clears the confidence floor, else None.\n"
    "    \"\"\"\n"
    "    import re as _re\n"
    "    if not intent or not tools:\n"
    "        return None\n"
    "    def _tokens(s):\n"
    "        s = _re.sub(r\"([a-z0-9])([A-Z])\", r\"\\1 \\2\", s or \"\")\n"
    "        return set(w.lower() for w in _re.findall(r\"[a-zA-Z0-9]+\", s))\n"
    "    intent_toks = _tokens(intent)\n"
    "    if not intent_toks:\n"
    "        return None\n"
    "    best_name = None\n"
    "    best_score = 0.0\n"
    "    for tool in tools:\n"
    "        name, tool_toks = _issue237_tool_tokens(tool)\n"
    "        if not name:\n"
    "            continue\n"
    "        name_toks = _tokens(name)\n"
    "        name_overlap = len(intent_toks & name_toks)\n"
    "        desc_overlap = len(intent_toks & tool_toks) - name_overlap\n"
    "        score = name_overlap * 2.0 + desc_overlap\n"
    "        if score > best_score:\n"
    "            best_score = score\n"
    "            best_name = name\n"
    "    if best_score < 2.0:\n"
    "        return None\n"
    "    return best_name\n"
    "\n"
    "def _issue237_synthesize_tool_call(name: str) -> object:\n"
    "    \"\"\"Build a DeltaMessage carrying a single synthesized tool call.\n"
    "\n"
    "    The arguments are an empty object: the model never emitted arguments,\n"
    "    so the client dispatches the tool with no arguments and the agent\n"
    "    framework fills in whatever it needs. The tool id is minted the same\n"
    "    way the parser mints ids for real calls.\n"
    "    \"\"\"\n"
    "    from vllm.entrypoints.chat_utils import make_tool_call_id\n"
    "    from vllm.entrypoints.openai.engine.protocol import (\n"
    "        DeltaFunctionCall,\n"
    "        DeltaMessage,\n"
    "        DeltaToolCall,\n"
    "    )\n"
    "    return DeltaMessage(\n"
    "        tool_calls=[\n"
    "            DeltaToolCall(\n"
    "                index=0,\n"
    "                id=make_tool_call_id(),\n"
    "                type=\"function\",\n"
    "                function=DeltaFunctionCall(name=name, arguments=\"{}\"),\n"
    "            )\n"
    "        ]\n"
    "    )\n"
    "\n"
    "_issue237_paragraph_state = {}\n"
    "_issue237_line_state = {}\n"
    "_issue237_toolcall_state = {}\n"
)

REGION_OLD = (
    "                    previous_texts[i] += delta_text\n"
)
REGION_NEW = (
    "                    previous_texts[i] += delta_text\n"
    "                    # [dspark-issue237-toolcall-on-loop] Complete the\n"
    "                    # skipped step: on a narration loop, match the narrated\n"
    "                    # intent against the declared tools and synthesize a\n"
    "                    # real tool call instead of merely stopping. Falls back\n"
    "                    # to the circuit-breaker abort+stop when no tool\n"
    "                    # matches. Fail-open on error.\n"
    "                    if _issue237_toolcall_on_loop_enabled():\n"
    "                        try:\n"
    "                            _txt = previous_texts[i]\n"
    "                            _para = _issue237_paragraph_loop_score(_txt, request_id)\n"
    "                            _pthr = _issue237_paragraph_loop_threshold()\n"
    "                            _line = _issue237_line_loop_score(_txt, request_id)\n"
    "                            _lthr = _issue237_line_loop_threshold()\n"
    "                            if _para >= _pthr or _line >= _lthr:\n"
    "                                if not _issue237_toolcall_state[request_id].get(\"tripped\"):\n"
    "                                    _issue237_toolcall_state[request_id][\"tripped\"] = True\n"
    "                                    _intent = _issue237_extract_intent(_txt)\n"
    "                                    _name = _issue237_match_tool(_intent, request.tools)\n"
    "                                    if _name:\n"
    "                                        _issue237_toolcall_on_loop_log(\n"
    "                                            f\"synthesize tool={_name} \"\n"
    "                                            f\"paragraphs={_para}/{_pthr} \"\n"
    "                                            f\"lines={_line}/{_lthr}\"\n"
    "                                        )\n"
    "                                        delta_message = _issue237_synthesize_tool_call(_name)\n"
    "                                        tools_streamed[i] = True\n"
    "                                    else:\n"
    "                                        _issue237_toolcall_on_loop_log(\n"
    "                                            f\"no-match paragraphs={_para}/{_pthr} \"\n"
    "                                            f\"lines={_line}/{_lthr}\"\n"
    "                                        )\n"
    "                                    # Abort the request at the engine so\n"
    "                                    # generation stops immediately instead\n"
    "                                    # of continuing to burn tokens. The\n"
    "                                    # finish_reason below is what the client\n"
    "                                    # sees; the abort is what actually halts\n"
    "                                    # the engine.\n"
    "                                    await self.engine_client.abort(request_id)\n"
    "                                output.finish_reason = \"stop\"\n"
    "                        except Exception:\n"
    "                            pass\n"
)

# Self-pins: the region constant must not drift inside this file.
REGION_OLD_SHA256 = "94de98ec8d8311897767cd54de6b1119a5944eae3b9c0328639e66ae1630913d"
REGION_NEW_SHA256 = "9f8751279a38c345199384f800203c61dd1eb9e9a6dbc5090bff8ba60cdb008b"


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


def _issue237_verify_matcher(
    extract_intent, match_tool, synthesize
) -> tuple[bool, str]:
    """Behavioral proof that the matcher maps a narrated action to a declared
    tool and stays quiet when no tool matches. Returns (ok, why)."""
    class _Fn:
        def __init__(self, name, description):
            self.name = name
            self.description = description

    class _Tool:
        def __init__(self, name, description):
            self.function = _Fn(name, description)

    tools = [
        _Tool("get_api_key", "Retrieve the API key from the running container."),
        _Tool("read_file", "Read the contents of a file on the local machine."),
        _Tool("run_command", "Execute a shell command in the container."),
    ]

    # A narrated "check the API key" must resolve to the get_api_key tool.
    intent = extract_intent("Let me check the API key.")
    if intent != "check the API key":
        return False, f"intent extraction failed: {intent!r}"
    name = match_tool(intent, tools)
    if name != "get_api_key":
        return False, f"did not match get_api_key, got {name!r}"

    # A narrated "read the file" must resolve to read_file.
    name = match_tool(extract_intent("Let me read the file."), tools)
    if name != "read_file":
        return False, f"did not match read_file, got {name!r}"

    # A narration with no tool overlap must not match anything.
    name = match_tool(extract_intent("Let me bake a sourdough loaf."), tools)
    if name is not None:
        return False, f"matched an unrelated tool: {name!r}"

    # No narration at all must not match.
    if extract_intent("The API key rotation failed and the loop continued.") is not None:
        return False, "extract_intent false-positived on non-narration text"

    # The synthesizer must produce a DeltaMessage-shaped object with one call.
    msg = synthesize("get_api_key")
    calls = getattr(msg, "tool_calls", None)
    if not calls or len(calls) != 1:
        return False, "synthesize did not produce one tool call"
    if getattr(calls[0].function, "name", None) != "get_api_key":
        return False, "synthesized call has the wrong name"

    return True, "matcher maps narration to declared tools and stays quiet otherwise"


def _self_check(patched_src: bytes) -> tuple[bool, str]:
    """Behavioral proof: the helpers are present, the module compiles, and the
    matcher maps a narrated action to a declared tool while staying quiet when
    no tool matches."""
    import importlib.util

    stock_src = patched_src.replace(REGION_NEW.encode(), REGION_OLD.encode(), 1)
    stock_src = stock_src.replace(HELPER_NEW.encode(), HELPER_ANCHOR.encode(), 1)
    with tempfile.TemporaryDirectory(prefix="dspark-issue237-tcol-") as tmpdir:
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

            patched = _load(patched_src, "serving_patched_tcol")
            stock = _load(stock_src, "serving_stock_tcol")

            for attr in (
                "_issue237_toolcall_on_loop_enabled",
                "_issue237_paragraph_loop_score",
                "_issue237_line_loop_score",
                "_issue237_extract_intent",
                "_issue237_match_tool",
                "_issue237_synthesize_tool_call",
                "_issue237_toolcall_on_loop_log",
            ):
                if not hasattr(patched, attr):
                    return False, f"helper {attr} missing in patched module"
            if hasattr(stock, "_issue237_toolcall_on_loop_enabled"):
                return False, "helper unexpectedly present in stock module"

            ok, why = _issue237_verify_matcher(
                patched._issue237_extract_intent,
                patched._issue237_match_tool,
                patched._issue237_synthesize_tool_call,
            )
            if not ok:
                return False, why
        except HotfixError:
            raise
        except Exception as err:  # broken/unimportable patch must fail closed
            return False, f"self-check raised {type(err).__name__}: {err}"
    return True, "toolcall-on-loop helpers verified; matcher maps narration to declared tools"


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
    fd, tmp_name = tempfile.mkstemp(prefix=".dspark-issue237-tcol-", dir=str(target.parent))
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
            print(f"dspark-issue237-toolcall-on-loop: {state} ({target})")
            return 0
        outcome = apply(target)
        print(f"dspark-issue237-toolcall-on-loop: {outcome} ({target})")
        return 0
    except HotfixError as error:
        print(f"dspark-issue237-toolcall-on-loop: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
