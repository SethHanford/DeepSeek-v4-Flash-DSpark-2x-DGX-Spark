#!/usr/bin/env python3
"""Issue #237: add a tool-call directive to the DeepSeek-V4 tools template.

Symptom
-------
The model enters a token-repetition loop in ``thinking_mode: "chat"`` with
large conversations. The model narrates an action it intends to take ("Let me
check the API key") but never emits the tool-call DSML structure, so it loops
on the narration. The tools are rendered into the prompt, but the model
narrates instead of calling them.

Fix
---
Append a directive to the ``TOOLS_TEMPLATE`` in
``vllm/tokenizers/deepseek_v4_encoding.py`` that steers the model to act
before speaking: emit a tool call the instant it intends to use one, and if no
available tool can do what it needs, say so once and stop. This discourages the
narration loop and gives the model a concrete fallback when a tool is missing.

The directive is phrased positively (no prohibitions) and uses a leading word
("Act, then speak") to anchor the target behavior.

Gating and fail-closed operation
--------------------------------
The compose entrypoint invokes this script only when
``DSPARK_ENABLE_ISSUE237_TOOLCALL_DIRECTIVE`` is exactly ``1`` (default ``0`` =
stock template, this script never runs) and chains it with ``|| exit 1``. The
anchored region (the closing lines of ``TOOLS_TEMPLATE``) must appear exactly
once; the region constants are sha256-pinned. After writing, a self-check must
pass or the original bytes are restored and the boot fails.

Usage (inside container, after vLLM source is on disk):
  python3 hotfix-vllm-issue237-toolcall-directive.py            # apply (or verify if already applied)
  python3 hotfix-vllm-issue237-toolcall-directive.py --status   # classify the served encoding module
  python3 hotfix-vllm-issue237-toolcall-directive.py --check    # preflight: classify the bytes the next boot will copy
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
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py"
)
MARK = "[dspark-issue237-toolcall-directive]"

REGION_OLD = (
    "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
    "\"\"\"\n"
)
REGION_NEW = (
    "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
    "\n"
    "# [dspark-issue237-toolcall-directive] Act, then speak. Call a tool the\n"
    "# instant you intend to use one. If no tool can do what you need, say it\n"
    "# once and stop: \"I need to [action] but no available tool can do this.\"\n"
    "\"\"\"\n"
)

# Self-pins: the region constants must not drift inside this file.
REGION_OLD_SHA256 = "815965a1ba7b35f4b9bc477f66de482deb9a38395015eb106a786d1670e13260"
REGION_NEW_SHA256 = "1c8740577c4211fff9ff3a756b613ee762bb4cb8b36164c47b1c255bc5f8625b"


class HotfixError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_self() -> None:
    if _sha256(REGION_OLD.encode()) != REGION_OLD_SHA256:
        raise HotfixError("REGION_OLD does not match its pinned sha256")
    if _sha256(REGION_NEW.encode()) != REGION_NEW_SHA256:
        raise HotfixError("REGION_NEW does not match its pinned sha256")


def inspect_bytes(data: bytes) -> str:
    """Classify encoding module bytes; anything but exactly-one anchor fails closed."""
    _verify_self()
    old_n = data.count(REGION_OLD.encode())
    new_n = data.count(REGION_NEW.encode())
    marked = MARK.encode() in data
    if new_n == 1 and old_n == 0 and marked:
        return "patched"
    if old_n == 1 and new_n == 0 and not marked:
        return "stock"
    raise HotfixError(
        f"unsupported encoding bytes (region_old x{old_n}, region_new x{new_n}, "
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
    patched = stock.replace(REGION_OLD.encode(), REGION_NEW.encode(), 1)
    compile(patched, "deepseek_v4_encoding.py", "exec")
    return patched


def _self_check(patched_src: bytes) -> tuple[bool, str]:
    """Behavioral proof: the directive is present and the module still compiles."""
    import importlib.util

    stock_src = patched_src.replace(REGION_NEW.encode(), REGION_OLD.encode(), 1)
    with tempfile.TemporaryDirectory(prefix="dspark-issue237-directive-") as tmpdir:
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

            patched = _load(patched_src, "enc_patched_directive")
            stock = _load(stock_src, "enc_stock_directive")

            # The directive must be present in the patched template
            if "Act, then speak." not in patched.TOOLS_TEMPLATE:
                return False, "directive not present in patched TOOLS_TEMPLATE"
            if "Act, then speak." in stock.TOOLS_TEMPLATE:
                return False, "directive unexpectedly present in stock TOOLS_TEMPLATE"
            if MARK not in patched_src.decode():
                return False, "mark not present in patched source"
            # The stock template must be unchanged
            if "You MUST strictly follow" not in stock.TOOLS_TEMPLATE:
                return False, "stock TOOLS_TEMPLATE anchor missing"
        except HotfixError:
            raise
        except Exception as err:  # broken/unimportable patch must fail closed
            return False, f"self-check raised {type(err).__name__}: {err}"
    return True, "directive verified; stock template unchanged"


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
    fd, tmp_name = tempfile.mkstemp(prefix=".dspark-issue237-directive-", dir=str(target.parent))
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
                        help="classify the served encoding module copy")
    parser.add_argument("--target", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        target = args.target or PRODUCTION_TARGET
        if args.check or args.status:
            state, data = inspect(target)
            ok, why = _self_check(data if state == "patched" else transform(data))
            if not ok:
                raise HotfixError(f"self-check failed: {why}")
            print(f"dspark-issue237-toolcall-directive: {state} ({target})")
            return 0
        outcome = apply(target)
        print(f"dspark-issue237-toolcall-directive: {outcome} ({target})")
        return 0
    except HotfixError as error:
        print(f"dspark-issue237-toolcall-directive: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
