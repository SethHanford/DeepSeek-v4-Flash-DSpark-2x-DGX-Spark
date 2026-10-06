#!/usr/bin/env python3
"""Issue #237: enforce a minimum repetition penalty in the vLLM sampling layer.

Symptom
-------
The model enters a token-level repetition loop in ``thinking_mode: "chat"``
with large conversations (300+ messages). The output repeats a fixed token
sequence until ``max_tokens`` is reached. The engine's sampling defaults are
temperature=1.0, top_p=1.0, top_k=0, and repetition_penalty=1.0 (no penalty),
so nothing prevents the model from repeating tokens.

Fix
---
In ``vllm/v1/worker/gpu/sample/penalties.py``, ``PenaltiesState.add_request``
stores the client's ``repetition_penalty`` verbatim and ``use_penalty()``
returns ``False`` when it equals 1.0 (and no frequency/presence penalty is
set), so the penalty kernel is never launched. This hotfix clamps a client
that leaves the default (``repetition_penalty <= 1.0``) up to a minimum value
(``DSPARK_ISSUE237_REPETITION_PENALTY``, default 1.05) and marks the request as
needing penalties, so the repetition penalty is always applied.

The clamp only raises a no-op/absent penalty; a client that explicitly sets a
penalty > 1.0 is left untouched. A client that sets a penalty < 1.0 (a
"bonus" for repetition) is also clamped to the minimum, since that is the
permissive direction that permits the loop.

Gating and fail-closed operation
--------------------------------
The compose entrypoint invokes this script only when
``DSPARK_ENABLE_ISSUE237_REPETITION_PENALTY`` is exactly ``1`` (default ``0`` =
stock behavior, this script never runs) and chains it with ``|| exit 1``. The
anchored regions (the ``RequestState`` import line and the ``add_request``
method) are disjoint from other patchers and must each appear exactly once;
the region constants are themselves sha256-pinned. An unrecognized file with
intact anchors is patchable; a missing or duplicated anchor fails closed.
After writing, a self-check must pass or the original bytes are restored and
the boot fails.

Usage (inside container, after vLLM source is on disk):
  python3 hotfix-vllm-issue237-repetition-penalty.py            # apply (or verify if already applied)
  python3 hotfix-vllm-issue237-repetition-penalty.py --status   # classify the served penalties.py
  python3 hotfix-vllm-issue237-repetition-penalty.py --check    # preflight: classify the bytes the next boot will copy
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
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/penalties.py"
)
MARK = "[dspark-issue237-repetition-penalty]"

# --- Module constant (inserted after the RequestState import) ---

IMPORT_OLD = (
    "from vllm.v1.worker.gpu.states import RequestState\n"
    "\n"
    "\n"
    "class PenaltiesState:\n"
)
IMPORT_NEW = (
    "from vllm.v1.worker.gpu.states import RequestState\n"
    "\n"
    "# [dspark-issue237-repetition-penalty] Issue #237: minimum repetition penalty\n"
    "# enforced when a client leaves the default (1.0 = no penalty). Configurable\n"
    "# via DSPARK_ISSUE237_REPETITION_PENALTY (default 1.05).\n"
    "import os as _os\n"
    "_MIN_REPETITION_PENALTY = float(_os.environ.get(\"DSPARK_ISSUE237_REPETITION_PENALTY\", \"1.05\"))\n"
    "\n"
    "\n"
    "class PenaltiesState:\n"
)

# --- add_request method ---

REGION_OLD = (
    "    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:\n"
    "        self.repetition_penalty.np[req_idx] = sampling_params.repetition_penalty\n"
    "        self.frequency_penalty.np[req_idx] = sampling_params.frequency_penalty\n"
    "        self.presence_penalty.np[req_idx] = sampling_params.presence_penalty\n"
    "\n"
    "        do_penalty = use_penalty(sampling_params)\n"
    "        self.use_penalty[req_idx] = do_penalty\n"
    "        if do_penalty:\n"
    "            self._new_penalties_reqs.append(req_idx)\n"
)
REGION_NEW = (
    "    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:\n"
    "        # [dspark-issue237-repetition-penalty] Issue #237: enforce a minimum\n"
    "        # repetition penalty so a client that leaves the default (1.0 = no\n"
    "        # penalty) still gets a penalty. This prevents the token-repetition\n"
    "        # loop seen at large context sizes in chat mode.\n"
    "        _rep = sampling_params.repetition_penalty\n"
    "        if _rep <= 1.0:\n"
    "            _rep = _MIN_REPETITION_PENALTY\n"
    "        self.repetition_penalty.np[req_idx] = _rep\n"
    "        self.frequency_penalty.np[req_idx] = sampling_params.frequency_penalty\n"
    "        self.presence_penalty.np[req_idx] = sampling_params.presence_penalty\n"
    "\n"
    "        do_penalty = use_penalty(sampling_params) or _rep != 1.0\n"
    "        self.use_penalty[req_idx] = do_penalty\n"
    "        if do_penalty:\n"
    "            self._new_penalties_reqs.append(req_idx)\n"
)

# Self-pins: the region constants must not drift inside this file.
IMPORT_OLD_SHA256 = "9476c8abaaf835b1b6a445d03ec67868550ceefab8123b46339267e322f71ab0"
IMPORT_NEW_SHA256 = "7f602214db50a51426a0b846ee0b687542f84854c99c2dfae282f3b241dac119"
REGION_OLD_SHA256 = "c09dd07820c6c58d85cf8a938a203188080dae16172c1cdbb97cca9b12714702"
REGION_NEW_SHA256 = "397079380d5ad5ed95ae8fdac7abedd75219972c9dc073592f66798bb6508ace"


class HotfixError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_self() -> None:
    if _sha256(IMPORT_OLD.encode()) != IMPORT_OLD_SHA256:
        raise HotfixError("IMPORT_OLD does not match its pinned sha256")
    if _sha256(IMPORT_NEW.encode()) != IMPORT_NEW_SHA256:
        raise HotfixError("IMPORT_NEW does not match its pinned sha256")
    if _sha256(REGION_OLD.encode()) != REGION_OLD_SHA256:
        raise HotfixError("REGION_OLD does not match its pinned sha256")
    if _sha256(REGION_NEW.encode()) != REGION_NEW_SHA256:
        raise HotfixError("REGION_NEW does not match its pinned sha256")


def inspect_bytes(data: bytes) -> str:
    """Classify penalties.py bytes; anything but exactly-one anchor fails closed."""
    _verify_self()
    import_old_n = data.count(IMPORT_OLD.encode())
    import_new_n = data.count(IMPORT_NEW.encode())
    old_n = data.count(REGION_OLD.encode())
    new_n = data.count(REGION_NEW.encode())
    marked = MARK.encode() in data
    if import_new_n == 1 and new_n == 1 and import_old_n == 0 and old_n == 0 and marked:
        return "patched"
    if import_old_n == 1 and old_n == 1 and import_new_n == 0 and new_n == 0 and not marked:
        return "stock"
    raise HotfixError(
        f"unsupported penalties.py bytes (import_old x{import_old_n}, import_new x{import_new_n}, "
        f"region_old x{old_n}, region_new x{new_n}, mark={marked}, sha256={_sha256(data)}); "
        f"expected exactly one of either"
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
    patched = stock.replace(IMPORT_OLD.encode(), IMPORT_NEW.encode(), 1)
    patched = patched.replace(REGION_OLD.encode(), REGION_NEW.encode(), 1)
    compile(patched, "penalties.py", "exec")
    return patched


def _self_check(patched_src: bytes) -> tuple[bool, str]:
    """Behavioral proof: the clamp raises a no-op penalty and marks the request."""
    import importlib.util

    stock_src = patched_src.replace(IMPORT_NEW.encode(), IMPORT_OLD.encode(), 1)
    stock_src = stock_src.replace(REGION_NEW.encode(), REGION_OLD.encode(), 1)
    with tempfile.TemporaryDirectory(prefix="dspark-issue237-check-") as tmpdir:
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

            stock = _load(stock_src, "pen_stock_i237")
            patched = _load(patched_src, "pen_patched_i237")

            class _SP:
                def __init__(self, rep, freq=0.0, pres=0.0):
                    self.repetition_penalty = rep
                    self.frequency_penalty = freq
                    self.presence_penalty = pres

            # A client that leaves the default (1.0) must be clamped and marked.
            sp = _SP(1.0)
            if stock.use_penalty(sp):
                return False, "stock use_penalty(1.0) unexpectedly True"
            if not patched.use_penalty(_SP(patched._MIN_REPETITION_PENALTY)):
                return False, "patched use_penalty(min) unexpectedly False"

            # A client that sets a penalty > 1.0 must be left untouched.
            if patched.use_penalty(_SP(1.2)) is not True:
                return False, "patched use_penalty(1.2) unexpectedly not True"

            # The clamp value must be the configured minimum.
            if not (patched._MIN_REPETITION_PENALTY > 1.0):
                return False, "patched _MIN_REPETITION_PENALTY not > 1.0"
        except HotfixError:
            raise
        except Exception as err:  # broken/unimportable patch must fail closed
            return False, f"self-check raised {type(err).__name__}: {err}"
    return True, "clamp verified; default penalty raised, explicit penalty untouched"


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
    fd, tmp_name = tempfile.mkstemp(prefix=".dspark-issue237-", dir=str(target.parent))
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
                        help="classify the served penalties.py copy")
    parser.add_argument("--target", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        target = args.target or PRODUCTION_TARGET
        if args.check or args.status:
            state, data = inspect(target)
            ok, why = _self_check(data if state == "patched" else transform(data))
            if not ok:
                raise HotfixError(f"self-check failed: {why}")
            print(f"dspark-issue237-repetition-penalty: {state} ({target})")
            return 0
        outcome = apply(target)
        print(f"dspark-issue237-repetition-penalty: {outcome} ({target})")
        return 0
    except HotfixError as error:
        print(f"dspark-issue237-repetition-penalty: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
