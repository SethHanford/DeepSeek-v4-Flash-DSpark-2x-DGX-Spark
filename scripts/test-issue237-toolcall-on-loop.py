#!/usr/bin/env python3
"""CPU tests for patches/hotfix-vllm-issue237-toolcall-on-loop.py (no vLLM, no GPU, no network).

Covers the pinned fixture identity, the pure transformation, the helper-block
behavioural contract (narration-intent extraction, tool matching, tool-call
synthesis, loop detection), the inspect/refusal behaviour, and the
compose/launcher/env/ci wiring locks.

The patcher's ``_self_check`` imports the full ``serving.py`` module (torch,
fastapi, pybase64, many vllm submodules), which is not available in the CPU
test environment.  Following the issue237 circuit-breaker test's pattern, the
behavioural contract is exercised on the extracted helper block instead.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXDIR = ROOT / "scripts" / "fixtures" / "issue191"
FIX_POST55 = FIXDIR / "chat_completion_serving-752a3a504-post-issue55.py"
FIX_PRISTINE = FIXDIR / "chat_completion_serving-752a3a504-pristine.py"
PATCHER = ROOT / "patches" / "hotfix-vllm-issue237-toolcall-on-loop.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"

POST55_STOCK_SHA256 = "4024243c259058c09f4d9340d42f410e68e160b96fcd5c09b1a68467b04f4f3a"
POST55_STOCK_SIZE = 51912
POST55_PATCHED_SHA256 = "84927c5a78d962811ecace19590755033686ba71f01a8b0f8737b5e8bd17e517"
POST55_PATCHED_SIZE = 63375
PRISTINE_STOCK_SHA256 = "6239fae503211193942d0e2037f7b0edcf71ed70271604701283bfc0453d202b"
PRISTINE_STOCK_SIZE = 49931
PRISTINE_PATCHED_SHA256 = "6173f25d55e4492376ba542a67a3fbfe734934061278c2c0b1a41c71a82fcb5b"
PRISTINE_PATCHED_SIZE = 61394


def _load_patcher():
    spec = importlib.util.spec_from_file_location("hotfix_issue237_tcol", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HF = _load_patcher()


class _ProtoBase:
    """Keyword-init stub for the vLLM pydantic protocol models."""

    _defaults: dict = {}

    def __init__(self, **kwargs):
        for key, value in self._defaults.items():
            setattr(self, key, list(value) if isinstance(value, list) else value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        pairs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({pairs})"


class _DeltaFunctionCall(_ProtoBase):
    _defaults = {"name": None, "arguments": None}


class _DeltaToolCall(_ProtoBase):
    _defaults = {"index": 0, "id": None, "type": None, "function": None}


class _DeltaMessage(_ProtoBase):
    _defaults = {"role": None, "content": None, "reasoning": None, "tool_calls": []}


def _install_vllm_fakes():
    """Register minimal fake vllm modules so the synthesizer can build a
    DeltaMessage without the real vllm package (a system boundary)."""
    protocol = types.ModuleType("vllm.entrypoints.openai.engine.protocol")
    protocol.DeltaMessage = _DeltaMessage
    protocol.DeltaToolCall = _DeltaToolCall
    protocol.DeltaFunctionCall = _DeltaFunctionCall
    chat_utils = types.ModuleType("vllm.entrypoints.chat_utils")
    chat_utils.make_tool_call_id = lambda: "call_fake"
    # Register parents so the dotted imports resolve.
    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.openai",
                 "vllm.entrypoints.openai.engine"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.entrypoints.openai.engine.protocol"] = protocol
    sys.modules["vllm.entrypoints.chat_utils"] = chat_utils


def _helpers():
    """Execute the helper block exactly as installed and return its namespace."""
    _install_vllm_fakes()
    block = HF.HELPER_NEW
    start = block.index("# " + HF.MARK)
    namespace: dict = {}
    exec(compile(block[start:], "issue237-tcol-helpers", "exec"), namespace)
    return namespace


class _Fn:
    def __init__(self, name, description):
        self.name = name
        self.description = description


class _Tool:
    def __init__(self, name, description):
        self.function = _Fn(name, description)


DECLARED_TOOLS = [
    _Tool("get_api_key", "Retrieve the API key from the running container."),
    _Tool("read_file", "Read the contents of a file on the local machine."),
    _Tool("run_command", "Execute a shell command in the container."),
]


class FixtureAndTransform(unittest.TestCase):
    def test_fixtures_match_identity_pins(self):
        post55 = FIX_POST55.read_bytes()
        pristine = FIX_PRISTINE.read_bytes()
        self.assertEqual(hashlib.sha256(post55).hexdigest(), POST55_STOCK_SHA256)
        self.assertEqual(len(post55), POST55_STOCK_SIZE)
        self.assertEqual(hashlib.sha256(pristine).hexdigest(), PRISTINE_STOCK_SHA256)
        self.assertEqual(len(pristine), PRISTINE_STOCK_SIZE)

    def test_region_constants_match_self_pins(self):
        HF._verify_self()
        self.assertEqual(
            hashlib.sha256(HF.REGION_OLD.encode()).hexdigest(), HF.REGION_OLD_SHA256
        )
        self.assertEqual(
            hashlib.sha256(HF.REGION_NEW.encode()).hexdigest(), HF.REGION_NEW_SHA256
        )

    def test_trip_block_synthesizes_tool_call(self):
        # The trip must synthesize a tool call (setting tools_streamed) and
        # abort at the engine, guarded by the one-shot "tripped" flag.
        region = HF.REGION_NEW
        self.assertIn("_issue237_synthesize_tool_call(_name)", region)
        self.assertIn("tools_streamed[i] = True", region)
        self.assertIn("await self.engine_client.abort(request_id)", region)
        self.assertEqual(region.count("await self.engine_client.abort(request_id)"), 1)
        # The abort must sit inside the one-shot trip guard so it fires once.
        self.assertLess(
            region.index("await self.engine_client.abort(request_id)"),
            region.index('output.finish_reason = "stop"'),
        )

    def test_trip_block_state_access_is_safe(self):
        # The one-shot "tripped" flag must be read/written via setdefault so a
        # first-time request_id does not raise KeyError (which the fail-open
        # except would swallow, silently disabling the intervention).
        region = HF.REGION_NEW
        self.assertIn(
            '_issue237_toolcall_state.setdefault(request_id, {}).get("tripped")',
            region,
        )
        self.assertIn(
            '_issue237_toolcall_state.setdefault(request_id, {})["tripped"] = True',
            region,
        )

    def test_transform_is_pinned_single_site_and_compiles(self):
        for fixture, patched_sha, patched_size in (
            (FIX_POST55, POST55_PATCHED_SHA256, POST55_PATCHED_SIZE),
            (FIX_PRISTINE, PRISTINE_PATCHED_SHA256, PRISTINE_PATCHED_SIZE),
        ):
            stock = fixture.read_bytes()
            self.assertEqual(stock.count(HF.REGION_OLD.encode()), 1)
            patched = HF.transform(stock)
            self.assertEqual(hashlib.sha256(patched).hexdigest(), patched_sha)
            self.assertEqual(len(patched), patched_size)
            self.assertGreaterEqual(patched.count(HF.MARK.encode()), 1)
            compile(patched, "serving.py", "exec")

    def test_transform_refuses_foreign_or_patched_bytes(self):
        with self.assertRaises(HF.HotfixError):
            HF.transform(b"def nothing():\n    pass\n")
        patched = HF.transform(FIX_POST55.read_bytes())
        with self.assertRaises(HF.HotfixError):
            HF.transform(patched)

    def test_inspect_classifies_stock_and_patched(self):
        stock = FIX_POST55.read_bytes()
        self.assertEqual(HF.inspect_bytes(stock), "stock")
        patched = HF.transform(stock)
        self.assertEqual(HF.inspect_bytes(patched), "patched")


class ToolMatchContract(unittest.TestCase):
    """Seam: _issue237_match_tool(loop_text, tools) -> str | None.

    The matcher is language-agnostic: it matches the loop text against the
    declared tool names (stable identifiers) and descriptions by token overlap,
    without requiring an English narration prefix. The tool name is weighted
    highest because it is a language-independent identifier the model tends to
    echo even when narrating in another language.
    """

    @classmethod
    def setUpClass(cls):
        cls.ns = _helpers()

    def test_matches_declared_tool_by_action(self):
        match = self.ns["_issue237_match_tool"]
        self.assertEqual(match("check the API key", DECLARED_TOOLS), "get_api_key")

    def test_matches_read_file(self):
        match = self.ns["_issue237_match_tool"]
        self.assertEqual(match("read the file", DECLARED_TOOLS), "read_file")

    def test_matches_container_phrase(self):
        match = self.ns["_issue237_match_tool"]
        self.assertEqual(
            match("get the API key from the container", DECLARED_TOOLS), "get_api_key"
        )

    def test_matches_tool_identifier_without_english_narration(self):
        # A multilingual model may narrate in another language but still echo
        # the tool identifier. No English "Let me" prefix is required.
        match = self.ns["_issue237_match_tool"]
        self.assertEqual(match("我需要调用 read_file 来读取文件", DECLARED_TOOLS), "read_file")

    def test_matches_tool_identifier_embedded_in_loop(self):
        # The tool name is a stable identifier; it should match even when the
        # surrounding loop text is long and in a non-English language.
        match = self.ns["_issue237_match_tool"]
        loop = "我需要获取 API key。\n" * 12 + "让我调用 get_api_key 来检查。\n"
        self.assertEqual(match(loop, DECLARED_TOOLS), "get_api_key")

    def test_no_match_on_unrelated_action(self):
        match = self.ns["_issue237_match_tool"]
        self.assertIsNone(match("bake a sourdough loaf", DECLARED_TOOLS))

    def test_no_match_on_empty_text_or_tools(self):
        match = self.ns["_issue237_match_tool"]
        self.assertIsNone(match("", DECLARED_TOOLS))
        self.assertIsNone(match("check the API key", None))

    def test_floor_is_env_tunable(self):
        # "run tests" overlaps run_command's name by exactly one token ("run"),
        # scoring 2.0 (name weighted double) with no description overlap. It is
        # accepted at the default floor but rejected when the floor is raised.
        match = self.ns["_issue237_match_tool"]
        original_floor = self.ns["_issue237_match_floor"]
        self.assertEqual(match("run tests", DECLARED_TOOLS), "run_command")
        self.ns["_issue237_match_floor"] = lambda: 3.0
        try:
            self.assertIsNone(match("run tests", DECLARED_TOOLS))
        finally:
            self.ns["_issue237_match_floor"] = original_floor

    def test_delta_text_combines_with_recent_window(self):
        # The matcher scores both the recent window (loop_text) and the
        # delta_text, taking the max score per tool. A narration loop with no
        # tool intent in either signal must circuit-break.
        match = self.ns["_issue237_match_tool"]
        self.assertIsNone(
            match(
                "Let me look at the matcher call site.",
                DECLARED_TOOLS,
                "Let me look at the matcher call site.",
            )
        )

    def test_delta_text_provides_tool_intent(self):
        # When the recent window is a weak narration loop but the delta_text
        # carries real tool intent, the matcher must resolve to that tool.
        match = self.ns["_issue237_match_tool"]
        self.assertEqual(
            match(
                "Let me look at the matcher call site.",
                DECLARED_TOOLS,
                "Let me read the file.",
            ),
            "read_file",
        )

    def test_recent_window_provides_tool_intent(self):
        # When the delta_text is weak but the recent window carries real tool
        # intent, the matcher must resolve to that tool.
        match = self.ns["_issue237_match_tool"]
        self.assertEqual(
            match(
                "Let me read the file.",
                DECLARED_TOOLS,
                "Let me look at the matcher call site.",
            ),
            "read_file",
        )


class SynthesisContract(unittest.TestCase):
    """Seam: _issue237_synthesize_tool_call(name) -> DeltaMessage-shaped object."""

    @classmethod
    def setUpClass(cls):
        cls.ns = _helpers()

    def test_produces_one_tool_call_with_name(self):
        msg = self.ns["_issue237_synthesize_tool_call"]("get_api_key")
        calls = getattr(msg, "tool_calls", None)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "get_api_key")
        self.assertEqual(calls[0].function.arguments, "{}")
        self.assertEqual(calls[0].type, "function")
        self.assertIsNotNone(calls[0].id)


class LoopDetectorContract(unittest.TestCase):
    """The loop detectors (reused from the circuit breaker) must trip on a loop
    and stay quiet on distinct prose."""

    @classmethod
    def setUpClass(cls):
        cls.ns = _helpers()

    def test_paragraph_loop_detects_repeated_paragraphs(self):
        score = self.ns["_issue237_paragraph_loop_score"]
        text = ""
        for _ in range(12):
            text += "The API key rotation failed and the token loop continued indefinitely.\n\n"
            score(text, "req-repeat")
        self.assertGreaterEqual(score(text, "req-repeat"), 10)

    def test_paragraph_loop_ignores_distinct_paragraphs(self):
        score = self.ns["_issue237_paragraph_loop_score"]
        distinct = [
            "Quantum computing leverages superposition and entanglement for computation.",
            "The weather forecast predicts rain showers across the coastal region tomorrow.",
            "Baking sourdough bread requires patience, hydration, and a warm kitchen.",
            "Mountain biking trails wind through pine forests and rocky switchbacks.",
        ]
        text = ""
        for p in distinct:
            text += p + "\n\n"
            score(text, "req-distinct")
        self.assertEqual(score(text, "req-distinct"), 0)

    def test_line_loop_detects_repeated_lines(self):
        score = self.ns["_issue237_line_loop_score"]
        text = ""
        for _ in range(12):
            text += '    _stub("vllm.entrypoints.openai.tool_parsers.tool_parsers_utils")\n'
            score(text, "req-line-repeat")
        self.assertGreaterEqual(score(text, "req-line-repeat"), 10)

    def test_line_loop_ignores_distinct_lines(self):
        score = self.ns["_issue237_line_loop_score"]
        text = ""
        for i in range(5):
            text += f"This is distinct line number {i} about a different topic entirely.\n"
            score(text, "req-line-distinct")
        self.assertEqual(score(text, "req-line-distinct"), 0)


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="i237-tcol-"))
        self.target = self.tmp / "serving.py"
        shutil.copyfile(FIX_POST55, self.target)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inspect_and_transform_on_temp_file(self):
        self.assertEqual(HF.inspect(self.target)[0], "stock")
        patched = HF.transform(self.target.read_bytes())
        self.assertEqual(hashlib.sha256(patched).hexdigest(), POST55_PATCHED_SHA256)

    def test_refuses_foreign_bytes(self):
        self.target.write_bytes(b"x = 1\n")
        with self.assertRaises(HF.HotfixError):
            HF.inspect(self.target)

    def test_refuses_symlink(self):
        link = self.tmp / "link.py"
        os.symlink(self.target, link)
        with self.assertRaises(HF.HotfixError):
            HF.inspect(link)

    def test_cli_check_and_status_report_stock(self):
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--check", "--target", str(self.target)],
            capture_output=True, text=True,
        )
        # The self-check needs the full module; if unavailable it fails closed.
        self.assertIn("FAIL-CLOSED", proc.stdout + proc.stderr)


class Wiring(unittest.TestCase):
    def test_compose_gate_default_off_fail_closed(self):
        compose = COMPOSE.read_text()
        self.assertIn(
            'DSPARK_ENABLE_ISSUE237_TOOLCALL_ON_LOOP: "${DSPARK_ENABLE_ISSUE237_TOOLCALL_ON_LOOP:-0}"',
            compose,
        )
        self.assertIn(
            'DSPARK_ISSUE237_TOOLCALL_ON_LOOP_LOG: "${DSPARK_ISSUE237_TOOLCALL_ON_LOOP_LOG:-0}"',
            compose,
        )
        self.assertIn(
            'DSPARK_ISSUE237_TOOLCALL_MATCH_FLOOR: "${DSPARK_ISSUE237_TOOLCALL_MATCH_FLOOR:-2.0}"',
            compose,
        )
        self.assertIn(
            'DSPARK_ISSUE237_TOOLCALL_RECENT_FRACTION: "${DSPARK_ISSUE237_TOOLCALL_RECENT_FRACTION:-0.25}"',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_ISSUE237_TOOLCALL_ON_LOOP:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-issue237-toolcall-on-loop.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_ISSUE237_TOOLCALL_ON_LOOP_HOTFIX:-./patches/hotfix-vllm-issue237-toolcall-on-loop.py}"
            ":/opt/hotfix-vllm-issue237-toolcall-on-loop.py:ro",
            compose,
        )

    def test_launcher_passthrough_and_preflight(self):
        start = START.read_text()
        self.assertIn(
            "DSPARK_ISSUE237_TOOLCALL_ON_LOOP_HOTFIX=\"${DSPARK_ISSUE237_TOOLCALL_ON_LOOP_HOTFIX:-$SCRIPT_DIR/patches/hotfix-vllm-issue237-toolcall-on-loop.py}\"",
            start,
        )
        self.assertIn("DSPARK_ENABLE_ISSUE237_TOOLCALL_ON_LOOP=$REMOTE_ISSUE237_TOOLCALL_ON_LOOP", start)
        self.assertIn(
            "DSPARK_ISSUE237_TOOLCALL_RECENT_FRACTION=$REMOTE_ISSUE237_TOOLCALL_RECENT_FRACTION", start
        )
        self.assertIn(
            "python3 vllm-dspark /opt/hotfix-vllm-issue237-toolcall-on-loop.py --check",
            start,
        )

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_ISSUE237_TOOLCALL_ON_LOOP=0", env)
        self.assertIn("DSPARK_ISSUE237_TOOLCALL_MATCH_FLOOR=2.0", env)
        self.assertIn("DSPARK_ISSUE237_TOOLCALL_RECENT_FRACTION=0.25", env)
        self.assertIn("scripts/test-issue237-toolcall-on-loop.py", CI.read_text())


class SelfCheckVerifyContract(unittest.TestCase):
    """Sabotage-rejection contract of the self-check's matcher verification."""

    @classmethod
    def setUpClass(cls):
        cls.ns = _helpers()

    def test_accepts_working_matcher(self):
        ok, why = HF._issue237_verify_matcher(
            self.ns["_issue237_match_tool"],
            self.ns["_issue237_synthesize_tool_call"],
        )
        self.assertTrue(ok, why)

    def test_rejects_matcher_that_never_matches(self):
        def no_match(*args, **kwargs):
            return None

        ok, why = HF._issue237_verify_matcher(
            no_match,
            self.ns["_issue237_synthesize_tool_call"],
        )
        self.assertFalse(ok)
        self.assertTrue(why)

    def test_rejects_matcher_that_always_matches(self):
        def always_match(*args, **kwargs):
            return "get_api_key"

        ok, why = HF._issue237_verify_matcher(
            always_match,
            self.ns["_issue237_synthesize_tool_call"],
        )
        self.assertFalse(ok)
        self.assertTrue(why)


if __name__ == "__main__":
    unittest.main()
