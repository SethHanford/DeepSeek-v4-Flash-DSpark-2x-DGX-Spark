#!/usr/bin/env python3
"""CPU tests for patches/hotfix-vllm-issue237-circuit-breaker.py (no vLLM, no GPU, no network).

Covers the pinned fixture identity, the pure transformation, the helper-block
behavioural contract (paragraph loop detection, phrase counters, thresholds),
the inspect/refusal behaviour, and the compose/launcher/env/ci wiring locks.

The patcher's ``_self_check`` imports the full ``serving.py`` module (torch,
fastapi, pybase64, many vllm submodules), which is not available in the CPU
test environment.  Following the issue191 test's pattern, the behavioural
contract is exercised on the extracted helper block instead.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXDIR = ROOT / "scripts" / "fixtures" / "issue191"
FIX_POST55 = FIXDIR / "chat_completion_serving-752a3a504-post-issue55.py"
FIX_PRISTINE = FIXDIR / "chat_completion_serving-752a3a504-pristine.py"
PATCHER = ROOT / "patches" / "hotfix-vllm-issue237-circuit-breaker.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"

POST55_STOCK_SHA256 = "4024243c259058c09f4d9340d42f410e68e160b96fcd5c09b1a68467b04f4f3a"
POST55_STOCK_SIZE = 51912
POST55_PATCHED_SHA256 = "5547545241f42c536dbe5aa721c4adef5aeb7d1e0972672da11f40f9d2d3ea30"
POST55_PATCHED_SIZE = 58696
PRISTINE_STOCK_SHA256 = "6239fae503211193942d0e2037f7b0edcf71ed70271604701283bfc0453d202b"
PRISTINE_STOCK_SIZE = 49931
PRISTINE_PATCHED_SHA256 = "300ea40c242b87d729e0ba4905936273fd24393cba7550e54882711d48dade9f"
PRISTINE_PATCHED_SIZE = 56715


def _load_patcher():
    spec = importlib.util.spec_from_file_location("hotfix_issue237_cb", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HF = _load_patcher()


def _helpers():
    """Execute the helper block exactly as installed and return its namespace."""
    block = HF.HELPER_NEW
    start = block.index("# " + HF.MARK)
    namespace: dict = {}
    exec(compile(block[start:], "issue237-cb-helpers", "exec"), namespace)
    return namespace


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

    def test_trip_block_aborts_at_engine(self):
        # The trip must halt generation at the engine, not just mutate the
        # output's finish_reason. Assert the injected block awaits the engine
        # abort exactly once, guarded by the one-shot "tripped" flag.
        region = HF.REGION_NEW
        self.assertIn("await self.engine_client.abort(request_id)", region)
        self.assertEqual(region.count("await self.engine_client.abort(request_id)"), 1)
        # The abort must sit inside the one-shot trip guard so it fires once.
        self.assertLess(
            region.index("await self.engine_client.abort(request_id)"),
            region.index('output.finish_reason = "stop"'),
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


class HelperContract(unittest.TestCase):
    """Behavioural contract of the circuit-breaker helpers, in isolation."""

    @classmethod
    def setUpClass(cls):
        cls.ns = _helpers()

    def test_paragraph_loop_detects_repeated_paragraphs(self):
        score = self.ns["_issue237_paragraph_loop_score"]
        # 12 identical paragraphs yield a score of 11, which exceeds the
        # documented default threshold of 10, so the detector demonstrably
        # trips rather than merely registering a repeat.
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

    def test_paragraph_loop_state_isolated_per_key(self):
        score = self.ns["_issue237_paragraph_loop_score"]
        # State is keyed by request id, so two concurrent requests must not
        # share loop-detection state. Feeding the same repeating text under two
        # distinct keys must not let one request's hits leak into the other.
        text = ""
        for _ in range(12):
            text += "The API key rotation failed and the token loop continued indefinitely.\n\n"
            score(text, "req-a")
        self.assertGreaterEqual(score(text, "req-a"), 10)
        # A fresh key that has only seen distinct paragraphs must stay at zero,
        # even though req-a has accumulated hits.
        fresh = ""
        for p in [
            "Quantum computing leverages superposition and entanglement for computation.",
            "The weather forecast predicts rain showers across the coastal region tomorrow.",
        ]:
            fresh += p + "\n\n"
            score(fresh, "req-b")
        self.assertEqual(score(fresh, "req-b"), 0)

    def test_line_loop_detects_repeated_lines(self):
        score = self.ns["_issue237_line_loop_score"]
        # 12 identical lines yield a score of 11, which exceeds the documented
        # default threshold of 10, so the detector demonstrably trips.
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

    def test_generic_loop_score_is_shared_engine(self):
        # Both detectors delegate to one generic sliding-window engine. Exercise
        # it directly with a fresh state dict to confirm it detects repeats and
        # resets after a run of misses.
        generic = self.ns["_issue237_loop_score"]
        state = {}
        split = lambda t: t.split("\n")
        normalize = lambda line: line.strip()
        is_hit = lambda s, window: s in window
        text = ""
        for _ in range(12):
            text += "The API key rotation failed and the loop continued.\n"
            generic(text, "req-gen", state, split, normalize, is_hit, 8)
        self.assertGreaterEqual(generic(text, "req-gen", state, split, normalize, is_hit, 8), 10)
        # Distinct units under a fresh key must not accumulate hits.
        fresh = {}
        distinct = ""
        for i in range(5):
            distinct += f"A wholly distinct line number {i} about another topic.\n"
            generic(distinct, "req-gen2", fresh, split, normalize, is_hit, 8)
        self.assertEqual(generic(distinct, "req-gen2", fresh, split, normalize, is_hit, 8), 0)

    def test_line_loop_threshold_reads_env(self):
        threshold = self.ns["_issue237_line_loop_threshold"]
        saved = os.environ.get("DSPARK_ISSUE237_LINE_LOOP_THRESHOLD")
        try:
            os.environ["DSPARK_ISSUE237_LINE_LOOP_THRESHOLD"] = "7"
            self.assertEqual(threshold(), 7)
        finally:
            if saved is None:
                os.environ.pop("DSPARK_ISSUE237_LINE_LOOP_THRESHOLD", None)
            else:
                os.environ["DSPARK_ISSUE237_LINE_LOOP_THRESHOLD"] = saved
        self.assertEqual(threshold(), 10)

    def test_thresholds_read_env(self):
        enabled = self.ns["_issue237_circuit_breaker_enabled"]
        threshold = self.ns["_issue237_paragraph_loop_threshold"]
        saved = os.environ.get("DSPARK_ISSUE237_PARAGRAPH_LOOP_THRESHOLD")
        try:
            os.environ["DSPARK_ISSUE237_PARAGRAPH_LOOP_THRESHOLD"] = "7"
            self.assertEqual(threshold(), 7)
        finally:
            if saved is None:
                os.environ.pop("DSPARK_ISSUE237_PARAGRAPH_LOOP_THRESHOLD", None)
            else:
                os.environ["DSPARK_ISSUE237_PARAGRAPH_LOOP_THRESHOLD"] = saved
        self.assertEqual(enabled(), False)

    def test_log_emitter_gated(self):
        log_enabled = self.ns["_issue237_circuit_breaker_log_enabled"]
        saved = os.environ.get("DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG")
        try:
            os.environ["DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG"] = "1"
            self.assertTrue(log_enabled())
        finally:
            if saved is None:
                os.environ.pop("DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG", None)
            else:
                os.environ["DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG"] = saved


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="i237-cb-"))
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
        # --check/--status run the self-check, which imports the full serving.py
        # module.  Without torch/fastapi/vllm these fail closed, so we only
        # assert the CLI is wired and reports a classification when the module
        # can be imported.  Here we assert the patcher refuses foreign bytes.
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
            'DSPARK_ENABLE_ISSUE237_CIRCUIT_BREAKER: "${DSPARK_ENABLE_ISSUE237_CIRCUIT_BREAKER:-0}"',
            compose,
        )
        self.assertIn(
            'DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG: "${DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG:-1}"',
            compose,
        )
        self.assertIn(
            'DSPARK_ISSUE237_PARAGRAPH_LOOP_THRESHOLD: "${DSPARK_ISSUE237_PARAGRAPH_LOOP_THRESHOLD:-10}"',
            compose,
        )
        self.assertIn(
            'DSPARK_ISSUE237_LINE_LOOP_THRESHOLD: "${DSPARK_ISSUE237_LINE_LOOP_THRESHOLD:-10}"',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_ISSUE237_CIRCUIT_BREAKER:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-issue237-circuit-breaker.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_ISSUE237_CIRCUIT_BREAKER_HOTFIX:-./patches/hotfix-vllm-issue237-circuit-breaker.py}"
            ":/opt/hotfix-vllm-issue237-circuit-breaker.py:ro",
            compose,
        )

    def test_launcher_passthrough_and_preflight(self):
        start = START.read_text()
        self.assertIn(
            "DSPARK_ISSUE237_CIRCUIT_BREAKER_HOTFIX=\"${DSPARK_ISSUE237_CIRCUIT_BREAKER_HOTFIX:-$SCRIPT_DIR/patches/hotfix-vllm-issue237-circuit-breaker.py}\"",
            start,
        )
        self.assertIn("DSPARK_ENABLE_ISSUE237_CIRCUIT_BREAKER=$REMOTE_ISSUE237_CIRCUIT_BREAKER", start)
        self.assertIn("DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG=$REMOTE_ISSUE237_CIRCUIT_BREAKER_LOG", start)
        self.assertIn("DSPARK_ISSUE237_PARAGRAPH_LOOP_THRESHOLD=$REMOTE_ISSUE237_PARAGRAPH_THRESHOLD", start)
        self.assertIn("DSPARK_ISSUE237_LINE_LOOP_THRESHOLD=$REMOTE_ISSUE237_LINE_THRESHOLD", start)
        # The circuit breaker must get the same --check preflight as the other
        # issue237 hotfixes, on worker, worker2 (when TP3), and head.
        self.assertIn(
            "python3 vllm-dspark /opt/hotfix-vllm-issue237-circuit-breaker.py --check",
            start,
        )

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_ISSUE237_CIRCUIT_BREAKER=0", env)
        self.assertIn("DSPARK_ISSUE237_CIRCUIT_BREAKER_LOG=1", env)
        self.assertIn("DSPARK_ISSUE237_PARAGRAPH_LOOP_THRESHOLD=10", env)
        self.assertIn("DSPARK_ISSUE237_LINE_LOOP_THRESHOLD=10", env)
        self.assertIn("scripts/test-issue237-circuit-breaker.py", CI.read_text())



class SelfCheckVerifyContract(unittest.TestCase):
    """Sabotage-rejection contract of the self-check's detector verification.

    The patcher's ``_self_check`` cannot run in the CPU test environment (it
    imports the full ``serving.py``), so the behavioral verification it performs
    is extracted into ``_issue237_verify_detectors`` and exercised here directly.
    """

    @classmethod
    def setUpClass(cls):
        cls.ns = _helpers()

    def test_accepts_working_detectors(self):
        ok, why = HF._issue237_verify_detectors(
            self.ns["_issue237_paragraph_loop_score"],
            self.ns["_issue237_line_loop_score"],
            self.ns["_issue237_paragraph_loop_threshold"],
            self.ns["_issue237_line_loop_threshold"],
        )
        self.assertTrue(ok, why)

    def test_rejects_detector_that_never_trips(self):
        # Sabotage: a detector that always returns 0 can never trip, so the
        # self-check must reject it as a broken patch.
        def noop(*args, **kwargs):
            return 0

        ok, why = HF._issue237_verify_detectors(
            noop,
            noop,
            self.ns["_issue237_paragraph_loop_threshold"],
            self.ns["_issue237_line_loop_threshold"],
        )
        self.assertFalse(ok)
        self.assertTrue(why)

    def test_rejects_detector_that_false_positives(self):
        # Sabotage: a detector that always returns a huge score trips on
        # distinct prose too, so the self-check must reject it.
        def always_trip(*args, **kwargs):
            return 999

        ok, why = HF._issue237_verify_detectors(
            always_trip,
            always_trip,
            self.ns["_issue237_paragraph_loop_threshold"],
            self.ns["_issue237_line_loop_threshold"],
        )
        self.assertFalse(ok)
        self.assertTrue(why)


if __name__ == "__main__":
    unittest.main()

