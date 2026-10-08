#!/usr/bin/env python3
"""CPU tests for patches/hotfix-vllm-issue237-toolcall-directive.py (no vLLM, no GPU, no network).

Covers both legitimate pre-states of the co-owned encoder module (the pristine
48095b345 snapshot and the live entrypoint chain product), the pinned fixture
identity, the pure transformation, the atomic apply/idempotency/refusal
behaviour, the directive-presence contract, and the compose/launcher/env/ci
wiring locks.
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
FIXDIR = ROOT / "scripts" / "fixtures" / "issue144-effort-align"
FIX_LIVE = FIXDIR / "deepseek_v4_encoding-48095b345-live-chain.py"
FIX_SNAP = FIXDIR / "encoding_dsv4-48095b345-snapshot.py"
PATCHER = ROOT / "patches" / "hotfix-vllm-issue237-toolcall-directive.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"

LIVE_STOCK_SHA256 = "07432ce4758143247b80ef269f5855cb4536fd610422e008cfeae36de65ad547"
LIVE_STOCK_SIZE = 39960
LIVE_PATCHED_SHA256 = "6b135647e2dec0b7e3a417c2d35d5f27757841d3969f078b1386d6ccc71d27c6"
LIVE_PATCHED_SIZE = 40179
SNAP_STOCK_SHA256 = "b4bbb74bbb11a9c8ada04daa30cc7de7dba3abba08e9ade06d38b51a3d0d1701"
SNAP_STOCK_SIZE = 36707
SNAP_PATCHED_SHA256 = "f83fb446c9ba4b502429026159dabe015e922dbe8d21cb3ee96f8499cfe145db"
SNAP_PATCHED_SIZE = 36926


def _load_patcher():
    spec = importlib.util.spec_from_file_location("hotfix_issue237_directive", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HF = _load_patcher()


class FixtureAndTransform(unittest.TestCase):
    def test_fixtures_match_identity_pins(self):
        live = FIX_LIVE.read_bytes()
        snap = FIX_SNAP.read_bytes()
        self.assertEqual(hashlib.sha256(live).hexdigest(), LIVE_STOCK_SHA256)
        self.assertEqual(len(live), LIVE_STOCK_SIZE)
        self.assertEqual(hashlib.sha256(snap).hexdigest(), SNAP_STOCK_SHA256)
        self.assertEqual(len(snap), SNAP_STOCK_SIZE)

    def test_region_constants_match_self_pins(self):
        HF._verify_self()
        self.assertEqual(
            hashlib.sha256(HF.REGION_OLD.encode()).hexdigest(), HF.REGION_OLD_SHA256
        )
        self.assertEqual(
            hashlib.sha256(HF.REGION_NEW.encode()).hexdigest(), HF.REGION_NEW_SHA256
        )

    def test_transform_is_pinned_single_site_and_compiles(self):
        for fixture, patched_sha, patched_size in (
            (FIX_LIVE, LIVE_PATCHED_SHA256, LIVE_PATCHED_SIZE),
            (FIX_SNAP, SNAP_PATCHED_SHA256, SNAP_PATCHED_SIZE),
        ):
            stock = fixture.read_bytes()
            self.assertEqual(stock.count(HF.REGION_OLD.encode()), 1)
            patched = HF.transform(stock)
            self.assertEqual(hashlib.sha256(patched).hexdigest(), patched_sha)
            self.assertEqual(len(patched), patched_size)
            self.assertEqual(patched.count(HF.MARK.encode()), 1)
            compile(patched, "deepseek_v4_encoding.py", "exec")

    def test_transform_refuses_foreign_or_patched_bytes(self):
        with self.assertRaises(HF.HotfixError):
            HF.transform(b"def nothing():\n    pass\n")
        patched = HF.transform(FIX_LIVE.read_bytes())
        with self.assertRaises(HF.HotfixError):
            HF.transform(patched)

    def test_self_check_accepts_both_pre_states(self):
        for fixture in (FIX_LIVE, FIX_SNAP):
            ok, why = HF._self_check(HF.transform(fixture.read_bytes()))
            self.assertTrue(ok, why)

    def test_self_check_rejects_a_broken_directive(self):
        # Sabotage: drop the directive text from the patched template.
        patched = HF.transform(FIX_LIVE.read_bytes())
        sabotaged = patched.replace(b"Act, then speak.", b"", 1)
        self.assertNotEqual(sabotaged, patched)
        ok, why = HF._self_check(sabotaged)
        self.assertFalse(ok)
        self.assertTrue(why)


class DirectiveContract(unittest.TestCase):
    """The directive must be present in the patched template and absent in stock."""

    def test_directive_present_in_patched_absent_in_stock(self):
        for fixture in (FIX_LIVE, FIX_SNAP):
            stock = fixture.read_bytes()
            patched = HF.transform(stock)
            self.assertIn("Act, then speak.", patched.decode())
            self.assertNotIn("Act, then speak.", stock.decode())
            self.assertIn("You MUST strictly follow", stock.decode())


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="i237-directive-"))
        self.target = self.tmp / "deepseek_v4_encoding.py"
        shutil.copyfile(FIX_LIVE, self.target)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_then_idempotent(self):
        self.assertEqual(HF.apply(self.target), "applied")
        data = self.target.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), LIVE_PATCHED_SHA256)
        self.assertEqual(HF.inspect(self.target)[0], "patched")
        self.assertEqual(HF.apply(self.target), "already-patched")
        self.assertEqual(self.target.read_bytes(), data)
        self.assertEqual([q.name for q in self.tmp.iterdir()], ["deepseek_v4_encoding.py"])

    def test_refuses_foreign_bytes(self):
        self.target.write_bytes(b"x = 1\n")
        with self.assertRaises(HF.HotfixError):
            HF.inspect(self.target)
        with self.assertRaises(HF.HotfixError):
            HF.apply(self.target)
        self.assertEqual(self.target.read_bytes(), b"x = 1\n")

    def test_refuses_symlink(self):
        link = self.tmp / "link.py"
        os.symlink(self.target, link)
        with self.assertRaises(HF.HotfixError):
            HF.inspect(link)
        with self.assertRaises(HF.HotfixError):
            HF.apply(link)

    def test_cli_check_and_status_do_not_write(self):
        before = self.target.read_bytes()
        for flag in ("--check", "--status"):
            proc = subprocess.run(
                [sys.executable, str(PATCHER), flag, "--target", str(self.target)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("stock", proc.stdout)
            self.assertEqual(self.target.read_bytes(), before)

    def test_cli_apply_and_status_roundtrip(self):
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--target", str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("applied", proc.stdout)
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--status", "--target", str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("patched", proc.stdout)


class Wiring(unittest.TestCase):
    def test_compose_gate_default_off_fail_closed(self):
        compose = COMPOSE.read_text()
        self.assertIn(
            'DSPARK_ENABLE_ISSUE237_TOOLCALL_DIRECTIVE: "${DSPARK_ENABLE_ISSUE237_TOOLCALL_DIRECTIVE:-0}"',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_ISSUE237_TOOLCALL_DIRECTIVE:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-issue237-toolcall-directive.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_ISSUE237_TOOLCALL_DIRECTIVE_HOTFIX:-./patches/hotfix-vllm-issue237-toolcall-directive.py}"
            ":/opt/hotfix-vllm-issue237-toolcall-directive.py:ro",
            compose,
        )

    def test_launcher_passthrough_and_preflight(self):
        start = START.read_text()
        self.assertIn(
            "DSPARK_ISSUE237_TOOLCALL_DIRECTIVE_HOTFIX=\"${DSPARK_ISSUE237_TOOLCALL_DIRECTIVE_HOTFIX:-$SCRIPT_DIR/patches/hotfix-vllm-issue237-toolcall-directive.py}\"",
            start,
        )
        self.assertIn("DSPARK_ENABLE_ISSUE237_TOOLCALL_DIRECTIVE=$REMOTE_ISSUE237_DIRECTIVE", start)
        # The patcher is synced to the worker's patches/ dir, where the compose
        # default (./patches/...) picks it up.
        self.assertIn(
            'dscp "$DSPARK_ISSUE237_TOOLCALL_DIRECTIVE_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-vllm-issue237-toolcall-directive.py"',
            start,
        )
        # The toolcall directive must get the same --check preflight as the
        # other issue237 hotfixes, on worker, worker2 (when TP3), and head.
        self.assertIn(
            "python3 vllm-dspark /opt/hotfix-vllm-issue237-toolcall-directive.py --check",
            start,
        )

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_ISSUE237_TOOLCALL_DIRECTIVE=0", env)
        self.assertIn("scripts/test-issue237-toolcall-directive.py", CI.read_text())


if __name__ == "__main__":
    unittest.main()
