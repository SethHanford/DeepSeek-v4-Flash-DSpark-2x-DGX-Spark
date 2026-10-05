#!/usr/bin/env python3
"""CPU tests for patches/hotfix-dsv4-drop-history-reasoning.py (no vLLM, no GPU, no network).

The patcher is a source-exact, fail-closed string replacement: it removes the
checkpoint encoder's tools override in `_encode_messages_text` so
`drop_thinking` is honored even when tool definitions are present. This breaks
the agent-mode reasoning self-contamination loop (issue #237).

These tests pin the patch contract without a live encoder:
- the OLD -> NEW replacement is exact and single-site,
- an already-patched source is reported `skipped` (idempotent),
- a source that has drifted (pattern absent) is reported `missing` and left
  untouched (fail-closed),
- `patch_file` writes only on `applied`,
- the `--status` path reports APPLIED / NOT APPLIED without writing,
- the CLI `main` apply path returns 0 on applied/skipped and 1 on missing.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "patches" / "hotfix-dsv4-drop-history-reasoning.py"

STOCK = (
    "    # Resolve drop_thinking: if any message has tools defined, don't drop thinking\n"
    "    effective_drop_thinking = drop_thinking\n"
    "    if any(m.get(\"tools\") for m in full_messages):\n"
    "        effective_drop_thinking = False\n"
)

PATCHED = (
    "    # Resolve drop_thinking: if any message has tools defined, don't drop thinking\n"
    "    # [drop-history-reasoning] The tools override is removed so earlier-turn\n"
    "    # reasoning_content is dropped even when tool definitions are present,\n"
    "    # matching the official DeepSeek API (which does not replay historical\n"
    "    # reasoning). This breaks the reasoning self-contamination loop seen in\n"
    "    # agent mode (issue #237). The live turn (index >= last user) still reasons.\n"
    "    effective_drop_thinking = drop_thinking\n"
)


def _load_patcher():
    spec = importlib.util.spec_from_file_location("dspark_issue237", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HF = _load_patcher()


class PatchText(unittest.TestCase):
    def test_applies_exact_replacement(self):
        updated, status = HF.patch_text(STOCK)
        self.assertEqual(status, "applied")
        self.assertEqual(updated, PATCHED)
        self.assertIn(HF.MARK, updated)
        self.assertNotIn("if any(m.get(\"tools\") for m in full_messages):", updated)

    def test_already_patched_is_skipped(self):
        updated, status = HF.patch_text(PATCHED)
        self.assertEqual(status, "skipped")
        self.assertEqual(updated, PATCHED)

    def test_drifted_source_is_missing_and_untouched(self):
        drifted = "    effective_drop_thinking = drop_thinking\n"
        updated, status = HF.patch_text(drifted)
        self.assertEqual(status, "missing")
        self.assertEqual(updated, drifted)

    def test_mark_is_single_site(self):
        self.assertEqual(PATCHED.count(HF.MARK), 1)


class PatchFile(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="i237-patcher-"))
        self.target = self.tmp / "deepseek_v4_encoding.py"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_writes_patched_bytes(self):
        self.target.write_text(STOCK, encoding="utf-8")
        self.assertEqual(HF.patch_file(self.target), "applied")
        self.assertEqual(self.target.read_text(encoding="utf-8"), PATCHED)

    def test_apply_is_idempotent(self):
        self.target.write_text(PATCHED, encoding="utf-8")
        before = self.target.read_bytes()
        self.assertEqual(HF.patch_file(self.target), "skipped")
        self.assertEqual(self.target.read_bytes(), before)

    def test_apply_fails_closed_on_drift(self):
        self.target.write_text("x = 1\n", encoding="utf-8")
        before = self.target.read_bytes()
        self.assertEqual(HF.patch_file(self.target), "missing")
        self.assertEqual(self.target.read_bytes(), before)


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="i237-cli-"))
        self.target = self.tmp / "deepseek_v4_encoding.py"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_status_reports_applied_on_stock(self):
        # Stock source carries the OLD pattern, so the patch would apply; the
        # --status path reports APPLIED and does not write.
        self.target.write_text(STOCK, encoding="utf-8")
        before = self.target.read_bytes()
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--status", str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("APPLIED", proc.stdout)
        self.assertEqual(self.target.read_bytes(), before)

    def test_status_reports_applied_on_patched(self):
        self.target.write_text(PATCHED, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--status", str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("APPLIED", proc.stdout)

    def test_status_reports_not_applied_on_drifted(self):
        # A source that has drifted (pattern absent) reports NOT APPLIED.
        self.target.write_text("x = 1\n", encoding="utf-8")
        before = self.target.read_bytes()
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--status", str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("NOT APPLIED", proc.stdout)
        self.assertEqual(self.target.read_bytes(), before)

    def test_status_missing_file_is_not_an_error(self):
        proc = subprocess.run(
            [sys.executable, str(PATCHER), "--status", str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("NOT APPLIED", proc.stdout)

    def test_apply_returns_zero_and_writes(self):
        self.target.write_text(STOCK, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(PATCHER), str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("applied", proc.stdout)
        self.assertEqual(self.target.read_text(encoding="utf-8"), PATCHED)

    def test_apply_already_present_returns_zero(self):
        self.target.write_text(PATCHED, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(PATCHER), str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("already present", proc.stdout)

    def test_apply_missing_pattern_returns_one(self):
        self.target.write_text("x = 1\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(PATCHER), str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("not found", proc.stderr)

    def test_apply_missing_file_returns_one(self):
        proc = subprocess.run(
            [sys.executable, str(PATCHER), str(self.target)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("not found", proc.stderr)


if __name__ == "__main__":
    unittest.main()
