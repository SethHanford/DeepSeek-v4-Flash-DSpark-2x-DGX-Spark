#!/usr/bin/env python3
"""CPU tests for patches/hotfix-vllm-issue237-repetition-penalty.py (no vLLM, no GPU, no network).

Covers the pinned fixture identity, the pure transformation, the atomic
apply/idempotency/refusal behaviour, the behavioural clamp contract, and the
compose/launcher/env-example/ci wiring locks.
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
FIXTURE = ROOT / "scripts" / "fixtures" / "issue237" / "penalties-stock.py"
PATCHER = ROOT / "patches" / "hotfix-vllm-issue237-repetition-penalty.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"

STOCK_SHA256 = "46c29434caf5f415129ef4a9bb2b8d4f62d2765596e3e07f73ad2b198c7def1b"
STOCK_SIZE = 10608
PATCHED_SHA256 = "9812321ac202e70611d6a2352db2fcc866f5d9c9433d647aa3536dc99e39db49"
PATCHED_SIZE = 11476


def _load_patcher():
    spec = importlib.util.spec_from_file_location("hotfix_issue237_rep", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HF = _load_patcher()


def _install_stubs():
    """Stub torch/triton/vllm so the patcher's self-check can import penalties.py.

    The patcher's ``_self_check`` imports the full ``penalties.py`` module, which
    pulls in torch, triton, and vllm.  These are not available in the CPU test
    environment, so we install minimal stand-ins that satisfy the module's
    top-level imports and the ``@triton.jit`` decorators.
    """
    def stub(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    # numpy is not guaranteed in the CPU test environment; stub the handful of
    # attributes penalties.py uses so the module can be imported.
    np = stub("numpy")
    np.zeros = lambda *a, **k: None
    np.ndarray = type("ndarray", (), {})
    np.any = lambda *a, **k: False
    np.fill = lambda *a, **k: None

    torch = stub("torch")
    torch.jit = types.SimpleNamespace(script=lambda f: f, script_if_tracing=lambda f: f)
    torch.Tensor = type("Tensor", (), {})
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.int32 = "int32"
    torch.int64 = "int64"
    torch.bool = "bool"
    torch.zeros = lambda *a, **k: None
    torch.tensor = lambda *a, **k: None
    torch.as_tensor = lambda *a, **k: None
    torch.cat = lambda *a, **k: None
    torch.stack = lambda *a, **k: None
    torch.arange = lambda *a, **k: None
    torch.full = lambda *a, **k: None
    torch.empty = lambda *a, **k: None
    torch.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda s: None, __exit__=lambda *a: None
    )

    tl = types.SimpleNamespace(
        constexpr=int, program_id=lambda *a: 0, load=lambda *a, **k: 0,
        arange=lambda *a, **k: 0, int64="int64", float32="float32",
        zeros=lambda *a, **k: 0, where=lambda *a, **k: 0, sum=lambda *a, **k: 0,
        max=lambda *a, **k: 0, min=lambda *a, **k: 0, exp=lambda *a, **k: 0,
        log=lambda *a, **k: 0, full=lambda *a, **k: 0, store=lambda *a, **k: None,
        atomic_add=lambda *a, **k: None, static_print=lambda *a, **k: None,
        device_assert=lambda *a, **k: None,
    )
    stub("vllm.triton_utils", tl=tl, triton=types.SimpleNamespace(jit=lambda f: f))
    stub("vllm")
    stub("vllm.sampling_params", SamplingParams=type("SamplingParams", (), {}))
    stub("vllm.utils")
    stub("vllm.utils.math_utils", cdiv=lambda a, b: (a + b - 1) // b)
    stub("vllm.utils.torch_utils", async_tensor_h2d=lambda *a, **k: None)
    stub("vllm.v1")
    stub("vllm.v1.worker")
    stub("vllm.v1.worker.gpu")
    stub("vllm.v1.worker.gpu.buffer_utils", UvaBackedTensor=type("UvaBackedTensor", (), {}))
    stub("vllm.v1.worker.gpu.states", RequestState=type("RequestState", (), {}))


class FixtureAndTransform(unittest.TestCase):
    def test_fixture_identity_matches_patcher_pins(self):
        data = FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), STOCK_SHA256)
        self.assertEqual(len(data), STOCK_SIZE)

    def test_region_constants_match_self_pins(self):
        HF._verify_self()
        self.assertEqual(
            hashlib.sha256(HF.IMPORT_OLD.encode()).hexdigest(), HF.IMPORT_OLD_SHA256
        )
        self.assertEqual(
            hashlib.sha256(HF.IMPORT_NEW.encode()).hexdigest(), HF.IMPORT_NEW_SHA256
        )
        self.assertEqual(
            hashlib.sha256(HF.REGION_OLD.encode()).hexdigest(), HF.REGION_OLD_SHA256
        )
        self.assertEqual(
            hashlib.sha256(HF.REGION_NEW.encode()).hexdigest(), HF.REGION_NEW_SHA256
        )

    def test_transform_is_pinned_single_site_and_compiles(self):
        stock = FIXTURE.read_bytes()
        self.assertEqual(stock.count(HF.IMPORT_OLD.encode()), 1)
        self.assertEqual(stock.count(HF.REGION_OLD.encode()), 1)
        patched = HF.transform(stock)
        self.assertEqual(hashlib.sha256(patched).hexdigest(), PATCHED_SHA256)
        self.assertEqual(len(patched), PATCHED_SIZE)
        self.assertEqual(patched.count(HF.MARK.encode()), 2)
        compile(patched, "penalties.py", "exec")

    def test_transform_refuses_foreign_or_patched_bytes(self):
        with self.assertRaises(HF.HotfixError):
            HF.transform(b"def nothing():\n    pass\n")
        patched = HF.transform(FIXTURE.read_bytes())
        with self.assertRaises(HF.HotfixError):
            HF.transform(patched)

    def test_self_check_accepts_patched(self):
        _install_stubs()
        ok, why = HF._self_check(HF.transform(FIXTURE.read_bytes()))
        self.assertTrue(ok, why)

    def test_self_check_rejects_a_broken_clamp(self):
        _install_stubs()
        # Sabotage: make the minimum penalty a no-op (<= 1.0) so the clamp
        # cannot raise a default penalty. The self-check must reject it.
        patched = HF.transform(FIXTURE.read_bytes())
        sabotaged = patched.replace(
            b'_MIN_REPETITION_PENALTY = float(_os.environ.get("DSPARK_ISSUE237_REPETITION_PENALTY", "1.05"))',
            b'_MIN_REPETITION_PENALTY = 1.0',
            1,
        )
        self.assertNotEqual(sabotaged, patched)
        ok, why = HF._self_check(sabotaged)
        self.assertFalse(ok)
        self.assertTrue(why)


class ClampContract(unittest.TestCase):
    """Behavioural contract of the clamp, exercised on the helper block."""

    def _namespace(self):
        # Extract just the module constant (skip the class definition that
        # follows in IMPORT_NEW).
        block = HF.IMPORT_NEW
        start = block.index("import os as _os")
        end = block.index("class PenaltiesState:")
        ns = {}
        exec(compile(block[start:end], "i237-rep-helpers", "exec"), ns)
        return ns

    def test_min_penalty_is_gt_one(self):
        ns = self._namespace()
        self.assertGreater(ns["_MIN_REPETITION_PENALTY"], 1.0)

    def test_min_penalty_reads_env(self):
        saved = os.environ.get("DSPARK_ISSUE237_REPETITION_PENALTY")
        try:
            os.environ["DSPARK_ISSUE237_REPETITION_PENALTY"] = "1.2"
            ns = self._namespace()
            self.assertAlmostEqual(ns["_MIN_REPETITION_PENALTY"], 1.2)
        finally:
            if saved is None:
                os.environ.pop("DSPARK_ISSUE237_REPETITION_PENALTY", None)
            else:
                os.environ["DSPARK_ISSUE237_REPETITION_PENALTY"] = saved

    def test_resolve_clamps_default_to_min(self):
        ns = self._namespace()
        self.assertEqual(ns["_issue237_resolve_rep_penalty"](1.0), ns["_MIN_REPETITION_PENALTY"])

    def test_resolve_clamps_sub_one_to_min(self):
        ns = self._namespace()
        self.assertEqual(ns["_issue237_resolve_rep_penalty"](0.9), ns["_MIN_REPETITION_PENALTY"])

    def test_resolve_leaves_explicit_penalty_untouched(self):
        ns = self._namespace()
        self.assertEqual(ns["_issue237_resolve_rep_penalty"](1.2), 1.2)


class Patcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_stubs()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="i237-rep-"))
        self.target = self.tmp / "penalties.py"
        shutil.copyfile(FIXTURE, self.target)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_then_idempotent(self):
        self.assertEqual(HF.apply(self.target), "applied")
        data = self.target.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), PATCHED_SHA256)
        self.assertEqual(HF.inspect(self.target)[0], "patched")
        self.assertEqual(HF.apply(self.target), "already-patched")
        self.assertEqual(self.target.read_bytes(), data)
        self.assertEqual([q.name for q in self.tmp.iterdir()], ["penalties.py"])

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

    def _cli(self, *args):
        """Run the patcher as a subprocess with the dependency stubs injected.

        The patcher's self-check imports the full penalties.py module (torch,
        triton, vllm), so we install a sitecustomize.py on PYTHONPATH that
        installs the stubs before the patcher module is imported.
        """
        site = self.tmp / "sitecustomize.py"
        site.write_text(
            "import sys, types\n"
            "def _stub(name, **attrs):\n"
            "    m = types.ModuleType(name)\n"
            "    for k, v in attrs.items(): setattr(m, k, v)\n"
            "    sys.modules[name] = m\n"
            "    return m\n"
            "np = _stub('numpy')\n"
            "np.zeros = lambda *a,**k:None; np.ndarray = type('ndarray', (), {})\n"
            "np.any = lambda *a,**k:False; np.fill = lambda *a,**k:None\n"
            "torch = _stub('torch')\n"
            "torch.jit = types.SimpleNamespace(script=lambda f: f, script_if_tracing=lambda f: f)\n"
            "torch.float32='float32'; torch.float16='float16'; torch.int64='int64'; torch.bool='bool'\n"
            "torch.zeros=lambda *a,**k:None; torch.tensor=lambda *a,**k:None\n"
            "torch.as_tensor=lambda *a,**k:None; torch.cat=lambda *a,**k:None\n"
            "torch.stack=lambda *a,**k:None; torch.arange=lambda *a,**k:None\n"
            "torch.full=lambda *a,**k:None; torch.empty=lambda *a,**k:None\n"
            "torch.no_grad=lambda: types.SimpleNamespace(__enter__=lambda s:None, __exit__=lambda *a:None)\n"
            "tl = types.SimpleNamespace(constexpr=int, program_id=lambda *a:0, load=lambda *a,**k:0,\n"
            "    arange=lambda *a,**k:0, int64='int64', float32='float32', zeros=lambda *a,**k:0,\n"
            "    where=lambda *a,**k:0, sum=lambda *a,**k:0, max=lambda *a,**k:0, min=lambda *a,**k:0,\n"
            "    exp=lambda *a,**k:0, log=lambda *a,**k:0, full=lambda *a,**k:0, store=lambda *a,**k:None,\n"
            "    atomic_add=lambda *a,**k:None, static_print=lambda *a,**k:None, device_assert=lambda *a,**k:None)\n"
            "_stub('vllm.triton_utils', tl=tl, triton=types.SimpleNamespace(jit=lambda f:f))\n"
            "_stub('vllm')\n"
            "_stub('vllm.sampling_params', SamplingParams=type('SamplingParams', (), {}))\n"
            "_stub('vllm.utils')\n"
            "_stub('vllm.utils.math_utils', cdiv=lambda a,b:(a+b-1)//b)\n"
            "_stub('vllm.utils.torch_utils', async_tensor_h2d=lambda *a,**k:None)\n"
            "_stub('vllm.v1'); _stub('vllm.v1.worker'); _stub('vllm.v1.worker.gpu')\n"
            "_stub('vllm.v1.worker.gpu.buffer_utils', UvaBackedTensor=type('UvaBackedTensor', (), {}))\n"
            "_stub('vllm.v1.worker.gpu.states', RequestState=type('RequestState', (), {}))\n"
        )
        env = dict(os.environ, PYTHONPATH=str(self.tmp))
        return subprocess.run(
            [sys.executable, str(PATCHER), *args, "--target", str(self.target)],
            capture_output=True, text=True, env=env,
        )

    def test_cli_check_and_status_do_not_write(self):
        before = self.target.read_bytes()
        for flag in ("--check", "--status"):
            proc = self._cli(flag)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("stock", proc.stdout)
            self.assertEqual(self.target.read_bytes(), before)

    def test_cli_apply_and_status_roundtrip(self):
        proc = self._cli()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("applied", proc.stdout)
        proc = self._cli("--status")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("patched", proc.stdout)


class Wiring(unittest.TestCase):
    def test_compose_gate_default_off_fail_closed(self):
        compose = COMPOSE.read_text()
        self.assertIn(
            'DSPARK_ENABLE_ISSUE237_REPETITION_PENALTY: "${DSPARK_ENABLE_ISSUE237_REPETITION_PENALTY:-0}"',
            compose,
        )
        self.assertIn(
            'DSPARK_ISSUE237_REPETITION_PENALTY: "${DSPARK_ISSUE237_REPETITION_PENALTY:-1.05}"',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_ISSUE237_REPETITION_PENALTY:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-issue237-repetition-penalty.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_ISSUE237_REPETITION_PENALTY_HOTFIX:-./patches/hotfix-vllm-issue237-repetition-penalty.py}"
            ":/opt/hotfix-vllm-issue237-repetition-penalty.py:ro",
            compose,
        )

    def test_launcher_passthrough_and_preflight(self):
        start = START.read_text()
        self.assertIn(
            "DSPARK_ISSUE237_REPETITION_PENALTY_HOTFIX='./patches/hotfix-vllm-issue237-repetition-penalty.py'",
            start,
        )
        self.assertIn("DSPARK_ENABLE_ISSUE237_REPETITION_PENALTY=$REMOTE_ISSUE237_REPETITION_PENALTY", start)
        self.assertIn("DSPARK_ISSUE237_REPETITION_PENALTY=$REMOTE_ISSUE237_REPETITION_PENALTY_VALUE", start)

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_ISSUE237_REPETITION_PENALTY=0", env)
        self.assertIn("DSPARK_ISSUE237_REPETITION_PENALTY=1.05", env)
        self.assertIn("scripts/test-issue237-repetition-penalty.py", CI.read_text())


if __name__ == "__main__":
    unittest.main()
