# Issue #237 — Investigation Notes

## Symptom
The model enters a token-level repetition loop in `thinking_mode: "chat"` with
large conversations (300+ messages). The output repeats a fixed token sequence
until `max_tokens` is reached. This affects multiple users of the repo, not just
Copilot.

## Reproducible evidence
- `sitecustomize-loaded-after-serve2.jsonl` — capture of generated tokens from the
  instrumented engine. Request `chatcmpl-8a6f29098c8f6811-adbc9789` shows 223
  repeat detections out of 278 tokens (a clear repetition loop).
- `scripts/analyze-issue237-capture.py` — detects the repetition pattern in the
  capture.

## Approaches tried and outcomes
1. **Drop-history-reasoning patch** (removes tools-override so `_drop_thinking_messages`
   runs). Result: functionally correct but does NOT fix the loop. The loop is in
   chat mode; the patch only affects `thinking_mode == "thinking"`. Confirmed by
   `scripts/diag-issue237-encode-compare.py` (patch produces identical output for
   the real conversation structure).
2. **MTP_NUM_TOKENS=0** — Result: invalid. Vision-Exp requires MTP_NUM_TOKENS >= 5
   and divisible by 3; the engine refuses to start.
3. **Disabling speculative decoding** (removing `--speculative-config`) — Result:
   does NOT fix the loop. The loop persists without spec.
4. **Sampling parameters** — the engine defaults are temperature=1.0, top_p=1.0,
   top_k=0, repetition_penalty=1.0 (no penalty). This is the likely root cause:
   nothing prevents the model from repeating tokens.

## Root cause hypothesis
The combination of permissive sampling defaults (top_p=1.0, top_k=0, and no
repetition penalty) allows the model to fall into a repetitive attractor state at
large context sizes. A repetition penalty (or reduced top_p) should prevent the
loop.

## Proposed fix
Apply a minimum repetition penalty in the vLLM sampling layer
(`vllm/v1/worker/gpu/sample/penalties.py`, `add_request`), so a penalty is always
applied even when the client does not specify one. This is a global fix that helps
all users, not just Copilot.

## Artifacts preserved
- `sitecustomize-loaded-after-serve2.jsonl` — the loop capture (reproducible case).
- `scripts/analyze-issue237-capture.py` — repetition detector.
- `scripts/instrument-generation-issue237.py` — sitecustomize token logger.
- `scripts/diag-issue237-encode-compare.py` — encoder comparison diagnostic.
- `scripts/instrument-issue237.py` — earlier encoder instrumentation.
- `issue237_capture.jsonl` — earlier encoder-level capture.
