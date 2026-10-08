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
5. **Repetition penalty (1.05)** — implemented as a hotfix clamping
   `repetition_penalty` to a minimum in `PenaltiesState.add_request`. Result:
   **partial mitigation only**. The loop still occurs (see A/B test below), but
   with a larger repeating period and more token variety. The penalty changes the
   loop's character but does not eliminate it.

## Root cause hypothesis (revised)
The loop is an **agentic "planning" loop**, not primarily a token-sampling issue.
The model gets stuck narrating an action it intends to take but cannot complete,
cycling through variations of a "Let me [do X]" preamble. The permissive sampling
defaults (top_p=1.0, top_k=0, no repetition penalty) allow the narration to repeat,
but the underlying driver is the model failing to resolve a planned action.

## A/B test: repetition penalty (1.05) does NOT eliminate the loop
A generation-phase instrumentation (sitecustomize token logger) was deployed to
capture token-level output, and the serve2 trigger was run with and without the
repetition penalty.

- **Baseline (no penalty):** request `chatcmpl-8a6f29098c8f6811-adbc9789` loops —
  223 repeat flags, repeating period of 11 tokens, unique-token ratio 0.108.
- **Config A (penalty 1.05):** request `chatcmpl-94be5a29acd9abf7-9a1362e6` still
  loops — 223 repeat flags, repeating period of 35 tokens, unique-token ratio 0.194.

The penalty **changes the loop's character** (larger period, more token variety,
later onset) but does **not eliminate it**. Both configurations show exactly one
request with >= 20 repeats and low token diversity (< 0.3).

## Decoded loop content (smoking gun)
The repeating token sequences were decoded with the model tokenizer:

- **Baseline repeating phrase:** `' me check the capture file and run the analysis.\n\nLet'`
- **Config A repeating phrases:**
  - `'Let me check the API key.\n\n'`
  - `'Let me get the API key from the container.\n\n'`
  - `'Let me get the API key from the running container.\n\n'`

The model is stuck **narrating an action it intends to take but never completes**.
It generates the "Let me [do X]" preamble over and over, cycling through variations
of the same planning phrase. This is most pronounced when the model is planning an
action it cannot take (e.g., running commands against containers that are not on
the local machine).

## Tool-call path investigation (root cause confirmed)
The looping request was decoded and checked for tool-call DSML structure:

- **`<tool_calls>`: False** — the model did NOT emit any tool-call DSML structure.
- **`<invoke`: False** — no tool invocation.
- **`Let me`: 41 occurrences** — the model narrated "Let me..." 41 times.
- **`check the API key`: 16 occurrences** — repeated the same phrase 16 times.

The model is **narrating actions it intends to take but never emits the tool-call
DSML structure** (`<tool_calls><invoke name="...">`). It says "Let me check the API
key" repeatedly but never makes the actual tool call.

The tool-call parser (`vllm/parser/deepseek_v4.py`) is a state machine that only
emits a `TOOL_CALL_START` event when it sees the DSML `<invoke name="...>` marker.
When the model generates "Let me [action]" as plain content (no DSML structure),
the parser extracts `tool_calls = []`, so the serving layer treats it as plain text
and `finish_reason` is not "tool_calls". The model, still expecting to make a tool
call, re-narrates the preamble and loops.

## Revised conclusion
The loop's root cause is a **behavioral planning loop**: the model narrates an
action it intends to take but never emits the tool-call DSML structure, so the
planned action never materializes and the model re-narrates. The implemented
mitigation combines a **repetition penalty** (suppresses the repeated preamble)
with a **circuit breaker** (aborts the request when near-duplicate paragraphs or
lines recur). Making the model actually emit the tool-call DSML structure
(tool-call emission) is a known limitation that this fix does not address.

## Proposed fix (revised)
1. **Repetition penalty** (implemented) — suppresses repeated preamble, partial
   mitigation.
2. **Loop-breaking on narration-without-call** (implemented) — a circuit breaker
   detects near-duplicate recent paragraphs or lines and aborts the request at the
   engine, forcing `finish_reason` to "stop" when the model narrates without
   emitting a tool call.

## Artifacts preserved
- `sitecustomize-loaded-after-serve2.jsonl` — the baseline loop capture (no penalty).
- `issue237_capture_A_penalty_on.jsonl` — the Config A capture (penalty 1.05), still loops.
- `scripts/analyze-issue237-capture.py` — repetition detector.
- `scripts/analyze-issue237-fresh.py` — refined loop detector (flags genuine loops only).
- `patches/sitecustomize.py` — the instrumentation deployed as sitecustomize.
- `scripts/diag-issue237-encode-compare.py` — encoder comparison diagnostic.
- `scripts/instrument-issue237.py` — earlier encoder instrumentation.
- `issue237_capture.jsonl` — earlier encoder-level capture.

## Loop patterns observed

Concrete loop shapes seen in the wild, used as test cases for a detector that
must catch them **without matching on language** (the model is multilingual, so
English phrase matching is not viable). Each pattern is a distinct failure mode
that a detector should trip on.

1. **Single-line echo loop** — the model regurgitates one source line over and
   over, separated by single newlines (the 47k-token `_stub(...)` loop). The
   lines are byte-identical. Caught by the line detector (exact match); missed
   by the paragraph detector because a single line never forms a complete
   paragraph.

   ```
       _stub("vllm.entrypoints.openai.tool_parsers.tool_parsers_utils")
       _stub("vllm.entrypoints.openai.tool_parsers.tool_parsers_utils")
       _stub("vllm.entrypoints.openai.tool_parsers.tool_parsers_utils")
   ```

2. **Short-phrase narration loop** — the model cycles through 2-3 short,
   non-identical "let me" statements repeatedly (observed in a session that ran
   for hours before being terminated). The statements are near-duplicate but
   not byte-identical, so exact match misses them; they are too short for the
   paragraph detector's token-set similarity to distinguish from legitimate
   near-duplicate prose.

   ```
   Let me check the API key.
   Let me get the API key from the container.
   Let me check the API key again.
   Let me check the API key.
   Let me get the API key from the container.
   ```

3. **Repetitive "Let me..." preamble chain** — the model opens successive
   turns with a short "Let me ..." preamble that repeats in structure even when
   the rest of the turn differs. The preamble alone is a loop signal.

   ```
   Let me check the imports of the serving.py fixture to gauge stub complexity.
   Let me look at the top imports of the serving fixture.
   Let me check the serving.py fixture imports.
   Let me view the imports of the serving fixture.
   ```

These three shapes are the acceptance cases for any future loop detector. The
current paragraph and line detectors cover shape 1 only; shapes 2 and 3 are
open gaps.
