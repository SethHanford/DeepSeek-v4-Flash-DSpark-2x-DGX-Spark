#!/usr/bin/env python3
"""Analyze the issue237 generation capture for token repetition / looping.

Reads sitecustomize-loaded-after-serve2.jsonl (one JSON record per generated
token) and reports:
- total records, distinct request_ids
- per-request: n_output_tokens, whether a repeat was flagged, max_tokens
- the longest runs and whether any request shows a repeating token pattern
"""
import json
from collections import defaultdict, Counter
from pathlib import Path

path = Path("sitecustomize-loaded-after-serve2.jsonl")
records = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

print(f"Total records: {len(records)}")
if not records:
    raise SystemExit("No records parsed.")

by_req = defaultdict(list)
for r in records:
    by_req[r.get("request_id")].append(r)

print(f"Distinct request_ids: {len(by_req)}")
print()

# Per-request summary
for rid, recs in sorted(by_req.items(), key=lambda kv: -len(kv[1]))[:10]:
    n = len(recs)
    last_tokens = [r.get("last_token") for r in recs]
    repeats = sum(1 for r in recs if r.get("repeat"))
    max_tokens = recs[0].get("max_tokens")
    # Detect a repeating tail window (last 20 tokens repeat prev 20)
    tail = last_tokens[-20:]
    prev = last_tokens[-40:-20]
    repeated = tail == prev and len(tail) == 20
    print(f"request_id={rid} n_tokens={n} max_tokens={max_tokens} repeats_flagged={repeats} tail_repeats_prev20={repeated}")
    if repeated:
        print(f"   REPEATING TAIL: {tail}")

# Overall: how many requests had any repeat flagged
any_repeat = sum(1 for recs in by_req.values() if any(r.get("repeat") for r in recs))
print(f"\nRequests with at least one repeat flagged: {any_repeat}/{len(by_req)}")

# Show the last 40 tokens of the longest request
longest = max(by_req.values(), key=len)
print(f"\nLongest request ({len(longest)} tokens) last 40 tokens:")
print([r.get("last_token") for r in longest[-40:]])
