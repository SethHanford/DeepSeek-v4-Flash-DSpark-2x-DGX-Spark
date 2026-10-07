#!/usr/bin/env python3
"""Analyze a fresh issue237 generation capture for repetition loops.

Usage: python3 analyze-issue237-fresh.py <capture.jsonl> [baseline.jsonl]

A genuine loop is a request where a fixed token period repeats MANY times
consecutively and consumes a large fraction of the output (the model
collapses into a deterministic cycle). Natural prose repetition (a phrase
repeated a few times) is NOT flagged.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

# A request is a genuine loop if the repeating period repeats >= MIN_REPEATS
# times consecutively AND consumes >= MIN_FRACTION of the output.
MIN_REPEATS = 10
MIN_FRACTION = 0.5


def find_period(seq, max_period=40):
    for p in range(1, max_period + 1):
        if len(seq) >= 3 * p:
            tail = seq[-3 * p:]
            if all(tail[i] == tail[i + p] for i in range(len(tail) - p)):
                return p
    return None


def load(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    by_req = defaultdict(list)
    for r in records:
        by_req[r.get("request_id")].append(r)
    return records, by_req


def detect_loop(toks):
    """Return (period, repeat_count, loop_start) if a genuine loop, else None."""
    period = find_period(toks)
    if not period:
        return None
    unit = toks[-period:]
    count = 0
    for i in range(len(toks) - period, -1, -period):
        if toks[i:i + period] == unit:
            count += 1
        else:
            break
    loop_tokens = count * period
    fraction = loop_tokens / len(toks) if toks else 0
    if count >= MIN_REPEATS and fraction >= MIN_FRACTION:
        return (period, count, len(toks) - loop_tokens)
    return None


def analyze(path, label):
    records, by_req = load(path)
    print(f"=== {label}: {path} ===")
    print(f"Total records: {len(records)}, distinct requests: {len(by_req)}")
    loops = []
    for rid, recs in by_req.items():
        toks = [r.get("last_token") for r in recs]
        result = detect_loop(toks)
        if result:
            period, count, loop_start = result
            loops.append((rid, len(toks), period, count, loop_start, recs[0].get("max_tokens")))
    if loops:
        print(f"\nGENUINE LOOPS DETECTED: {len(loops)}")
        for rid, n, period, count, loop_start, mt in sorted(loops, key=lambda x: -x[3])[:10]:
            print(f"  {rid}: n={n} period={period} repeats={count} loop_start={loop_start} max_tokens={mt}")
    else:
        print("\nNO GENUINE LOOPS DETECTED")
    return loops


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    loops = analyze(sys.argv[1], "FRESH CAPTURE")
    if len(sys.argv) > 2:
        analyze(sys.argv[2], "BASELINE")
    print("\n=== VERDICT ===")
    if loops:
        print(f"LOOP STILL PRESENT: {len(loops)} request(s) looped. Repetition penalty did NOT fully prevent it.")
    else:
        print("NO LOOP DETECTED. Repetition penalty appears to have prevented the loop.")
