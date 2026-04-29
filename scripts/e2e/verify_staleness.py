#!/usr/bin/env python3
"""
Trace-based validation of distributed staleness properties.

Parses router log (structured key=value tracing format) and checks:
  P-i:   Logical clocks (step_id, epoch) are monotonic per engine
  P-ii:  For every CORRECTION with diff != 0, decision_epoch < current_epoch
  P-iii: Staleness is not loss — every non-zero diff has a positive epoch gap

Compatible with new structured tracing (target=scheduling/correction/cache_tracking).
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Decision:
    timestamp: datetime
    request_id: int
    engine: int
    predicted_hits: int
    radix_epoch: int
    new_tokens: int


@dataclass
class Correction:
    timestamp: datetime
    request_id: int
    engine: int
    predicted: int
    actual: int
    diff: int
    decision_epoch: int
    current_epoch: int


@dataclass
class SseEvent:
    timestamp: datetime
    engine: int
    step_id: int
    evicted: int
    inserted: int
    epoch_before: int
    epoch_after: int


@dataclass
class Eviction:
    timestamp: datetime
    engine: int
    evicted_blocks: int


# Match: <ISO ts>  INFO <target>: <file>:<line>: <MSG_KEYWORD> key1=v1 key2=v2 ...
LINE_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+INFO\s+(?P<target>[a-z_:]+):\s+\S+:\s+"
    r"(?P<msg>DECISION|CORRECTION|SSE_EVENT|EVICTION)\s+(?P<kv>.*)"
)
KV_RE = re.compile(r"(\w+)=(-?\d+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_kv(kv_str: str) -> dict:
    return {k: int(v) for k, v in KV_RE.findall(kv_str)}


def parse_log(path: str):
    decisions, corrections, sse_events, evictions = [], [], [], []

    with open(path) as f:
        for line in f:
            line = strip_ansi(line.strip())
            if not line:
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            ts = parse_ts(m.group("ts"))
            msg = m.group("msg")
            kv = parse_kv(m.group("kv"))

            try:
                if msg == "DECISION":
                    decisions.append(Decision(
                        ts, kv["request_id"], kv["engine"],
                        kv["predicted_hits"], kv["radix_epoch"], kv["new_tokens"],
                    ))
                elif msg == "CORRECTION":
                    corrections.append(Correction(
                        ts, kv["request_id"], kv["engine"],
                        kv["predicted"], kv["actual"], kv["diff"],
                        kv["decision_epoch"], kv["current_epoch"],
                    ))
                elif msg == "SSE_EVENT":
                    sse_events.append(SseEvent(
                        ts, kv["engine"], kv["step_id"],
                        kv["evicted"], kv["inserted"],
                        kv["epoch_before"], kv["epoch_after"],
                    ))
                elif msg == "EVICTION":
                    evictions.append(Eviction(
                        ts, kv["engine"], kv.get("evicted_blocks", 0),
                    ))
            except KeyError as e:
                print(f"Warning: malformed {msg} line missing key {e}: {line[:120]}",
                      file=sys.stderr)

    return decisions, corrections, sse_events, evictions


def check_pi(sse_events: list[SseEvent]) -> tuple[bool, list[str]]:
    """P-i: Logical clocks (step_id consecutive, epoch non-regressing) per engine."""
    violations = []
    by_engine: dict[int, list[SseEvent]] = defaultdict(list)
    for e in sse_events:
        by_engine[e.engine].append(e)

    for eng, events in sorted(by_engine.items()):
        events.sort(key=lambda e: e.timestamp)

        # step_id consecutive
        for i in range(1, len(events)):
            prev, curr = events[i - 1], events[i]
            if curr.step_id != prev.step_id + 1:
                violations.append(
                    f"engine={eng}: step_id gap at {curr.timestamp} — "
                    f"expected {prev.step_id + 1}, got {curr.step_id}"
                )

        # epoch non-regression within step
        for e in events:
            if e.epoch_after < e.epoch_before:
                violations.append(
                    f"engine={eng}: epoch regression in step {e.step_id} — "
                    f"before={e.epoch_before} after={e.epoch_after}"
                )

        # epoch non-regression across steps
        for i in range(1, len(events)):
            prev, curr = events[i - 1], events[i]
            if curr.epoch_before < prev.epoch_after:
                violations.append(
                    f"engine={eng}: cross-step epoch regression at step {curr.step_id} — "
                    f"prev_after={prev.epoch_after} curr_before={curr.epoch_before}"
                )

    return len(violations) == 0, violations


def check_pii(corrections: list[Correction]) -> tuple[bool, list[str]]:
    """P-ii: For every CORRECTION with diff != 0, decision_epoch < current_epoch."""
    violations = []
    for c in corrections:
        if c.diff != 0 and c.decision_epoch >= c.current_epoch:
            violations.append(
                f"request_id={c.request_id} engine={c.engine}: diff={c.diff} but "
                f"decision_epoch={c.decision_epoch} >= current_epoch={c.current_epoch} "
                f"— staleness NOT explained by epoch gap"
            )
    return len(violations) == 0, violations


def check_piii(corrections: list[Correction]) -> tuple[bool, list[str]]:
    """P-iii: Staleness is not loss — every non-zero diff has positive epoch gap."""
    violations = []
    for c in corrections:
        if c.diff != 0 and c.current_epoch <= c.decision_epoch:
            violations.append(
                f"request_id={c.request_id} engine={c.engine}: diff={c.diff} with "
                f"zero/negative epoch gap (decision={c.decision_epoch}, current={c.current_epoch})"
            )
    return len(violations) == 0, violations


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <router-log-path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    decisions, corrections, sse_events, evictions = parse_log(path)

    print(f"Parsed: {len(decisions)} DECISIONs, {len(corrections)} CORRECTIONs, "
          f"{len(sse_events)} SSE_EVENTs, {len(evictions)} EVICTIONs")
    print()

    nonzero = [c for c in corrections if c.diff != 0]
    under = [c for c in corrections if c.diff > 0]
    over = [c for c in corrections if c.diff < 0]
    exact = [c for c in corrections if c.diff == 0]
    if corrections:
        print(f"Corrections: {len(exact)} exact ({100 * len(exact) / len(corrections):.1f}%), "
              f"{len(under)} under-predict, {len(over)} over-predict")
    print()

    # Sanity check: if no SSE_EVENTs, the log filter is wrong
    if len(sse_events) == 0:
        print("WARNING: 0 SSE_EVENTs parsed. Likely cause: router was launched without")
        print("  LOG_LEVEL=info,cache_tracking=info  — cache_tracking target is OFF by default.")
        print("  Without SSE_EVENTs, P-i cannot be verified.")
        print()

    if len(corrections) == 0:
        print("WARNING: 0 CORRECTIONs parsed. The router did not produce structured")
        print("  correction events. P-ii / P-iii cannot be verified.")
        print()

    all_pass = True

    # P-i
    print("=" * 60)
    print("P-i: Logical clocks (step_id, epoch) monotonic per engine")
    print("=" * 60)
    if len(sse_events) == 0:
        print("  SKIP — no SSE_EVENTs to check")
    else:
        ok, viols = check_pi(sse_events)
        if ok:
            by_engine = defaultdict(list)
            for e in sse_events:
                by_engine[e.engine].append(e)
            for eng, evts in sorted(by_engine.items()):
                evts.sort(key=lambda e: e.step_id)
                print(f"  engine={eng}: {len(evts)} steps, "
                      f"step_id [{evts[0].step_id}..{evts[-1].step_id}], "
                      f"epoch [{evts[0].epoch_before}..{evts[-1].epoch_after}]")
            print("  PASS")
        else:
            all_pass = False
            print(f"  FAIL — {len(viols)} violations:")
            for v in viols[:10]:
                print(f"    {v}")
            if len(viols) > 10:
                print(f"    ... and {len(viols) - 10} more")
    print()

    # P-ii
    print("=" * 60)
    print("P-ii: Causal ordering (decision_epoch < current_epoch when diff != 0)")
    print("=" * 60)
    if len(corrections) == 0:
        print("  SKIP — no CORRECTIONs to check")
    else:
        ok, viols = check_pii(corrections)
        if ok:
            if nonzero:
                gaps = [c.current_epoch - c.decision_epoch for c in nonzero]
                print(f"  All {len(nonzero)} non-zero corrections have epoch gap > 0")
                print(f"  Epoch gap: min={min(gaps)}, max={max(gaps)}, mean={sum(gaps)/len(gaps):.1f}")
            else:
                print("  No non-zero corrections (all predictions exact)")
            print("  PASS")
        else:
            all_pass = False
            print(f"  FAIL — {len(viols)} violations:")
            for v in viols[:10]:
                print(f"    {v}")
            if len(viols) > 10:
                print(f"    ... and {len(viols) - 10} more")
    print()

    # P-iii
    print("=" * 60)
    print("P-iii: No loss (every non-zero diff has positive epoch gap)")
    print("=" * 60)
    total_ins = sum(e.inserted for e in sse_events)
    total_evict = sum(e.evicted for e in sse_events)
    print(f"  Total blocks inserted: {total_ins}, evicted: {total_evict}")
    if len(corrections) == 0:
        print("  SKIP — no CORRECTIONs to check")
    else:
        ok, viols = check_piii(corrections)
        if ok:
            print(f"  All {len(nonzero)} non-zero corrections have positive epoch gap")
            print("  PASS")
        else:
            all_pass = False
            print(f"  FAIL — {len(viols)} unexplained corrections:")
            for v in viols[:10]:
                print(f"    {v}")
            if len(viols) > 10:
                print(f"    ... and {len(viols) - 10} more")
    print()

    # Vacuous-pass guard: if all three checks were SKIPped, the validation is meaningless
    skipped_all = (len(sse_events) == 0 and len(corrections) == 0)
    if skipped_all:
        print("=" * 60)
        print("VACUOUS PASS — no events parsed, validation is meaningless")
        print("Check log filter and router instrumentation")
        print("=" * 60)
        sys.exit(2)

    print("=" * 60)
    print("ALL PROPERTIES PASS" if all_pass else "SOME PROPERTIES FAILED")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
