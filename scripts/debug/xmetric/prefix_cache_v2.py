#!/usr/bin/env python3
import re
import argparse
from datetime import datetime, timedelta
from collections import defaultdict


def parse_ts(s, default_year):
    s = s.strip()
    if "T" in s and "Z" in s:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    try:
        return datetime.strptime(f"{default_year}-{s}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


# Regex patterns
RE_ROUTER_PRESUME = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z).*?Request_(?P<rid>\d+)\s+with\s+(?P<hits>\d+)\s+presumed hit",
    re.IGNORECASE,
)
RE_BACKEND_CACHED = re.compile(
    r"(?P<ts>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?\[cached\].*?req id:\s*(?P<rid>\d+).*?cached bids:\s*\[(?P<bids>[^\]]*)\]",
    re.IGNORECASE,
)
RE_INSERT = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z).*?insert.*?\[\[([^\]]*)\]\]", re.IGNORECASE
)
RE_EVICT = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z).*?removes bids\s*\[(?P<bids>[^\]]*)\]",
    re.IGNORECASE,
)


def parse_int_list(s):
    return [int(x) for x in re.findall(r"\d+", s)]


def parse_router_log(path, year):
    inserts = defaultdict(list)
    evicts = defaultdict(list)
    presumes = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if m := RE_INSERT.search(line):
                ts = parse_ts(m["ts"], year)
                bids = parse_int_list(m.group(2))
                for b in bids:
                    inserts[b].append(ts)
            elif m := RE_EVICT.search(line):
                ts = parse_ts(m["ts"], year)
                bids = parse_int_list(m["bids"])
                for b in bids:
                    evicts[b].append(ts)
            elif m := RE_ROUTER_PRESUME.search(line):
                rid = m["rid"]
                ts = parse_ts(m["ts"], year)
                hits = int(m["hits"])
                presumes[rid] = {"ts": ts, "hits": hits}
    return inserts, evicts, presumes


def load_backend_log(path, year):
    actuals = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if m := RE_BACKEND_CACHED.search(line):
                rid = m["rid"]
                ts = parse_ts(m["ts"], year)
                bids = parse_int_list(m["bids"])
                actuals[rid] = {"ts": ts, "bids": bids}
    return actuals


def last_not_after(ts_list, ts):
    """Return the latest timestamp in ts_list that is <= ts."""
    return max([t for t in ts_list if t <= ts], default=None)


def first_not_before(ts_list, ts):
    return min([t for t in ts_list if t >= ts], default=None)

def last_before(ts_list, ts):
    """Return the last timestamp before ts."""
    return max([t for t in ts_list if t < ts], default=None)


def analyze(router_inserts, router_evicts, presumes, actuals, skew_ms):
    skew = timedelta(milliseconds=skew_ms)
    clues = []
    for rid, actual in actuals.items():
        if rid not in presumes:
            print(f"Request_{rid} not found in router::queue, this is a bug!")
            continue
        presume = presumes[rid]
        phits = presume["hits"]
        ahits = len(actual["bids"])
        if ahits <= phits:
            continue
        # first false negative bid
        false_bid = actual["bids"][phits]
        p_ts = presume["ts"]
        a_ts = actual["ts"]

        insert_times = router_inserts.get(false_bid, [])
        evict_times = router_evicts.get(false_bid, [])
        if len(insert_times) == 0:
            raise ValueError(f"Request_{rid}::bid={false_bid} has not been ever inserted!")

        insert_before = last_not_after(insert_times, p_ts)
        evict_before = last_before(evict_times, p_ts)

        insert_after = first_not_before(insert_times, p_ts)
        
        verdict = "clue"
        reason = ""
        if insert_before is None:
            reason = "never_inserted_before_presume"
        elif evict_before and (insert_before < evict_before):
            verdict = "staleness"
            reason = "evicted_before_presume"
        elif insert_before > (p_ts + skew):
            verdict = "staleness"
            reason = "insert_after_presume"
        else:
            verdict = "clue"
            reason = "inserted_before_presume_no_eviction"

        clues.append(
            {
                "request_id": rid,
                "false_bid": false_bid,
                "presume_ts": p_ts.isoformat(sep=" "),
                "actual_ts": a_ts.isoformat(sep=" "),
                "insert_before": (
                    insert_before.isoformat(sep=" ") if insert_before else ""
                ),
                "evict_before": evict_before.isoformat(sep=" ") if evict_before else "",
                "verdict": verdict,
                "reason": reason,
                "presumed_hits": phits,
                "actual_hits": ahits,
                "first_insert_ts": insert_after.isoformat(sep=" ") if insert_after else "",
            }
        )
    return clues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-log", required=True)
    parser.add_argument("--vllm-log", required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--skew-ms", type=int, default=50)
    parser.add_argument("--csv", help="optional CSV output file")
    args = parser.parse_args()

    inserts, evicts, presumes = parse_router_log(args.router_log, args.year)
    actuals = load_backend_log(args.vllm_log, args.year)
    clues = analyze(inserts, evicts, presumes, actuals, args.skew_ms)

    if not clues:
        print("No clues found.")
        return

    print(f"Found {len(clues)} clues:\n")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=clues[0].keys())
            writer.writeheader()
            writer.writerows(clues)
        print(f"\nCSV written: {args.csv}")
    else:
        for c in clues:
            print(
                f"- req={c['request_id']} bid={c['false_bid']} "
                f"verdict={c['verdict']} reason={c['reason']} "
                f"presume={c['presume_ts']} insert={c['insert_before']} evict={c['evict_before']}"
            )


if __name__ == "__main__":
    main()
