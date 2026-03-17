#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按以下流程分析 cache 命中低估并找出所有“非 staleness”线索:
1) 每个 Request_id:收集 presumed 命中数与 ts;
2) 读取执行器 [cached] 的实际命中 bid 列表与 ts;
3) 当 actual > presumed:找第一个 false negative bid(在 presume_ts 的控制器已知集合之外的第一个 actual bid);
4) 找出该 bid 在控制器侧的插入时间(优先来自“全量快照首次包含该 bid”的时间,否则来自 insert 首次见到的时间);
5) 若插入时间 > presume_ts + skew → staleness,否则记为“线索(clue)”;
6) 输出所有线索并可导出 CSV。

用法:
  python analyze_cache_inconsistency.py --log combined.log --skew-ms 50 --year 2025 --csv clues.csv
"""

import re
import sys
import csv
import argparse
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ---------- Time parsing utils ---------- #

ISOZ = re.compile(r'^\d{4}-\d{2}-\d{2}T[\d:.]+Z$') # Ok
MD_HMS = re.compile(r'^\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$') 

def parse_ts_str(ts_str: str, default_year: int) -> datetime:
    ts_str = ts_str.strip()
    # ISO 8601 with trailing Z
    if ISOZ.match(ts_str):
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.replace(tzinfo=None)
    # "MM-DD HH:MM:SS"(从执行器 python 日志)
    # TODO: leading 'INFO'
    if MD_HMS.match(ts_str):
        dt = datetime.strptime(f"{default_year}-{ts_str}", "%Y-%m-%d %H:%M:%S")
        return dt  # 视为 naive UTC
    # 兜底:尝试更宽松的 ISO 解析
    try:
        if ts_str.endswith("Z"):
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        return datetime.fromisoformat(ts_str)
    except Exception('Timestamp match error!'):
        return None

# ---------- Regular Expressions ---------- #

# (1) Router presumed hit number
#
# 2025-11-07T06:28:00.768253Z DEBUG router_v2::queue: router_v2/src/queue.rs:1013: vLLM#0::Request_32 with 1 presumed hit blocks
# (?P<name>pattern) => named capture group
# .*? => non-greedy wild match
RE_ROUTER_PRESUME = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z).*?Request_(?P<rid>\d+)\s+with\s+(?P<hits>\d+)\s+presumed hit blocks',
    re.IGNORECASE
)

# (4) vLLM cached bids
#
# INFO 11-07 06:27:56 [scheduler.py:421] [cached] for req id: 0 cached bids: []
# ^ => Negation
# [^\]]* => a char class `[...]`, with set "not `]`"
RE_BACKEND_CACHED = re.compile(
    r'(?P<ts>\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?\[cached\].*?req id:\s*(?P<rid>\d+).*?cached bids:\s*\[(?P<bids>[^\]]*)\]',
    re.IGNORECASE
)

# (3) Router is acknowledged of backend bids
#
# 2025-11-07T06:27:56.276600Z  INFO router_v2::replica::colocation::task_assignment: router_v2/src/replica/colocation.rs:497: Entry(0) update backend bids [1, 2, 3, 4]
RE_SSE_UPDATE_SNAPSHOT = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z).*?Entry\((?P<entry>\d+)\)\s+update backend bids\s*\[(?P<bids>[^\]]*)\]',
    re.IGNORECASE
)

# (5) Router binds bid with hashes
# 
# Entry(k) insert (...) |-> [[ ... ]]
RE_SSE_INSERT = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z).*?Entry\((?P<entry>\d+)\)\s+insert.*',
    re.IGNORECASE
)

def parse_int_list(s: str):
    return [int(x) for x in re.findall(r'\d+', s)]

def extract_last_bracket_int_list(line: str):
    groups = re.findall(r'\[([^\[\]]*)\]', line)
    if not groups:
        return []
    return parse_int_list(groups[-1])

# ---------- Data Structures ---------- #

class RequestView:
    __slots__ = ("rid", "presume_hits", "presume_ts", "actual_bids", "actual_ts")
    def __init__(self, rid: str):
        self.rid = rid
        self.presume_hits = None
        self.presume_ts = None
        self.actual_bids = None
        self.actual_ts = None

class SseState:
    """维护全量快照时间线 + 首次出现时间(从快照或插入推导)"""
    def __init__(self):
        self.snap_times = []   # [ts1, ts2, ...] 递增
        self.snap_sets  = []   # [set(bids at ts1), set(bids at ts2), ...]
        self.first_seen = {}   # bid -> ts(尽量取来源于“首次被包含的快照”;没有快照时用 insert 首见)

    def add_snapshot(self, ts: datetime, bids):
        s = set(bids)
        self.snap_times.append(ts)
        self.snap_sets.append(s)
        # 用快照来完善 first_seen(更可靠)
        for b in s:
            if b not in self.first_seen or ts < self.first_seen[b]:
                self.first_seen[b] = ts

    def add_insert(self, ts: datetime, bids):
        # 仅在无快照能覆盖时,作为 first_seen 的补充
        for b in bids:
            if b not in self.first_seen:
                self.first_seen[b] = ts

    def known_set_at(self, ts: datetime):
        """返回在 ts 时刻控制器“已知的全集”(优先基于快照)"""
        if not self.snap_times:
            # 无快照:退化为所有 first_seen <= ts 的集合
            return {b for b, t0 in self.first_seen.items() if t0 is not None and t0 <= ts}
        idx = bisect_right(self.snap_times, ts) - 1
        if idx < 0:
            return set()
        return self.snap_sets[idx]

    def first_insert_ts(self, bid: int):
        """返回 bid 首次进入控制器视图的时间(优先使用快照推导;否则使用 insert)"""
        if bid in self.first_seen:
            return self.first_seen[bid]
        # 兜底:从快照里线性查找(通常不会走到这里,因为 add_snapshot 已填充)
        for ts, s in zip(self.snap_times, self.snap_sets):
            if bid in s:
                self.first_seen[bid] = ts
                return ts
        return None

# ---------- 主流程 ----------

def analyze(lines, default_year=2025, skew_ms=50, csv_out=None):
    reqs = defaultdict(lambda: RequestView(rid=None))
    sse = SseState()

    # 先逐行解析
    for line in lines:
        # (1) presumed
        m = RE_ROUTER_PRESUME.search(line)
        if m:
            rid = m["rid"]
            ts = parse_ts_str(m["ts"], default_year)
            hits = int(m["hits"])
            rv = reqs[rid]
            rv.rid = rid
            rv.presume_hits = hits
            rv.presume_ts = ts
            continue

        # (4) backend cached bids
        m = RE_BACKEND_CACHED.search(line)
        if m:
            rid = m["rid"]
            ts = parse_ts_str(m["ts"], default_year)
            bids = parse_int_list(m["bids"])
            rv = reqs[rid]
            rv.rid = rid
            rv.actual_bids = bids
            rv.actual_ts = ts
            continue

        # (3) update snapshot
        m = RE_SSE_UPDATE_SNAPSHOT.search(line)
        if m:
            ts = parse_ts_str(m["ts"], default_year)
            bids = parse_int_list(m["bids"])
            sse.add_snapshot(ts, bids)
            continue

        # (5) insert delta
        m = RE_SSE_INSERT.search(line)
        if m:
            ts = parse_ts_str(m["ts"], default_year)
            bids = extract_last_bracket_int_list(line)
            if bids:
                sse.add_insert(ts, bids)
            continue

    # 逐请求找“线索”
    skew = timedelta(milliseconds=skew_ms)
    clues = []  # 收集所有“非 staleness”线索
    stats = {"req_total": 0, "req_candidates": 0, "clue_count": 0, "staleness_count": 0, "no_fn_found": 0}

    for rid, rv in reqs.items():
        # 基础信息完整性
        if rv.presume_hits is None or rv.presume_ts is None or rv.actual_bids is None:
            continue
        stats["req_total"] += 1

        actual_hits = len(rv.actual_bids)
        if actual_hits <= rv.presume_hits:
            continue
        stats["req_candidates"] += 1

        # 找到 presume_ts 时刻控制器“已知集合”
        known = sse.known_set_at(rv.presume_ts)

        # 找第一个 false negative bid
        false_bid = None
        for b in rv.actual_bids:
            if b not in known:
                false_bid = b
                break

        if false_bid is None:
            stats["no_fn_found"] += 1
            continue

        insert_ts = sse.first_insert_ts(false_bid)
        verdict = "clue"
        reason = "insert_ts<=presume_ts"
        if insert_ts is None:
            verdict = "clue"
            reason = "no_insert_ts"  # 未知更倾向标为线索,交由人工复核
        elif insert_ts > (rv.presume_ts + skew):
            verdict = "staleness"
            reason = "insert_ts>presume_ts+skew"

        rec = {
            "request_id": rid,
            "presume_ts": rv.presume_ts.isoformat(sep=" "),
            "presume_hits": rv.presume_hits,
            "actual_ts": rv.actual_ts.isoformat(sep=" ") if rv.actual_ts else "",
            "actual_hits": actual_hits,
            "first_false_negative_bid": false_bid,
            "bid_insert_ts": insert_ts.isoformat(sep=" ") if insert_ts else "",
            "verdict": verdict,                  # clue / staleness
            "reason": reason,                    # 用于解释
            "used_snapshot": bool(sse.snap_times),  # 是否使用了快照来判定已知集合/插入时间
        }
        clues.append(rec)
        if verdict == "clue":
            stats["clue_count"] += 1
        else:
            stats["staleness_count"] += 1

    # 输出摘要
    clues_sorted = sorted(clues, key=lambda r: r["presume_ts"])
    print(f"[Summary] total_reqs={stats['req_total']} candidates(actual>presume)={stats['req_candidates']}, "
          f"clues={stats['clue_count']} staleness={stats['staleness_count']} no_fn_found={stats['no_fn_found']}")
    if clues_sorted:
        first = clues_sorted[0]
        print("\n[First CLUE]")
        for k in ["request_id","presume_ts","presume_hits","actual_ts","actual_hits",
                  "first_false_negative_bid","bid_insert_ts","verdict","reason","used_snapshot"]:
            print(f"  {k:26s}: {first[k]}")
    else:
        print("\n[First CLUE] not found")

    # 列表输出
    if clues_sorted:
        print("\n[All CLUES]")
        for r in clues_sorted:
            print(f"- req={r['request_id']} presume_ts={r['presume_ts']} "
                  f"presumed={r['presume_hits']} actual={r['actual_hits']} "
                  f"bid={r['first_false_negative_bid']} insert_ts={r['bid_insert_ts']} verdict={r['verdict']} ({r['reason']})")

    # 导出 CSV(可选)
    if csv_out and clues_sorted:
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(clues_sorted[0].keys()))
            w.writeheader()
            w.writerows(clues_sorted)
        print(f"\nCSV written: {csv_out}")

    return clues_sorted, stats


def main():
    ap = argparse.ArgumentParser(description="Visualize non-staleness causes of cache hit underestimation.")
    ap.add_argument("--log", nargs="+", required=True, help="日志文件(可多个,顺序即拼接顺序)")
    ap.add_argument("--year", type=int, default=2025, help="无年份时间戳(如 '11-07 06:29:36')的默认年份")
    ap.add_argument("--skew-ms", type=int, default=50, help="时钟偏差容忍阈值(ms),insert_ts > presume_ts+skew 视为 staleness")
    ap.add_argument("--csv", dest="csv_out", help="将线索输出为 CSV")
    args = ap.parse_args()

    lines = []
    for p in args.log:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            lines.extend(f.readlines())

    analyze(lines, default_year=args.year, skew_ms=args.skew_ms, csv_out=args.csv_out)


if __name__ == "__main__":
    main()
