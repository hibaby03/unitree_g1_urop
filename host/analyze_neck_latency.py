"""Join one Host/PC2 run: python3 analyze_neck_latency.py --host ... --pc2 ..."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path


def read_log(path):
    records, bad = [], 0
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("not an object")
                records.append(record)
            except (ValueError, TypeError):
                bad += 1
    return records, bad


def elapsed(record, end, start):
    a, b = record.get(end), record.get(start)
    return None if a is None or b is None else (a - b) / 1e6


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"count": 0}
    def percentile(p):
        x = (len(values) - 1) * p
        i = int(x)
        return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)
    return {"count": len(values), "min": values[0], "p50": percentile(.5),
            "p95": percentile(.95), "max": values[-1]}


LOCAL_INTERVALS = {
    "host_pose_age_ms": ("send_start_host_ns", "pose_rx_host_ns"),
    "host_send_call_ms": ("send_end_host_ns", "send_start_host_ns"),
    "pc2_receive_to_target_ms": ("target_ready_pc2_ns", "rx_pc2_ns"),
    "pc2_target_to_write_ms": ("write_start_pc2_ns", "target_ready_pc2_ns"),
    "motor_write_ms": ("write_end_pc2_ns", "write_start_pc2_ns"),
    "encoder_read_ms": ("read_end_pc2_ns", "read_start_pc2_ns"),
    "pc2_service_ms": ("loop_end_pc2_ns", "rx_pc2_ns"),
}


def analyze(host_records, pc2_records, max_age_s=10.0):
    if max_age_s <= 0:
        raise ValueError("max clock age must be positive")
    host_packets = [r for r in host_records if r.get("event") == "host_packet"]
    by_uid = {r["packet_uid"]: r for r in host_packets}
    clocks = sorted((r for r in host_records if r.get("event") == "clock_sample"),
                    key=lambda r: r["t3"])
    clock_times = [r["t3"] for r in clocks]
    rows, unmatched, matched = [], 0, set()
    repeated_pose_packets = 0
    previous_pose = None
    for r in host_packets:
        if r.get("pose_id") is not None:
            if r["pose_id"] == previous_pose:
                repeated_pose_packets += 1
            previous_pose = r["pose_id"]
    for pc in pc2_records:
        if pc.get("event") != "pc2_packet":
            continue
        h = by_uid.get(pc.get("packet_uid"))
        if h is None:
            unmatched += 1
        else:
            matched.add(h["packet_uid"])
        # Retain raw times, targets, encoders and errors in the output table.
        row = {**(h or {}), **pc}
        row.pop("event", None)
        for name, (end, start) in LOCAL_INTERVALS.items():
            row[name] = elapsed(row, end, start)
        tail_start = "read_end_pc2_ns" if row.get("read_end_pc2_ns") is not None else (
            "write_end_pc2_ns" if row.get("write_end_pc2_ns") is not None else "target_ready_pc2_ns")
        row["feedback_and_logging_ms"] = elapsed(row, "loop_end_pc2_ns", tail_start)
        row["clock_status"] = "unmatched_host" if h is None else "unavailable_or_stale"
        if h is not None:
            now = h["send_start_host_ns"]
            lo = bisect.bisect_left(clock_times, now - int(max_age_s * 1e9))
            hi = bisect.bisect_right(clock_times, now)
            candidates = clocks[lo:hi]
            if candidates:
                c = min(candidates, key=lambda r: r["rtt_ns"])
                offset = c["offset_ns"]
                row.update(clock_status="estimated", clock_offset_pc2_minus_host_ns=offset,
                           clock_rtt_ms=c["rtt_ns"] / 1e6,
                           clock_uncertainty_ms=c["uncertainty_ns"] / 1e6,
                           clock_sample_age_ms=(now - c["t3"]) / 1e6)
                # Includes PC2 kernel queue wait and scheduling, not just wire time.
                row["send_to_pc2_dequeue_ms"] = (pc["rx_pc2_ns"] - offset - now) / 1e6
                if h.get("pose_rx_host_ns") is not None and pc.get("write_end_pc2_ns") is not None:
                    row["pose_to_command_sent_ms"] = (
                        pc["write_end_pc2_ns"] - offset - h["pose_rx_host_ns"]) / 1e6
        rows.append(row)
    metrics = list(LOCAL_INTERVALS) + ["feedback_and_logging_ms", "send_to_pc2_dequeue_ms",
                                     "pose_to_command_sent_ms", "clock_rtt_ms", "clock_uncertainty_ms"]
    summary = {
        "host_packets": len(host_packets), "pc2_packets": len(rows),
        "matched_host_packets": len(matched), "unmatched_pc2_packets": unmatched,
        "host_without_pc2_record": len(set(by_uid) - matched),
        "repeated_pose_packets": repeated_pose_packets,
        "clock_samples": len(clocks),
        "rows_without_clock_estimate": sum(r["clock_status"] != "estimated" for r in rows),
        "negative_corrected_delay_rows": sum(r.get("send_to_pc2_dequeue_ms", 0) < 0 for r in rows),
        "encoder_error_rows": sum(bool(r.get("encoder_error")) for r in rows),
        "host_send_errors": sum(bool(r.get("send_error")) for r in host_packets),
        "metrics_ms": {key: stats(r.get(key) for r in rows) for key in metrics},
        "notes": [
            "Offsets map monotonic clocks; they are not wall-clock corrections.",
            "Half RTT is probe uncertainty under stable clock rates; drift and asymmetry remain.",
            "send_to_pc2_dequeue includes transport, OS queueing and scheduling.",
            "write_end is API completion, not motor acceptance or physical arrival.",
            "pose_rx is Host callback entry, not Quest sensor acquisition.",
            "Missing packets can be loss, log drops, or recording start/stop boundaries.",
        ],
    }
    for label, records in (("host", host_records), ("pc2", pc2_records)):
        ends = [r for r in records if r.get("event") == "trace_end"]
        summary[label + "_clean_log_close"] = bool(ends)
        summary[label + "_dropped_log_records"] = sum(r.get("dropped", 0) for r in ends) if ends else None
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="one host-*.jsonl")
    parser.add_argument("--pc2", required=True, help="matching pc2-*.jsonl copied from PC2")
    parser.add_argument("--out", default="neck_latency_report", help="new output directory")
    parser.add_argument("--max-clock-age", type=float, default=10.0)
    args = parser.parse_args(argv)
    host, host_bad = read_log(args.host)
    pc2, pc2_bad = read_log(args.pc2)
    rows, summary = analyze(host, pc2, args.max_clock_age)
    summary.update(host_malformed_lines=host_bad, pc2_malformed_lines=pc2_bad)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["packet_uid"]
    with (out / "packets.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
