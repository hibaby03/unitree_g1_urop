"""Two-way Host/PC2 DDS heartbeat on isolated test topics; no robot commands."""
import argparse
from dataclasses import dataclass
import json
import math
import re
import threading
import time
import uuid


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("host", "pc2"))
    parser.add_argument("--interface", required=True)
    parser.add_argument("--seconds", type=positive, default=30)
    parser.add_argument("--hz", type=positive, default=5)
    parser.add_argument("--domain", type=int, default=42, help="same on both machines; default 42")
    parser.add_argument("--test-id", default="link_check", help="same on both machines; letters, digits, underscore")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]{1,48}", args.test_id):
        parser.error("--test-id must contain 1-48 letters, digits or underscores")

    from cyclonedds.idl import IdlStruct
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber

    @dataclass
    class DiagnosticHeartbeat(IdlStruct, typename="UropDiagnosticHeartbeatV1"):
        payload: str

    peer = "pc2" if args.role == "host" else "host"
    prefix = f"urop_diagnostics/{args.test_id}"
    send_topic, receive_topic = f"{prefix}/{args.role}", f"{prefix}/{peer}"
    instance = uuid.uuid4().hex
    lock = threading.Lock()
    state = {"peer_received": 0, "last": None, "peer_instance": None,
             "peer_sequence": None, "malformed": 0}

    def receive(message):
        try:
            value = json.loads(message.payload)
            if value["role"] != peer or value["test_id"] != args.test_id:
                return
            if not isinstance(value["sequence"], int) or not isinstance(value["instance"], str):
                raise ValueError("invalid heartbeat fields")
            with lock:
                state.update(peer_received=state["peer_received"] + 1,
                             last=time.monotonic(), peer_instance=value["instance"],
                             peer_sequence=value["sequence"])
        except Exception:
            with lock:
                state["malformed"] += 1

    publisher = subscriber = None
    print(json.dumps({"role": args.role, "interface": args.interface,
                      "domain": args.domain, "send_topic": send_topic,
                      "receive_topic": receive_topic, "instance": instance}), flush=True)
    try:
        ChannelFactoryInitialize(args.domain, networkInterface=args.interface)
        subscriber = ChannelSubscriber(receive_topic, DiagnosticHeartbeat)
        subscriber.Init(receive, 10)
        publisher = ChannelPublisher(send_topic, DiagnosticHeartbeat)
        publisher.Init()
        start = time.monotonic()
        deadline = start + args.seconds
        next_send = next_report = start
        sequence = writes_ok = writes_failed = 0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                sequence += 1
                message = DiagnosticHeartbeat(json.dumps({"role": args.role,
                    "test_id": args.test_id, "instance": instance, "sequence": sequence}))
                # Write success alone does NOT prove delivery to the other PC.
                if publisher.Write(message):
                    writes_ok += 1
                else:
                    writes_failed += 1
                next_send = now + 1 / args.hz
            if now >= next_report:
                with lock:
                    report = dict(state)
                last = report.pop("last")
                report.update(elapsed_s=round(now - start, 2), local_writes_ok=writes_ok,
                              local_writes_failed=writes_failed,
                              last_peer_age_s=None if last is None else round(now - last, 3))
                print(json.dumps(report), flush=True)
                next_report = now + 1
            time.sleep(max(0, min(next_send, next_report, deadline) - time.monotonic()))
        with lock:
            final = dict(state)
        last = final.pop("last")
        age = None if last is None else time.monotonic() - last
        # Freshness allows three missed heartbeat periods, with a 3s minimum.
        max_age = max(3, 3 / args.hz)
        ok = final["peer_received"] > 0 and age is not None and age <= max_age
        final.update(local_writes_ok=writes_ok, local_writes_failed=writes_failed,
                     last_peer_age_s=None if age is None else round(age, 3), pass_inbound=ok)
        print(json.dumps(final), flush=True)
        print(f"{'PASS' if ok else 'FAIL'}: {peer} -> {args.role}. "
              "Check the OTHER terminal too; both must PASS for a two-way result.", flush=True)
        return 0 if ok else 1
    finally:
        try:
            if subscriber is not None:
                subscriber.Close()
        finally:
            if publisher is not None:
                publisher.Close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped by operator.", flush=True)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        raise SystemExit(2)
