"""Receive G1 rt/lowstate only. Never creates a publisher or changes robot mode."""
import argparse
import json
import math
import threading
import time


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True, help="PC2: eth0; Host: actual robot-facing interface")
    parser.add_argument("--seconds", type=positive, default=10)
    parser.add_argument("--domain", type=int, default=0)
    args = parser.parse_args()

    # Import after parsing so --help works without the robot SDK installed.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    lock = threading.Lock()
    state = {"received": 0, "last": None, "motor_count": None,
             "first_motor_q": None, "decode_errors": 0, "last_error": None}

    def receive(message):
        now = time.monotonic()
        with lock:
            state["received"] += 1
            state["last"] = now
            try:
                state["motor_count"] = len(message.motor_state)
                state["first_motor_q"] = float(message.motor_state[0].q)
                if not math.isfinite(state["first_motor_q"]):
                    raise ValueError("first motor q is not finite")
            except Exception as error:
                state["decode_errors"] += 1
                state["last_error"] = f"{type(error).__name__}: {error}"

    subscriber = None
    print(json.dumps({"mode": "receive_only", "interface": args.interface,
                      "domain": args.domain, "topic": "rt/lowstate"}), flush=True)
    try:
        ChannelFactoryInitialize(args.domain, networkInterface=args.interface)
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(receive, 10)
        start = time.monotonic()
        deadline = start + args.seconds
        while time.monotonic() < deadline:
            time.sleep(min(1, max(0, deadline - time.monotonic())))
            now = time.monotonic()
            with lock:
                result = dict(state)
            last = result.pop("last")
            age = None if last is None else round(now - last, 3)
            result.update(elapsed_s=round(now - start, 3), last_receive_age_s=age,
                          average_callback_hz=round(result["received"] / (now - start), 2))
            print(json.dumps(result), flush=True)
        ok = (result["received"] > result["decode_errors"]
              and result["decode_errors"] == 0 and age is not None and age <= 3)
        print("PASS: valid lowstate received recently." if ok else
              "FAIL: no recent valid lowstate, or message inspection failed.", flush=True)
        return 0 if ok else 1
    finally:
        if subscriber is not None:
            subscriber.Close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped by operator.", flush=True)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        raise SystemExit(2)
