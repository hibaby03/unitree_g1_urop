import unittest
from types import SimpleNamespace

from host.teleop_record_dex3_tactile import (
    TOPIC_DEX3_LEFT_STATE,
    TOPIC_DEX3_RIGHT_STATE,
    Dex3TactileSubscriber,
    KeyState,
    build_states_actions,
    parse_args,
    press_sensors_to_dict,
)


def make_sensor(offset, lost=0):
    return SimpleNamespace(
        pressure=[offset + i for i in range(12)],
        temperature=[30.0 + i for i in range(12)],
        lost=lost,
    )


class FakeSubscriber:
    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.handler = None
        self.queue_len = None
        self.closed = False

    def Init(self, handler=None, queueLen=0):
        self.handler = handler
        self.queue_len = queueLen

    def Close(self):
        self.closed = True


class PressSensorTests(unittest.TestCase):
    def test_converts_every_sensor(self):
        result = press_sensors_to_dict([make_sensor(0), make_sensor(100, lost=2)])

        self.assertEqual(len(result["pressure"]), 2)
        self.assertEqual(result["pressure"][1][:3], [100.0, 101.0, 102.0])
        self.assertEqual(len(result["temperature"][0]), 12)
        self.assertEqual(result["lost"], [0, 2])

    def test_subscriber_reads_both_hands(self):
        subscribers = []

        def factory(topic, message_type):
            subscriber = FakeSubscriber(topic, message_type)
            subscribers.append(subscriber)
            return subscriber

        tactile = Dex3TactileSubscriber(factory, "HandState_")
        self.assertEqual([s.topic for s in subscribers],
                         [TOPIC_DEX3_LEFT_STATE, TOPIC_DEX3_RIGHT_STATE])
        self.assertTrue(all(s.handler is not None and s.queue_len > 0 for s in subscribers))
        self.assertEqual(tactile.snapshot(), {"left_ee": None, "right_ee": None})

        # Only the right hand reports: the left must stay None, not block it.
        subscribers[1].handler(SimpleNamespace(press_sensor_state=[make_sensor(2)]))
        snapshot = tactile.snapshot()
        self.assertIsNone(snapshot["left_ee"])
        self.assertEqual(snapshot["right_ee"]["pressure"][0][0], 2.0)

        subscribers[0].handler(SimpleNamespace(press_sensor_state=[make_sensor(1)]))
        snapshot = tactile.snapshot()
        tactile.close()
        self.assertTrue(all(s.closed for s in subscribers))

        self.assertEqual(snapshot["left_ee"]["pressure"][0][0], 1.0)
        self.assertEqual(snapshot["right_ee"]["pressure"][0][0], 2.0)
        self.assertGreaterEqual(snapshot["left_ee"]["age_ms"], 0.0)


class StatesActionsTests(unittest.TestCase):
    def test_g1_29_dex3_layout(self):
        arm_q = [float(i) for i in range(14)]
        arm_dq = [0.1 * i for i in range(14)]
        sol_q = [100.0 + i for i in range(14)]
        sol_tauff = [-float(i) for i in range(14)]
        hand_state = [200.0 + i for i in range(14)]
        hand_action = [300.0 + i for i in range(14)]

        states, actions = build_states_actions(arm_q, arm_dq, sol_q, sol_tauff,
                                               hand_state, hand_action)

        self.assertEqual(states["left_arm"]["qpos"], arm_q[:7])
        self.assertEqual(states["right_arm"]["qpos"], arm_q[7:])
        self.assertEqual(states["right_arm"]["qvel"], arm_dq[7:])
        self.assertEqual(actions["left_arm"]["qpos"], sol_q[:7])
        self.assertEqual(actions["right_arm"]["torque"], sol_tauff[7:])
        self.assertEqual(states["left_ee"]["qpos"], hand_state[:7])
        self.assertEqual(actions["right_ee"]["qpos"], hand_action[7:])
        self.assertEqual(set(states), {"left_arm", "right_arm", "left_ee", "right_ee", "body"})

    def test_wrong_hand_size_is_rejected(self):
        with self.assertRaises(ValueError):
            build_states_actions([0.0] * 14, [0.0] * 14, [0.0] * 14, [0.0] * 14,
                                 [0.0] * 12, [0.0] * 14)


class KeyAndArgumentTests(unittest.TestCase):
    def test_record_toggle_requires_start(self):
        keys = KeyState()
        keys.on_press("s")
        self.assertFalse(keys.record_toggle)
        keys.on_press("r")
        keys.on_press("s")
        self.assertTrue(keys.record_toggle)
        keys.on_press("q")
        self.assertTrue(keys.stop)

    def test_defaults(self):
        args = parse_args(["--record"])
        self.assertTrue(args.record)
        self.assertEqual(args.neck_pose_port, 5005)
        self.assertEqual(args.frequency, 30.0)


if __name__ == "__main__":
    unittest.main()
