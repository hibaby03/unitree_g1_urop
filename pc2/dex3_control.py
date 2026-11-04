"""Dex3-1 raw command / state / tactile over DDS for the Orin bridge.

unitree_sdk2py only (no dex_retargeting, no xr_teleoperate): the Host does the
hand retargeting and sends joint q; this publishes it, and reads hand q and the
press-sensor tactile back. The command publisher mirrors Dex3_1_Controller's
RIS mode, kp/kd and per-cycle slew limit so motion matches the Host fork.
"""

import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

DEX3_NUM_MOTORS = 7
TOPIC_LEFT_CMD = "rt/dex3/left/cmd"
TOPIC_RIGHT_CMD = "rt/dex3/right/cmd"
TOPIC_LEFT_STATE = "rt/dex3/left/state"
TOPIC_RIGHT_STATE = "rt/dex3/right/state"
DEX3_KP = 1.5
DEX3_KD = 0.2
MAX_DELTA_Q_PER_CYCLE = 0.2  # slew limit, matches Dex3_1_Controller


class Dex3LeftJointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6


class Dex3RightJointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


class _RIS_Mode:
    def __init__(self, id=0, status=0x01, timeout=0):
        self.motor_mode = 0
        self.id = id & 0x0F
        self.status = status & 0x07
        self.timeout = timeout & 0x01

    def to_uint8(self):
        mode = self.id & 0x0F
        mode |= (self.status & 0x07) << 4
        mode |= (self.timeout & 0x01) << 7
        return mode


def press_sensors_to_dict(press_sensor_state):
    pressure, temperature, lost = [], [], []
    for sensor in press_sensor_state:
        pressure.append([float(v) for v in sensor.pressure])
        temperature.append([float(v) for v in sensor.temperature])
        lost.append(int(sensor.lost))
    return {"pressure": pressure, "temperature": temperature, "lost": lost}


class Dex3CommandPublisher:
    """Publish Dex3-1 joint q targets (raw DDS), with the fork's RIS mode/gains."""

    def __init__(self):
        self._pubs, self._msgs, self._indices = [], [], []
        for topic, index in ((TOPIC_LEFT_CMD, Dex3LeftJointIndex),
                             (TOPIC_RIGHT_CMD, Dex3RightJointIndex)):
            publisher = ChannelPublisher(topic, HandCmd_)
            publisher.Init()
            message = unitree_hg_msg_dds__HandCmd_()
            for motor in index:
                message.motor_cmd[motor].mode = _RIS_Mode(id=motor, status=0x01).to_uint8()
                message.motor_cmd[motor].q = 0.0
                message.motor_cmd[motor].dq = 0.0
                message.motor_cmd[motor].tau = 0.0
                message.motor_cmd[motor].kp = DEX3_KP
                message.motor_cmd[motor].kd = DEX3_KD
            self._pubs.append(publisher)
            self._msgs.append(message)
            self._indices.append(index)

    def send(self, left_q, right_q):
        for publisher, message, index, q in zip(
                self._pubs, self._msgs, self._indices, (left_q, right_q)):
            for idx, motor in enumerate(index):
                message.motor_cmd[motor].q = float(q[idx])
            publisher.Write(message)


class Dex3StateReader:
    """Keep the latest Dex3-1 joint q and tactile for both hands (raw DDS)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._q = {"left_ee": None, "right_ee": None}
        self._tactile = {"left_ee": None, "right_ee": None}
        self._subs = []
        for key, topic, index in (("left_ee", TOPIC_LEFT_STATE, Dex3LeftJointIndex),
                                  ("right_ee", TOPIC_RIGHT_STATE, Dex3RightJointIndex)):
            subscriber = ChannelSubscriber(topic, HandState_)
            subscriber.Init(self._make_handler(key, index), 10)
            self._subs.append(subscriber)

    def _make_handler(self, key, index):
        def handler(message):
            try:
                q = [message.motor_state[i].q for i in index]
                tactile = press_sensors_to_dict(message.press_sensor_state)
            except (AttributeError, IndexError, TypeError, ValueError):
                return
            with self._lock:
                self._q[key] = q
                self._tactile[key] = tactile
        return handler

    def joints(self):
        """(left_q, right_q); each is a 7-list or None."""
        with self._lock:
            return self._q["left_ee"], self._q["right_ee"]

    def tactile(self):
        with self._lock:
            return (dict(self._tactile["left_ee"]) if self._tactile["left_ee"] else None,
                    dict(self._tactile["right_ee"]) if self._tactile["right_ee"] else None)

    def close(self):
        for subscriber in self._subs:
            subscriber.Close()
