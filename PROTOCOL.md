# Active-camera UDP protocol v1

Each UDP datagram contains exactly one 48-byte head-pose packet. All multi-byte
values use network byte order (big endian).

| Offset | Size | Type | Name | Description |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | bytes | magic | ASCII `ACAM` |
| 4 | 1 | uint8 | version | `1` |
| 5 | 1 | uint8 | flags | Validity and control flags |
| 6 | 2 | uint16 | reserved | Must be zero |
| 8 | 4 | uint32 | sequence | Increments once per sent pose |
| 12 | 8 | uint64 | sender_time_ns | Host Unix epoch time in nanoseconds |
| 20 | 12 | 3 x float32 | position | OpenXR x/y/z position in metres |
| 32 | 16 | 4 x float32 | orientation | OpenXR quaternion x/y/z/w |

The corresponding Python format string is:

```text
!4sBBHIQ3f4f
```

## Flags

| Bit | Hex | Meaning |
| ---: | ---: | --- |
| 0 | `0x01` | Orientation is valid |
| 1 | `0x02` | Position is valid |
| 2 | `0x04` | Pose is actively tracked |
| 3 | `0x08` | Recenter request |

The receiver requires both orientation-valid and tracked bits before updating
the target. Position is transported for future translational camera stages but
is intentionally ignored by the current 2-DoF yaw/pitch receiver.

## Coordinates

Packets use the native OpenXR right-handed convention:

- +X: right
- +Y: up
- -Z: forward

Quaternion order is `(x, y, z, w)`. The receiver normalizes accepted
quaternions and rejects non-finite or significantly non-unit values.

`sender_time_ns` is useful for logging only when Host and PC2 wall clocks are
synchronized. Packet ordering always uses `sequence`, not timestamps.

# Neck state packet v1 (PC2 → Host)

`pc2/head_pose_receiver.py` sends one 60-byte datagram after each accepted
head pose, to the head-pose sender's address on UDP `5006` by default
(`--neck-state-host`, `--neck-state-port`, `--no-neck-state`). The Host
decoder is `host/neck_state_receiver.py`. Network byte order.

| Offset | Size | Type | Name | Description |
| ---: | ---: | --- | --- | --- |
| 0 | 4 | bytes | magic | ASCII `ANCK` |
| 4 | 1 | uint8 | version | `1` |
| 5 | 1 | uint8 | flags | See below |
| 6 | 2 | uint16 | reserved | Must be zero |
| 8 | 4 | uint32 | pose_sequence | `sequence` of the head-pose packet that produced this command |
| 12 | 8 | uint64 | command_pc2_monotonic_ns | PC2 `time.monotonic_ns()` just before the Sync Write (pose accept time if monitor-only) |
| 20 | 8 | uint64 | present_pc2_monotonic_ns | PC2 `time.monotonic_ns()` just before the encoder Sync Read |
| 28 | 8 | 2 x float32 | command_deg | Commanded yaw, pitch in degrees |
| 36 | 8 | 2 x float32 | present_deg | Encoder yaw, pitch in degrees |
| 44 | 8 | 2 x int32 | goal_position | Goal counts written to yaw, pitch motors |
| 52 | 8 | 2 x int32 | present_position | Present Position counts read from yaw, pitch motors |

Python format string: `!4sBBHIQQ2f2f2i2i`

| Bit | Hex | Meaning |
| ---: | ---: | --- |
| 0 | `0x01` | Command is valid (always set) |
| 1 | `0x02` | Present time/angles/positions are valid |
| 2 | `0x04` | Motor torque is enabled; goal_position is valid |

Angles use the same convention as the commands: yaw is right positive and
pitch is up positive, `angle = (count - center) × sign × 360 / 4096`. Invalid
fields are sent as zero. Both timestamps are in the PC2 monotonic domain and
must not be compared with Host clocks.
