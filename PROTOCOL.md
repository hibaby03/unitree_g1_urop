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
