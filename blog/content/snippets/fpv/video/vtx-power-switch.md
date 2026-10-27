---
title: "VTX Power Control via Transmitter"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "vtx", "power", "switch", "aux", "cli"]
---

Map a transmitter lever, wheel, or switch to cycle through VTX power levels in Betaflight. Useful for pit mode on startup, low power for close proximity flying, and full power for long range.

---

## How It Works

Betaflight reads an AUX channel and maps PWM ranges to VTX power table indices. The VTX must be connected to the FC via SmartAudio, IRC Tramp, or MSP VTX — direct control with no FC passthrough won't work.

---

## VTX Power Table

Define allowed power levels in the VTX table. Betaflight Configurator → **Video Transmitter** tab → edit the power levels to match what your VTX supports.

Via CLI:
```
vtxtable powervalues 25 100 200 400 600
vtxtable powerlabels 25 100 200 400 600
```

---

## Mode Condition Setup

Power-from-switch is configured in Configurator → **VTX** tab (saved as `vtx` CLI lines); pit mode uses the `VTX PIT MODE` mode in the **Modes** tab.

For a 3-position switch cycling through 25 / 200 / 600 mW:
```
# AUX2 range mapping example (1000-2000 µs typical)
# Position LOW  (1000-1300) → pit / 25 mW
# Position MID  (1300-1700) → 200 mW
# Position HIGH (1700-2000) → 600 mW
```

---

## Radiomaster Pocket — Lever / Wheel Example

The Radiomaster Pocket has a scrolling wheel (S1) that outputs a smooth 1000–2000 µs range on an AUX channel — ideal for VTX power.

First confirm the VTX power table in the CLI (these are real commands):

```
vtxtable powervalues 25 100 200 400 600
vtxtable powerlabels 25 100 200 400 600
save
```

Map the wheel to **AUX3** in the transmitter's mixer, then add the bands in Configurator → **VTX** tab, or with `vtx <slot> <aux> <band> <channel> <power> <start> <end>` lines (AUX is 0-based, so AUX3 = 2; band/channel/power 0 = don't change; ranges in 25 µs steps; power is the 1-based `vtxtable` level). Split the 1000–2000 µs wheel travel into non-overlapping bands:

| Wheel band | µs range | Assign |
|-----------|----------|--------|
| LOW | 1000–1325 | `VTX PIT MODE` (Modes tab) |
| MID | 1350–1650 | `vtx 0 2 0 0 3 1350 1650` (level 3 = 200 mW) |
| HIGH | 1675–2000 | `vtx 1 2 0 0 5 1675 2000` (level 5 = 600 mW) |

Outside every band the last selection stays; you don't need exact midpoints, just non-overlapping ranges.

---

## Pit Mode on Arm / Disarm

Keep the wheel/lever at minimum (pit mode) while in the pit. The VTX will drop to minimum power (or RF off, if your VTX supports it) until you push the lever up. This avoids blasting other pilots' goggles at full power while you're still on the bench.

---

## Verify

After setup, check in the OSD or Betaflight Configurator → Video Transmitter tab that the power level changes as you move the wheel. Verify with a signal meter or frequency scanner if available.

---

## Notes

- **SmartAudio v2.1+** is required for reliable power switching in flight. Earlier versions may need a save/reboot cycle.
- Some VTX units require a reboot to apply power changes — test before relying on it in the field.
- IRC Tramp is also fully supported; select it as the VTX peripheral in Configurator → **Ports** tab (VTX (Tramp)) on the UART wired to the VTX, rather than any `set` variable.
