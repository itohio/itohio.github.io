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

In Configurator → **Modes** tab, assign `VTX PIT MODE` and/or `VTX POWER LEVEL n` to AUX channel ranges.

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

Map the wheel to **AUX3** in the transmitter's mixer, then assign the ranges in Configurator → **Modes** tab (not via raw CLI — mode ranges are edited in the GUI or with the numeric `aux` command). Split the 1000–2000 µs wheel travel into non-overlapping bands:

| Wheel band | µs range | Assign |
|-----------|----------|--------|
| LOW | 1000–1333 | `VTX PIT MODE` |
| MID | 1334–1666 | `VTX POWER LEVEL 2` (200 mW) |
| HIGH | 1667–2000 | `VTX POWER LEVEL 4` (600 mW) |

The `VTX POWER LEVEL n` mode conditions handle the stepping — you don't need exact midpoints, just non-overlapping ranges.

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
