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

## Radiomaster Pocket — Lever / Wheel CLI Example

The Radiomaster Pocket has a scrolling wheel (S1) that outputs a smooth 1000–2000 µs range on an AUX channel — ideal for VTX power.

Map wheel to **AUX3** in the transmitter's mixer, then in Betaflight CLI:

```
# Confirm VTX table is set
vtxtable powervalues 25 100 200 400 600
vtxtable powerlabels 25 100 200 400 600

# Map AUX3 (channel 7 internally, 0-indexed as 6) to VTX power
# Betaflight uses 'vtx' resource; power is driven via the mode system
# Example using a 3-band split across the wheel travel:

# LOW band   (1000-1333): pit mode / 25 mW
auxn mode VTX_PIT_MODE range 0 900 1333

# MID band   (1334-1666): 200 mW  (power index 2)
# HIGH band  (1667-2000): 600 mW  (power index 4)

# Save
save
```

For a smooth wheel or a lever, split the 1000–2000 µs range into equal thirds (or halves for two levels). The "POWER LEVEL" mode conditions in the Modes tab handle the stepping — you don't need to set exact midpoints, just non-overlapping ranges.

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
- IRC Tramp protocol is also fully supported; configure via `vtx_type = TRAMP` in CLI.
