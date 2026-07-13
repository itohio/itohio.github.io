---
title: "Betaflight Stick Commands"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "stick-commands", "configuration", "cli"]
---

Stick commands let you trigger Betaflight functions from your transmitter sticks without a computer. Useful in the field.

All commands require the quad to be **disarmed**. Throttle is always at its minimum position unless noted.

---

## Stick Positions

```
         PITCH UP
            ↑
YAW LEFT ←     → YAW RIGHT
            ↓
         PITCH DOWN

ROLL maps to: ← LEFT  |  RIGHT →
```

---

## Common Commands

| Action                        | Throttle | Yaw   | Pitch | Roll  |
|-------------------------------|----------|-------|-------|-------|
| Enter CLI / Config mode       | LOW      | LOW   | HIGH  | HIGH  |
| Save config (in CLI mode)     | LOW      | LOW   | HIGH  | LOW   |
| Enter Accelerometer Calibration | LOW    | LOW   | HIGH  | CENTER|
| Enable/Disable Blackbox       | LOW      | LOW   | LOW   | HIGH  |
| Enter OSD menu                | CENTER   | LOW   | CENTER| CENTER (hold) |

> Exact stick positions vary by Betaflight version and whether you use mode 1/2. The table above is for **Mode 2** (throttle left stick).

---

## Entering CLI via Sticks (Field Config)

Hold: **Throttle LOW · Yaw LOW · Pitch HIGH · Roll HIGH** for ~5 seconds.

The FC enters a configuration state. From here you can:
- Access the OSD menu (if configured)
- Trigger calibration

To enter the actual CLI you still need USB + Betaflight Configurator or a Bluetooth/WiFi adapter. Stick-based "config mode" is primarily for OSD menu access.

---

## OSD Menu Access (Most Common Field Use)

With **CONFIGURATOR MSP** mode enabled on a switch, or via stick command:

Hold **Throttle CENTER · Yaw LEFT · Pitch CENTER** for ~3 seconds.

Navigate:
- **Pitch UP/DOWN** — move cursor
- **Roll RIGHT** — select / enter
- **Roll LEFT** — back
- **Yaw LEFT** — exit menu

This lets you change PID profiles, rate profiles, VTX power, and other settings without a phone or laptop.

---

## Notes

- Stick commands only fire reliably with sticks held at exact endpoints (>95% deflection). Calibrate your transmitter endpoints.
- Some commands are guarded behind a double-input sequence in newer Betaflight versions to prevent accidental triggers.
- Arming is typically on **Yaw RIGHT** (or a dedicated arm switch). Never mix arm with config stick positions.

---

## Reference

Full stick command table: [Betaflight Wiki — Stick Commands](https://betaflight.com/docs/wiki/guides/current/stick-commands)
