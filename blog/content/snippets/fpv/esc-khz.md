---
title: "ESC PWM Frequency — 24 vs 48 vs 96 kHz"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "esc", "betaflight", "blheli", "dshot", "pwm", "frequency", "tune"]
---

ESC PWM frequency controls how often the ESC updates the power delivered to the motor. Higher frequency = smoother power delivery, lower frequency = less heat but coarser control.

---

## What PWM Frequency Does

The ESC uses PWM to modulate the voltage applied to each motor phase. The frequency determines how many times per second this switching happens.

- **24 kHz** — coarser switching, less switching loss, lower ESC temperature
- **48 kHz** — balanced; default on most modern ESCs; smooth control, moderate heat
- **96 kHz** — finest switching, best motor smoothness, higher ESC heat

Higher frequency reduces the "buzzing" feeling in the motors at hover and gives the flight controller more granular motor control, which helps filtering and PID stability.

---

## Trade-offs

| Frequency | Smoothness | ESC heat | Motor efficiency | Desync risk |
|-----------|------------|----------|-----------------|-------------|
| 24 kHz    | Lower      | Low      | Slightly higher  | Lower       |
| 48 kHz    | Good       | Moderate | Good             | Low         |
| 96 kHz    | Best       | Higher   | Slightly lower   | Higher (some ESCs) |

- **24 kHz** is preferred for long-range / efficiency builds where flight time matters and the tune is conservative.
- **48 kHz** is the default for most freestyle and 5" racing quads.
- **96 kHz** is popular for 5" freestyle where the smoother motor response helps with propwash, but verify your ESC supports it (not all BLHeli_32 ESCs are stable at 96 kHz at full throttle).

---

## Setting PWM Frequency

### Via BLHeli Configurator / ESC-Configurator

Connect via passthrough or direct USB, open ESC-Configurator or BLHeli Configurator, and set **PWM Frequency** per ESC (or all at once).

### Via Betaflight CLI (BLHeli_32 with telemetry)

Some setups allow setting via Betaflight:
```
# Check current ESC settings (BLHeli_32 with telemetry)
# Usually done through BLHeli Suite / ESC Configurator, not BF CLI
```

### Via AM32 ESC

AM32 (open-source alternative to BLHeli_32) exposes PWM frequency directly:
```
# In AM32 configurator
# Set "Input PWM Frequency" to 24 / 48 / 96 kHz
```

---

## With RPM Filter (DSHOT Bidirectional)

If you're using the RPM filter, the ESC must support bidirectional DSHOT. Most BLHeli_32 ESCs do.

**RPM filter + 48 kHz** is the standard pairing for modern freestyle builds. Going to 96 kHz with RPM filter gives marginal additional improvement and increases ESC heat — not always worth it.

In Betaflight CLI:
```
set dshot_bidir = ON
set motor_pwm_protocol = DSHOT300  # or DSHOT600
save
```

After enabling bidirectional DSHOT, verify in the Motors tab that RPM telemetry is visible (each motor shows RPM readout).

---

## Notes

- Some ESCs claim 96 kHz support but become unstable under heavy load. Test with aggressive throttle punches before flying over anything expensive.
- Changing PWM frequency requires a power cycle on most ESCs.
- PWM frequency is separate from DSHOT protocol — DSHOT is a digital protocol; the PWM frequency setting affects the analog phase switching on the motor windings.
