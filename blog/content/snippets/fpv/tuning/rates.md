---
title: "Betaflight Rates Explained"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "tuning", "freestyle"]
---

Rates control how fast the drone rotates in response to stick input. Higher rates = faster rotation = more aggressive feel. A rate profile is shaped by three parameters — **RC Rate**, **Super Rate**, and **Expo** — plus the choice of rate *system* (Betaflight, Actual, etc.).

---

## Common Shorthand Profiles

The `533 / 633 / 733` shorthand names a profile by its **maximum rotation rate in °/s** — `733` peaks near 730 °/s, `533` near 530 °/s:

| Profile   | Max rate  | Feel             | Use case                              |
|-----------|-----------|------------------|---------------------------------------|
| **733**   | ~733 °/s  | Punchy           | Freestyle, flips, confident pilots    |
| **633**   | ~633 °/s  | Balanced         | General freestyle / cruising          |
| **533**   | ~533 °/s  | Mellow           | Learning acro, racing lines, cinematic |

All three use a fixed **Super Rate 0.70** with RC Rate stepped up (0.80 / 0.95 / 1.10). For the exact values in both rate systems, with copy-paste CLI and graphs, see [Rate Presets](../rate-presets/).

---

## What Each Parameter Does

**RC Rate** — base sensitivity at center stick. Higher values make the whole stick feel more responsive.

**Super Rate** — adds extra rotation speed toward the stick edges. This is the "high end" speed you hit at full deflection. Range: 0.0 – 1.0 (0 = linear, higher = more curve toward edges).

**Expo** — softens the center stick region, giving finer control around hover/center while keeping full deflection fast. Range: 0.0 – 1.0.

---

## Max Rotation Rate Formula

For the default Betaflight rate style:

```
maxRate = (RC Rate × 200) × (1 / (1 - Super Rate))
```

Example — profile **733** (RC Rate 1.1, Super Rate 0.7):
```
maxRate = (1.1 × 200) × (1 / (1 - 0.7))
        = 220 × 3.33
        ≈ 733 °/s
```

This is why the shorthand number *is* the max rate: with Super Rate fixed at 0.70 the denominator is 0.30, so `maxRate = 666.7 × RC Rate`.

> **Expo does not change the maximum.** Expo only softens the curve between center and full stick; at full deflection the output returns to the same `maxRate`. To go faster you raise RC Rate or Super Rate, not Expo. Use the Rates Preview tab in Betaflight Configurator to see the real curve for any values.

---

## Rate Styles

Betaflight supports multiple rate styles — each uses the same three sliders but applies them differently:

| Style        | Characteristic                                     |
|--------------|----------------------------------------------------|
| Betaflight   | Default. Predictable, widely used.                 |
| Actual       | RC Rate maps directly to max °/s. Intuitive.       |
| Quickrates   | Simplified two-parameter curve.                    |
| Kiss         | Classic Kiss FC style.                             |

**Actual Rates** are increasingly popular because RC Rate directly equals max rotation speed in °/s — no math needed.

---

## Practical Tips

- Start with **533** when learning; the lower max rate keeps rotations slow and controllable.
- Move to **633** or **733** when you want faster flips and a livelier center.
- Add a little **Expo** (0.10–0.20) if the center feels twitchy — it softens fine control without changing the max rate. Above ~0.5 it starts to feel "laggy" at center.
- Rates are per-axis. Most pilots copy the same profile across Roll, Pitch, Yaw — but yaw is often set slightly lower for cleaner spins.

---

## CLI

```
# View current rates
rates

# Set a 733 profile (legacy Betaflight style) on Roll
set roll_rc_rate = 110
set roll_srate = 70
set roll_expo = 0

# Copy same values to Pitch
set pitch_rc_rate = 110
set pitch_srate = 70
set pitch_expo = 0

# Lower yaw slightly for cleaner spins
set yaw_rc_rate = 95
set yaw_srate = 70
set yaw_expo = 0

save
```
