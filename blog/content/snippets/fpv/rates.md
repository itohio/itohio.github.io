---
title: "Betaflight Rates Explained"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "tuning", "freestyle"]
---

Rates control how fast the drone rotates in response to stick input. Higher rates = faster rotation = more aggressive feel. Three numbers describe a rate profile in shorthand: **RC Rate · Super Rate · Expo** (or equivalent in other rate styles).

---

## Common Shorthand Profiles

| Profile   | Feel             | Use case                              |
|-----------|------------------|---------------------------------------|
| **733**   | Fast & snappy    | Freestyle, flips, experienced pilots  |
| **633**   | Medium-fast      | Freestyle / racing crossover          |
| **533**   | Moderate         | Cruising, cinematic, learning tricks  |

The three digits map to **RC Rate · Super Rate · Expo** in Betaflight's default *Betaflight* rate style.

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

Example — profile **733** (RC Rate 0.7, SR 0.33, Expo 0.3):
```
maxRate = (0.7 × 200) × (1 / (1 - 0.33))
        = 140 × 1.49
        ≈ 209 °/s ... but full-stick with expo lands near 600 °/s
```

> Betaflight's actual calculation includes expo shaping; the formula is an approximation. Use the Rates Preview tab in Betaflight Configurator to see the real curve.

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

- Start with **533** when learning; the center softness (Expo 0.3) keeps hover precise.
- Move to **633** or **733** when you want to throw the quad into snappy flips without full-deflection slow-down.
- Expo above 0.5 feels "laggy" at center for experienced pilots — keep it ≤ 0.35 for freestyle.
- Rates are per-axis. Most pilots copy the same profile across Roll, Pitch, Yaw — but yaw is often set slightly lower for cleaner spins.

---

## CLI

```
# View current rates
rates

# Set Betaflight style rates on Roll axis
set roll_rc_rate = 70
set roll_super_rate = 33
set roll_expo = 30

# Copy same values to Pitch
set pitch_rc_rate = 70
set pitch_super_rate = 33
set pitch_expo = 30

# Lower yaw slightly
set yaw_rc_rate = 60
set yaw_super_rate = 30
set yaw_expo = 25

save
```
