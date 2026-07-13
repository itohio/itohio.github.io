---
title: "KV and Prop Matching — Tip Speed, Load, and Selection"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "motor", "kv", "props", "tip-speed", "efficiency", "thrust", "matching"]
---

Matching motor KV to prop size is the single most important factor in build efficiency and motor longevity. The goal: keep tip speed in the efficient range and keep motor temperature reasonable.

---

## Prop Tip Speed

Propeller tip speed is the velocity at which the blade tip moves through the air. As tip speed approaches the speed of sound (~343 m/s), efficiency drops sharply and noise increases dramatically.

**Practical efficient range: 100–150 m/s tip speed at hover/cruise throttle.**

### Calculation

```
Tip Speed (m/s) = (π × Prop Diameter [m] × RPM) / 60
```

Example — 5" prop (0.127 m diameter) at 20,000 RPM:
```
Tip Speed = (π × 0.127 × 20,000) / 60
           = (3.1416 × 0.127 × 20,000) / 60
           ≈ 7,980 / 60
           ≈ 133 m/s
```
→ 133 m/s is within the efficient range.

---

## Quick Reference Table

| Prop diameter | Max efficient RPM (150 m/s) | Typical KV on 4S (14.8V) |
|---------------|----------------------------|--------------------------|
| 3" (76 mm)    | ~37,600 RPM                | ~3,500–4,500 KV          |
| 4" (102 mm)   | ~28,100 RPM                | ~2,600–3,200 KV          |
| 5" (127 mm)   | ~22,500 RPM                | ~2,000–2,500 KV          |
| 5" (127 mm)   | ~22,500 RPM                | ~1,500–1,800 KV on 6S    |
| 7" (178 mm)   | ~16,100 RPM                | ~1,300–1,600 KV on 4S–6S |
| 10" (254 mm)  | ~11,300 RPM                | ~700–900 KV on 6S        |

---

## Motor Load Index (Thrust-to-Power Efficiency)

A useful sanity-check: at hover throttle (~50%), the motor should produce ~250–350 g of thrust per motor for a ~500–700 g all-up-weight (AUW) 5" quad.

**Hover thrust per motor:**
```
Hover thrust = AUW / 4        (for a quad)
Required thrust-to-weight per motor = AUW / 4 / motor_weight
```

A **thrust-to-weight ratio (TWR)** of 4:1 (total thrust: AUW) at 50% throttle is typical for freestyle. Cinematic: 3:1 is fine. Racing: 6:1+.

---

## Prop Pitch and Motor Selection

Pitch is the theoretical distance a prop advances per revolution. Higher pitch = more aggressive bite = more speed but more drag = motor works harder.

| Use case         | Pitch recommendation         |
|------------------|------------------------------|
| Efficiency / long range | Low pitch (3.8"–4.3") |
| Freestyle        | Moderate (4.8"–5.1")         |
| Racing / top speed | Higher pitch (5.1"–6.0")  |

**Higher pitch requires more torque → lower KV motor on higher voltage** for the same efficiency.

---

## Matching Workflow

1. **Choose frame size** → sets prop diameter range
2. **Choose battery voltage** → sets voltage input to motor
3. **Choose use case** → sets target RPM range and pitch preference
4. **Calculate required KV:**
   ```
   KV = Target Cruise RPM ÷ (Voltage × 0.75)
   ```
5. **Verify tip speed** at estimated max RPM:
   ```
   Max RPM = KV × Max Voltage
   Tip Speed = (π × Diameter_m × MaxRPM) / 60
   ```
   Should stay below ~170 m/s; ideally below 150 m/s.
6. **Check stator size** — larger stator (e.g. 2306 vs 2204) handles more heat at a given KV, so heavier prop combinations need a bigger stator.

---

## Worked Example — 5" Freestyle 4S

- Frame: 5", battery: 4S (16.8 V max)
- Target hover RPM ≈ 16,000–18,000 RPM (50% throttle)
- Max RPM ≈ 24,000–28,000 RPM (full throttle)

```
KV = 17,000 ÷ (16.8 × 0.75) = 17,000 ÷ 12.6 ≈ 1,350 KV
```
→ But 1,350 KV × 16.8 V = 22,680 RPM, which maps to 150 m/s tip speed on 5" prop. Good.

In practice, **2306–2450 KV** motors are the established sweet spot for 5" on 4S because manufacturers tune stator mass and winding inductance for this application.

---

## Prop Selection Rules of Thumb

- **Diameter** — determined by frame arm length and motor mount spacing. Don't exceed frame limits.
- **Pitch** — higher pitch for speed and responsiveness; lower pitch for efficiency and hover time.
- **Blade count** — 3-blade: efficiency and handling balance. 4/5-blade: more thrust in same diameter, more noise and less efficiency.
- **Tip speed** — always check your calculated tip speed at max throttle. If it exceeds 180 m/s you're leaving efficiency on the table and generating unnecessary noise.

---

## Notes

- These calculations give theoretical RPM. Real loaded RPM with a prop is typically 70–80% of no-load KV × voltage.
- Motor thrust tables from manufacturers are measured at specific voltages on specific props — always cross-reference for your exact combination.
- Small differences in prop manufacturer (HQ, Gemfan, DAL) affect actual RPM, thrust, and efficiency even on nominally identical props.
