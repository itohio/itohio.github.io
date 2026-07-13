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

## Motor Load Index (Thrust-to-Weight)

A useful sanity-check at hover: each motor carries a quarter of the all-up weight.

**Hover thrust per motor:**
```
Hover thrust per motor = AUW / 4        (for a quad)
```

For a ~500–700 g 5" quad that is ~125–175 g per motor, which usually lands around 40–50% throttle on a healthy build.

**Thrust-to-weight ratio (TWR)** compares *total full-throttle* thrust to AUW:
```
TWR = (4 × max thrust per motor) / AUW
```

A TWR of 4:1 is typical for freestyle, 3:1 is fine for cinematic, and racing wants 6:1+. A 4:1 quad lifts its own weight using only a quarter of its available thrust — the rest is headroom for punch-outs.

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

- Frame: 5" (0.127 m), battery: 4S (16.8 V full charge)
- Target full-throttle *loaded* RPM ≈ 24,000–28,000 RPM

Work backwards from the loaded max RPM to KV (loaded ≈ 75% of no-load `KV × Voltage`):
```
KV = Max loaded RPM ÷ (Voltage × 0.75)
KV = 26,000 ÷ (16.8 × 0.75) = 26,000 ÷ 12.6 ≈ 2,060 KV
```
→ Tip speed at that loaded max: `π × 0.127 × 26,000 / 60 ≈ 173 m/s`. That is above the 150 m/s efficient ceiling — normal for freestyle, which trades some efficiency for punch (racing runs higher still).

In practice, **2000–2450 KV** motors are the established sweet spot for 5" on 4S, matching the quick-reference table above. (A designation like "2306" is the *stator size* — 23 mm × 6 mm — not a KV value.)

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
