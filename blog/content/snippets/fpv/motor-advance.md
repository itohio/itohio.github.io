---
title: "Motor Timing Advance — 15° vs 22°"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "esc", "motor", "timing", "advance", "efficiency", "blheli"]
---

Motor timing advance shifts the ESC's commutation point forward in the electrical cycle relative to the motor's rotor position. Higher advance = more power at the cost of efficiency and heat.

---

## What Timing Advance Does

A brushless motor's ESC switches phase power based on rotor position. "On time" — switching exactly when the rotor aligns — is 0° advance. Advancing the switch point anticipates rotor movement, increasing torque in the power band at the expense of running hotter.

Think of it like ignition timing in an internal combustion engine — advance too far and you get knock and heat; set it right and you get peak power; retard it and you lose power but run cool.

---

## 15° vs 22° (Common ESC Settings)

| Setting | Efficiency | Power  | Heat  | Use case                       |
|---------|------------|--------|-------|--------------------------------|
| **15°** | Higher     | Lower  | Lower | Efficiency, long range, cruising|
| **22°** | Moderate   | Higher | Higher| Freestyle, racing, punchy builds|

**15° advance** is more thermally efficient — the ESC switches at a point that minimizes iron losses in the motor stator. Less heat means less energy wasted, translating directly into longer flight times.

**22°** (and higher, up to ~30°) pushes more power and RPM out of the same motor/prop combination, but increases motor and ESC temperature, especially at sustained high throttle.

---

## When to Use 15°

- Long-range builds where flight time matters
- Efficiency-focused 5" cruising setups
- Motors with tight tolerances (oversized stators) that run hot at 22°
- Any build where motor temperature at landing feels high

**Rule of thumb:** if your motors are warm to the touch (can hold finger on > 3 seconds) after a full-throttle punch session, lower timing advance is one lever to reduce heat before reaching for a bigger ESC.

---

## When to Use 22°

- Freestyle and racing where peak power matters more than efficiency
- Builds with adequate cooling (open frames, larger ESCs)
- When you've tested 15° and peak throttle feel is noticeably sluggish

---

## Setting Timing Advance

In BLHeli_32 / AM32 configurator:

| ESC Firmware | Setting Name        | Values                |
|--------------|--------------------|-----------------------|
| BLHeli_32    | Motor Timing       | Low / MedLow / Med / MedHigh / High |
| AM32         | Motor Timing (deg) | Direct degree input   |

BLHeli_32 "Motor Timing" approximate mapping:
- Low ≈ 15°
- MedLow ≈ 18°
- Medium ≈ 22°
- MedHigh ≈ 25°
- High ≈ 30°

For efficiency: select **Low**. For freestyle: **Medium**.

---

## Interaction with RPM and Demag

BLHeli_32 and AM32 also have a **Demag Compensation** setting. Demag handles the back-EMF spike when a phase switches off. Higher demag compensation can help prevent desync at high RPM changes.

If you run 15° timing for efficiency, pair it with normal or moderate demag settings. Aggressive demag with low timing can occasionally cause hesitation on fast throttle-up.

---

## Notes

- Timing advance interacts with motor KV and prop load. A high-KV motor on a small prop hitting peak RPM frequently benefits less from advance timing than a mid-KV motor on a larger prop.
- Always run a full-throttle punch session and check motor temps after changing timing. Catch thermal issues before they cook motor windings.
