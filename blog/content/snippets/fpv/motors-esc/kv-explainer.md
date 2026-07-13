---
title: "Motor KV Explained"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "motor", "kv", "voltage", "rpm", "efficiency", "selection"]
---

KV is the most misunderstood motor specification in FPV. It is **not** a measure of motor quality or power — it is a ratio that determines how much RPM you get per volt.

---

## Definition

**KV = RPM per Volt (unloaded)**

A motor rated 2400 KV spinning at 16.8 V (4S fully charged) will spin at:

```
RPM = KV × Voltage
RPM = 2400 × 16.8 ≈ 40,320 RPM (no load)
```

Under load (with a prop), RPM will be lower — typically 65–85% of no-load RPM depending on prop size and throttle.

---

## What KV Affects

| Higher KV               | Lower KV                    |
|-------------------------|-----------------------------|
| Higher RPM              | Lower RPM                   |
| More suited to small props | More suited to large props |
| Less torque             | More torque                 |
| Runs hotter under load  | Runs cooler under load      |
| Needs lower voltage (2S, 3S) | Needs higher voltage (4S, 6S) |

The stator windings determine KV: fewer, thicker windings = higher KV; more, thinner windings = lower KV.

---

## Voltage and KV Together

KV is meaningless without knowing the operating voltage. The same RPM can be achieved with different KV × Voltage combinations:

```
2400 KV × 14.8 V (4S nominal) = 35,520 RPM
1750 KV × 22.2 V (6S nominal) = 38,850 RPM
```

Both setups deliver similar RPM — but the 6S motor has more torque available to turn a larger prop efficiently.

---

## Practical KV Guide by Frame Size

| Frame size | Common voltage | Typical KV range | Prop size      |
|------------|---------------|-----------------|----------------|
| 1" whoop   | 1S (3.7 V)    | 15,000–20,000 KV| 31 mm          |
| 2.5" micro | 2S–3S         | 5,000–8,000 KV  | 2.5"           |
| 3" toothpick | 3S–4S       | 3,000–5,000 KV  | 3"             |
| 5" freestyle | 4S–6S       | 1,700–2,500 KV  | 5"             |
| 7" long range| 4S–6S       | 1,300–1,800 KV  | 7"             |
| 10" cinematic| 4S–6S       | 700–1,000 KV    | 9"–10"         |

---

## KV Calculator

**Target RPM from KV and voltage:**
```
RPM = KV × Voltage
```

**KV needed for a target RPM at a known voltage:**
```
KV = Target RPM ÷ Voltage
```

**Loaded RPM estimate (approximate):**
```
Loaded RPM ≈ KV × Voltage × 0.75
```

Example: You want ~25,000 loaded RPM on 4S (14.8 V nominal):
```
KV ≈ 25,000 ÷ (14.8 × 0.75) ≈ 25,000 ÷ 11.1 ≈ 2,250 KV
```
→ A 2300–2450 KV motor on 4S is a reasonable target.

---

## KV and Efficiency

Lower KV motors running at higher voltage are generally more efficient for the same power output. The motor's copper windings have the same resistance regardless of KV, but higher-voltage lower-current operation reduces I²R losses (heat in the windings).

This is why 6S builds on a 5" frame achieve better flight times despite similar physical size — the lower KV motor on 6S runs cooler and wastes less energy as heat.

---

## Notes

- KV is a no-load spec. Real-world RPM depends on prop pitch, prop diameter, throttle, and air density.
- Stator dimensions (e.g., 2306 = 23 mm diameter × 6 mm height) determine torque capacity and heat dissipation, independent of KV.
- Two motors with the same stator and same KV from different manufacturers can behave very differently — winding quality, magnet strength, and bearing precision all matter.
