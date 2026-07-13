---
title: "LiPo Battery Basics — C Rating, Cell Count, and Care"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "battery", "lipo", "c-rating", "cell-count", "storage", "safety"]
---

LiPo batteries are the most maintenance-sensitive component in an FPV build. Misuse kills them fast or causes fires.

---

## Cell Voltage

| State          | Voltage per cell |
|----------------|-----------------|
| Fully charged  | 4.20 V           |
| Nominal        | 3.70 V           |
| Storage        | 3.80–3.85 V      |
| Low cutoff     | 3.50 V           |
| Dead / damaged | < 3.30 V         |

**Never discharge below 3.5 V per cell** under load. Set your FC low-battery warning to trigger at ~3.5–3.6 V per cell (voltage under load, not at rest).

---

## Cell Count

| Config | Nominal | Max   | Typical use              |
|--------|---------|-------|--------------------------|
| 1S     | 3.7 V   | 4.2 V | Tiny whoops              |
| 2S     | 7.4 V   | 8.4 V | 2.5" micros              |
| 3S     | 11.1 V  | 12.6 V| 3" micros                |
| 4S     | 14.8 V  | 16.8 V| 5" standard              |
| 6S     | 22.2 V  | 25.2 V| 5" high performance, 7"+ |

---

## C Rating

C rating is the discharge rate multiplier relative to capacity.

```
Max continuous current (A) = Capacity (Ah) × C rating
```

Example — 1500 mAh, 100C battery:
```
Max current = 1.5 Ah × 100 = 150 A
```

**The C rating is marketing** on most budget packs. Treat it as a rough guide. A real-world usable rule:

- 5" freestyle quad: aim for **1500–2200 mAh, 80C+** on 4S
- 5" efficiency: 2200–3000 mAh, 50–80C on 4S or 6S
- Racing: 650–1300 mAh, 100C+ on 4S–6S

If your pack sags heavily under throttle punch (voltage drops 1+ V per cell), C rating is insufficient for your build's current draw.

---

## Storage Charge

Always store LiPo batteries at **3.80–3.85 V per cell**. Never store fully charged or fully depleted.

- Fully charged storage accelerates capacity loss and causes puffing.
- Fully depleted storage risks cell damage from continued self-discharge below minimum voltage.

Most chargers have a **Storage** charge mode — use it if you won't fly within 24–48 hours.

---

## Puffing

A puffed battery has internal gas buildup — a sign of cell damage, usually from overdischarge, overcharge, or high current stress.

- Minor puff: still usable but monitor closely. Retire soon.
- Significant puff: retire immediately. Do not charge.
- Dispose of at an electronics recycling facility — discharge fully in a salt water bucket first.

---

## Safe Handling

- Charge on a **LiPo-safe bag or fireproof surface** — never unattended.
- Never charge a damaged, puffed, or crashed pack without inspection.
- Charge at **1C** as the default (i.e., a 1500 mAh pack → 1.5 A charge rate). Higher rates reduce cycle life.
- Never leave a charging LiPo unattended.

---

## Battery Life

Expect 150–300 cycles from a quality pack if treated well. Indicators of end-of-life:
- Significant capacity loss under load
- Increased internal resistance (measure with your charger)
- Consistent puffing after flights
- Voltage sag exceeds 0.5 V/cell under load at moderate throttle
