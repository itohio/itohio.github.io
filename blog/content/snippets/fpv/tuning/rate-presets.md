---
title: "Rate Presets — 733 / 633 / 533 in Betaflight & Actual"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "actual-rates", "presets", "cli", "tuning", "733", "633", "533"]
---

`533`, `633`, `733` are community shorthand for three popular freestyle rate profiles, and the number **is the maximum rotation rate in °/s** — `733` tops out around 730 °/s, `533` around 530 °/s. They're all built the same way in the legacy Betaflight system: a fixed **Super Rate 0.70** with the **RC Rate** stepped up. This snippet gives each in both rate systems (legacy Betaflight *and* Actual) with copy-paste CLI and the real curve. For the formulas behind the systems see [Rate Modes](../rate-modes/); for the center/mid/edge zones see the [Rates Deep Dive](../rates-deep-dive/).

---

## Why the number is the max rate

In the legacy Betaflight system the ceiling at full stick is:

```
maxRate = 200 × RC Rate / (1 − Super Rate)
```

With Super Rate fixed at 0.70, the denominator is `0.30`, so `maxRate = 666.7 × RC Rate`:

| Profile | RC Rate | Super Rate | 200 × RC ÷ (1 − SR) | Max rate |
|---------|---------|-----------|---------------------|----------|
| **533** | 0.80    | 0.70      | 200 × 0.80 ÷ 0.30   | **533 °/s** |
| **633** | 0.95    | 0.70      | 200 × 0.95 ÷ 0.30   | **633 °/s** |
| **733** | 1.10    | 0.70      | 200 × 1.10 ÷ 0.30   | **733 °/s** |

Raising RC Rate lifts the whole curve (and the ceiling); the shared Super Rate 0.70 gives them the same "soft-ish center, hard edge" shape. Expo is **not** part of the shorthand — it's a personal add-on (commonly 0.10–0.20) that softens the center without changing the max.

> These are honest freestyle rates. For reference: racing tends to run flatter ~550–650 °/s, general freestyle ~650–900 °/s, and aggressive/pro freestyle 900–1200 °/s. `533` is the mellow end, `733` is solidly punchy.

---

## At a glance

| Profile | Feel                    | Legacy BF (RC · SR · Expo) | Actual (center / max / expo) | Max rate | Use case                          |
|---------|-------------------------|----------------------------|------------------------------|----------|-----------------------------------|
| **733** | Punchy, fast flips      | `1.10 · 0.70 · 0`          | `220 / 730 / 55`             | ~733 °/s | Confident freestyle, snappy tricks |
| **633** | Balanced all-rounder    | `0.95 · 0.70 · 0`          | `190 / 630 / 55`             | ~633 °/s | General freestyle / cruising       |
| **533** | Mellow, smooth, linear  | `0.80 · 0.70 · 0`          | `160 / 530 / 54`             | ~533 °/s | Learning acro, racing lines, cine  |

The Actual columns reproduce the same curves to within ~0.5 % (verified against Betaflight's `applyBetaflightRates` / `applyActualRates`).

---

## The three curves

```chart
{
  "type": "line",
  "data": {
    "labels": ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
    "datasets": [
      {
        "label": "533 — center 160, max 533 °/s",
        "data": [0, 17.2, 37.2, 60.8, 88.9, 123.1, 165.5, 219.6, 290.9, 389.2, 533.3],
        "borderColor": "rgba(34,197,94,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5, "tension": 0.3, "pointRadius": 3
      },
      {
        "label": "633 — center 190, max 633 °/s",
        "data": [0, 20.4, 44.2, 72.2, 105.6, 146.2, 196.6, 260.8, 345.5, 462.2, 633.3],
        "borderColor": "rgba(59,130,246,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5, "tension": 0.3, "pointRadius": 3
      },
      {
        "label": "733 — center 220, max 733 °/s",
        "data": [0, 23.7, 51.2, 83.5, 122.2, 169.2, 227.6, 302.0, 400.0, 535.1, 733.3],
        "borderColor": "rgba(249,115,22,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5, "tension": 0.3, "pointRadius": 3
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": { "display": true, "text": "733 / 633 / 533 — commanded °/s vs stick deflection" },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": { "title": { "display": true, "text": "Stick deflection" } },
      "y": { "beginAtZero": true, "max": 760, "title": { "display": true, "text": "Rotation rate (°/s)" } }
    }
  }
}
```

Same shape, three heights. Super Rate 0.70 keeps the first ~60 % of stick relatively tame, then the curve ramps hard toward full deflection — that's where the "snap" for flips and rolls lives. Higher RC Rate raises both the center liveliness and the ceiling.

---

## 733 — punchy

Center sensitivity ~220 °/s and a ceiling of ~733 °/s. Full-stick flips and rolls come around fast, while the Super-Rate curve keeps mid-stick manageable for line work. A common "I can actually freestyle now" profile. Aggressive pilots push RC Rate higher still (833/933+).

**Betaflight (legacy):**
```
set rates_type = BETAFLIGHT
set roll_rc_rate = 110
set pitch_rc_rate = 110
set yaw_rc_rate = 110
set roll_srate = 70
set pitch_srate = 70
set yaw_srate = 70
set roll_expo = 0
set pitch_expo = 0
set yaw_expo = 0
save
```

**Actual:**
```
set rates_type = ACTUAL
set roll_rc_rate = 22
set pitch_rc_rate = 22
set yaw_rc_rate = 22
set roll_srate = 73
set pitch_srate = 73
set yaw_srate = 73
set roll_expo = 55
set pitch_expo = 55
set yaw_expo = 55
save
```

---

## 633 — balanced

Center sensitivity ~190 °/s, max ~633 °/s. The middle-ground freestyle profile — responsive enough for tricks, calm enough for long cruising lines. A great default while you decide whether you want more (733) or less (533).

**Betaflight (legacy):**
```
set rates_type = BETAFLIGHT
set roll_rc_rate = 95
set pitch_rc_rate = 95
set yaw_rc_rate = 95
set roll_srate = 70
set pitch_srate = 70
set yaw_srate = 70
set roll_expo = 0
set pitch_expo = 0
set yaw_expo = 0
save
```

**Actual:**
```
set rates_type = ACTUAL
set roll_rc_rate = 19
set pitch_rc_rate = 19
set yaw_rc_rate = 19
set roll_srate = 63
set pitch_srate = 63
set yaw_srate = 63
set roll_expo = 55
set pitch_expo = 55
set yaw_expo = 55
save
```

---

## 533 — mellow

Center sensitivity ~160 °/s, max ~533 °/s. The gentlest of the three: slower rotations and a calmer center make it forgiving for learning acro, and its flatter feel suits racing-style lines and smooth cinematic flying. Still a real freestyle rate, just relaxed.

**Betaflight (legacy):**
```
set rates_type = BETAFLIGHT
set roll_rc_rate = 80
set pitch_rc_rate = 80
set yaw_rc_rate = 80
set roll_srate = 70
set pitch_srate = 70
set yaw_srate = 70
set roll_expo = 0
set pitch_expo = 0
set yaw_expo = 0
save
```

**Actual:**
```
set rates_type = ACTUAL
set roll_rc_rate = 16
set pitch_rc_rate = 16
set yaw_rc_rate = 16
set roll_srate = 53
set pitch_srate = 53
set yaw_srate = 53
set roll_expo = 54
set pitch_expo = 54
set yaw_expo = 54
save
```

---

## Legacy and Actual describe the same curve

Both rate systems store into the *same* CLI fields (`rc_rate`, `srate`, `expo`) — `rates_type` just changes how those numbers are interpreted:

| Field       | Legacy Betaflight            | Actual                          |
|-------------|------------------------------|---------------------------------|
| `rc_rate`   | RC Rate × 100 (a multiplier) | Center sensitivity ÷ 10 (°/s)   |
| `srate`     | Super Rate × 100             | Max rate ÷ 10 (°/s)             |
| `expo`      | Expo × 100                   | Expo × 100 (shifts the kink)    |

Because the underlying math differs, the **expo numbers are not the same** between systems even for an identical curve — that's why 733's `expo=0` in legacy becomes `expo=55` in Actual (the Actual expo recreates the shape that Super Rate produces in the legacy model). The curves themselves overlap:

```chart
{
  "type": "line",
  "data": {
    "labels": ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
    "datasets": [
      {
        "label": "733 — Betaflight (RC 1.10, SR 0.70, E 0)",
        "data": [0, 23.7, 51.2, 83.5, 122.2, 169.2, 227.6, 302.0, 400.0, 535.1, 733.3],
        "borderColor": "rgba(249,115,22,1)",
        "backgroundColor": "transparent",
        "borderWidth": 3, "tension": 0.3, "pointRadius": 3
      },
      {
        "label": "733 — Actual (center 220, max 730, expo 55)",
        "data": [0, 24.3, 53.2, 86.9, 125.9, 171.8, 227.7, 299.5, 396.4, 533.0, 730.0],
        "borderColor": "rgba(99,102,241,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2, "borderDash": [6,3], "tension": 0.3, "pointRadius": 2
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": { "display": true, "text": "Same feel, two rate systems — 733 legacy vs Actual" },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": { "title": { "display": true, "text": "Stick deflection" } },
      "y": { "beginAtZero": true, "max": 760, "title": { "display": true, "text": "Rotation rate (°/s)" } }
    }
  }
}
```

---

## Notes

- **Expo** is not part of the shorthand. Add `expo` (0.10–0.20 legacy, ~30–50 in Actual) to soften the center for finer hover/line control — it does not change the max rate.
- **Yaw** is shown equal to roll/pitch for clean copy-paste. Many pilots drop yaw slightly for cleaner spins — lower the `yaw_*` values to taste.
- After pasting, open the **Rates** tab in Betaflight Configurator and confirm the live curve and the max-rate readout match the table above before flying.
- Switching `rates_type` does not erase the other system's values — Betaflight keeps them per profile, so you can flip between BETAFLIGHT and ACTUAL to compare.

---

## Related

- [Betaflight Rates Explained](../rates/) — what RC Rate, Super Rate, and Expo each do
- [Rate Modes — Formulas & Conversion](../rate-modes/) — the math for all five rate systems
- [Rates Deep Dive](../rates-deep-dive/) — the center/mid/edge zones and throttle expo
- [FPV Terminology](../../reference/fpv-terminology/) — glossary including rates, expo, Acro
