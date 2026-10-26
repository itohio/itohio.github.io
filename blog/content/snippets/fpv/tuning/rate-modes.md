---
title: "Betaflight Rate Modes — Formulas, Comparison, and Conversion"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "actual-rates", "kiss-rates", "quickrates", "rc-rate", "expo", "tuning"]
---

Betaflight supports five rate mode formulas (Betaflight, Raceflight, KISS, Actual, Quick). They all do the same job — map stick deflection (0 to ±1) to a commanded rotation rate (°/s) — but they parameterize the curve differently. Switching modes changes what the sliders mean, not what the quad feels like if you set equivalent parameters.

---

## Overview

| Mode | CLI value | Parameters | Best for |
|------|-----------|-----------|---------|
| **Betaflight** (legacy) | `BETAFLIGHT` | RC Rate, Super Rate, Expo | Default; most community presets use this |
| **Actual** | `ACTUAL` | Center Sensitivity, Max Rate, Expo | Direct control over specific feel metrics |
| **Raceflight** | `RACEFLIGHT` | Rate, Acro+, Expo | Matching a RaceFlight feel |
| **KISS** | `KISS` | RC Rate, Rate, RC Curve | Matching a KISS FC feel |
| **Quickrates** | `QUICK` | Center Rate, Max Rate, Expo | Center and max set independently |

**Switch modes via CLI:**
```
set rates_type = ACTUAL    # or BETAFLIGHT, RACEFLIGHT, KISS, QUICK
save
```

Betaflight remembers your last-used values per mode. Switching modes does not overwrite the other mode's stored values.

---

## Betaflight (Legacy) Rates

**Parameters:** RC Rate, Super Rate, Expo

**Formula:**
```
stick = |rcCommand|       // 0 to 1

// Expo pre-shaping (reduces center sensitivity):
if expo > 0:
    stick = stick × (expo × |stick|³ + (1 − expo))

// Rate with super rate denominator (hockey-stick at extremes):
output = sign(rcCommand) × stick × rcRate × 200 / (1 − superRate × stick)
```

The denominator `(1 − superRate × stick)` is what creates the characteristic "hockey stick" — near center (stick≈0) the effect is minimal, but near full deflection (stick→1) the denominator approaches `(1 − superRate)`, amplifying the output sharply.

| Parameter | Typical range | Effect |
|-----------|--------------|--------|
| RC Rate | 0.8–1.8 | Scales the entire curve; 1.0 ≈ 200 °/s near center with SR=0 |
| Super Rate | 0.50–0.75 | Controls the acceleration at extremes; higher = more hockey-stick |
| Expo | 0–0.5 | Softens center feel; most pilots use 0 or low values |

**Limitation:** RC Rate and Super Rate interact — changing Super Rate also changes the center feel, and vice versa. There is no single parameter that controls only center sensitivity and another that controls only max rate.

---

## Actual Rates

**Parameters:** Center Sensitivity, Max Rate, Expo

**Formula (expo = 0):**
```
output = sign(rcCommand) × ((max_rate − center) × |rcCommand|² + center × |rcCommand|)
```

This satisfies two independent constraints simultaneously:
- Slope at stick = 0: equals `center` (center sensitivity in °/s per unit)
- Value at stick = 1: equals `max_rate`

With expo > 0, the middle portion of the curve is blended toward a sine-shaped curve, creating a gentler mid-range transition.

**Advantage:** Each parameter controls exactly one thing. You can change center feel without touching the max rate, and vice versa. This makes Actual rates easier to tune to a specific feel.

| Parameter | Effect |
|-----------|--------|
| Center Sensitivity | Degrees/s per stick unit at center. 70–120 is common. |
| Max Rate | Hard ceiling at full deflection. 600–900 °/s typical freestyle. |
| Expo | 0 = pure quadratic curve; higher = softer mid-transition |

---

## KISS Rates

**Parameters:** RC Rate, Rate, RC Curve

**Formula:**
```
curve  = rcCurve × x³ + (1 − rcCurve) × x            # x = stick, −1…1
output = 2000 × (rcRate / 10) × curve / (1 − rate × |x|)   # rcRate 1.00, rate 0.70 → ~667 °/s
```

Structurally close to the BF legacy formula: RC Rate scales the whole curve, Rate sets the 1/(1 − rate·|x|) end boost, and RC Curve adds cubic expo.

KISS rates are named after the KISS ESC/FC ecosystem where this formula originated. Betaflight's implementation matches the original closely.

---

## Quickrates

**Parameters:** Center Rate (RC Rate × 200 °/s), Max Rate, Expo

**Formula:**
```
superFactor = 1 / (1 − |x| × (1 − centerRate / maxRate))
output      = sign(x) × centerRate × |x| × superFactor   # expo 0
```

Set the center sensitivity and the full-stick rate directly; a super-factor bends the curve between them. It is linear only when center rate equals max rate.

**When to use:**
- Cinematography and slow-motion builds where you want predictable, linear control
- Tuning data collection flights where you want the setpoint signal to be clean and linear (makes step response analysis slightly cleaner)
- New pilots learning acro — linear feel is initially easier to understand

---

## Rate Comparison Chart

Typical settings matched for roughly comparable feel:

```chart
{
  "type": "line",
  "data": {
    "labels": ["0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
    "datasets": [
      {
        "label": "Betaflight (rcRate=1.2, SR=0.70) — max 800°/s",
        "data": [0,26,56,91,133,185,248,329,436,584,800],
        "borderColor": "rgba(239,68,68,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "tension": 0.2,
        "pointRadius": 2
      },
      {
        "label": "Actual (Center=100, Max=700, Expo=0)",
        "data": [0,16,44,84,136,200,276,364,464,576,700],
        "borderColor": "rgba(34,197,94,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "tension": 0.2,
        "pointRadius": 2
      },
      {
        "label": "KISS (rate=0.26) — max 702°/s",
        "data": [0,53,110,169,232,299,369,445,525,610,702],
        "borderColor": "rgba(99,102,241,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2,
        "tension": 0.2,
        "pointRadius": 2
      },
      {
        "label": "Quickrates (center = max = 700) — linear in this case",
        "data": [0,70,140,210,280,350,420,490,560,630,700],
        "borderColor": "rgba(249,115,22,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2,
        "borderDash": [5,3],
        "tension": 0,
        "pointRadius": 2
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": { "display": true, "text": "Rate mode comparison — commanded °/s vs stick deflection (0 to 1)" },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": { "title": { "display": true, "text": "Stick deflection (0 = center, 1 = full throw)" } },
      "y": {
        "min": 0,
        "max": 850,
        "title": { "display": true, "text": "Commanded rotation rate (°/s)" }
      }
    }
  }
}
```

**Key observations:**
- **BF (red)**: starts shallow (low center sensitivity at these settings) but accelerates hard at extremes — the hockey stick is visible above 0.7 stick
- **Actual (green)**: defined quadratic — clean, predictable slope from center to max
- **KISS (purple)**: similar shape to BF (RC Rate × end boost from Rate); mid-stick feel slightly more linear than BF
- **Quickrates (orange dashed)**: linear only because center = max here; a lower center rate bends it like the others

---

## BF ↔ Actual Conversion

The two curves have different shapes, so a perfect conversion does not exist. These formulas give a close starting point:

### BF → Actual

```
Actual.center    ≈ BF.rcRate × 200
Actual.max_rate  = read from BF configurator (full-stick output value)
Actual.expo      ≈ BF.superRate / 2.0   (rough starting point — tune by feel)
```

**Example:** BF rcRate=1.5, superRate=0.70, expo=0
- `Actual.center = 1.5 × 200 = 300`
- Read configurator: max output ≈ 900°/s → `Actual.max_rate = 900`
- `Actual.expo = 0.70 / 2.0 = 0.35`

### Actual → BF

```
BF.rcRate      ≈ Actual.center / 200
BF.superRate   ≈ Actual.expo × 2.0   (starting point)
```

After computing, open the configurator and visually compare the two curves on the Rate graph. Adjust `superRate` until the full-stick output matches your desired max rate. The curves will not be identical — choose the one that matches the feel you want at mid-stick.

---

## Equivalent Feel — Example Presets

| Feel | Betaflight | Actual | KISS |
|------|-----------|--------|------|
| Freestyle medium | RC=1.4, SR=0.70, E=0 | Center=100, Max=750, E=0.30 | Rate=0.28 |
| Racing / sharp | RC=1.8, SR=0.75, E=0 | Center=140, Max=900, E=0.25 | Rate=0.32 |
| Smooth / cinematic | RC=0.8, SR=0.40, E=0.20 | Center=50, Max=350, E=0.40 | Rate=0.15 |
| 2" ripper | RC=1.2, SR=0.65, E=0 | Center=90, Max=650, E=0.30 | Rate=0.24 |

These are starting points — adjust to taste. The configurator Rate graph updates live, so compare visually rather than trusting the numbers exactly.

---

## Which Mode Should I Use?

**Use Betaflight (legacy)** if you're loading community presets or following a tune guide that specifies RC Rate and Super Rate values. The vast majority of published presets assume this mode.

**Use Actual** if you want to independently control "how sharp is center" vs "how fast does full stick go". Particularly useful when copying the feel of a previous build to a new one with different motor/prop — you can match center sensitivity and max rate independently.

**Use KISS** if you are coming from a KISS flight controller and want to carry over its RC Rate / Rate / RC Curve numbers.

**Use Quickrates** with center = max rate for cinematography or tuning data collection flights (linear setpoint = cleanest step response data).

> All five modes can produce an identical real-world flying feel if parameterized correctly. Mode choice is a matter of how you prefer to think about and adjust the curve — not which one is "better".

---

## Related

- [Betaflight Tuning Math](../betaflight-tuning-math/) — what happens to rates inside the PID loop
- [Wobble-Test PID Protocol](../pid-tuning-wobble-test/) — tuning workflow that references these rate settings
- [Tuning Flight Protocol](../tuning-flight-protocol/) — why Quickrates or FF=0 matters for clean BBL data
- [FPV Terminology](../../reference/fpv-terminology/) — glossary including Acro mode, rates, PID
- **Rylo** — rate setup help and PID tuning guidance → [app.sintra.ai/community/helpers/rylo](https://app.sintra.ai/community/helpers/rylo)
