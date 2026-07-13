---
title: "Rates Deep Dive — Expo Curve, Zones, and Throttle"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "expo", "throttle", "tuning", "freestyle", "cinematic"]
---

Rates are not just a "how fast does it spin" setting. They shape *where on the stick* control lives — and understanding the curve lets you tune for precision hover, natural cruising, and explosive tricks, all in the same profile.

---

## The Rate Curve

Moving a gimbal stick from center to full deflection does not produce a linear increase in rotation speed. The rate curve defines that relationship. Here are the three common profiles plotted:

```chart
{
  "type": "line",
  "data": {
    "labels": ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
    "datasets": [
      {
        "label": "Stability Zone (0–20%)",
        "data": [760, 760, 760, null, null, null, null, null, null, null, null],
        "backgroundColor": "rgba(34, 197, 94, 0.13)",
        "borderColor": "transparent",
        "borderWidth": 0,
        "pointRadius": 0,
        "fill": "origin",
        "order": 10
      },
      {
        "label": "Normal Flying (20–70%)",
        "data": [null, null, 760, 760, 760, 760, 760, 760, null, null, null],
        "backgroundColor": "rgba(59, 130, 246, 0.10)",
        "borderColor": "transparent",
        "borderWidth": 0,
        "pointRadius": 0,
        "fill": "origin",
        "order": 10
      },
      {
        "label": "Tricks & Flicks (70–100%)",
        "data": [null, null, null, null, null, null, null, 760, 760, 760, 760],
        "backgroundColor": "rgba(249, 115, 22, 0.13)",
        "borderColor": "transparent",
        "borderWidth": 0,
        "pointRadius": 0,
        "fill": "origin",
        "order": 10
      },
      {
        "label": "533 — Mellow / Learning (max 533°/s)",
        "data": [0, 17.2, 37.2, 60.8, 88.9, 123.1, 165.5, 219.6, 290.9, 389.2, 533.3],
        "borderColor": "rgba(34, 197, 94, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.35,
        "fill": false,
        "order": 1
      },
      {
        "label": "633 — All-round Freestyle (max 633°/s)",
        "data": [0, 20.4, 44.2, 72.2, 105.6, 146.2, 196.6, 260.8, 345.5, 462.2, 633.3],
        "borderColor": "rgba(59, 130, 246, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.35,
        "fill": false,
        "order": 1
      },
      {
        "label": "733 — Punchy Freestyle (max 733°/s)",
        "data": [0, 23.7, 51.2, 83.5, 122.2, 169.2, 227.6, 302.0, 400.0, 535.1, 733.3],
        "borderColor": "rgba(249, 115, 22, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.35,
        "fill": false,
        "order": 1
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": {
        "display": true,
        "text": "Rotation Rate vs Stick Position — 533 / 633 / 733 (Super Rate 0.70, Expo 0)"
      },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": {
        "title": { "display": true, "text": "Stick deflection" }
      },
      "y": {
        "beginAtZero": true,
        "max": 760,
        "title": { "display": true, "text": "Rotation rate (°/s)" }
      }
    }
  }
}
```

---

## The Three Zones

### Zone 1 — Stability (0–20% stick, green band)

The very beginning of stick travel is the flattest part of the curve. Super Rate 0.70 saves the steep response for the stick edges, so the center stays gentle and forgiving (adding a little Expo flattens it further still).

At 5% stick deflection on a 533 profile, the rotation rate is only **8.3 °/s** — about **1.6 °/s per 1% of stick movement**. On a 733 profile it's **2.2 °/s per 1%** in this zone. Either way, small unintentional stick movements cause little rotation: hand tremor, wind buffeting the quad back to the sticks, or an arm not fully relaxed — mostly absorbed here.

**This zone is your hover zone.** A quad hovering in place sits somewhere in this range. The lower the sensitivity here, the easier it is to hold a stable position or make slow, cinematic pans. Beginners benefit most — every training flight is spent in this zone.

### Zone 2 — Normal Flying (20–70% stick, blue band)

The middle section of the curve is where you spend most of your flight time. This region should feel **roughly linear** — consistent response to stick input so the quad goes where you point it without surprises.

At 50% stick on 533 the rate is 123 °/s; at 50% on 733 it's 169 °/s. The difference between profiles becomes more noticeable here — 533 feels gentle and smooth, 733 feels punchy and alive.

**Why "roughly linear" matters:** if the middle section has too much curvature, transitions from hover into medium-speed flight feel jerky — the stick suddenly becomes much more sensitive as you push out of the center zone. A good tune keeps the transition seamless enough that you don't notice it. A modest amount of Expo (0.10–0.20) helps; Expo above ~0.5 starts to create a noticeable "dead center → suddenly alive" feel.

### Zone 3 — Tricks & Flicks (70–100% stick, orange band)

Full stick deflection is where Super Rate kicks in hardest. The curve steepens noticeably — on 733, the last 5% of stick travel (95%→100%) adds **~22 °/s per 1%** compared to **~2.2 °/s per 1%** at the center — roughly a 10× swing in sensitivity from center to edge.

This zone provides the "snap" for split-S maneuvers, power loops, and flips. You don't spend much time here during a flight — these are momentary full-deflection inputs. Having the rate high in this zone means tricks are snappy and fast without the entire stick feeling aggressive.

**Practical implication:** Super Rate concentrates the extra speed at the edges, so a high max rate does not have to mean a twitchy center. This is why a skilled pilot can run 733 (or higher) while still landing smoothly — precision lives in Zone 1, speed lives in Zone 3.

---

## Sensitivity Comparison at Extremes

| Profile | Center sensitivity (°/s per 1% stick) | Full deflection sensitivity |
|---------|---------------------------------------|-----------------------------|
| 533     | 1.6 °/s / 1%                          | 15.9 °/s / 1%               |
| 633     | 1.9 °/s / 1%                          | 18.9 °/s / 1%               |
| 733     | 2.2 °/s / 1%                          | 21.9 °/s / 1%               |

A 733 pilot's center stick is only ~38% more sensitive than a 533 pilot's, and full deflection is ~38% faster. Because only RC Rate changes, the whole curve scales together — the center-to-edge *ratio* (~10×) is identical across all three profiles.

---

## Throttle Expo

Rotation rate isn't the only curve that matters. The throttle channel — by default completely linear — can also be shaped. Betaflight calls this **Throttle Mid** and **Throttle Expo**.

```chart
{
  "type": "line",
  "data": {
    "labels": ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
    "datasets": [
      {
        "label": "Linear (no expo)",
        "data": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "borderColor": "rgba(156, 163, 175, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2,
        "borderDash": [6, 3],
        "pointRadius": 3,
        "tension": 0,
        "fill": false
      },
      {
        "label": "Expo 0.3 (mild curve)",
        "data": [0, 7.0, 14.2, 21.8, 29.9, 38.7, 48.5, 59.3, 71.4, 84.9, 100],
        "borderColor": "rgba(59, 130, 246, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.3,
        "fill": false
      },
      {
        "label": "Expo 0.5 (aggressive curve)",
        "data": [0, 5.1, 10.4, 16.3, 23.2, 31.2, 40.8, 52.1, 65.6, 81.5, 100],
        "borderColor": "rgba(249, 115, 22, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.3,
        "fill": false
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": {
        "display": true,
        "text": "Throttle Output vs Stick Position — Effect of Expo"
      },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": {
        "title": { "display": true, "text": "Throttle stick position" }
      },
      "y": {
        "beginAtZero": true,
        "max": 100,
        "title": { "display": true, "text": "Motor output (%)" }
      }
    }
  }
}
```

With linear throttle, moving from 40% to 60% stick raises motor output by 20%. With Expo 0.5, the same movement raises output from 23% to 41% — an 18% change spread across a larger stick range, giving finer granularity in the hover zone.

### Throttle Mid

**Throttle Mid** sets the stick position that produces 50% motor output. By default it is 50% (center stick = half power). This single value has the biggest practical impact on how a new build feels.

Most freestyle quads hover somewhere between 30–50% throttle depending on AUW, prop size, and battery voltage. If your hover point is at 40% stick, a Throttle Mid of 0.5 means the top half of your throttle stick only covers a 10% range — 40% to 50% to 100% — making high-throttle maneuvers jerky. Shift Throttle Mid toward the hover point and the stick range expands symmetrically around it.

---

## Battery Depletion Shifts the Hover Point

A fully charged 4S pack sits at ~16.8 V. As the battery discharges, voltage drops — and the same motor + prop combination produces less thrust per unit of throttle. To maintain altitude, you push the stick higher.

```chart
{
  "type": "bar",
  "data": {
    "labels": ["Full pack (16.8V)", "75% capacity (15.8V)", "50% capacity (15.0V)", "20% capacity (14.2V)"],
    "datasets": [
      {
        "label": "Required hover throttle (%)",
        "data": [38, 43, 49, 57],
        "backgroundColor": [
          "rgba(34, 197, 94, 0.7)",
          "rgba(59, 130, 246, 0.7)",
          "rgba(249, 115, 22, 0.7)",
          "rgba(239, 68, 68, 0.7)"
        ],
        "borderColor": [
          "rgba(34, 197, 94, 1)",
          "rgba(59, 130, 246, 1)",
          "rgba(249, 115, 22, 1)",
          "rgba(239, 68, 68, 1)"
        ],
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "responsive": true,
    "plugins": {
      "title": {
        "display": true,
        "text": "Hover Throttle Shift vs Battery State — Typical 5\" 4S Build"
      },
      "legend": { "display": false }
    },
    "scales": {
      "y": {
        "beginAtZero": true,
        "max": 80,
        "title": { "display": true, "text": "Throttle stick position (%)" }
      }
    }
  }
}
```

On a 5" freestyle quad the hover point shifts by ~15–20 percentage points across a full pack. This has two consequences:

1. **Throttle feel changes through the flight.** A Throttle Mid tuned for a fresh pack will feel wrong on a low pack and vice versa.
2. **The transition from hover to climb becomes more abrupt** as the battery depletes, because the hover point gets closer to the high-output zone of the curve.

Setting Throttle Mid to the **average** hover point across the pack (roughly 46–48% for a typical 5" 4S) is the practical compromise. A mild expo (0.3) around it gives enough center softness to absorb the shift without re-tuning mid-session.

---

## Finding Your Optimal Throttle Mid

### Method 1 — OSD Recording

Enable the throttle OSD element and record a flight. Find a stable hover section of the footage and read the throttle percentage shown on screen. Do this at the start, middle, and end of the pack. Average the three values — that's your Throttle Mid starting point.

In Configurator → OSD tab, enable **Throttle Position** (shown as a percentage) and place it in a corner that's visible in your recordings.

### Method 2 — Blackbox Analysis

With blackbox logging enabled, pull a log from a full flight and plot the `rcCommand[3]` (throttle) channel against time. Identify stable hover sections (low variation, constant altitude) and compute the mean throttle value in those windows.

In Betaflight Configurator's Blackbox Explorer, use the axis filter to show only throttle. Find a 5–10 second window of level flight and read the average rcCommand value. Scale to percentage: rcCommand throttle range is typically 1000–2000 µs; subtract 1000 and divide by 10 to get percent.

This gives you a per-flight average. After 3–5 flights, you'll have a stable estimate across fresh and depleted packs.

---

## Putting It Together

| Setting         | What to set it to                          |
|-----------------|--------------------------------------------|
| RC Rate         | 0.80–1.10 for the 533–733 family (sets the max rate) |
| Super Rate      | 0.70 for the 533/633/733 family; higher = harder edge |
| Expo            | 0 by default; add 0.10–0.20 to soften the center |
| Throttle Expo   | 0.3 for most builds                        |
| Throttle Mid    | Measured hover point (method above) ≈ 0.45–0.55 |

Tune rates and expo first in a simulator where crashes have no cost. Then move to the field and use OSD/blackbox to dial in throttle mid.

See also: [Rate Presets (733 / 633 / 533)](../rate-presets/) for the exact values and copy-paste CLI in both rate systems.
