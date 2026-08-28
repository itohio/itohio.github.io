---
title: "Lost in the Baltic: What the Telemetry Says"
date: 2026-08-28
description: "Post-mortem on losing a Pavo20 Pro II to the Baltic Sea: measured telemetry, two-ray multipath over water, why every callout failed, and the hardening strategy that comes out of it."
draft: false
toc: true
categories:
  - FPV
tags:
  - fpv
  - edgetx
  - telemetry
  - gps
  - battery
  - long-range
  - callouts
  - risk-management
keywords: ["fpv battery callouts", "edgetx telemetry", "gps rescue", "point of no return", "long range fpv", "pavo20 pro ii ocean loss", "fpv sea crash", "two-ray multipath"]
thumbnail: "https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/39e00838-047a-453b-b5b8-443a54420a5e/ocean_flight_analysis.png"
---

## What Happened

I flew the Pavo20 Pro II 2.47 km out over the Baltic Sea at dusk to find out
what clear-horizon ELRS range looked like. It is a reasonable experiment.
The result was informative.

The drone is now on the sea floor.

Telemetry went dark at t = 79 s, altitude 0 m, roughly 920 m out. It came back
at t = 230 s when the drone had climbed to 75 m on the return leg. Two-ray
multipath over seawater at grazing incidence: the sea is a near-perfect RF
reflector, and at low altitude the direct and reflected paths arrive nearly
equal in amplitude and opposite in phase. Climbing breaks the cancellation.
By the time telemetry restored, 555 mAh had been consumed. 266 mAh remained.
The point of no return was 1323 m from home. The drone was 1946 m from home.
Already 623 m past it.

## Flight Analysis

![Five-panel flight analysis from EdgeTX telemetry. Top: GPS and dead-reckoned flight path. Middle: consumed capacity and current vs distance. Bottom: consumed capacity vs time, and RSSI / link quality vs distance. Grey shading marks the telemetry dark period.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/39e00838-047a-453b-b5b8-443a54420a5e/ocean_flight_analysis.png)

## What the Numbers Say

The pack was a LAVA II 680 mAh 3S LiHV. The flight controller had it logged as
821 mAh, which overstates capacity by 20% and silently inflates every SoC figure
the telemetry records. Battery SoC read 85% remaining when the link first went dark.

Outbound rate was 126 mAh/km. The drone had a 10 km/h tailwind at approximately
50 km/h airspeed, so that number is as good as it gets. The return leg, same
airspeed into the same wind, cost 183 mAh/km: 1.45 times more expensive per
kilometre. Total telemetry dark: 171 s out of 350 s. 49% of the flight,
invisible to the radio.

RSSI at launch was −36 dBm. At 60 m out, still at sea level, it was −84 dBm.
48 dB of loss over 60 horizontal metres, with the link quality gauge reading
100% throughout. RSSI was the ramp; LQ was the cliff.

The mental model failure: I had flown a 2 km round trip in heavy field winds
earlier the same day and landed with 20% remaining. I assumed open water would
be easier. What I did not account for: the return leg costs 1.45× the outbound
at matching airspeed, the wind reversed after I turned, and I was flying at
0 m altitude over a near-perfect RF reflector.

## What the Callouts Did and Did Not Do

The `rth` spoken callout reached the radio at t = 229.8 s, which was 150 s after
the pack had crossed 3.8 V per cell. EdgeTX logical switches sourced from
telemetry sensors are forced FALSE during a blackout. A level-comparison switch
cannot fire while the link is down. The callout was silent for the entire dark
period.

An RSSI dBm alarm set at −85 dBm would have fired at t = 38 s, around 235 m
outbound. That is correct for this flight: the link was already anomalously lossy
at short range because of the sea-surface geometry. A warning at 235 m would have
changed the flight. The radio had `rssiSource` set to `none`. There was no RF alarm
wired at all.

RQly held 100% until total blackout. On this flight, link quality told me nothing.
RSSI was telling a story from the first 40 seconds.

## Goggles Footage

![Earliest recovered frame. 2.47 km out, 66 km/h, pack at 11.0 V.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/f031f1b9-00b9-420e-899f-1e9ee2405f31/vlcsnap-2026-08-28-00h18m36s708.png)

![Turning for home. 1.89 km from launch, 10.9 V.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/355d3028-6331-4409-8c1e-9fd772811eac/vlcsnap-2026-08-28-00h18m53s493.png)

![Returning. 1.33 km. LOW BATTERY on OSD.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/0d4f62f8-43d0-480a-8bc0-14b69a3198e1/vlcsnap-2026-08-28-00h19m11s482.png)

![1.04 km from home. LAND NOW on OSD. 6.1 V total, 2.03 V per cell.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/4a54fb8f-71fe-4ecf-a946-08cb51a7d5c2/vlcsnap-2026-08-28-00h19m25s148.png)

![Last frame before water. 1.04 km from home, descending.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/ab568c4f-a08e-41ba-9341-6bdb2544a9ec/vlcsnap-2026-08-28-00h19m56s313.png)

## What I Am Changing

The table below is strategy and design intent. Each row is a failure mode the
telemetry identified. Implementation is a separate post.

| Condition | Old behaviour | New behaviour |
|---|---|---|
| Battery voltage crosses 4.2 / 4.0 / 3.8 V/cell on descent | Single voltage-threshold logical switch, fires once on the first crossing, gated on the battery button | Spoken number ("one", "two", "three") on each 0.1 V crossing below 3.8 V; no tone (beep volume is 0 on this radio) |
| Battery < 3.6 V/cell | `lowbat` track via logical switch | Same, but now also comes from the background script — first line of defence even through a blackout |
| Return leg costs more than 1.3× the outbound per km | No callout existed | Wind asymmetry warning after ≥30 s outbound + ≥15 s return: "warning close {ratio}%" |
| Specific power elevated + specific speed normal (internal fault: motor, prop, bearing) | No callout existed | "warning power {ratio}%" once, after a cruise baseline is established |
| Specific power elevated + specific speed depressed (external: headwind, drag) | No callout existed | "warning speed {ratio}%" once |
| Link RSSI below −92 dBm | `rssiSource: none` — no RF alarm wired at all. RQly held 100% until total loss | `siglow` + dBm value; `sigcrt` at −100 dBm. RSSI is the ramp; LQ is the cliff |
| Link RSSI degrading + altitude < 25 m | Nothing | "tolow" — climb. Measured: link died at 0 m, restored at 75 m, two-ray multipath over water |
| Telemetry dark for > 4 s | Logical switches frozen FALSE; all battery callouts silenced | Background Lua script continues dead-reckoning distance and SoC, re-announces state on restore |
| Point of no return approaching (outbound, GPS available) | Nothing | "close {metres remaining}" counting down while PNR − dHome < 400 m |
| Arming with GPS rescue not ready (FC says "?") | Satellite count comparison against a hardcoded threshold in the script | FC's own verdict: Betaflight appends `?` to the CRSF flight-mode string when `numSat < gps_rescue_min_sats`. Script reads that directly. |
| Impedance rises beyond what depth-of-discharge explains (excess ≥ 1.4×) | No callout | "warning bad {mΩ}" once, while SoC > 25% |

## Closing

The drone is on the sea floor. The telemetry is not. One of those is more useful
for not repeating the experiment.

The scripts and YAML come in a separate post. This one is the flight data and
the decisions it forced.
