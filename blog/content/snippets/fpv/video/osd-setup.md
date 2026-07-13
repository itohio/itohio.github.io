---
title: "OSD Setup — Analog, Digital, GPS"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "osd", "analog", "digital", "gps", "dji", "walksnail"]
---

A clean OSD gives you essential flight data without cluttering the image for recording review. The goal: show only what you'll actually need after landing.

---

## Analog OSD (MAX7456)

The MAX7456 chip on most FC stacks handles OSD overlay on analog video.

**Minimal useful set for recording investigation:**

| Element            | Reason                                        |
|--------------------|-----------------------------------------------|
| Battery voltage    | Correlate voltage sag to behavior             |
| Current (mA)       | Spot motor or ESC stress                      |
| mAh consumed       | Know when you're past 80% of pack capacity    |
| Arming flags       | Catch why an arm was refused                  |
| Timer (on / total) | Timeline for clip review                      |
| RSSI / Link quality| Spot RF events in footage                     |

Add GPS elements only if a GPS module is installed.

**Avoid cluttering with:** artificial horizon, crosshairs, throttle bar, G-force, heading tape — these rarely help post-flight investigation and eat screen space.

Configurator → **OSD** tab → drag elements to the corners. Keep the center frame clean.

---

## Digital OSD (DJI O3 / Walksnail Avatar / HDZero)

Digital systems overlay OSD differently:

- **DJI O3 / O4**: OSD is rendered on the goggles. Betaflight still sends MSP OSD data; DJI renders it. Configure in Betaflight OSD tab as normal.
- **Walksnail Avatar**: Same MSP OSD path. Full custom element positioning.
- **HDZero**: MSP OSD. Fine-grained control via HDZero goggles menu.

The same minimalist element list applies. With digital you have more screen real-estate but still keep center clean.

For DJI, enable MSP OSD in CLI:
```
set osd_displayport_device = MSP
set displayport_msp_serial = <serial port number of DJI air unit>
save
```

---

## With GPS

GPS adds meaningful elements for investigation:

| Element            | Reason                                        |
|--------------------|-----------------------------------------------|
| GPS speed          | Verify max speed in blackbox vs OSD           |
| Altitude           | Legal ceiling check, altitude spikes          |
| GPS coords         | Where a crash happened                        |
| Home arrow + distance | Return-to-home margin check               |
| GPS fix / sats     | Catch weak GPS lock before flying             |

Position GPS elements at screen edges (top or bottom). Keep altitude and speed in a corner, not center.

---

## Without GPS (Minimal)

If no GPS: remove all GPS-related elements. Keep:
- Voltage
- mAh
- Timer
- Link quality / RSSI

Four elements in two corners. Uncluttered, all relevant.

---

## Cinematic Analog OSD

Cinematic setups prioritize clean footage — the OSD should nearly disappear.

**Recommended layout:**
- Battery voltage: bottom-left corner, small
- mAh used: bottom-right corner, small
- Timer: top-right, small
- Nothing else

Turn off all warnings (low battery text pop-up can be enabled but set threshold conservatively so it doesn't fire mid-shot).

In CLI:
```
# Reduce OSD element clutter
set osd_vbat_pos = 2400       # bottom left
set osd_mah_drawn_pos = 2424  # bottom right
set osd_warnings_pos = 14731  # off-screen or disabled
save
```

> Position values encode row/column as `row*32 + col + 0x800` for blink bit. Use Configurator drag-and-drop instead of manual calculation.

---

## Stats Screen (Post-Disarm)

The stats screen shows after disarm if `stats = ON`. Useful elements:

```
set stats = ON
set stats_min_voltage = ON
set stats_max_current = ON
set stats_used_mah = ON
set stats_max_speed = ON    # requires GPS
set stats_max_altitude = ON # requires GPS or baro
set stats_total_flights = ON
save
```

See also: [Rate Profiles & Persistent Stats](../../tuning/rate-profiles-persistent-stats/) for the interaction between rate profile switching and stat persistence.
