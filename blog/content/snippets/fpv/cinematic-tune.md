---
title: "Cinematic Tune from Betaflight Presets"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "cinematic", "tune", "presets", "pid", "filters"]
---

Betaflight ships with a Presets library that includes ready-made cinematic tunes. These lower aggression, soften filters, and reduce propwash artifacts for smooth, floaty footage.

---

## Loading a Preset

Betaflight Configurator → **Presets** tab → search `cinematic`.

Good starting points:
- **Cinematic Freestyle** — moderate filtering, low D-term, smooth step response
- **Cinematic HD** — heavier filtering for gyro noise rejection, suits larger quads with gimbals
- **Slow Flyer / Whoop Cinematic** — tuned for small 2" or toothpick frames

Click a preset → **Preview** to see what CLI commands it will apply → **Apply and Save**.

---

## What a Cinematic Tune Changes

| Parameter        | Freestyle typical  | Cinematic target   |
|------------------|--------------------|--------------------|
| P-term           | Moderate–high      | Lower (less snappy)|
| D-term           | Moderate           | Lower              |
| I-term           | Moderate           | Similar / slightly higher (holds position) |
| Feedforward      | Moderate–high      | Low (reduces stick response sharpness) |
| RPM filter       | ON                 | ON (essential)     |
| Dynamic notch    | ON                 | ON, wider bands    |
| TPA breakpoint   | ~1250              | ~1150 (earlier softening) |
| Rates            | 633–733            | 333–433            |

---

## Manual Adjustments After Preset

Presets are a starting point. After applying:

1. **Verify RPM filter** is enabled and bidirectional DSHOT is working (`dshot_bidir = ON`, check `rpmfilter` in CLI).
2. **Lower RC smoothing** if the footage looks like it's hunting at center:
   ```
   set rc_smoothing_auto_factor = 50
   ```
3. **Reduce feedforward** if the quad feels twitchy on slow pans:
   ```
   set feedforward_transition = 100
   set ff_interpolate_sp = AVERAGED_4
   ```
4. **Tune rates to 333 or 433** — cinematic flying rarely needs more than 400 °/s max roll.
5. **Enable I-term relax** if you see bounce-back after rolls:
   ```
   set iterm_relax = RP
   set iterm_relax_type = SETPOINT
   ```

---

## Blackbox Verification

After your first cinematic flight, pull a blackbox log and look for:
- Clean gyro trace without high-frequency spikes
- Low motor output variance (no oscillation)
- Smooth setpoint vs. gyro tracking on slow moves

Propwash on fast direction changes is expected and harder to eliminate with filtering alone — throttle management and flying technique matter more for cinematic work.

---

## Notes

- RPM filter is mandatory for modern cinematic tunes — without it you'll need much heavier lowpass filtering that adds latency and phase delay.
- Preset tunes are version-specific. Always check which Betaflight version the preset targets before applying.
