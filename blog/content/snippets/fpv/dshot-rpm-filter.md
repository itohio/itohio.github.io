---
title: "DSHOT and RPM Filter"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "dshot", "rpm-filter", "bidirectional", "noise", "filtering"]
---

DSHOT is a digital ESC protocol. When used bidirectionally, it feeds motor RPM back to the flight controller, enabling the RPM filter — the single biggest improvement to motor noise rejection in modern Betaflight tunes.

---

## DSHOT Variants

| Protocol   | Speed       | Notes                                   |
|------------|-------------|-----------------------------------------|
| DSHOT150   | 150 kbps    | Legacy; slow; not recommended           |
| DSHOT300   | 300 kbps    | Default; compatible with most ESCs      |
| DSHOT600   | 600 kbps    | Faster updates; required for some 8K loops |
| DSHOT1200  | 1200 kbps   | Rarely used; few ESCs support it        |

**DSHOT300** is the default and works on every modern ESC. Use DSHOT600 only if running 8K/8K or the ESC supports it explicitly.

---

## Bidirectional DSHOT

Bidirectional DSHOT adds a return signal from ESC to FC — each ESC sends its motor RPM back at the same time as it receives throttle commands.

Enable in CLI:
```
set dshot_bidir = ON
set motor_pwm_protocol = DSHOT300
save
```

After enabling, go to **Motors** tab in Configurator — each motor should show live RPM readout when spun by hand (props off). If any motor shows 0 RPM, check that the ESC firmware supports bidirectional DSHOT (BLHeli_32 ≥ 32.7, or AM32 with bidir support).

---

## RPM Filter

The RPM filter uses real-time motor RPM telemetry to place notch filters exactly on motor noise harmonics. This eliminates the need for wide-band lowpass filters that add phase delay.

Enable:
```
set rpm_filter_harmonics = 3
set rpm_filter_q = 500
save
```

The filter automatically tracks the fundamental motor frequency and its harmonics (2× and 3× typically). As RPM changes during flight, the notch positions move with it.

**Effect on tune:**
- Significantly reduces motor noise in the gyro signal
- Allows higher P/D gains without oscillation
- Enables lower static notch filter burden (dynamic notch can be reduced)
- Makes the tune more consistent across throttle ranges

---

## RPM Filter and Betaflight Version

- **BF 4.1+**: RPM filter available and stable
- **BF 4.2+**: Improved tracking; multi-harmonic support
- **BF 4.3+**: Works alongside Dynamic Notch v2 (wider, smarter notch)

Do not run RPM filter on BF versions below 4.1.

---

## Dynamic Notch Filter (Companion)

Even with RPM filter, the dynamic notch filter handles non-motor noise (frame resonance, prop wash, bearing noise). Leave it on:

```
set dyn_notch_count = 4
set dyn_notch_q = 250
set dyn_notch_min_hz = 100
set dyn_notch_max_hz = 600
save
```

With RPM filter handling motor harmonics, the dynamic notch can be set conservatively (fewer notches, wider Q) to avoid excessive phase delay.

---

## Lowpass Filters

With RPM filter enabled, you can reduce lowpass filter aggression:

```
# Gyro lowpass — more permissive with RPM filter
set gyro_lowpass_hz = 0           # disable static lowpass (RPM filter handles it)
set gyro_lowpass2_hz = 0

# D-term lowpass — keep this for D-noise
set dterm_lowpass_hz = 100
set dterm_lowpass2_hz = 200
```

The Betaflight cinematic and freestyle presets set these automatically when you apply them.

---

## Troubleshooting

| Symptom                            | Cause                                       |
|------------------------------------|---------------------------------------------|
| RPM shown as 0 in Motors tab       | ESC doesn't support bidir DSHOT; bidir not enabled; wrong protocol |
| Oscillations worse after enabling  | RPM filter Q too high; harmonics too many   |
| ESC beeps oddly after enabling     | Some ESCs need reflash to enable bidir mode |

Check ESC firmware changelog for bidirectional DSHOT support before assuming hardware failure.
