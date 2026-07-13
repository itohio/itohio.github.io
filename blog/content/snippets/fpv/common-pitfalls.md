---
title: "Common FPV Build Pitfalls"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "troubleshooting", "motors", "esc", "build"]
---

A collection of "first 10 minutes of a new build" failures and how to diagnose them.

---

## Wrong Motor Direction (Single Motor)

**Symptom:** Quad lifts but drifts or rotates on the yaw axis; one arm tilts down immediately on throttle.

**Cause:** One motor spinning the wrong direction. Motors come as CW and CCW variants; some have reversed bullet connectors. If you swap any two of the three motor wires, the motor reverses.

**Fix:**  
With DSHOT protocol, reverse direction in Betaflight without rewiring:
```
# In Motors tab — with props OFF, run motor direction test
# Or in CLI:
set motor_direction = REVERSED   # reverses ALL motors

# For individual motor reversal (BF 4.2+):
set motor_1_direction = REVERSED
save
```
Or physically swap any two of the three phase wires on the offending motor.

---

## All Motors Spinning the Wrong Direction

**Symptom:** Quad immediately flips on throttle in the same direction every time. Might arm and immediately crash.

**Cause:** All four motors are reversed — either the ESC was flashed with reversed defaults, or all motors were wired backwards.

**Diagnose:** In Betaflight Motors tab (props OFF), spin each motor and confirm rotation against the [Betaflight motor layout diagram](https://betaflight.com/docs/wiki/guides/current/Motor-Spin-Directions) for your frame.

**Fix in CLI:**
```
# Reverse all motors at once
set motor_direction = REVERSED
save
```
If only some are wrong, use `set motor_n_direction = REVERSED` per motor.

---

## Props On Wrong Motors (Quad Flips on Arm)

**Symptom:** Quad immediately flips toward one side the moment throttle is applied, no matter the tune.

**Cause:** CW prop on a CCW motor, or props assigned to the wrong arms.

**Fix:** Check the Betaflight motor layout diagram. Each motor position is labeled 1–4 with expected rotation direction. CW props (bent leading edge clockwise when viewed from above) go on CCW-spinning motors, and vice versa.

Always spin motors with props OFF first and verify direction visually before installing props.

---

## ESC Desync

**Symptom:** Motor stutters or stops momentarily under load; sudden loss of throttle response; audible click/stutter followed by a flip.

**Cause:** ESC loses synchronization with the motor's back-EMF signal. Common triggers: aggressive acceleration, worn bearings, too-high RPM for the ESC's PWM frequency, bad filtering.

**Fix options:**
1. Lower RPM filtering demand — enable RPM filter (bidirectional DSHOT)
2. Increase ESC PWM frequency: 48 kHz or 96 kHz (see [ESC kHz](../esc-khz/))
3. Reduce motor timing if using advance timing
4. Check motor bearings — worn bearings cause irregular back-EMF
5. Reduce D-term if oscillation is forcing rapid motor reversals

---

## FC Boots With Props Arming Immediately

**Symptom:** Motors spin immediately on power-up.

**Cause:** Arm switch was left in ARM position during boot, or `motor_stop` is off and throttle calibration is off.

**Fix:** Always power up with arm switch in DISARMED position. Enable pre-arm or arm switch mode in Modes tab. Never use throttle-stick arming in flight without a dedicated arm switch.

---

## OSD Not Showing

**Symptom:** Black video or video with no OSD overlay.

**Cause (analog):** OSD chip not initialized, wrong UART, or camera not connected to FC video input (connected directly to VTX instead).

**Cause (digital):** MSP OSD not enabled, wrong serial port assigned.

```
# For digital systems — set MSP OSD
set osd_displayport_device = MSP
set displayport_msp_serial = 1   # match the UART connected to air unit
save
```

---

## Video Noise / Lines on Analog Feed

**Symptom:** Horizontal lines, rolling noise pattern, or complete white-out on the video feed.

**Common causes:**
- VTX powering up before camera has stable voltage — add a small LC filter on VTX power
- ESC switching noise coupling into video ground — separate power rails, add capacitor on battery leads
- VTX and camera sharing a noisy 5V rail — use a dedicated LC-filtered regulator for camera

---

## No Arming (Prearm / Throttle High)

**Symptom:** Quad refuses to arm; OSD shows RXLOSS, THROTTLE, or ANGLE arming flag.

**Fix checklist:**
1. Throttle stick at minimum
2. Arm switch in DISARM position before applying power
3. RC link connected (check RXLOSS flag)
4. Angle mode (if enabled) — horizon must be roughly level
5. GPS fix required if `GPS_FIX` arming requirement is set

Run `status` in CLI to see all active arming flags.
