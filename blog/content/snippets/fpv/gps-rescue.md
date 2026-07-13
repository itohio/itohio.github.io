---
title: "GPS Rescue Configuration"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "gps", "rescue", "failsafe", "return-to-home", "long-range"]
---

GPS Rescue is Betaflight's return-to-home failsafe. When triggered (link loss, low battery, or manual activation), the quad climbs to a configured altitude and flies back to where it armed.

---

## Prerequisites

- GPS module connected and assigned to a UART (enable **GPS** in the Ports tab for that UART)
- 6+ satellite lock before arming (configurable minimum)
- GPS Rescue mode added in the Modes tab (on a switch or tied to failsafe)

---

## Basic Configuration (CLI)

```
set gps_provider = UBLOX       # or NMEA for generic modules
set gps_baudrate = 115200      # match GPS module baud rate

set gps_rescue_angle = 30      # max tilt angle during rescue (degrees)
set gps_rescue_initial_climb = 15  # climb height above arm point (m)
set gps_rescue_altitude = 30   # target cruise altitude during rescue (m)
set gps_rescue_return_alt = 30 # altitude for return flight
set gps_rescue_speed = 300     # return speed (cm/s = 3 m/s)
set gps_rescue_descent_dist = 20 # meters from home to start descending
set gps_rescue_landing_alt = 5  # altitude to switch to landing mode
set gps_rescue_disarm_threshold = 15  # crash detection on landing

save
```

---

## Failsafe Integration

GPS Rescue integrates with Betaflight's failsafe system. In the Failsafe tab:

1. Set **Failsafe Stage 2** to `GPS RESCUE`
2. Set a reasonable Stage 1 guard time (2–3 seconds of link loss before Stage 2 triggers)
3. Enable `GPS_RESCUE` mode on an AUX switch for manual activation

```
set failsafe_procedure = GPS-RESCUE
set failsafe_delay = 20       # 20 × 0.1s = 2 seconds
set failsafe_recovery_delay = 20
```

---

## Arming Checks

With GPS Rescue set as failsafe, Betaflight requires a GPS fix before arming (configurable):

```
set gps_rescue_min_sats = 6    # minimum satellites required
set gps_rescue_allow_arming_without_fix = OFF  # don't arm without fix
```

For test flights in known safe areas you can temporarily allow arming without fix — but never fly with GPS Rescue as failsafe without an actual GPS fix.

---

## Tuning Tips

- **Initial climb** should clear local obstacles. In open fields, 15 m is fine. Near trees or structures, increase to 30–50 m.
- **Return altitude** sets the altitude used during the homeward flight. Should be above the highest obstacle on the return path.
- **Return speed** (default 300 cm/s = 3 m/s) is conservative. For long-range builds, increase to 500–700 cm/s.
- **Descent distance** is how far from home the quad starts its final descent. Increase for windy conditions.

---

## Testing

Always test GPS Rescue on a calm day over open ground before relying on it:

1. Arm and take off; get a GPS home point locked
2. Fly 100 m away
3. Activate GPS Rescue via switch (not by killing link)
4. Verify the quad climbs, turns toward home, and descends

Be ready to take back control if behavior is wrong.

---

## Notes

- GPS Rescue requires a magnetometer (compass) for reliable heading. Some builds use the GPS module's built-in compass (enable in Betaflight's GPS tab).
- Without a compass, the quad uses accelerometer + GPS velocity for heading estimate — less accurate but functional.
- GPS accuracy varies with satellite count and atmospheric conditions. HDOP < 2.0 is acceptable for rescue; < 1.5 is ideal.
