---
title: "FPV Terminology Reference"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "reference", "glossary", "terminology", "beginner"]
---

A single-page reference for the acronyms and terms that fill every FPV forum post and Discord server. Organized by domain so related terms stay together.

---

## Hardware Abbreviations

| Acronym | Full name | What it does |
|---------|-----------|--------------|
| **FC** | Flight Controller | The brain — runs Betaflight/INAV, reads gyro, outputs motor commands |
| **ESC** | Electronic Speed Controller | Converts FC motor commands to three-phase AC for the brushless motors |
| **4-in-1 ESC** | Four-in-one ESC | All four ESC channels on one board; sits under or above the FC in a stack |
| **VTX** | Video Transmitter | Sends FPV video wirelessly to your goggles; typically 5.8 GHz analog or 2.4/5.8 GHz digital |
| **VRX** | Video Receiver | The goggle-side receiver for analog video; replaced by a display unit in digital systems |
| **OSD** | On-Screen Display | Overlays telemetry (battery, RSSI, flight mode) onto the FPV video feed |
| **RX / Receiver** | RC Receiver | Receives your RC link from the radio and passes stick commands to the FC |
| **TX / Transmitter** | RC Transmitter | Your radio controller; outputs a control link (ELRS, CRSF, etc.) |
| **GPS** | Global Positioning System | Provides position, altitude, speed; required for GPS Rescue and navigation modes |
| **IMU** | Inertial Measurement Unit | Gyro + accelerometer combined on the FC; measures rotation rate and linear acceleration |
| **MPU / ICM** | Motion Processing Unit | Specific gyro chip brands: MPU-6000, ICM-42688-P — the IMU chip |
| **BEC** | Battery Eliminator Circuit | Voltage regulator on the stack; provides 5 V or 9 V rails for FC, VTX, camera |
| **PDB** | Power Distribution Board | Distributes battery voltage to all four ESCs and the BEC; often integrated into 4-in-1 ESC |
| **XT60 / XT30** | Xt connectors (60 A / 30 A) | Standard LiPo battery connectors; XT60 for 5" and larger, XT30 for micro/whoop |
| **Cap** | Capacitor | Low-ESR electrolytic soldered to battery pads; suppresses voltage spikes that crash the FC |
| **AIO** | All-in-one | FC + ESC + (sometimes) VTX on a single board; common in whoops and micros |
| **Stack** | FC stack | FC board and ESC board bolted together with M2/M3 standoffs |

---

## Motors and Props

| Term | Meaning |
|------|---------|
| **KV** | Motor velocity constant: RPM per volt with no load. 2400 KV on 4S → ~40,000 RPM no-load at 16.8 V. Lower KV = more torque, larger props. |
| **Stator size** | The motor's stator diameter × height in mm. "2306" = 23 mm diameter, 6 mm stator height. Larger stator = more power. |
| **Pole count** | Number of magnet poles; affects RPM vs electrical frequency. 12N14P = 12 stator teeth, 14 magnet poles. |
| **eRPM** | Electrical RPM = mechanical RPM × (poles / 2). Used by RPM filter and DSHOT telemetry. |
| **Prop notation** | e.g., **5148**: first two digits = diameter in tenths of an inch (5.1"), next two = pitch in tenths (4.8"). |
| **Pitch** | How far a prop advances per revolution in ideal air. Higher pitch = more speed, more load. |
| **Tri-blade / Bi-blade** | Blade count. Tri-blade = more thrust per RPM, more noise. Bi-blade = smoother, more efficient at speed. |
| **CW / CCW** | Clockwise / Counter-clockwise rotation. Standard Betaflight motor layout: M1 CCW, M2 CW, M3 CW, M4 CCW (Betaflight props-in). |
| **T-mount** | Prop attachment with a central bolt. Used on 5" and larger. |
| **Press-fit** | Prop slides directly onto motor shaft — common on whoops. Check fit torque before every session. |
| **AUW** | All-up weight. Total flying mass including battery. TWR = thrust / AUW. |
| **TWR** | Thrust-to-weight ratio (total full-throttle thrust ÷ AUW). ~4:1 for comfortable freestyle; 6:1+ for racing/maximum agility. |

---

## Battery and Power

| Term | Meaning |
|------|---------|
| **LiPo** | Lithium Polymer. Standard FPV battery chemistry. High discharge rate, high energy density, fire risk if damaged. |
| **Li-Ion** | Lithium Ion (18650/21700 cells). Lower C-rating than LiPo, higher capacity, better for long-range builds. |
| **Cell count** | Number of cells in series. "4S" = 4 cells × 4.2 V = 16.8 V fully charged; 14.8 V nominal. |
| **mAh** | Milliamp-hours. Capacity. 1300 mAh = can supply 1.3 A for one hour, or 13 A for 6 minutes. |
| **C-rating** | Max continuous discharge multiplier. 75C on a 1300 mAh = 97.5 A max continuous. Take C-ratings with skepticism — manufacturers inflate them. |
| **IR / Internal Resistance** | Voltage drop inside the cell under load. Low IR = healthy cell. Measure with a battery checker at rest and under load. |
| **Sag** | Voltage drop under full throttle pull. Severe sag → brownout reset of the FC. |
| **Storage voltage** | 3.8 V per cell. Store LiPos here if not flying for more than a day. Storing full or empty accelerates degradation. |
| **Balance lead** | JST-XH connector with one wire per cell. Used to balance-charge each cell individually. Never fly an unbalanced pack. |
| **UBEC** | Universal BEC. Switching regulator; more efficient than linear BECs at higher voltage drops. |

---

## RC Link and Radio

| Term | Meaning |
|------|---------|
| **ELRS** | ExpressLRS. Open-source long-range RC link. 2.4 GHz or 900 MHz. Extremely low latency (3–6 ms at 500 Hz), long range, free. |
| **CRSF** | Crossfire Serial protocol by TBS. Full-duplex 400 kbps serial between TX module and FC; also the frame format used by ELRS. |
| **FrSky** | Radio manufacturer; D8/D16/ACCESS protocols. Legacy but still very common (Taranis/Jumper). |
| **SBUS** | Digital serial protocol (inverted UART, 100 kbps) for RX → FC. One wire, 16 channels. |
| **PWM** | Pulse Width Modulation. Legacy one-wire-per-channel RC signal. 1000–2000 µs. Rarely used on modern FC stacks. |
| **RSSI** | Received Signal Strength Indicator. Signal strength in dBm or as 0–100% (0 = lost, −90 dBm typical minimum). |
| **LQ** | Link Quality. Percentage of received packets over the last 100 (ELRS). More useful than RSSI alone; LQ <70% = warning, <50% = fly home. |
| **SNR** | Signal-to-Noise Ratio. How far the signal is above the noise floor in dB. Negative SNR is still usable in ELRS. |
| **Packet rate** | ELRS update rate: 50 Hz, 150 Hz, 250 Hz, 500 Hz. Higher = lower latency, shorter range. |
| **Telemetry** | Downlink data from quad to radio: battery voltage, GPS position, RSSI, LQ, flight mode. |
| **Failsafe** | Behavior when RC link is lost. Betaflight Stage 2 procedures: Drop (motors off immediately), Land (controlled descent), GPS Rescue (return to home — needs GPS). Separately, a *receiver's* "Hold" mode keeps outputting the last stick values (dangerous — the FC never sees the loss). |
| **BVLOS** | Beyond Visual Line of Sight. Requires special authorization in almost all regulatory frameworks. |

---

## Betaflight / Firmware

| Term | Meaning |
|------|---------|
| **Betaflight** | Most popular FPV flight controller firmware. Acro (rate mode) focused. `github.com/betaflight/betaflight` |
| **INAV** | iNav. Navigation-focused fork of Betaflight. Adds GPS waypoints, fixed-wing support, RTH, cruise modes. |
| **Acro mode** | Pure rate control — FC only corrects gyro drift, stick = rotation rate. Default FPV flying mode. |
| **Horizon / Angle** | Self-leveling modes. FC uses accelerometer to maintain a level attitude. Useful for beginners only. |
| **Arming** | State where motors are enabled. Most setups require: RC link good, no arming flags, arm switch. |
| **PID** | Proportional-Integral-Derivative. The control loop at the core of Betaflight. See [PID Basics](../../tuning/pid-basics/). |
| **Rates** | How stick deflection maps to rotation speed (°/s). Four styles: Betaflight, Actual, KISS, Quickrates. See [Rate Modes](../../tuning/rate-modes/). |
| **Blackbox** | Flight data recorder built into Betaflight. Logs gyro, setpoint, motors, PIDs at up to 4 kHz. See [Blackbox Logging](../../tuning/blackbox-logging/). |
| **CLI** | Command Line Interface. Text terminal in Betaflight Configurator for direct parameter access. `diff all` to backup. |
| **RPM filter** | Dynamic notch filters locked to motor electrical RPM via DSHOT telemetry. Requires bidirectional DSHOT. |
| **Dynamic notch** | Auto-adapting notch filter that tracks dominant noise frequencies in the gyro signal. |
| **TPA** | Throttle PID Attenuation. Reduces P (and optionally D) at high throttle to compensate for faster motor response at high RPM. |
| **iterm_relax** | Suppresses I-term integration during fast stick inputs (flips/rolls). Prevents I-term windup and bounce-back. |
| **anti_gravity** | Temporarily boosts I-term when throttle changes rapidly. Prevents altitude drop on sudden throttle chops. |
| **d_min / d_max** | D-term that scales between a lower value at rest and a higher value during fast maneuvers. Reduces motor heat at hover. |
| **FF / Feedforward** | Adds a motor command proportional to stick movement speed. Reduces following lag. |
| **DSHOT** | Digital motor protocol (DSHOT150/300/600/1200). Replaces analog PWM with a binary frame. Required for RPM filter. |
| **Bidirectional DSHOT** | DSHOT with telemetry back from ESC to FC. Provides eRPM per motor → enables RPM filter. |
| **Turtle mode** | Flip-over-after-crash. Reverses motors to flip the quad right-side up without walking to it. |
| **Air Mode** | Keeps PID active at zero throttle. Prevents yaw twitch on throttle-off. Required for acrobatics. |
| **GPS Rescue** | Emergency return-to-home using GPS. Requires GPS lock and home point set. |
| **Master multiplier** | Scales all PID gains proportionally. Useful when switching motor/prop combinations. |
| **diff all** | CLI command that outputs only non-default settings. Use for backups. Smaller than `dump`, still complete. |

---

## Video Systems

| Term | Meaning |
|------|---------|
| **Analog** | Traditional FPV: 5.8 GHz FM video, NTSC/PAL. Low latency (~1 ms), low resolution (~600 TVL), cheap. Goggles: Fatshark HDO2, Skyzone. |
| **Digital HD** | DJI O3/O4, Walksnail Avatar, HDZero. 1080p video, higher latency (20–35 ms), better image quality. |
| **Latency (glass-to-glass)** | Total delay from camera lens to goggle display. Analog: 3–7 ms. Digital: 20–40 ms. Critical for proximity flying. |
| **TVL** | TV Lines. Analog camera resolution. 1200 TVL is roughly equivalent to SD video. |
| **WDR** | Wide Dynamic Range. Camera feature that compresses highlights and shadows. Useful in mixed indoor/outdoor light. |
| **DVR** | Digital Video Recorder. Records the FPV feed in the goggles for crash review and content. |
| **OSD** | Overlays from FC superimposed on video. Betaflight OSD elements: battery, RSSI, LQ, mode, speed, position. |
| **VTX power** | Transmit power in mW. 25 mW/200 mW typical. Above 25 mW is illegal in many countries/environments without authorization. |
| **vtxtable** | Betaflight CLI table mapping VTX power levels to mW values. Needed for smart audio / Tramp control. |

---

## Aerodynamics (Quick Reference)

| Term | Meaning |
|------|---------|
| **Downwash** | Column of accelerated air pushed downward by the rotor. Extends several prop diameters below the craft. |
| **Propwash** | Turbulence experienced when the craft descends into its own downwash. Causes the characteristic oscillation on dive exits. See [Propwash](../../aerodynamics/propwash/). |
| **Ground effect** | Increased lift efficiency when flying within ~1 prop diameter of the ground. Downwash can't fully develop; spreads radially. |
| **Tip vortex** | Pressure leak at the blade tip. Reduces effective disk area. Ducted fans eliminate this — see [Ducted Fans](../../aerodynamics/ducted-fans/). |
| **AoA** | Angle of Attack. The angle between the prop blade chord line and the oncoming airflow. Related to thrust and stall. |
| **Blade pitch** | Angle of the prop blade relative to the plane of rotation. Higher pitch = more pitch per revolution, more load. |
| **Thrust curve** | Relationship between motor throttle command and actual thrust output. Not linear — thrust scales roughly with RPM². |

---

## Prop Notation Decoder

```
 5 1 4 8
 │ │ │ │
 │ │ └─┴── Pitch × 10 in inches → 48 = 4.8"
 └─┴────── Diameter × 10 in inches → 51 = 5.1"
```

Examples: `5148` = 5.1" dia, 4.8" pitch | `4045` = 4.0" dia, 4.5" pitch | `3020` = 3.0" dia, 2.0" pitch

Some manufacturers append a third segment: `5148-3` = 5.1" × 4.8", tri-blade.

---

## Motor Designation Decoder

Motor designations follow `DDHHkv-NNNN` loosely:

```
 2 3 0 6   2 4 5 0 K V
 │ │ │ │   └───────┘
 └─┴─┴─┴── Stator: 23mm diameter × 06mm height
           KV rating: 2450 RPM/V
```

The stator size drives maximum power: bigger stator → more torque → heavier prop, higher voltage.

| Build type | Typical motor | Typical prop | Battery |
|-----------|--------------|-------------|---------|
| 5" freestyle | 2306–2407, 1700–2450 KV | 5145–5148 tri | 4S–6S |
| 5" racing | 2204–2306, 2400–2600 KV | 5040–5140 bi | 4S–6S |
| 3.5" / Cinelog | 1404–1507, 3600–4000 KV | 3540–3545 | 3S–4S |
| Tinywhoop 75mm | 0802–0803, 19000–25000 KV | 40mm ducted | 1S |
| Pavo20 / Meteor | 1102–1103, 8700–11000 KV | 2" ducted | 1S |
