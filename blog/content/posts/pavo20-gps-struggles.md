---
title: "Pavo20 Pro II GPS Struggles — Noise, Harmonics, and the Search for Better Isolation"
date: 2026-07-13
description: "The Pavo20 Pro II can barely find 3 GPS satellites where a 1S digital build finds 20+. Here is what I found on the spectrum analyser, what I've tried, and what remains unsolved."
draft: true
toc: true
categories:
  - FPV
  - Hardware
tags:
  - fpv
  - gps
  - pavo20
  - noise
  - rf
  - tinysa
  - betaflight
  - inav
series:
  - FPV Builds
---

The Pavo20 Pro II is a capable 2.5" whoop with GPS built in. On paper that should mean GPS Rescue on a micro build — a genuine safety net. In practice the GPS is nearly useless in most flying conditions: I watch the satellite count sit at 2 or 3 while a different quad on the same field, at the same time, locks onto 20+. This is the story of what I found and where I am with it.

---

## The Symptom

First flight of the day. Field is open, sky is clear, no buildings. I power up both quads and wait:

| Quad | Time to first fix | Satellites at fix |
|------|------------------|------------------|
| 1S Matrix 3-in-1 digital build | ~90s | 20–22 |
| Pavo20 Pro II | >5min | 2–4 |

The 1S build is smaller. Its electronics are arguably denser. The GPS module is the same generation. The difference is the video system: the 1S build runs a DJI O3 Air Unit on a whoop stack with a camera module, while the Pavo20 runs an integrated stack where the VTX, FC, ESC, and GPS share a single compact board.

A whoop chassis has almost no space between the GPS module and everything else generating noise.

---

## First Look: Physical Setup

Before touching a spectrum analyser I did the obvious checks.

<!-- IMAGE: photo of Pavo20 Pro II with GPS module and buzzer soldered, showing physical proximity to VTX area -->
*[TODO: Photo — soldered GPS and buzzer on Pavo20 Pro II stack]*

The GPS antenna on the Pavo20 sits on the top of the stack, directly above the ESC/VTX board. The antenna ground plane is the PCB copper — which is also carrying motor switching currents and VTX RF ground. There is no physical shield between the GPS module's LNA and the VTX output stage.

<!-- IMAGE: photo of GPS wiring and filtering attempt — ferrite beads, filtering capacitors on power line -->
*[TODO: Photo — GPS wiring with ferrite beads and power line filtering]*

I added ferrite beads on the GPS power line and a 100µF cap at the module's power pins. This is the standard low-frequency noise fix. It made no measurable difference to satellite count.

---

## Spectrum Analysis — 1S Build vs Pavo20

Time to actually measure the noise floor where GPS operates. GPS L1 band is at **1575.42 MHz**. The constellation signals arriving at the antenna are extraordinarily weak — typically around −130 dBm. Any local interference in the 1.5–1.6 GHz range drowns them out.

I connected a TinySA to a short wire antenna positioned near the stack on each quad, with the quads powered on and armed (motors running via a motor test jig, no props).

<!-- IMAGE: TinySA screenshot — 1S Matrix 3-in-1 digital build, 1.2GHz–1.8GHz span, showing noise floor -->
*[TODO: TinySA screenshot — 1S digital build, 1.2–1.8 GHz span]*

<!-- IMAGE: TinySA screenshot — Pavo20 Pro II, same span and settings, showing elevated noise in GPS band -->
*[TODO: TinySA screenshot — Pavo20, same 1.2–1.8 GHz span]*

The contrast is stark. The 1S build shows a clean noise floor in the GPS band with only the expected atmospheric background. The Pavo20 shows a raised noise floor across the entire 1.2–1.8 GHz range, with several distinct spurs in the 1.4–1.6 GHz region.

---

## The Switching Harmonic Problem

ESCs run at a PWM switching frequency — 24, 48, or 96 kHz on modern stacks. Harmonics of these frequencies should be at audio frequencies and their multiples, nothing close to GPS L1.

The actual interference mechanism is different: **motor PWM generates fast-edge current transitions**, and those transitions excite resonances in the power distribution traces, solder joints, and capacitor parasitics. The result is broadband conducted and radiated noise that appears at unpredictable frequencies well above the fundamental switching frequency.

Additionally: **video transmitters** on 5.8 GHz can generate sub-harmonics and mixing products. A 5.8 GHz VTX at 200mW can produce detectable energy at 5800/4 = 1450 MHz — right in the GPS band.

I confirmed this in a different environment:

<!-- IMAGE: TinySA screenshot taken in basement/shielded environment — 1.5GHz region showing switching harmonics leaking in -->
*[TODO: TinySA screenshot — basement test, 1.4–1.6 GHz span, showing harmonic spurs from the Pavo20 stack]*

<!-- IMAGE: TinySA screenshot — GPS L1 signal reference, showing what the GPS signal actually looks like at 1575.42 MHz -->
*[TODO: TinySA screenshot — 1575.42 MHz GPS L1 signal reference, demonstrating the signal level the module is trying to receive]*

The GPS L1 signal is genuinely tiny. The noise spurs I measured are 40–50 dB above the GPS signal level. The GPS module's LNA is fighting a losing battle.

---

## What I Have Tried

### 1. Ferrite Beads on GPS Power

Ferrite beads on the VCC and GND lines to the GPS module. Effective for conducted noise on the power rail at lower frequencies. No effect on radiated interference from the VTX at 1.5 GHz — RF doesn't travel via the power line at that frequency.

**Result: No improvement in satellite count.**

### 2. Reducing VTX Power

Setting VTX to pit mode (0 mW) or lowest power (25 mW) during the GPS acquisition phase. This reduces the 5.8 GHz fundamental and therefore its sub-harmonics.

**Result: Marginal improvement.** Acquisition sometimes reaches 6–8 satellites with VTX off, but that is not a practical flight scenario.

### 3. ESC PWM Frequency Reduction

Dropped the ESC PWM frequency from 48 kHz to 24 kHz. Lower frequency means fewer harmonics per unit frequency range — the harmonic density drops.

**Result: Minimal difference.** The noise profile shifted but did not disappear from the GPS band.

### 4. Physical Shielding Attempts

Wrapped the GPS module and antenna area with copper foil tape connected to ground. This creates a Faraday shield around the module, but the antenna still needs a line of sight to the sky — and the antenna is right next to the noise source.

**Result: Slight improvement, but the geometry makes it nearly impossible to shield the antenna without blocking the sky-facing GPS signals.**

---

## Root Cause Assessment

The Pavo20 Pro II's integrated stack design prioritizes compactness over RF isolation. This is a deliberate trade-off for a 2.5" chassis — there is simply no room for the separation that would make a difference.

The interference has at least two components:

1. **Conducted noise** on the GPS module power rail from the switching regulators and ESC current peaks
2. **Radiated RF** from the VTX at sub-harmonics that land in the GPS L1 band

Ferrite beads address component 1 partially. Component 2 requires either physical distance (not available on a whoop) or a shielded GPS module with its own ground plane isolated from the main stack.

The GPS module used in the Pavo20 is a standard M8N/M10 variant in a miniaturised SMD package — there is no shield can over the RF LNA. This is common on whoop-class GPS builds; the assumption is that flying outdoors provides enough sky view to overcome the degraded SNR.

That assumption holds for some flying environments and fails completely in others.

---

## Where I Am Now

I am still searching for a reliable noise isolation solution. Current experiments in progress:

- **Longer GPS cable**: moving the module 5–8 cm further from the stack via a thin cable. Even small physical separation dramatically reduces near-field coupling. The trade-off is weight and mechanical complexity on a build that is supposed to be compact.
- **Active re-radiation (GPS repeater)**: using an external active GPS antenna with a separate LNA, outside the aircraft envelope, connected via a thin coax. This is overkill for a whoop, but it would confirm whether the problem is purely proximity-based.
- **INAV migration**: INAV's navigation stack handles degraded GPS lock more gracefully than Betaflight GPS Rescue. If I cannot eliminate the noise, a better software stack might at least work reliably at 6–8 satellites instead of requiring 12+.

The 1S Matrix build continues to embarrass the Pavo20 on satellite count every single session. Until I find a fix, GPS Rescue on the Pavo20 remains in the "emergency backup that might not work" category rather than a reliable safety feature.

---

## Next Steps

I'll update this article as experiments progress. If you've solved this on a similar integrated-stack whoop build, I want to hear about it.

The TinySA files and Betaflight configuration dumps for both builds are available — I'll link them once I've cleaned up the directory structure.
