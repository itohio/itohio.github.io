---
title: "Giving My Quad a Cockpit Voice, Part 1: Why a Drone Should Talk to You"
date: 2026-08-16T09:00:00+03:00
description: "A low-battery number in the corner of the OSD is an interface failure, not a pilot failure. Why I made my RadioMaster GX12 speak, and the one flight-controller setting the whole thing depends on."
summary: "A low-battery number in the corner of the OSD is an interface failure, not a pilot failure. Why I made my RadioMaster GX12 speak, and the one flight-controller setting the whole thing depends on."
draft: false
toc: true
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - radiomaster-gx12
  - elrs
  - crsf
  - telemetry
  - betaflight
  - report-cell-voltage
  - lihv
keywords: ["EdgeTX audio battery warning", "report_cell_voltage Betaflight", "FPV per cell voltage telemetry", "RadioMaster GX12 setup", "EdgeTX speak battery voltage"]
series:
  - EdgeTX Cockpit Voice
thumbnail: "cover.jpg"
---

> **EdgeTX Cockpit Voice**, part 1 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [Part 2: The Calibration Every Battery Warning Rests On ›](/fpv/edgetx-cockpit-voice-calibration/)

You know the flight. You are a long way out, the terrain is good, the lines are
flowing, and you are entirely inside the goggles. Somewhere in the corner of the
OSD a voltage number has been quietly dropping for the last ninety seconds and
you have not looked at it once, because you were busy flying. Then the OSD starts
blinking at you, and you do the arithmetic: distance home, headwind, remaining
sag. And the arithmetic says no.

That flight ends in a walk. Sometimes it ends in a walk with a bag.

The thing that always bothered me about this failure mode is that it is _purely_
an interface problem. The data was there the whole time. The radio knew. The
quad knew. The only broken link in the chain was that the information was
rendered as small glowing digits in the periphery of a human who was
concentrating on something else.

## An aircraft would never do this to you

Here is the part that struck me as absurd. Put a pilot in a Cessna and the
aircraft will not let a low-fuel state be a visual detail you might miss. It
will say so. Out loud. Repeatedly. Gear-up warnings, stall warnings, altitude
callouts, terrain warnings. An entire century of aviation human-factors
engineering converged on one conclusion: **for time-critical state changes,
audio beats vision, because audio does not require the pilot to look
somewhere.**

And yet the default FPV configuration for a 250-gram aircraft with a four-minute
endurance is... a number in the corner of the screen.

So I fixed it. My GX12 now talks to me. Not with a Lua script, not with anything
exotic, just EdgeTX logical switches and special functions, which have been
sitting in the firmware the whole time.

This is the first time I have built this, and I want to be upfront: **there are
less clumsy ways to do parts of it.** I will show you exactly where mine is
clumsy and why, because that is more useful than pretending I got it right
first time. But the core of it works, and one specific warning, the
half-capacity "return home" callout, has genuinely saved flights for me on long
range missions. It gives me the cue to start planning the trip back while I
still have the budget to make it, instead of discovering the problem when the
budget is already spent.

![RadioMaster GX12](cover.jpg "RadioMaster GX12")

## Step zero: make every drone speak the same language

This is the single change that makes the whole system possible, and it happens
on the flight controller, not on the radio.

By default, the CRSF battery frame reports **pack voltage**. That is useless as
a fleet-wide trigger, because my fleet spans 1S to 4S. A "3.5 V" threshold means
nothing when one aircraft runs a single 18650 and another runs a 4S LiHV pack.
I would need a different threshold set per model, maintained by hand, forever.

So I set every aircraft to report **average cell voltage** instead. In
Betaflight this is a single parameter:

```text
set report_cell_voltage = ON
save
```

It is also exposed in the Betaflight configurator under _Power & Battery_ as
"Report cell voltage instead of pack voltage in telemetry". The FC divides pack
voltage by its detected cell count before it ever reaches the telemetry frame.

Now `3.5 V` means the same physical thing on the 1S whoop, the 2S rippers and
the 4S 3-inch. One threshold ladder, whole fleet.

> **Note on the INAV equivalent:** I run Betaflight on everything relevant here,
> so I have only verified this on Betaflight. If you are on INAV, check the
> parameter name before assuming it is identical — I have not measured it.

### Why not do the division in EdgeTX instead?

You can. EdgeTX lets you set a custom **Ratio** on a telemetry sensor, so you
could leave the FC reporting pack voltage and divide by cell count in the radio.

I deliberately did not, and you can see the decision in my config, the RxBt
sensor has no correction applied at all:

```yaml
telemetrySensors:
   14:
      id1:
         id: 8              # CRSF frame 0x08, BATTERY_SENSOR
      id2:
         instance: 0
      label: "RxBt"
      unit: 1               # volts
      prec: 1               # one decimal place
      cfg:
         custom:
            ratio: 0        # no scaling
            offset: 0       # no offset
```

Two reasons for doing it on the aircraft instead:

1. **The ratio is per-model in the radio, but the cell count is per-battery.**
   A radio-side divide-by-four is wrong the moment I fly the same airframe on a
   3S pack.
2. **LiHV breaks a hardcoded guess.** My 3-inch runs 4S LiHV at 4.35 V/cell
   fully charged, 17.4 V on the pack. A radio that has been told "assume 4S"
   copes fine, but a radio doing cell-count _detection_ from a divided number
   does not. The FC already knows its own cell count from the actual detection
   logic. Let the thing that knows do the maths.

The trade-off is honest: doing it FC-side means every new aircraft needs that
CLI line, and if you forget it, your warnings fire at absurd times. So it belongs
on the setup checklist for a new build, next to the things you cannot see either.

One threshold ladder now means the same physical thing on every aircraft I own.
Next: the three buttons that decide which warnings are allowed to speak, and the
AND gate that keeps them from tripping over each other.


---

> **Series:** EdgeTX Cockpit Voice, part 1 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [Part 2: The Calibration Every Battery Warning Rests On ›](/fpv/edgetx-cockpit-voice-calibration/)
