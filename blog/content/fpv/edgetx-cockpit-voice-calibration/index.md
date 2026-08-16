---
title: "Part 2: The Calibration Every Battery Warning Rests On"
date: 2026-08-16T10:00:00+03:00
description: "A miscalibrated voltage reading does not look broken, it looks plausible. Why report_cell_voltage makes a vbat_scale error propagate twice, and what the 3S-to-4S move does to your alarms."
draft: false
toc: true
weight: 2
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - betaflight
  - vbat-scale
  - calibration
  - battery
  - lihv
  - edgetx
  - telemetry
keywords: ["Betaflight vbat_scale calibration", "battery voltage calibration drone", "report_cell_voltage cell count detection", "Betaflight configurator calibration motors", "3S to 4S battery warning wrong"]
series:
  - EdgeTX Cockpit Voice
---

Every battery warning in this series is a comparison against a number. So before
any of it, the number has to be true. This part is about the setting that decides
whether it is, and it is the one I have most obviously failed to get right.

## The whole ladder rests on a calibration you have probably skipped

I need to put this immediately after the previous section, because everything
that follows depends on it and I do not want anyone building this on a bad
foundation.

**Your battery warnings are exactly as good as your voltage calibration.**

That sounds obvious written down. It is not obvious in practice, because a
miscalibrated voltage reading does not look broken. It looks like a perfectly
plausible number that happens to be wrong by 200 mV, and every threshold in the
ladder above inherits that error silently.

I have two aircraft that are miscalibrated right now, which means **my warnings
on those two fire too late.** Not "slightly imprecisely" — too late, in the
direction that costs you a pack. I know this and I have not fixed it yet, which
is the kind of admission this blog exists for.

The knob is `vbat_scale` in Betaflight. It corrects the ADC divider ratio for
the actual resistors on your board, which vary between boards, and it is set to
a generic default that is right for nobody in particular.

### The 3S-to-4S trap

The specific way this bit me is worth spelling out, because it is a natural
thing to do and there is no warning.

I had aircraft set up and flying on **3S**, then moved them to **4S** for
testing. Nothing in that transition tells you your calibration is now costing
you more. But it is, for a compounding reason.

`report_cell_voltage = ON` means the FC divides pack voltage by its **detected**
cell count. And that detection is itself derived from the measured pack voltage
at power-on, the FC divides what it reads by a maximum-cell-voltage constant
and rounds. So a voltage error propagates **twice**:

1. Directly, into the reported per-cell figure.
2. Potentially again, by pushing the detected cell count to the wrong integer.

That second path is the nasty one, because it fails *silently and plausibly*. If
a badly-scaled 4S pack reads low enough that the FC decides it is looking at 3S,
then it divides by three instead of four, and hands the radio a per-cell number
that sits comfortably in the normal range while being completely fictional. Every
threshold in my ladder would then be measuring a quantity that does not exist,
and the `ready` self-test would happily fire, because a wrong number above 4.2 V
is still a number above 4.2 V.

The self-test I was so pleased with earlier in this post checks that the signal
path works. **It does not check that the number is true.** Those are different
claims and I want to be clear about which one I have.

### The regression in the new configurator

Here is the practical annoyance, and it is the reason this is getting its own
post rather than a paragraph.

The way I used to calibrate was to spin the motors up to a modest load —
something drawing on the order of 2 A from the pack, and then switch to the
calibration tab **with the motors still running**, so I was calibrating at a
realistic operating point rather than at idle. That matters: you want the reading
trustworthy where you actually use it, under load, not just at rest on the bench.

In the current Betaflight configurator you cannot do that any more. **Leaving the
tab cuts the motors.** The workflow is simply gone.

I have not yet worked out the right replacement procedure, so I am not going to
invent one here. That is the next post: proper voltage calibration with the
current configurator, what changed, and how to get a trustworthy reading under
load without the old trick.

### One honesty note about a number earlier in this post

The 3.065 V/cell sag figure I quote further down, from an 83 A punch-out on my
3-inch, carries this same dependency. It is what the flight controller
*recorded*, and its accuracy rests on that aircraft's voltage calibration being
sound. I have not independently verified that particular airframe's `vbat_scale`
against a reference meter. Treat it as a strong indication of the shape of the
problem rather than a metrologically clean measurement.

If you build the warning system in this post and skip the calibration, you have
built something that will confidently tell you the wrong thing in a calm voice.
That is arguably worse than a number in the corner of the screen.

Two of my aircraft are still telling me the truth late. I know which two, and I
have not fixed it, which is the sort of thing that belongs in a lab notebook
rather than a tutorial.

**Next:** [Part 3, three buttons, three colours, and the AND gate](/fpv/edgetx-cockpit-voice-buttons/)
