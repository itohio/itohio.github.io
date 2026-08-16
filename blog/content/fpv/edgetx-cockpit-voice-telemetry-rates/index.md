---
title: "Part 6: Telemetry Logging, and the One Number You Have to Measure Yourself"
date: 2026-08-16T14:00:00+03:00
description: "ELRS telemetry ratio, CRSF frame round-robin and the EdgeTX log period sit in series. Why the arithmetic is not the answer, how to get the real per-sensor rate out of your own CSV, and the 3D viewer I built to read it."
draft: false
toc: true
weight: 6
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - crsf
  - elrs
  - telemetry-ratio
  - logging
  - rxmap
  - antenna-diversity
  - data-analysis
keywords: ["ELRS telemetry ratio explained", "CRSF frame types round robin", "EdgeTX SD logs period", "EdgeTX telemetry update rate", "FPV RX blind spot 3D viewer", "CRSF link statistics 1RSS 2RSS"]
series:
  - EdgeTX Cockpit Voice
thumbnail: "rxmap-sphere-airframe.png"
---

The red button in this setup writes a CSV to the SD card. That log turns out to be
the most interesting object in the whole project, and also the one I understood
worst.

## Telemetry logging, and the number you have to measure yourself

The red button drives `LOGS` with `def: "3,1"`, a **0.3 second** log period,
writing a CSV to the SD card. This is where I have to stop making claims and
start pointing at homework, because the honest answer is that I have not
measured the thing that matters.

Log fidelity is not set by the log period. It is bounded by three things in
series, and the log period is only the _last_ one:

1. **The ELRS telemetry ratio**, how often the RF link gives the downlink a
   slot at all.
2. **CRSF frame round-robin**, the FC has several different frame types to
   send, and each telemetry opportunity carries one.
3. **The EdgeTX log period**, how often the radio samples whatever value it
   most recently received.

My own sensor list makes point 2 concrete. Grouping my sensors by their CRSF
frame ID:

| CRSF ID | Frame type       | Sensors it carries                                                            |
| ------- | ---------------- | ----------------------------------------------------------------------------- |
| `0x02`  | GPS              | `GPS`, `GSpd`, `Hdg`, `GAlt`, `Sats`                                          |
| `0x08`  | BATTERY\_SENSOR  | `RxBt`, `Curr`, `Capa`, `Bat%`                                                |
| `0x1E`  | ATTITUDE         | `Ptch`, `Roll`, `Yaw`                                                         |
| `0x21`  | FLIGHT\_MODE     | `FM`                                                                          |
| `0x14`  | LINK\_STATISTICS | `1RSS`, `2RSS`, `RQly`, `RSNR`, `ANT`, `RFMD`, `TPWR`, `TRSS`, `TQly`, `TSNR` |

Note that `Sats` and `GAlt` arrive **together**, in the same frame, they can
never be out of sync with each other. But `RxBt` lives in a different frame
entirely, so it updates independently, and slower than the raw telemetry slot
rate.

```wave
{ "signal": [
  { "name": "RF packets",        "wave": "p..............." },
  { "name": "downlink slot 1:4", "wave": "0.10.10.10.10" },
  { "name": "CRSF frame",        "wave": "x.3x.4x.5x.6x",
    "data": ["GPS 0x02", "BATT 0x08", "ATT 0x1E", "FM 0x21"] },
  { "name": "RxBt fresh",        "wave": "0....1........." }
],
  "head": { "text": "Telemetry slots round-robin between CRSF frame types" }
}
```

The naive arithmetic: at a 500 Hz packet rate with a 1:4 telemetry ratio you get
125 downlink slots per second, and with four flight-data frame types
round-robining, `RxBt` would refresh about 31 times a second. Against that, a
0.3 s log period is _massively_ undersampling. I would be logging one sample in
ten and would miss every sag transient.

**But I do not believe that number, and neither should you.** It is arithmetic
from the frame structure, not a measurement. It ignores the fact that ELRS
telemetry slots carry a small payload while a CRSF GPS frame is comparatively
large, so a single frame is fragmented across multiple slots. The real
per-sensor rate is lower than 31 Hz, possibly by a lot, and I have not
established by how much.

Here is the thing though — **the measurement is sitting on my SD card already,
and on yours.** The log period is 0.3 s. If a sensor is genuinely arriving
faster than that, every row has a fresh value. If it is arriving slower, the CSV
contains _runs of identical consecutive values_, and the mean run length is
exactly the ratio between the true arrival interval and the log period.

So: count the duplicate runs per column. That gives you the real update rate of
every sensor, per aircraft, per telemetry ratio, with no assumptions. Then set
your log period to match, and set your telemetry ratio deliberately, knowing
that a low ratio buys you link robustness at the direct cost of log resolution.

That is the next thing I am going to actually do, and it will get its own post
with real numbers in it.

### I built a thing that reads these logs

Since the whole point of the red button is producing a CSV, I should mention that
I have written a browser tool that eats exactly this file:

**[RX Blind-Spot Viewer](https://rxmap-viewer.sintra.site/rxmap/)**, load an
EdgeTX SD-Logs CSV and it renders your **control link** in 3D. It runs entirely in
the browser: nothing is uploaded, there is no account, and the log never leaves
your machine.

![RX Blind-Spot Viewer. Sphere view in the airframe frame, RSSI plotted as an empirical antenna pattern](rxmap-sphere-airframe.png "RX Blind-Spot Viewer. Sphere view, airframe frame, RSSI (worst of 1RSS/2RSS)")

Three views:

- **Cloud**, true 3D flight positions, coloured by whatever link metric you pick
- **Sphere**, the one above, and the one I actually built the tool for. Every
  sample is placed **in the direction of the transmitter as seen from the
  aircraft**, so the axes are NOSE / STBD / TAIL / PORT rather than compass
  directions. **Radius is signal strength.** That makes the result an
  *empirically measured antenna pattern* for your specific airframe, and an
  **inward dent is a real RX blind spot in a real orientation.** Rings mark 0°,
  30° and 60° elevation. There is a frame toggle — *From TX* for the spatial view,
  *Airframe frame* for the antenna-pattern view, and a render toggle: *Points*
  for raw samples, *Surface* for a smoothed shell that goes grey where there is no
  data. The white and green ticks along the track are heading markers, white for
  nose and green for starboard.
- **Path**, the trajectory, with marker size and colour inversely proportional to
  link quality, so bad moments are literally bigger and redder

The metric list is data-driven, it detects which sensors are actually in your log
and offers those: worst-of-`1RSS`/`2RSS`, `RSNR`, `RQly`, `TRSS`, `TSNR`, and
`TPWR` (treated as *higher = worse*, since ELRS ramps transmit power up as the link
degrades). Any raw column is selectable too. It also splits multiple flights out of
a single log file automatically.

It closes the loop on this whole post. The radio tells me about a limit in the
moment, in one word, while I am flying. The viewer tells me *why* afterwards, with
the geometry attached. Same telemetry stream, two ends of the same problem.

Two details in it are worth calling out, because they are the analysis-side
solutions to problems I hit earlier in this post.

**It has a robust ground reference for altitude**, and that exists precisely
because of the `GAlt` problem from the L6 section above. `GAlt` is metres above
MSL, and its *first* samples are its worst, because the fix is fresh. Zero the
whole flight on one fresh-fix sample and the entire log reads negative. So the
viewer offers Auto / at-start / lowest / manual referencing, with an optional
median filter for GPS altitude spikes, and it treats exact zeros in a `GAlt`
column as "no fix" rather than as sea level. Same physics as the altitude warning
problem, attacked from the other end.

You can see that logic firing in the screenshot above, the amber note is the tool
reporting that the log begins about 154 m above its own lowest point, so it took
zero from the lowest 2 % of the flight instead of trusting the first sample. On a
naive at-start reference, that one fresh-fix sample would have made the entire
flight read as negative altitude.

**It has a current-sensor correction factor**, which is the calibration section
of this post, made actionable. If the FC current sensor is mis-scaled then every
mAh figure in the log is wrong by a fixed multiplier, and so is every derived
number. You set the correction to `actual ÷ logged` and the whole battery model
rescales with it. (In Betaflight the knob is `ibata_scale`, and note the direction:
*lower* scale means *higher* reported current.) On top of that it computes
**return-to-home radius rings at the tightest moment of the flight**, given pack
capacity, usable percentage, and a reserve you declare safe.

Which is the rigorous version of the `rth` callout at the top of this post. The
radio gives me a crude voltage proxy for half capacity while I am airborne, in one
word, with no maths. The viewer tells me afterwards whether that word arrived early
enough, and on which part of the flight it would not have.

One more measured detail worth flagging: the ELRS telemetry ratio is **not in
the model YAML**. My `moduleData` block contains only this:

```yaml
moduleData:
   0:
      type: TYPE_CROSSFIRE
      subType: 0
      channelsStart: 0
      channelsCount: 16
      failsafeMode: NOT_SET
      mod:
         crsf:
            telemetryBaudrate: 0
```

No ratio field, because the ratio lives on the TX module itself, configured
through the ELRS Lua script. Which means **sharing a model YAML does not share
your telemetry ratio.** If you copy my config and your logs look different to
mine, that is the first place to look.

So the honest state of this section is: I know the shape of the answer, I know
exactly which measurement settles it, and I have not published the number yet. It
is going to get its own post.

**Next:** [Part 7, two antennas, two bands, and the quad I lost to polarisation](/fpv/edgetx-cockpit-voice-antennas/)
