---
title: "Part 8: Four Things Wrong With It"
date: 2026-08-16T10:00:00+03:00
description: "Stacked thresholds that layer sirens, zero debounce against punch-out sag, an alarm that fires before the first telemetry frame arrives, and the link-quality warning I never wired up."
summary: "Stacked thresholds that layer sirens, zero debounce against punch-out sag, an alarm that fires before the first telemetry frame arrives, and the link-quality warning I never wired up."
draft: false
toc: true
weight: 8
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - logical-switches
  - debounce
  - voltage-sag
  - link-quality
  - blackbox
  - troubleshooting
keywords: ["EdgeTX logical switch debounce duration delay", "voltage sag false alarm FPV", "EdgeTX warning fires at startup", "RQly link quality warning EdgeTX", "EdgeTX a<x level versus edge"]
series:
  - EdgeTX Cockpit Voice
---

> **EdgeTX Cockpit Voice**, part 8 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [‹ Part 7: Two Antennas, Two Bands](/fpv/edgetx-cockpit-voice-antennas/)  ·  [Part 9: The Rebuild, Grouped in Flight Order ›](/fpv/edgetx-cockpit-voice-rebuild/)  ·  [Start at part 1](/fpv/edgetx-cockpit-voice-why/)

Everything so far is what I actually fly. This part is the honest accounting. I
said at the start there were less clumsy ways to do parts of this, and here is
exactly where mine is clumsy.

## What's clumsy about this

I said I would be specific. Reading my own config back with fresh eyes, here is
what is wrong with it.

### 1. The thresholds stack, and the sirens pile up

`a < x` is a **level** test, not an edge test. Once you drop below 3.5 V, you are
also still below 3.6 V, and below 4.0 V, and below 3.8 V. Every one of those
logical switches is true simultaneously.

At 3.4 V/cell my radio is running:

| Switch     | State | Sound    | Repeat                      |
| ---------- | ----- | -------- | --------------------------- |
| L1 (< 4.0) | true  | `Wrn1`   | once — already fired, quiet |
| L3 (< 3.8) | true  | `rth`    | once — already fired, quiet |
| L2 (< 3.6) | true  | `Sirn`   | **every 1 s, forever**      |
| L8 (< 3.5) | true  | `lowbat` | **every 5 s, forever**      |

So below 3.5 V I get a siren every second _and_ a spoken "low battery" every
five seconds, layered on top of each other. In the moment that is arguably fine
— it is certainly attention-getting, but it is not _informative_. A siren that
never stops carries no more information than a siren that fires once.

The fix is to make each rung exclusive, by ANDing each threshold with the
negation of the one below it: "below 3.6 **and not** below 3.5". EdgeTX can do
this with a second logical switch layer. I have not rebuilt it yet.

### 2. Zero debounce, everywhere

Every logical switch has `delay: 0, duration: 0`. There is no filtering at all,
which means **any transient sag triggers a permanent warning.**

This is not theoretical for me. On my 3-inch 4S build, a blackbox log of a hard
punch-out showed the pack collapse to **3.065 V/cell** under an 83 A draw, a
momentary event, fully recovered a fraction of a second later. That is 165 mV
of margin from tripping my 2.9 V "you have damaged the pack" alarm, on a pack
that was completely fine.

Sag is not state of charge. A voltage threshold with no time qualifier cannot
tell the difference. The 3.5 V and 2.9 V rungs are the exposed ones, because
those are the values you transit under load long before you reach them at rest.

EdgeTX gives you the tools: **Duration** requires the condition to hold for N
seconds before the switch goes true, and **Delay** postpones the transition.
Putting a duration of a second or two on the low rungs would eliminate
sag-triggered false alarms almost entirely.

I am not going to publish numbers for this, because I have not derived them from
my own logs yet, and picking them by feel would be exactly the kind of guess
this whole post is arguing against. The right way to set them is to look at the
sag duration distribution in your own blackbox and choose a duration longer than
your longest punch. That is a measurement, and I have the logs to do it.

### 3. It shouts at me before it says hello

This is the most annoying flaw in daily use, and the one I have not solved.

**When I plug a battery in, the radio announces the warnings and the low-battery
track _before_ it says "ready".** Every time. It sounds like the aircraft is in
trouble the instant it wakes up.

The cause is a property of `a<x` that is obvious in hindsight and invisible when
you are building the thing: **a level comparison cannot tell "critically low" apart
from "no data yet".**

At link-up the radio has a connection, but the CRSF battery frame has not arrived
yet, so the `RxBt` sensor is still sitting at its initialisation value of **0.0 V**.
And `0.0` is less than 4.0, and less than 3.6, and less than 3.5, and, the good
part, less than **2.9**. So the entire ladder fires at once, bottom rung included:
the radio cheerfully informs me I have destroyed the pack, on a fresh battery,
before the first real voltage reading has ever arrived.

Then the battery frame lands, `RxBt` jumps to its true value, every switch goes
false, `L10` sees `> 4.2 V` and says "ready" — and everything is fine. But the
first thing I hear is an alarm.

This interacts badly with the frame rate from the previous section, too. The dead
window is not milliseconds, it lasts until the first battery frame arrives, and
those frames are not frequent.

The fix I have not yet applied is trivial: **AND every low-voltage switch with a
validity condition**, something like `RxBt > 0.5`, so that "no telemetry" reads
as "no opinion" instead of "catastrophe". A `Duration` long enough to outlast the
startup gap would also work.

The fix I have actually _started_ is more interesting, and it explains a switch
that would otherwise look like junk. **L9 is `RxBt < 3.8 V AND SE-`**, gated on the
middle position of the 3-position SE switch rather than on the green `bat` button.
That is deliberate: I have put **arming on SE**, and **warnings on SE mid**, so that
the whole warning system is armed at *prearm* rather than while the aircraft is
sitting on the ground doing nothing. Prearm is the correct place for a preflight
voltage check, it is the moment you are about to commit.

**I have not configured prearm yet.** I know exactly when I will: the first time I
pick up a drone, the radio bumps my chest, and the arm switch flips. I am fairly
confident that will be a memorable enough lesson to get it done that same evening,
assuming I still have all my fingers to type with.

Which is a bad plan. It is, however, an honest description of my actual plan.

### 4. No link-quality warning at all

This is the biggest actual gap. My model has:

```yaml
rssiSource: none
rfAlarms:
   warning: 65
   critical: 35
```

There are RF alarm thresholds configured, but `rssiSource` is `none`, so
nothing is wired up to trigger them. Meanwhile I have `RQly`, `RSNR`, `ANT` and
both `1RSS`/`2RSS` sensors sitting right there in the sensor list, fully
populated, `logs: 1` on every one of them, and completely unused by any logical
switch.

Given that this whole project exists to stop me flying past a limit I was not
looking at, the fact that I have not applied it to **link quality**, the limit
that actually ends flights, in a hedge, a long way from the car, is an
oversight I noticed while writing this up. `RQly < 70 → PLAY_TRACK "link"` is one
logical switch and one special function, and it is next on the list.

And it is worse than that, because of what those particular sensors are. See
below.

The irony is not lost on me: those same sensors are the ones my
[RX Blind-Spot Viewer](https://rxmap-viewer.sintra.site/rxmap/) reads to build a
3D antenna pattern. I will happily spend an evening analysing link quality in
three dimensions after the flight, and I have not wired up the one switch that
would make the radio say "link" during it.

Four flaws, all of them mine, all of them fixable in an evening. Writing them down
is the part that makes them fixable.


---

> **Series:** EdgeTX Cockpit Voice, part 8 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [‹ Part 7: Two Antennas, Two Bands](/fpv/edgetx-cockpit-voice-antennas/)  ·  [Part 9: The Rebuild, Grouped in Flight Order ›](/fpv/edgetx-cockpit-voice-rebuild/)  ·  [Start at part 1](/fpv/edgetx-cockpit-voice-why/)
