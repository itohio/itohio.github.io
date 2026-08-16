---
title: "ELRS Gemini and True Diversity Antennas (Part 7)"
date: 2026-08-16T15:00:00+03:00
description: "ELRS control links are linearly polarised, not circular like video. Why one horizontal and one vertical antenna beats a single one on diversity."
summary: "ELRS control-link antennas are linearly polarised, not circular like video. Why one horizontal and one vertical on a true diversity receiver beats a single antenna, and why my telemetry already measures it."
draft: false
toc: true
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - elrs
  - antenna-diversity
  - polarisation
  - radiomaster-gx12
  - gemini
  - crsf
  - telemetry
keywords: ["ELRS antenna polarisation linear", "true diversity receiver FPV", "ELRS Gemini dual band", "FPV antenna null dipole", "1RSS 2RSS active antenna CRSF", "RadioMaster GX12 vs Boxer"]
series:
  - EdgeTX Cockpit Voice
---

> **EdgeTX Cockpit Voice**, part 7 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [‹ Part 6: Telemetry Logging and the Number You Must Measure](/fpv/edgetx-cockpit-voice-telemetry-rates/)  ·  [Part 8: Four Things Wrong With It ›](/fpv/edgetx-cockpit-voice-whats-wrong/)  ·  [Start at part 1](/fpv/edgetx-cockpit-voice-why/)

The extra buttons are what made this project pleasant. They are not why I bought
the radio. That decision came out of losing an aircraft, and it is worth a part of
its own because the physics is the same physics the warnings are built on.

## The other reason I bought this radio: two antennas, two bands

The extra buttons are what made this project pleasant. They are not why I bought
the radio. I bought it for **dual-band operation with two antennas**, and that
decision came out of losing an aircraft.

### The quad that fell into the weeds

On the Pocket I ended up with a **polarisation mismatch** between the radio's
antenna and the receiver's, and at distance the drone simply dropped out of the
sky into the weeds.

The mechanism is worth being precise about, because FPV people are used to
thinking about polarisation in the *video* context, where the convention is
circular. LHCP on both ends, and mixing LHCP with RHCP costs you around 20 dB.
The control link is a different animal. **ELRS antennas are linearly
polarised**, dipoles and monopoles, not helicals. And two linear antennas at
90° to each other are cross-polarised, which is a loss of the same brutal order.

Linear antennas have a second problem that circular ones share but which is
easier to forget: a dipole radiates in a torus with **deep nulls along its own
axis**. Point the end of the antenna at the other station and there is
essentially nothing there. On the ground that is easy to avoid. Mid-dive, with
the aircraft rotating through every attitude it has, you cannot avoid it, you
can only make sure the null is never in the same place on both antennas at once.

### One horizontal, one vertical

So on my newest build, a **folding 4-inch**, which is getting its own post once I
have flown it enough to say anything honest about it. I run a **true diversity
receiver with two dual-band antennas, one mounted horizontal and one vertical.**

That orthogonal pairing is the whole trick, and it buys two independent things
from one arrangement:

- **Polarisation coverage.** Whatever the radio's polarisation is at that
  instant, one of the two receive antennas is reasonably aligned with it. There
  is no orientation where both are cross-polarised.
- **Null coverage.** The two antennas' nulls point in orthogonal directions, so
  no single aircraft attitude can put both of them in a null simultaneously.

"True diversity" is the part that makes this work rather than just sound good. A
true diversity receiver has two independent receive chains, one per antenna, and
picks the better one **per packet**. It is not a passive combiner and it is not a
single receiver with a switch it flips occasionally.

The result, in the air: diving Norwegian waterfalls, rotating through every
attitude the airframe has, it switches between antennas cleanly and I do not get
the dropout the geometry says I should.

Notably this works **even when Gemini is not available on the aircraft.** ELRS
Gemini mode transmits on both bands simultaneously and needs a Gemini-capable
receiver at the other end. Without that, the radio still has two antennas and
still selects between them, so I get the benefit of the radio's diversity on
builds that cannot do full Gemini.

### Your telemetry already measures this — and mine is not using it

Here is the part that made me slightly annoyed at myself while writing this
section, and it connects straight back to the missing link-quality warning.

Three of the sensors already sitting in my model are exactly the diversity
instrumentation:

| Sensor | What it actually is |
|--------|--------------------|
| `1RSS` | RSSI at the **receiver's antenna 1** |
| `2RSS` | RSSI at the **receiver's antenna 2** |
| `ANT`  | Which antenna the receiver is currently **using** |

Be precise about whose antennas those are: `1RSS`, `2RSS` and `ANT` come from
the CRSF link-statistics frame and describe the **diversity receiver on the
aircraft**, not the two antennas on the radio. The radio-side benefit I
described above is a separate mechanism, and I have not instrumented it, the
downlink figures I do have (`TRSS`, `TQly`, `TSNR`) are measured at the radio
but do not break out per-antenna.

All three have `logs: 1`, so **they are already being written to the CSV every
0.3 s.** Which means the claim I just made — "it switches between antennas
perfectly" — is currently a field impression, not a measurement, and I have the
data to turn it into one. The Sphere view in the
[RX Blind-Spot Viewer](https://rxmap-viewer.sintra.site/rxmap/) is built for
exactly this: it plots the worst of `1RSS`/`2RSS` by azimuth and elevation in the
airframe's own frame, so an orthogonal antenna pair that is genuinely working
should show up as a rounder sphere with fewer dents than a single antenna would. Count the `ANT` transitions against the `1RSS`/`2RSS`
difference and you get the real switching behaviour: how often it swaps, whether
one antenna is systematically doing all the work, and whether the swaps line up
with the attitude changes in the blackbox.

If one antenna is carrying the link and the other is contributing nothing, that
is a mounting problem, and it is invisible from the goggles. I have a Lua script
in my telemetry suite for antenna diversity balance; what I do not yet have is
an **audible** version. A logical switch on the difference between `1RSS` and
`2RSS` would tell me about a dead or badly-routed antenna on the bench, before
it becomes a walk in the weeds.

That is the second thing going on the list, right after the link-quality
callout, and it is the same lesson as the rest of this post. The information
was already arriving. Nobody was listening to it.

## A short aside on the radio itself

The GX12 is my third radio, and I am going to be unprofessionally enthusiastic
about it for a paragraph.

I fell for it the moment I saw it. It sits between the RadioMaster Pocket and
the Boxer, not as compact as the Pocket, but _far_ more ergonomic, and it feels
genuinely good in the hands in a way the Pocket does not. The six extra
top-mounted buttons with individually addressable RGB are what made this entire
project pleasant instead of tedious.

I did briefly fly a colleague's 5-inch on a Boxer, and the Boxer is better.
Better gimbals, better ergonomics, no argument. My first flight with it went
directly, immediately and vertically into a tree, to considerable amusement from
its owner. I redeemed myself somewhat with a few power loops through gates
afterwards, but the tree is the part he remembers.

The reason I do not own a Boxer is prosaic: it does not fit. Most of my flying
happens on motorcycle trips, and I already barely fit two drones, goggles,
batteries and the radio into the GS Adventure's top box. The DJI Mini 3 era of
packing, where the whole kit left room for sandwiches and a bottle of water —
is long gone. For long trips I am going to have to pack even more ruthlessly,
and a Boxer-sized radio is exactly the wrong direction.

The GX12 is the compromise that stopped feeling like a compromise.

The information was already arriving. Nobody was listening to it. Which is, more
or less, the thesis of this entire series, and it leads directly into the part
where I audit my own work.


---

> **Series:** EdgeTX Cockpit Voice, part 7 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [‹ Part 6: Telemetry Logging and the Number You Must Measure](/fpv/edgetx-cockpit-voice-telemetry-rates/)  ·  [Part 8: Four Things Wrong With It ›](/fpv/edgetx-cockpit-voice-whats-wrong/)  ·  [Start at part 1](/fpv/edgetx-cockpit-voice-why/)
