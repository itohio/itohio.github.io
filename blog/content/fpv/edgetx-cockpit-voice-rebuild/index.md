---
title: "EdgeTX Telemetry Rebuilt in Flight Order (Part 9)"
date: 2026-08-16T17:00:00+03:00
description: "A clean-sheet layout: validity helpers first, then recording, battery and GPS in the order I use them, plus a readout that speaks punch-out sag."
summary: "A clean-sheet layout: validity helpers at L1 to L4, then recording, battery and GPS in the order I use them. Plus a minimum-voltage readout that speaks punch-out sag while I am still flying."
draft: false
toc: true
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - logical-switches
  - prearm
  - telemetry
  - minimum-voltage
  - gps-rescue
  - refactor
keywords: ["EdgeTX logical switch AND helper", "EdgeTX RxBt minimum readout", "EdgeTX reset telemetry special function", "EdgeTX prearm warning switch", "EdgeTX play value minimum voltage"]
series:
  - EdgeTX Cockpit Voice
---

> **EdgeTX Cockpit Voice**, part 9 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [‹ Part 8: Four Things Wrong With It](/fpv/edgetx-cockpit-voice-whats-wrong/)  ·  [Start at part 1](/fpv/edgetx-cockpit-voice-why/)

[Part 8](/fpv/edgetx-cockpit-voice-whats-wrong/) listed four things wrong with the
config I fly today. This part is the fix, designed but not yet flashed, written
down partly so I actually do it.

## I'll try this one next: a full regroup

Everything above is what I actually fly today, mess included. This section is the
rebuild I have designed but not yet flashed, written down partly so I actually do it.

> **Note on numbering:** this section is a clean-sheet layout, so the `L` numbers
> below do **not** mean what they mean earlier in the post. In the config I fly
> today `L1` is `RxBt < 4.0 V`; in the rebuild `L1` is the validity helper. Read the
> two layouts as separate documents.

Because the honest problem with my current config is not any individual switch, it
is that **it accreted.** I added battery set points as I thought of them, then
wedged GPS and altitude in between, and the result is eleven switches in the order
I happened to invent them. Nothing is wrong. Nothing is findable either.

So: three groups, in the order I use them in real life. **Recording, then battery,
then GPS.** That is the sequence of a flight, start the log, check the pack, wait
for a fix.

### Helpers first, at the bottom of the range

Renumbering gives me something for free. Last time I sketched these helpers at
L12–L16 and had to warn that they would lag their consumers by one evaluation
cycle. Putting them at **L1–L4** removes that caveat entirely: EdgeTX walks L1 → L64
once per cycle, so a helper at L1 is always fresh by the time L5 reads it.

```yaml
0:                               # = L1  "telemetry is actually present"
   func: FUNC_VPOS
   def: "tele(14),5"             # RxBt > 0.5   (prec:1, so 5 = 0.5 V)
   andsw: "NONE"

1:                               # = L2  battery warnings armed AND valid
   func: FUNC_AND
   def: "SW52,L1"                # green bat button + validity

2:                               # = L3  GPS callouts armed AND valid
   func: FUNC_AND
   def: "SW62,L1"                # blue gps button + validity

3:                               # = L4  preflight staged AND valid
   func: FUNC_AND
   def: "SE1,L1"                 # SE middle + validity
```

`L1` is the one that matters most. It is what stops the whole ladder shouting at me
on a fresh battery, because `RxBt` sitting at its `0.0` initialisation value fails
`RxBt > 0.5`, so every gate downstream is false until real data arrives.

### Group 1 — Recording

Recording needs no logical switch at all; it is a special function driven straight
off the red button. It goes **first** in the special function list purely so the
list reads in flight order.

| Trigger | Function |
|---|---|
| `SW42` (red `log`) | `LOGS` 0.3 s |

### Group 2 — Battery, L5 → L13

The ladder in descending voltage order, which is finally also numerical order.

| LS | Test | Gate | Sound |
|---|---|---|---|
| **L5** | `RxBt > 4.2` | `L2` | `ready` — fresh pack self-test |
| **L6** | `RxBt < 4.0` | `L2` | `Wrn1` |
| **L7** | `RxBt < 3.8` | **`L3`** | `rth` — see note |
| **L8** | `RxBt < 3.6` | `L2` | `Sirn`, 1 s |
| **L9** | `RxBt < 3.5` | `L2` | `lowbat`, 5 s |
| **L10** | `RxBt < 2.9` | `L2` | `Alrm` |
| **L11** | `\|Δ\|≥ RxBt- 0.1` | `L2` | **speaks the new minimum** |
| **L12** | `RxBt < 3.8` | `L4` | preflight fail — `Sirn`, 2 s |
| **L13** | `RxBt > 3.8` | `L4` | preflight pass, spoken confirmation |

**The `L7` gate is a deliberate exception.** It sits in the battery group because it
is a voltage threshold, but it is gated on the *gps* helper rather than the battery
one, because "turn around" is a long-range warning. On a whoop in a hotel room it
would be noise. Group by what a switch measures; gate by when you want to hear it.
Those are different questions and it is fine for them to disagree.

### L11: the minimum-voltage readout

This one is new, and it is the reason I started rebuilding rather than patching.

EdgeTX tracks a running minimum for every telemetry sensor, exposed as a separate
source — `RxBt-`. My telemetry screens already display it. What I have never done is
make it *talk*.

```yaml
10:                              # = L11
   func: FUNC_ADIFFEGREATER      # |Δ| >= x
   def: "tele(-14),1"            # RxBt MINIMUM, step 0.1 V
   andsw: "L2"
```

Paired with `PLAY_VALUE` on `tele(-14)`, that means: **every time the flight
records a new lowest cell voltage, the radio speaks it.** Not a threshold, not a
warning, a measurement, read out loud, at the moment it happens.

Which gives me sag data from punch-outs while I am flying, instead of afterwards in
blackbox. A hard punch drops the pack, `RxBt-` drops with it, and I hear "three
point four". That is the number I care about most and have never had in the air.

Two details make it work properly:

**Use `|Δ|`, not `Δ`.** A minimum only ever decreases, so the delta is always
negative — `Δ≥x` would never fire. The absolute-value form catches it.

**Reset the tracker per pack, or it is useless.** A running minimum that never
resets just remembers the worst moment of the day. So `L5`, the fresh-pack
detector, gets a *second* special function alongside `ready`:

| Trigger | Function |
|---|---|
| `L5` | `PLAY_TRACK ready` |
| `L5` | `RESET Telemetry` |

Plug in a fresh battery, the radio says "ready" and simultaneously wipes the min/max
trackers, so `RxBt-` now tracks *this* pack. One switch, two jobs.

A 0.1 V step is the sensor's own quantisation, so this announces every single new
minimum. If that turns out chatty during freestyle, raise the step to 0.2 V, the
threshold is the volume knob.

### Why L13 exists: silence is not a pass

`L12` and `L13` are a pair, and the second one matters more than it looks.

The workflow I want is: flick SE to the middle, wait a moment, listen. If nothing
complains, arm and fly. The problem is that **silence means two different things
wearing the same costume**:

1. the battery is fine, which is the answer I want
2. telemetry has not arrived yet, so nothing has an opinion

Those are indistinguishable by ear. Flick to middle and arm quickly and I hear
nothing, conclude the pack is good, and take off on a battery no one has measured.
The check passes hardest in exactly the case where it told me nothing at all.

Aviation settled this long ago and the rule is worth stealing verbatim: **a
preflight check must give a positive indication, not the absence of a negative
one.** A test that passes by being quiet also passes when it is broken.

So `L13` fires on `RxBt > 3.8` while staged, and says so out loud. Now flicking SE
to the middle produces exactly one of two outcomes: a pass I can hear, or a
warning. Nothing is no longer an answer. It means wait longer.

### Group 3 — GPS, L14 → L16

| LS | Test | Gate | Sound |
|---|---|---|---|
| **L14** | `Sats > 10` | `L3` | **speaks the satellite count** |
| **L15** | `Sats < 6` | `L3` | `gpsoff` — fix degraded |
| **L16** | `\|Δ\|≥ GAlt 120` | `NONE` | `warnng` — altitude |

**Acquire at 10, warn at 6, the gap is deliberate.** Ten satellites is "solid
enough to launch on". Six is "GPS Rescue is no longer something I would trust".
Setting both to the same number would make it chatter every time the count wobbled
across the boundary; a four-satellite deadband means each announcement is a real
state change. Set the lower number to match your own `gps_rescue_min_sats`.

`L14` replaces the old pair of Sats switches. Announcing the count at 10 rather
than 6 is the change I wanted: below ten I do not want a running commentary while
I wait, I want to know when it is *ready*.

If you would rather have a continuous count as satellites come and go, that is a
`|Δ|≥1` on `Sats` gated behind a `Sats > 10` helper, one more switch, and rather
more talking.

`L16` stays ungated on purpose. The altitude limit applies on every aircraft on
every flight, so it is the one warning that should not have an off switch.

### Special function order

The list finally reads in flight order: log, battery, GPS.

```text
 0  SW42  LOGS         0.3s          <- recording
 1  L5    PLAY_TRACK   ready         <- battery
 2  L5    RESET        Telemetry
 3  L6    PLAY_SOUND   Wrn1
 4  L7    PLAY_TRACK   rth
 5  L8    PLAY_SOUND   Sirn    1s
 6  L9    PLAY_TRACK   lowbat  5s
 7  L10   PLAY_SOUND   Alrm
 8  L11   PLAY_VALUE   tele(-14)     <- minimum voltage readout
 9  L12   PLAY_SOUND   Sirn    2s
10  L13   PLAY_TRACK   <pass callout>
11  L14   PLAY_VALUE   tele(22)      <- GPS, satellite count
12  L15   PLAY_TRACK   gpsoff
13  L16   PLAY_TRACK   warnng
```

### Why the helper indirection is worth it

The gating policy now lives in **one place per subsystem** instead of being
copy-pasted into eleven switches. When I get round to a proper prearm switch,
probably `SA`, staging moves off SE with a single edit: `L4`'s second operand
changes and every preflight behaviour follows. No threshold gets touched.

Threshold logic, arming logic and validity logic each get their own layer, and none
of them know about each other.

### What to verify rather than trust

I am confident about the structure, and `L<n>` is definitely the reference form:
my existing `customFn` block already uses `swtch: "L3"`. Three things are predictions:

- the exact YAML spelling of `FUNC_AND` and its two-operand `def`, since my current
  config contains no AND-type switch to copy from
- that `tele(-14)` is selectable as a logical switch operand. It definitely exists
  as a *source*, my telemetry screens use it, but I have not yet confirmed the
  picker offers min/max variants inside a logical switch
- the `RESET Telemetry` special function's `def` format

Build it in the radio UI, export the model, and read back what EdgeTX actually
wrote. Then trust that, not this.

### Where this leaves the series

Every warning in these nine parts is built from telemetry that was already arriving
at the radio, from sensors that were already discovered, using firmware that was
already installed. Nothing was added to any aircraft. No Lua, no extra hardware,
not one gram of takeoff weight.

The only thing that changed is that the information now goes into my ears instead
of into a corner of a screen I am not looking at.

The flight where the voltage quietly slid past the point of no return while I was
busy enjoying myself does not happen any more. Somewhere around half capacity a
voice says "return home", and I turn around with fuel in the tank.

The aircraft knew all along. It just needed a way to say so.

---

*If you build a cleaner version of any of this, particularly a proper
launch-relative altitude warning or a threshold ladder that does not stack, I would
genuinely like to see it.*

---

> **Series:** EdgeTX Cockpit Voice, part 9 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [‹ Part 8: Four Things Wrong With It](/fpv/edgetx-cockpit-voice-whats-wrong/)  ·  [Start at part 1](/fpv/edgetx-cockpit-voice-why/)
