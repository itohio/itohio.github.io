---
title: "Part 5: Where the Callouts Come From, and Why You Cannot Just Copy My Config"
date: 2026-08-16T13:00:00+03:00
description: "The callouts come from the stock EdgeTX voice pack, not from anything I recorded. What to scrub from a model YAML before publishing it, and the positional sensor-index trap that makes a shared config misbehave silently."
summary: "The callouts come from the stock EdgeTX voice pack, not from anything I recorded. What to scrub from a model YAML before publishing it, and the positional sensor-index trap that makes a shared config misbehave silently."
draft: false
toc: true
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - wav
  - sounds
  - model-yaml
  - privacy
  - elrs
  - telemetry
keywords: ["EdgeTX voice pack sounds folder", "EdgeTX model yml portability", "EdgeTX telemetry sensor index tele()", "scrub EdgeTX config before sharing", "ELRS binding phrase privacy"]
series:
  - EdgeTX Cockpit Voice
---

> **EdgeTX Cockpit Voice**, part 5 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [‹ Part 4: What the Radio Actually Says](/fpv/edgetx-cockpit-voice-callouts/)  ·  [Part 6: Telemetry Logging and the Number You Must Measure ›](/fpv/edgetx-cockpit-voice-telemetry-rates/)  ·  [Start at part 1](/fpv/edgetx-cockpit-voice-why/)

The callouts in [Part 4](/fpv/edgetx-cockpit-voice-callouts/) come from the voice
pack that ships with the radio. Two practical things follow from that: where the
audio actually lives, and why handing you my config file is less useful than it
sounds.

## The callouts: rth, gpson, gpsoff, lowbat, warnng, ready

The spoken callouts come from the **voice pack that ships with the radio**. I did
not record anything and I did not generate anything. Six of them do the work here:
`rth`, `gpson`, `gpsoff`, `lowbat`, `warnng`, `ready`.

That is worth saying plainly, because "the radio talks to me" sounds like a project
in its own right and it is not. `PLAY_TRACK` takes a filename from the
language-specific sounds directory on the SD card, `/SOUNDS/en/` on an English
radio, and the stock pack already contains a usable vocabulary of short callouts.
Everything in this series is threshold logic pointed at files that were already
there.

Which is the cheapest part of the whole build, and the part I would have expected
to be the most work.

The one place it constrains you is when you want a callout the pack does not
contain. The regrouped config in Part 9 wants a spoken "check ok" for the
preflight pass, and there is no such track in the stock vocabulary. Two ways round
it: pick an existing pack track that is close enough in meaning, or add your own
WAV to `/SOUNDS/en/` and select that. I have not needed to add one yet, so I am not
going to write a format guide from guesswork.

Two audio settings from `radio.yml` are worth knowing about, because they change
how the callouts land:

```yaml
wavVolume: 4
beepVolume: 0
audioMuteEnable: 1
```

`beepVolume: 0` means the beeps are off entirely while the spoken tracks stay up.
`audioMuteEnable: 1` powers the amplifier down between sounds, which reduces hiss
at the cost of the amp needing a moment to come back. If you ever find short tracks
losing their first syllable, that setting is the first thing to try at `0`. Mine
sound fine, so I am flagging it as a thing to know rather than a problem I have.

## Sharing the config: what is portable and what to scrub

I want to make this replicable, so: yes, publish your YAML. But two warnings.

### Scrub these fields before you publish

EdgeTX 2.9 and later store the radio config as YAML on the SD card —
`radio.yml` for the radio and one file per model in `/MODELS/`. Both of mine
contain a registration ID:

```yaml
# radio.yml
ownerRegistrationID: " 24P42P-"

# model00.yml
modelRegistrationID: " 24P42P-"
```

Before publishing a config, check for and scrub:

* `ownerRegistrationID` / `modelRegistrationID`
* `bluetoothName`
* Your **ELRS binding phrase**, this one is not in the model YAML, it lives on
  the TX module, but if you are also sharing a module backup, that phrase is
  effectively the key to your aircraft
* Model names, if they identify you
* Stick calibration (`calib:`), harmless but meaningless to anyone else, and
  copying mine will make your sticks feel wrong

### The YAML is less portable than it looks

Here is the trap, and it is a real one. Logical switches reference telemetry
sensors **by position**, not by name:

```yaml
def: "tele(14),40"     # sensor slot 14, which is RxBt *in my file*
```

`tele(14)` is not "RxBt". It is "whatever ended up in slot 14 during sensor
discovery". The slot order depends on which frames arrived first when you
discovered sensors, which depends on your FC configuration and the order you
powered things up. **On your radio, slot 14 may well be something else**, and
if it is, my logical switches will silently compare a voltage threshold against
your heading, and the whole thing will misbehave in ways that look like magic.

For reference, my slot order is:

```text
0  1RSS   1  2RSS   2  RQly   3  RSNR   4  ANT    5  RFMD
6  TPWR   7  TRSS   8  TQly   9  TSNR  10  FM    11  Ptch
12 Roll  13  Yaw   14  RxBt  15  Curr  16  Capa  17  Bat%
18 GPS   19  GSpd  20  Hdg   21  GAlt  22  Sats
```

So my honest advice, in order of preference:

1. **Read the tables in this post and re-enter them by hand**, using your own
   sensor names. It is fifteen minutes and you will actually understand the
   result, which matters when you want to change a threshold at a field.
2. If you drop in my YAML wholesale: delete your discovered sensors, re-discover
   them, then **verify slot-by-slot** that the numbers in the logical switch
   page point at the sensors you think they do. The UI shows names, so this is
   easy to check, just do not skip it.
3. `radio.yml` is board-specific (mine says `board: gx12`) and version-tagged
   (`semver: 2.12.2`). Do not copy it to a different radio.

Read the tables, use your own sensor names, and you will end up with something you
actually understand. That matters at a field, in the wind, when you want to move a
threshold by 0.1 V.


---

> **Series:** EdgeTX Cockpit Voice, part 5 of 9. Making a RadioMaster GX12 speak its own telemetry, so a low battery is something I hear instead of something I forgot to look at.
>
> [‹ Part 4: What the Radio Actually Says](/fpv/edgetx-cockpit-voice-callouts/)  ·  [Part 6: Telemetry Logging and the Number You Must Measure ›](/fpv/edgetx-cockpit-voice-telemetry-rates/)  ·  [Start at part 1](/fpv/edgetx-cockpit-voice-why/)
