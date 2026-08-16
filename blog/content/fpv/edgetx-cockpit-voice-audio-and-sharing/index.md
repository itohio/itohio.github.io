---
title: "Part 5: Custom Voice Files, and Why You Cannot Just Copy My Config"
date: 2026-08-16T13:00:00+03:00
description: "Where EdgeTX keeps custom WAV callouts, what to scrub from a model YAML before publishing it, and the positional sensor-index trap that makes a shared config misbehave silently."
draft: false
toc: true
weight: 5
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
keywords: ["EdgeTX custom wav sounds folder", "EdgeTX model yml portability", "EdgeTX telemetry sensor index tele()", "scrub EdgeTX config before sharing", "ELRS binding phrase privacy"]
series:
  - EdgeTX Cockpit Voice
---

The callouts in [Part 4](/fpv/edgetx-cockpit-voice-callouts/) are custom WAV files,
not built-in sounds. Two practical things follow from that: where the audio lives,
and why handing you my config file is less useful than it sounds.

## Custom audio: rth, gpson, gpsoff, lowbat, warnng, ready

The spoken callouts are custom WAV files, not built-in sounds. Six of them today:
`rth`, `gpson`, `gpsoff`, `lowbat`, `warnng`, `ready`, and a seventh, `checkok`,
once I build the regrouped config above.

They live in the language-specific sounds directory on the SD card, alongside
the voice pack, for an English radio, `/SOUNDS/en/`. The filename minus the
`.wav` extension is what you select in the special function, which is why they
are all abbreviated: **the name is limited to six characters**, hence `warnng`
rather than `warning`.

I generated mine with text-to-speech and converted them to the format EdgeTX
expects. If your tracks play but sound wrong, clipped, sped up, or silent —
the format is the first thing to check, because EdgeTX plays WAVs directly with
no resampling.

One thing worth checking in `radio.yml` if your tracks sound truncated at the
start, which I have not conclusively verified as the cause on mine:

```yaml
audioMuteEnable: 1      # amplifier muted between sounds
wavVolume: 4
beepVolume: 0
```

`audioMuteEnable: 1` powers the amplifier down between sounds to reduce hiss.
The trade-off is that the amp needs a moment to come back up, which can eat the
first syllable of a short track. Setting it to `0` is the test. I mention it as
a candidate, not a diagnosis.

Also note `beepVolume: 0`. I have the beeps turned all the way down and the
WAV volume up. If everything is going to talk to me, I do not also want it
beeping at me.

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

**Next:** [Part 6, telemetry logging and the one number you have to measure yourself](/fpv/edgetx-cockpit-voice-telemetry-rates/)
