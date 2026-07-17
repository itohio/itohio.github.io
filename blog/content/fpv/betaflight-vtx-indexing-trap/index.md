---
title: "The Betaflight VTX Trap: aux_channel Is 0-Based, vtx_power Is 1-Based"
date: 2026-07-17
description: "VTX stuck in pit mode, 400mW unreachable, power always one level too low? In Betaflight's vtx command, aux_channel counts from 0 but vtx_power counts from 1. Nothing warns you. Here is the fix."
draft: true
toc: true
categories:
  - FPV
  - Betaflight
tags:
  - fpv
  - betaflight
  - vtx
  - vtx-power
  - smartaudio
  - cli
  - pit-mode
  - emax-th3
  - vtxtable
  - dbm
  - restricted-mode
  - troubleshooting
keywords: ["Betaflight VTX always pit mode", "Betaflight can't set 400mW", "vtx_power 1-based", "Betaflight vtx command indexing", "SmartAudio power one level too low", "Betaflight VTX power switch not working", "vtxtable index off by one", "Emax TH3 V02 vtxtable", "Betaflight vtxtable dBm values", "VTX restricted mode 400mW"]
series:
  - FPV Builds
---

Every programmer learns to count from zero. Betaflight agrees: AUX channels are 0-based, so AUX1 is `0` and AUX6 is `5`. Type a few CLI lines and your fingers stop thinking about it. Count from zero. It's the rule the whole configurator seems to run on.

Then I lost an evening to a VTX that flatly refused to output 400 mW, and found the one field where Betaflight quietly counts from *one* instead. Nothing warns you. The failure looked like a dead VTX, not a typo — which is exactly why it ate the whole evening.

---

## The Symptom

The build: a 2" analog ripper on a **SUB250F411** running **Betaflight 4.5.4**, with an **Emax TH3 V02** VTX (25 / 100 / 400 mW) on SmartAudio, hardware UART2. I wanted power on a rotary wheel — **AUX6** on the radio — so I could dial output up for range and back down for close-in without diving into a menu.

What I got instead:

- Every wheel position produced power **one level lower** than I'd configured.
- **400 mW was unreachable.** No wheel position, anywhere in its travel, would select it.
- At the extremes of the wheel, the VTX dropped into what looked like **pit mode** — near-zero output, the classic "why can't my buddy see my feed on the bench" symptom.

I did the usual hardware witch-hunt first. Re-flashed SmartAudio settings. Checked the UART2 solder joints under a loupe. Swapped the VTX onto a known-good SmartAudio line. Power-cycled between every change. Nothing moved. The VTX itself was fine the whole time — it was doing exactly what I told it to. I just didn't realise what I'd told it.

---

## The Trap

Here is the `vtx` command format:

```
vtx <index> <aux_channel> <vtx_band> <vtx_channel> <vtx_power> <start_range> <end_range>
```

Two of those fields use **opposite indexing conventions**, and nothing in the docs or in `help vtx` says so:

| Field | Convention | Example |
|-------|-----------|---------|
| `aux_channel` | **0-based** | AUX6 → `5` |
| `vtx_power` | **1-based** | first vtxtable entry → `1` |

Read that again, because it's the whole article. In *one command*, field 2 counts from zero and field 5 counts from one. If you carry the "everything is 0-based" rule across the whole line — which is the natural thing to do — every power value you write lands one slot too low.

`vtx_power` is an index into your `vtxtable`, and that index starts at **1**:

```mermaid
graph LR
    subgraph CLI["vtx_power value you type"]
        direction TB
        P0["0"]
        P1["1"]
        P2["2"]
        P3["3"]
        P4["4"]
    end
    subgraph TBL["what it actually addresses"]
        direction TB
        T0["invalid → PIT"]
        VT0["vtxtable[0]\nPIT"]
        VT1["vtxtable[1]\n25 mW"]
        VT2["vtxtable[2]\n100 mW"]
        VT3["vtxtable[3]\n400 mW"]
    end
    P0 --> T0
    P1 --> VT0
    P2 --> VT1
    P3 --> VT2
    P4 --> VT3

    classDef bad fill:#3a1a1a,stroke:#c0392b,color:#f5d5d5;
    classDef good fill:#16281a,stroke:#27ae60,color:#d5f0d8;
    class T0,VT0 bad;
    class VT1,VT2,VT3 good;
```

The mapping is `vtx_power = N` selects `vtxtable[N-1]`. Value `0` isn't "the first entry" — it's out of range, and out-of-range means pit. Value `1` *is* valid, but on a standard Emax-style table `vtxtable[0]` is the **PIT** entry. So the two lowest values I "naturally" reached both meant pit, and the real 400 mW slot at `vtxtable[3]` needed a `4` I never typed.

---

## The vtxtable It Indexes Into

`vtx_power` is a 1-based index — but an index into what, exactly? Here is the actual power table I run on the Emax TH3 V02:

```
vtxtable powervalues 1 14 20 26
vtxtable powerlabels PIT 25 100 400
```

And now the *second* number system shows up. The entries in `powervalues` aren't indices, and they aren't milliwatts — they're **dBm**, the units SmartAudio v2 actually speaks. The conversion is `dBm = 10·log10(mW)`:

| dBm I set | Power | How it's computed |
|----------:|-------|-------------------|
| `1` | PIT (~1.3 mW) | pit / essentially off |
| `14` | 25 mW | 10·log10(25) ≈ 13.98 |
| `20` | 100 mW | 10·log10(100) = 20 |
| `26` | 400 mW | 10·log10(400) ≈ 26.02 |

So the full addressing chain has two layers, and each layer counts in a different number system:

| `vtx_power` (1-based index) | vtxtable slot | dBm value | Label |
|:---------------------------:|:-------------:|:---------:|:-----:|
| `1` | 1st | `1` | PIT |
| `2` | 2nd | `14` | 25 mW |
| `3` | 3rd | `20` | 100 mW |
| `4` | 4th | `26` | 400 mW |

This is what makes the trap so easy to fall into. To *define* a level you type its dBm (`26` for 400 mW). To *select* that level from an aux switch you type its 1-based index (`4`). Same VTX, same feature — two completely different numbers for "400 mW" depending on which command you are in, and neither of them is the 0-based value your fingers want to type.

---

## What I Actually Typed vs What I Got

My wrong config carried the 0-based assumption straight through:

```
vtx 0 5 0 0 0 1000 1250
vtx 1 5 0 0 1 1250 1500
vtx 2 5 0 0 2 1500 1750
vtx 3 5 0 0 3 1750 2000
set vtx_power = 1
```

Note that the `5` in field 2 is *correct* — AUX6, 0-based. It's the last-but-two field, the power index, that's wrong on every line. Here's the damage, wheel position by wheel position:

| Wheel zone | I typed `vtx_power` | Addresses | I expected | I actually got |
|-----------|--------------------:|-----------|-----------|----------------|
| lowest    | `0` | invalid | PIT (fine) | **PIT** |
| low       | `1` | vtxtable[0] | 25 mW | **PIT** |
| mid       | `2` | vtxtable[1] | 100 mW | **25 mW** |
| high      | `3` | vtxtable[2] | 400 mW | **100 mW** |
| —         | `4` | vtxtable[3] | — | **400 mW (never addressed)** |

Everything shifted down exactly one row. And the top of my table topped out at 100 mW while the entry I actually wanted, `vtxtable[3]` = 400 mW, sat there unreachable because no line in my config ever passed a `4`.

There was a second, sneakier casualty: `set vtx_power`.

```
set vtx_power = 1
```

This is the **fallback** power Betaflight applies when the AUX value doesn't match any configured range. I set it to `1` thinking "lowest real power." But `set vtx_power` is *also* 1-based and points at the same table — so `1` = `vtxtable[0]` = **PIT**. That's why any wheel position that slipped between my ranges didn't fall back to a low power; it fell back to pit. Two separate 1-based fields, one wrong assumption, compounding.

---

## The Fix

Shift every power index up by one, and clean up the ranges while I'm in here:

```
vtx 0 5 0 0 2 900 1333
vtx 1 5 0 0 3 1333 1666
vtx 2 5 0 0 4 1666 2100
set vtx_power = 2
```

- **`2` / `3` / `4`** now correctly select 25 / 100 / 400 mW. 400 mW is finally reachable at the top of the wheel.
- **`set vtx_power = 2`** makes the fallback a real 25 mW instead of pit, so an unmatched wheel position drops to low power, not to a dead feed.
- **Ranges tightened to three zones** across the wheel's travel with no gaps between them — each range's `end` is the next range's `start`.
- **`900` and `2100`** at the extremes, not `1000`–`2000`, on purpose: on CRSF the channel endpoints overshoot slightly — the top of the wheel reports around **~2012 µs**, not a clean 2000, and the bottom dips below 1000. A range that ends at exactly `2000` leaves the very top of the wheel unmatched, which — thanks to the fallback trap above — used to dump me straight into pit at max wheel. Widening to `900`–`2100` swallows the overshoot so the endpoints stay locked to the zones I want.

Reload, and the wheel does what the physical rotation implies: bottom = low, middle = medium, top = 400 mW. No pit surprises.

---

## One More Wall: Restricted Mode

Fix the indexing and 400 mW *should* appear at the top of the wheel. If it still doesn't, the VTX itself may be holding it back. Many SmartAudio VTX — the TH3 included — ship with a **restricted (region-locked) mode** that caps output for regulatory compliance. In that state the high-power entries in your `vtxtable` are defined correctly, selected correctly, and simply refused by the hardware.

Restricted mode lives on the VTX, not in Betaflight, so no amount of CLI editing clears it. You disable it on the unit itself — typically a button hold on the VTX (or the matching SmartAudio "unlock") to drop out of the region-locked profile. [TODO: confirm the exact TH3 V02 unlock sequence before publish.]

The tell is diagnostic: if low and mid power select correctly but the top level alone stays dead after the index fix, suspect restriction, not configuration.

---

## The One Thing to Remember

Count from zero. It's the rule your fingers already know, it's the rule almost every field in the Betaflight CLI obeys — and it's the rule `vtx_power` breaks. That field counts from one, because it's an index into a `vtxtable` that starts at 1, and `set vtx_power` breaks the same way for the same reason. Same command, one line apart, opposite conventions, zero warnings.

So if your VTX is stuck in pit mode, or 400 mW won't show up no matter how you turn the knob: don't reach for the soldering iron. Count that one field from **1**, check whether `vtxtable[0]` is a PIT entry, and the "dead" VTX comes back to life. The hardware was never broken. I just counted from zero one field too many.
