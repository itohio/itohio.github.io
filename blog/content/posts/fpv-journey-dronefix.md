---
title: "How I Finally Gave Into FPV — Dronefix.lt and Diving Into a New Hobby"
date: 2026-07-13
description: "I resisted FPV for years. Then I signed up for the Dronefix.lt training programme, flew the simulator, soldered my first build, and completely rearranged my project priorities. The spectrometer can wait."
draft: true
toc: false
categories:
  - FPV
  - Personal
tags:
  - fpv
  - dronefix
  - hobby
  - betaflight
  - fleet
  - personal
series:
  - FPV Builds
---

I had been watching FPV content for a couple of years before I did anything about it. The builds looked expensive, the learning curve looked steep, and I already had a backlog of hardware projects that weren't getting finished. The spectrometer. The DARP mesh networking work. The drone detector training pipeline that kept needing more data.

And then I signed up for the Dronefix.lt training programme and everything changed.

---

## Dronefix.lt

Dronefix.lt runs structured FPV pilot training in Lithuania. Not "here's a drone, go crash it" — an actual programme: simulator hours, theory on regulations and airspace, hands-on with real builds, guidance on first purchases.

<!-- IMAGE: photo from Dronefix.lt academy session — group, equipment, or flying area -->
*[TODO: Photo from Dronefix.lt training]*

<!-- IMAGE: photo of simulator setup or early training flight -->
*[TODO: Photo — simulator session or early training]*

The simulator hours matter more than I expected. I came in thinking I had decent spatial reasoning and a background in electronics — how hard could it be? The answer is: hard enough that spending time in the simulator before touching a real quad saves you from a lot of expensive crashes. Acro mode is humbling. There is no substitute for flying hours, and simulator hours count.

What the programme did that I couldn't have done alone:

- **Forced me to start with basics before jumping to what I wanted to do.** My instinct was to immediately build a 5" freestyle quad. The programme started with a 2.5" whoop. That was the right call — whoop crashes are survivable.
- **Gave me a framework for choosing gear.** The FPV market is full of competing standards, incompatible ecosystems, and products that were good two years ago. Having instructors who flew daily cut through a lot of the noise.
- **Connected me to other pilots.** The community aspect was unexpected. Other pilots are the fastest way to fix problems, find flying spots, and understand what actually matters versus what gets argued about online.

---

## The Fleet

After Dronefix.lt I did not buy one quad. I built several.

<!-- IMAGE: fleet photo — all current quads laid out -->
*[TODO: Fleet photo — Pavo20, 1S build, and other quads]*

<!-- IMAGE: individual build photos -->
*[TODO: Individual build photos]*

The current roster:

**Pavo20 Pro II** — 2.5" GPS whoop. My main tool for testing GPS configurations and the subject of a [separate article about GPS struggles](../pavo20-gps-struggles/). It is not the most capable quad I own, but it taught me the most about RF interference and ESC noise.

**1S Matrix 3-in-1 digital** — the build that embarrasses the Pavo20 on satellite count in every session. Tiny, runs DJI O3 air unit, finds 20+ GPS satellites without issue. The fact that a 1S build outperforms a dedicated GPS whoop in signal quality is part of why I got interested in the noise problem.

More builds followed. Each one taught something specific — motor direction failures, ESC protocol mismatches, blackbox analysis, PID tuning. The hobby is genuinely educational in a way that feels more hands-on than most software work.

---

## What I Postponed

The spectrometer project was mid-build when I found FPV. I had a working visible-light spectrometer on a Raspberry Pi with TOSLINK fiber coupling, a calibrated pipeline, and plans for a fiber-optic front end for backscattering experiments.

That project is still in the lab. The fiber optics work is progressing slowly, the 405nm and 535nm laser experiments are ongoing, and the software has evolved considerably — but the pace dropped when FPV arrived. I have no regret about this. You can only be obsessed with one thing at a time, and right now FPV is that thing.

A follow-up article on the spectrometer is planned. The short version: TOSLINK plastic fiber fluoresces strongly under 405nm UV, which rules out Raman spectroscopy through that pathway, and Windows doesn't expose 10-bit video from the camera with adequate exposure control, which means the handheld device needs to be redesigned for a proper fiber optic front end before the Raman work can continue.

---

## Why FPV Connects

I am not primarily interested in FPV as a filming tool or a sport. What keeps me engaged is the systems work: RF link design, noise analysis, GPS signal integrity, motor timing, PID theory. Every build is a small embedded systems project with real physics at stake.

The Pavo20 GPS problem is a genuine RF engineering problem. The ELRS link budget question is antenna theory. Blackbox analysis is signal processing. The community is full of people who fix things empirically, which is the fastest kind of engineering.

And the flying is genuinely fun. That part surprised me more than anything else.

---

## What's Coming

The spectrometer article will be a proper technical post — fiber optics, beam splitter geometry, backscattering experiments, why TOSLINK was the wrong choice, what 300µm fiber splitters do better. I'll get to it when the GPS problem is solved or when I run out of new things to break.

In the meantime: the Pavo20 still can't find GPS satellites reliably, the 1S build continues to mock it, and I'm running out of excuses to not try INAV.
