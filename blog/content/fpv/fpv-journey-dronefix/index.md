---
title: "FPV: how it all began"
date: 2026-07-13
description: "I resisted FPV for years. Then I signed up for the Dronefix.lt training programme, flew the simulator, soldered my first build, and completely rearranged my project priorities. The spectrometer can wait."
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

For years my drones travelled with me. A DJI Air first, then a Mini 3 — camera platforms that rode along on every trip, packed next to the rest of the gear. They were good at exactly one thing: hovering somewhere legal and taking a clean photo.

That was the problem. Hovering at the permitted altitude, framing a mountain from a respectful distance, felt underwhelming every single time. The footage was fine. It was also lifeless — the drone observed from a polite distance and brought back postcards. Eventually I stopped packing one at all. On motorcycle trips especially, the Mini 3 stayed home; the weight and the faff weren't worth another set of hover shots.

What I actually wanted was the opposite of what those drones were built for: dives down a rock face, threading tight gaps, fast flybys skimming the water, runs along a ridgeline. Motion and presence. The feeling of flying rather than filming.

I had been watching FPV content for a couple of years, knowing full well it was the answer and doing nothing about it. The builds looked expensive, the learning curve looked steep, and I already had a backlog of hardware projects that weren't getting finished. The spectrometer. The DARP mesh networking work. The drone detector training pipeline that kept needing more data.

And then I signed up for the Dronefix.lt training programme and everything changed.

---

## Dronefix.lt

Dronefix.lt runs structured FPV pilot training in Lithuania. Not "here's a drone, go crash it" — an actual programme: simulator hours, theory on regulations and airspace, hands-on with real builds, guidance on first purchases.

<!-- IMAGE: photo from Dronefix.lt academy session — group, equipment, or flying area -->
*[TODO: Photo from Dronefix.lt training]*

<!-- IMAGE: photo of simulator setup or early training flight -->
*[TODO: Photo — simulator session or early training]*

The academy had all sorts of drones on hand, but the ones that caught my eye were the small ones — 2", even a few 3" builds. I was fascinated that something that small and that light could pack such a punch. In the hands of Faustas and Karolis those tiny quads were astonishing to watch, and that was the moment the whole thing clicked for me.

Acro mode is humbling. I came in thinking I had decent spatial reasoning and a background in electronics — how hard could it be? Hard. The simulator is where you start, and the early hours are worth it — though, as it turned out, the real quad would end up suiting me far better than the sim ever did.

What the programme did that I couldn't have done alone:

- **Kept me from getting ahead of myself.** I didn't build anything straight away — I just bought an Air65 in its freestyle version to get a hang of it. Learning to fly first and build later was the right order.
- **Gave me a framework for choosing gear.** The FPV market is full of competing standards, incompatible ecosystems, and products that were good two years ago. Having instructors who flew daily cut through a lot of the noise.
- **Connected me to other pilots.** The community aspect was unexpected. Other pilots are the fastest way to fix problems, find flying spots, and understand what actually matters versus what gets argued about online.

---

## The Fleet

After Dronefix.lt I did not stop at one quad. I bought one to learn on, then started hoarding motors, frames and flight controllers and building the rest.

<!-- IMAGE: fleet photo — all current quads laid out -->
*[TODO: Fleet photo — Air65, 2" rippers, 2.5" LR, Pavo20, BabyApe]*

<!-- IMAGE: individual build photos -->
*[TODO: Individual build photos]*

The current roster, more or less in the order it happened:

**Air65 (freestyle)** — bought, not built. My trainer, and the one I learned on before I trusted myself with a soldering iron and a parts pile.

**My first long-range experimental frame** — heavy, ambitious, and gone on its maiden flight. I lost it literally three meters to the side of where I was sitting. Between the inertia and not enough thrust I couldn't pull it back, clipped a tree, and it simply vanished. I searched for three days — literally three days. I studied the last few milliseconds before the feed cut out in my goggles and went through the 360 recording trying to work out where it fell after it hit a branch. Nothing. It's still out there.

**Two 2" rippers (analog and digital)** — the ones that earn the name. The 2" absolutely rips: a solid six minutes on a 2S 580mAh pack. Oddly, ever since I started flying these I can barely fly in the simulator anymore — the sim feels weird and awkward, even with identical rates, even the Air65 in Liftoff. It's just not the same.

**A 2.5" long-range experimental platform** — the next step up in range and endurance.

**Pavo20 Pro II** — 2.5" GPS whoop, my main tool for testing GPS configurations and the subject of a [separate article about GPS struggles](../pavo20-gps-struggles/). Not the most capable quad I own, but it taught me the most about RF interference and ESC noise.

**A 4" foldable "BabyApe" long-range build** — still on the bench, possibly my travel drone. Whether it earns that role depends entirely on whether its flight controller handles GPS better than the Pavo20 does — the Pavo20's GPS jamming problem is the bar it has to clear.

Each build taught something specific — motor direction failures, ESC protocol mismatches, blackbox analysis, PID tuning. The hobby is genuinely educational in a way that feels more hands-on than most software work.

---

## What I Postponed

The spectrometer project was mid-build when I found FPV. I had a working visible-light spectrometer on a Raspberry Pi with TOSLINK fiber coupling, a calibrated pipeline, and plans for a fiber-optic front end for backscattering experiments.

That project is still in the lab. The fiber optics work is progressing slowly, the 405nm and 535nm laser experiments are ongoing, and the software has evolved considerably — but the pace dropped when FPV arrived. I have no regret about this. You can only be obsessed with one thing at a time, and right now FPV is that thing.

A follow-up article on the spectrometer is planned. The short version: TOSLINK plastic fiber fluoresces strongly under 405nm UV, which rules out fluorescence spectroscopy through that pathway, and Windows doesn't expose 10-bit video from the camera with adequate exposure control, which means the handheld device needs to be redesigned for a proper fiber optic front end before the fluorescence work can continue.

---

## Why FPV Connects

I am not primarily interested in FPV as a filming tool or a sport. What keeps me engaged is the systems work: RF link design, noise analysis, GPS signal integrity, motor timing, PID theory. Every build is a small embedded systems project with real physics at stake.

The Pavo20 GPS problem is a genuine RF engineering problem. The ELRS link budget question is antenna theory. Blackbox analysis is signal processing. The community is full of people who fix things empirically, which is the fastest kind of engineering.

And the flying is genuinely fun. That part surprised me more than anything else.

---

## What's Coming

The spectrometer article will be a proper technical post — fiber optics, beam splitter geometry, backscattering experiments, why TOSLINK was the wrong choice, what 300µm fiber splitters do better. I'll get to it when the GPS problem is solved or when I run out of new things to break.

In the meantime: the Pavo20 still can't find GPS satellites reliably, the BabyApe is waiting on a flight controller I can trust outdoors, and I'm running out of excuses to not try INAV.
