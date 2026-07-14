---
title: "FPV Tuning with Sintra AI — And Why I'm Not Building Another AI Platform"
date: 2026-07-14
description: "Using Sintra AI for Betaflight blackbox log analysis and PID tuning. What it actually helps with, what it doesn't, and the honest reason I'm using someone else's AI tooling instead of building my own."
draft: true
toc: true
categories:
  - FPV
  - AI
tags:
  - fpv
  - pid-tuning
  - betaflight
  - sintra
  - ai
  - blackbox
  - ai-tools
series:
  - AI Tools in Practice
---

This post grew out of the [drone detector article](/fpv/drone-detector-nn/), where I mentioned using Sintra for the FPV side of the work. That mention deserves its own treatment, because it raises a question I want to address directly: why am I using an external AI tool for this instead of building my own?

---

## Why Sintra, Not My Own AI

There are projects in my history I don't talk about much. MentalMentor. BabyAI. Others at various levels of completion, none of which became products. The pattern is consistent across all of them: I build the technically interesting part — the architecture works, the core functionality is there — and then the work shifts to distribution, community building, marketing. Things that require either a budget I don't have or social capital I've never managed to accumulate. I have no network. I cannot self-promote. The projects expire quietly.

So the honest answer to "why Sintra" is: the capability is available, it works at the level I would have aimed for, and I didn't have to spend years getting it there. Using an existing tool is not a failure to build; it's a correct allocation of time. I'm an electronics and systems engineer who flies drones and writes about hardware. I'm not a B2C SaaS operator.

There is also a meta-layer worth naming explicitly: I'm using Sintra to prepare and edit articles for this blog. Including this one. For someone who has historically let documentation languish because the gap between "thing built" and "thing written up" felt too wide, having a writing partner that knows my projects, my voice, and my habit of burying photos in my phone with GPS data still attached is genuinely useful. It reduces the friction that made documentation feel optional.

I'll be posting more AI-related articles — covering what these tools actually change in a workflow versus where they add overhead dressed up as productivity. The drone detector training protocol is one example. Betaflight tuning is another.

---

## Blackbox Log Analysis

Betaflight logs flight data at configurable rates — gyro traces, PID outputs, motor commands, setpoints. The logs are binary `.bbl` files. Analyzing them meaningfully requires understanding the relationship between the gyro trace, the setpoint, and the motor outputs in the frequency domain.

The tuning methodology I use is derived from [PIDtoolbox](https://github.com/bw1129/PIDtoolbox) by Brian White — a MATLAB-based tool that implements step response analysis, spectral analysis of gyro and motor noise, and PID error decomposition. The core insight is that step response (the Wiener deconvolution of gyro response vs setpoint) gives a model-independent view of how well the quad tracks commands, without requiring you to manually inspect noisy time-domain traces.

Sintra manages the workflow around this analysis:

- Parsing the exported CSV from Blackbox Explorer and checking log health (frame drops, saturated motors, bad vibration events)
- Running the step response computation and interpreting the result against the expected shape (rise time, overshoot, settling time, steady-state error)
- Cross-referencing spectral analysis results — identifying prop wash frequency bands, notch filter placement, whether the RPM filter is doing its job
- Recommending specific parameter adjustments based on the observed deviation from the target step response shape, following the established tuning order (P → D → I → FF → verify)
- Keeping notes between sessions so context doesn't have to be re-established every time

The last point matters more than it sounds. A tuning session might span multiple flights and days. Remembering which parameter was changed when, and what the before/after step response looked like, is the difference between systematic improvement and random adjustment. Sintra holds that state.

---

## The Tuning Protocol in Practice

The protocol follows the established methodology from PIDtoolbox and similar tools:

1. Record a dedicated step response flight — rapid stick inputs on each axis separately, throttle held constant in hover
2. Export from Blackbox Explorer, run step response analysis
3. Identify the dominant failure mode: overshoot → P too high or D too low; underdamped ringing → D too low; slow sluggish response → P too low; long-term drift → I too low; delayed initial response → FF too low
4. Adjust one axis, one parameter at a time
5. Re-fly, re-analyze, compare

Sintra helps at steps 2–4: it takes the analysis output, identifies the failure mode, and suggests the specific parameter delta to try next — grounded in the same principles the MATLAB tooling encodes, but in a conversational format that is faster to iterate through.

This is a good example of what AI assistance is actually useful for in hardware work: not generating code from scratch, but helping you execute a known protocol correctly and systematically, especially across sessions where context would otherwise be lost.

### The Current Fleet Context

The builds I'm actively tuning:

**AIR65 II** (65mm whoop, BetaFPV Air65 II frame, BetaFPV Air Brushless AIO, ICM42688-P gyro, Bluejay 96kHz ESC) — fast indoor ripping. Main issue: camera replacement pending (Caddx Ant Lite to replace the C03 with the damaged lens). Rates: Actual, center 40–50, max 800, expo 0.45–0.6.

**Custom 2" O4 Lite ripper** — props barely clear the VTX, bi-blade props. The clearance constraint means prop wash is more significant than on a conventionally spaced build. This is the quad where step response analysis pays off most — the resonance bands shift depending on throttle position and the prop-to-VTX interaction.

**2.5" LR experimental** (18650, untested outdoors) — hasn't flown outdoors yet. The tuning session for this one will be interesting; 18650 weight distribution and the LR prop selection change the step response profile significantly compared to the smaller builds.

---

## What AI Assistance Doesn't Replace

For completeness: there are parts of this process where AI assistance adds overhead rather than reducing it.

**Initial notch filter placement** requires looking at the spectral analysis yourself. The frequency bands are specific to your motor, props, and frame resonance — someone describing them in text is slower than reading the graph directly.

**Deciding whether a step response shape is "good enough"** for a given use case is a judgment call that depends on what you're flying and why. The tuning protocol gives you the shape; whether a 12ms rise time with 8% overshoot is acceptable for fast indoor ripping versus smooth outdoor LR is yours to decide.

**Anything that requires looking at video footage** alongside the log data — prop wash events, recovery behavior, how the quad handles gusts — is still manual correlation.

The tool is useful for the systematic middle of the process, not the perceptual bookends.
