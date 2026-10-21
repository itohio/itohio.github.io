---
title: "ESP32-C5 CSI Sensing with LHCP and RHCP Antennas"
date: 2026-10-21
description: "Two ESP32-C5 boards wired together over UART, sensing motion, presence and breathing from WiFi CSI. The direct path between the antennas is winning."
draft: false
toc: true
categories:
  - Hardware
  - WiFi Sensing
tags:
  - esp32-c5
  - csi
  - wifi-sensing
  - presence-detection
  - breathing-detection
  - lhcp
  - rhcp
  - antennas
keywords: ["ESP32-C5 CSI", "WiFi CSI presence detection", "WiFi breathing detection", "LHCP RHCP antenna isolation", "ESP32 channel state information"]
---

I finally had some free time to try WiFi sensing myself, with a few tweaks. It works well enough to show breathing on a plot. It also has a problem that I have not solved: the two antennas hear each other far better than they hear me.

This is a teaser. The code is ugly and experimental, and I will clean it up and publish a GitHub repo later. Further CSI experiments are coming, and some of them have a lot to do with FPV.

## The setup

Two ESP32-C5 boards, soldered side by side. They share UART and power, so they can talk to each other.

- The boards decide on their own which one is the **leader** and which is the **follower**.
- The leader hosts a web app over WiFi.
- Both boards send raw WiFi packets at 100 Hz. That illuminates the environment and, at the same time, each board senses what the other one sent.
- Each board passes its CSI data to the other over UART. The filtering that finds motion, human presence and breathing runs on the boards.

[TODO: Andrius — what firmware stack is this: ESP-IDF C, or TinyGo? Which board model, and are the two sides really "TX and RX at once on each board" or alternating?]

[TODO: Photo — the proof of concept: both C5 boards soldered side by side, UART and power wires visible]

## LHCP on one side, RHCP on the other

Motion is predominantly a reflection. The idea is to use a left-hand circularly polarized antenna on one board and a right-hand one on the other, so the polarization acts as a rough filter on reflections.

[TODO: Andrius — which antenna transmits and which receives, and are both boards doing both? Also: is the working assumption that a reflection flips the handedness, so the opposite-handed antenna prefers reflected energy over the direct path?]

[TODO: Photo — the dual LHCP and RHCP antennas]

## The direct path problem

The direct signal between the two antennas is far too strong. What I know is wrong with the geometry:

- The antennas are close together, so the separation is a near-field one.
- The shield between them is smaller than the minimum size it should be, and it sits at a non-ideal distance.
- A metal shelf is right next to the antennas.
- The antennas are poorly aligned.

For the shield, I had fume extractor filter foam lying around. I measured about 600 kΩ over 5 mm, so I figured it would be better than nothing as a makeshift RF absorber. It did not fix the direct signal.

[TODO: Photo — the separator with the foam absorber between the antennas]

I tried moving the antennas around and reached **26 dB** of separation. Then the glue settled, and the separation went back to about **4 to 8 dB**.

[TODO: Andrius — how did you measure the separation? Which quantity: RSSI between the boards, CSI amplitude, something else? A single number or a spectrum?]

[TODO: Andrius — the foam resistance: how was it measured (probe spacing, multimeter)? And is it known whether foam at 600 kΩ per 5 mm actually absorbs at 5 GHz, or is it only a guess? I will label it a guess unless you have a measurement.]

## What the sensing looks like

The leader's web app shows the data in real time: the CSI subcarriers, spectrograms, and the breathing trace.

[TODO: Photo — web app screenshot of breathing detection]

[TODO: Photo — spectrogram screenshot]

[TODO: Photo — CSI traces / subcarrier plot]

[TODO: Andrius — what did you do to get the breathing plot (distance, posture, how many seconds of window)? Which filters do you use for motion, presence and breathing? Give me anything you want stated as a fact here, with numbers.]

## The router bug

Hosting an access point and streaming binary frames to the web app, with real-time sensing data and the channel subcarriers, works at the same time. That part is fine.

Connecting the leader to a router is not. CSI freezes on the leader after 3 minutes. Always. I have not found out why.

[TODO: Andrius — what exactly freezes: do the packets stop arriving, does the CSI callback stop, does the board hang or reboot? Is the follower affected? And what is the router band and channel versus the AP channel?]

I am not going to guess a cause in the post. If you have seen this on the C5, I would like to hear about it.

## Where things stand

Presence, motion and breathing are detected. The isolation between the antennas is the thing standing in the way of better data, and the glue demonstrated how fragile 26 dB is. The router freeze is open.

Next: a proper mechanical mount that does not depend on glue, a better shield, and the cleaned-up repo.
