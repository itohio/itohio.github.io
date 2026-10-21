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

I finally had some free time to try WiFi sensing myself, with a few tweaks. It also has two problems that I have not solved: the direct signal between the antennas is too strong, and CSI freezes whenever the leader connects to a router.

This is a teaser. The code is ugly and experimental, and I will clean it up and publish a GitHub repo later. Further CSI experiments are coming, and some of them have a lot to do with FPV.

## The setup

Two Seeed Studio XIAO ESP32-C5 boards, soldered side by side. They share UART and power, so they can talk to each other.

- The boards decide on their own which one is the **leader** and which is the **follower**.
- The leader hosts a web app over WiFi.
- The boards take turns. One sends raw WiFi packets at 100 Hz to illuminate the environment while the other senses, then the roles reverse.
- Each board passes its CSI data to the other over UART. The filtering that finds motion, human presence and breathing runs on the boards.

I wanted to be able to switch WiFi channels, but then the access point would not work.

The firmware is ESP-IDF. I opted out of updating TinyGo for this, since that needed too much effort for a one-night experiment.

[TODO: Photo — the proof of concept: both C5 boards soldered side by side, UART and power wires visible]

## LHCP on one side, RHCP on the other

Motion is predominantly a reflection. The idea is to use a left-hand circularly polarized antenna on one board and a right-hand one on the other, so the polarization acts as a rough filter on reflections.

My working assumption is that a reflection flips the handedness, so the opposite-handed antenna should prefer reflected energy over the direct path. It is an assumption. I have not verified it here.

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

The separation figures are RSSI.

I have no measurement of whether the foam absorbs anything at WiFi frequencies. Checking that properly needs two directional or horn antennas, which I do not have, though I might build one. It would also need an anechoic chamber, which I do not have either. So the foam is a guess.

## What the sensing looks like

The leader's web app shows the data in real time: the CSI subcarriers, spectrograms, and the breathing trace.

[TODO: Photo — web app screenshot of breathing detection]

[TODO: Photo — spectrogram screenshot]

[TODO: Photo — CSI traces / subcarrier plot]

I got the breathing plot when the separation was high. The spectrograms, however, show noise, so the breathing trace might be just noise. I cannot tell yet, and I am not claiming it is breathing.

[TODO: Andrius — which filters do you use for motion, presence and breathing? Anything with numbers you want stated as fact.]

## The router bug

Hosting an access point and streaming binary frames to the web app, with real-time sensing data and the channel subcarriers, works at the same time. That part is fine.

Connecting the leader to a router is not. CSI freezes on the leader after 3 minutes. Always. I have not found out why.

The details of the freeze:

- CSI keeps reporting valid frames, but all the values are identical. The packets are byte-for-byte the same.
- It survives soft resets and resets.
- It goes away only after a power cycle.

I do not know the cause yet, so I am not going to guess one here.

## Where things stand

Presence, motion and breathing detection runs on the boards. Whether the breathing is real is open, because the spectrograms show noise. The direct path between the antennas is too strong, and the glue showed how fragile 26 dB was. The router freeze is open.

These are the first steps into something more interesting, and the GitHub repo follows once the code is cleaned up.
