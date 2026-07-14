---
title: "Spectrometer Followup — Fiber Optics, Beam Splitters, and the TOSLINK Fluorescence Problem"
date: 2026-07-13
description: "The spectrometer rig grew a fiber optic front end. 50µm vs 300µm fibers, beam splitter cubes vs fiber splitters for backscattering, 405nm and 535nm lasers for Raman and fluorescence experiments, and why Windows 10-bit video killed the Raman work for now."
draft: true
toc: true
categories:
  - Science Instruments
  - Hardware
tags:
  - spectrometer
  - fiber-optics
  - raman
  - fluorescence
  - laser
  - spectroscopy
  - hardware
  - raspberry-pi
  - 405nm
  - 535nm
---

The original spectrometer was a Raspberry Pi build with a diffraction grating, OV9281 camera, and TOSLINK plastic optical fiber for light delivery. It worked well enough for reflectance and transmittance measurements. The plan was always to extend it toward backscattering experiments — Raman and fluorescence spectroscopy — which require a different optical geometry entirely.

That extension is what this article is about. It is early-stage experimental work. I do not have clean results to present. What I have is a description of what was built, what got ruled out, and where the constraints are.

---

## What Changed: A Dedicated Spectrometer Rig

The original handheld device was a single-purpose instrument. The new setup is a bench rig designed around fiber optic cables — the kind with FC connectors that plug into actual optical equipment.

<!-- IMAGE: photo of the bench spectrometer rig with fiber optic inputs -->
*[TODO: Photo — bench spectrometer rig with FC fiber inputs]*

The rig accepts standard FC/PC and FC/APC connectors, which opens up the full range of commercial fiber optic probes, splitters, and couplers designed for spectroscopy. This was the right architectural decision — building custom coupling every time a probe geometry changes is slower than plugging in a different fiber.

---

## 50µm vs 300µm Fiber Experiments

The first question was fiber core diameter. Spectroscopy fibers generally fall into two regimes:

| Parameter | 50 µm core | 300 µm core |
|-----------|-----------|-------------|
| Numerical aperture | 0.22 typical | 0.22–0.37 typical |
| Light collection (at equal NA) | Lower | Higher |
| Coupling efficiency into spectrometer slit | Better (tighter beam) | Harder (large exit cone) |
| Flexibility / handling | More fragile | Robust |
| Crosstalk in splitter | Lower | Higher |

For the spectrometer slit geometry I'm working with, 300µm produced better signal throughput in practice. The larger core collects more light from a diffuse sample, which matters more than coupling efficiency losses at the slit when the sample is dim.

I tested 50µm cables specifically for the backscattering geometry, where the tight beam was expected to improve spatial selectivity. The signal was too weak through 50µm to produce usable Raman features at the laser powers I was using without risking sample damage. I settled on 300µm.

---

## Beam Splitter Cube vs Fiber Splitter for Backscattering

Raman and fluorescence backscattering require the laser and collection paths to share the same axis — you illuminate the sample and collect the scattered light coming back along the same optical path. There are two common ways to do this with fiber optics.

### Beam Splitter Cube Probe

A beam splitter cube sits at the probe tip. The laser fiber and collection fiber both couple into the cube. The cube sends 50% of the laser to the sample and 50% to the collection path (wasted), and routes 50% of the returning scatter to the collection fiber and 50% back toward the laser (also wasted).

```
Laser fiber ──→ [BS cube] ──→ sample
                    ↕
Collection fiber ←── scattered light
```

I built several beam splitter probe geometries. The fundamental problem: 50% loss at input and 50% loss at collection means 25% efficiency at best — 75% of the signal is lost in the splitter. With Raman signals that are already 6–8 orders of magnitude weaker than the excitation laser, this is a significant penalty.

The cube also introduces reflections and stray light — the back-reflection of the laser at the cube surfaces appears in the collection fiber as a background signal that partially overlaps the weak Raman features.

<!-- IMAGE: photo of beam splitter cube probe geometry -->
*[TODO: Photo — beam splitter cube probe assembly]*

### Fiber Splitter (2-in-1 Connector)

The alternative: a single fiber that carries both the excitation and collection paths, separated at the splitter junction. The connector at the sample end has two fibers in a single ferrule — one for excitation, one for collection — positioned close together to share the illumination volume.

This is the approach I settled on after the beam splitter experiments. The key advantages:

- **No free-space optics** — no reflections, no dust surfaces, no alignment
- **Better coupling** — all light is in fiber, losses are only splice losses
- **Lower stray light** — the excitation and collection fibers are physically separated inside the ferrule, reducing the back-reflection problem

The downside: spatial separation between excitation and collection fibers inside the ferrule means the collection volume is not perfectly coaxial with the excitation beam. For bulk liquid samples this does not matter. For surface measurements it introduces some geometric efficiency loss.

<!-- IMAGE: photo of fiber splitter / bifurcated fiber probe tip showing two-fiber ferrule -->
*[TODO: Photo — fiber splitter bifurcated probe, two-fiber ferrule end]*

---

## Laser Configuration: 405nm and 535nm

Two laser lines are in use:

| Laser | Wavelength | Use case |
|-------|-----------|---------|
| 405 nm violet | UV/near-UV | Raman excitation; fluorescence excitation of UV-active compounds |
| 535 nm green | Visible | Raman excitation at longer wavelength; less fluorescence background |

The choice of excitation wavelength affects Raman signal strength (Raman cross-section scales as ~1/λ⁴, favoring shorter wavelengths) and fluorescence background (shorter wavelengths excite more fluorescence, often overwhelming Raman features).

In practice, 405nm produced stronger Raman signals on inorganic samples but was completely overwhelmed by fluorescence on organic samples. 535nm gave cleaner spectra on organics but weaker Raman signals overall.

Both lasers require bandpass filters on the excitation path (to clean up laser line broadening) and notch or longpass filters on the collection path (to reject the excitation laser line while passing the Raman-shifted photons).

<!-- IMAGE: photo of laser + filter + fiber coupling setup at the excitation end -->
*[TODO: Photo — 405nm laser coupling into fiber with bandpass filter]*

---

## The TOSLINK Fluorescence Problem

The original handheld spectrometer used TOSLINK plastic optical fiber. TOSLINK is cheap, flexible, and easy to terminate. It works well for white light reflectance and transmittance measurements in the visible range.

It is completely unsuitable for 405nm UV excitation.

TOSLINK plastic fiber (PMMA core) fluoresces strongly under 405nm illumination. The fluorescence emission covers a broad band from about 430nm to 550nm — exactly where weak Raman features from many compounds of interest would appear. The fiber itself becomes the largest background signal source.

This was discovered after the fiber splitter work was already running with silica fiber. Going back to test TOSLINK with the 405nm laser confirmed it immediately: the background from the fiber alone was comparable in intensity to a moderately strong fluorescence sample.

**For UV excitation: silica fiber only.** PMMA fiber is ruled out for anything below about 450nm excitation.

---

## The Windows 10-bit Video Problem

The spectrometer camera is an OV9281 monochrome rolling shutter sensor. It is capable of 10-bit output — more dynamic range, better ability to distinguish weak spectral features against a background.

On Windows, the 10-bit video path from the OV9281 is not exposed correctly. The driver delivers 8-bit or converts from 10-bit to 8-bit without adequate control. Additionally, exposure time control is extremely limited — the exposure increments are coarse, which makes it difficult to set the integration time to avoid both saturation of the laser line and underexposure of weak Raman features.

On Linux, the V4L2 driver exposes the full 10-bit output and fine-grained exposure control. This is not a software problem I can work around on Windows — it's a driver limitation.

The practical consequence: Raman spectroscopy experiments with the current hardware require Linux. The development workstation runs Windows. This means the Raman work is blocked until either:

1. The handheld device (Raspberry Pi-based, runs Linux) is updated with a proper fiber optic front end replacing the TOSLINK coupling
2. A Linux machine is set up specifically for the bench rig

Option 1 is the planned path. The handheld device needs to be rebuilt with an FC fiber input port rather than a TOSLINK socket. This is not mechanically complex but requires careful alignment of the fiber-to-slit coupling in the spectrometer housing.

---

## Where the Software Is

Despite the hardware being in an early experimental state, the software has evolved considerably. The processing pipeline handles:

- Wavelength calibration from known spectral lines (neon, mercury)
- Dark frame subtraction and flat field correction
- Rolling average for noise reduction on weak signals
- Peak detection and simple library matching against a small reference database
- Fluorescence background estimation and subtraction (polynomial baseline fitting)

<!-- IMAGE: screenshot of spectrometer software showing a spectrum with background subtraction -->
*[TODO: Software screenshot — spectrum with background subtraction visible]*

The background subtraction is the part that has been most revised. Polynomial baseline fitting works adequately for smooth fluorescence backgrounds but fails when the background has structure. The next iteration will use asymmetric least squares (ALS) fitting, which handles structured backgrounds better and is the standard approach in Raman preprocessing.

---

## Current State Summary

| Component | Status |
|-----------|--------|
| 300µm silica fiber splitter | Working — preferred over beam splitter cube |
| 50µm fiber experiments | Completed — ruled out (too little signal) |
| Beam splitter cube probes | Built — ruled out (stray light, efficiency loss) |
| 405nm laser + bandpass filter | Working |
| 535nm laser + bandpass filter | Working |
| Longpass/notch filter on collection | Working |
| TOSLINK fiber for UV | Ruled out — fluoresces strongly at 405nm |
| Windows 10-bit video | Blocked — driver limitation |
| Handheld device fiber upgrade | Planned — not started |
| Raman spectrum acquisition | Blocked pending Windows fix or Linux setup |
| Fluorescence spectroscopy | Partially working — clean spectra on some samples |

The fluorescence spectroscopy is producing usable data on some samples. Raman is blocked on the Windows video issue. The fiber splitter geometry is confirmed as the right approach for backscattering.

I'll update this when the handheld device gets the fiber optic front end. That is the next hardware milestone that unblocks the Raman experiments.
