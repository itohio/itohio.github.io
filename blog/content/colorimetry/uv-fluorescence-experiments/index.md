---
title: "Glowing Things: First UV Fluorescence Experiments"
date: 2025-11-09T18:00:00+02:00
description: "Building a fiber-coupled UV fluorescence setup — and why TOSLINK, despite its convenience, can never be the light carrier for fluorescence spectroscopy"
thumbnail: "vitamin-b-cuvette.jpg"
author: admin
categories:
  - Spectroscopy
  - Hardware
tags:
  - Fluorescence
  - UV
  - Fiber Optics
  - TOSLINK
  - 3D Printing
  - Spectroscopy
series:
  - Color Science
---

With the spectrometer alignment sorted and PySpectrometer3 taking shape, the obvious next experiment was UV fluorescence. Minerals, biological samples, gemstones — a lot of interesting chemistry shows up as emission spectra under UV excitation that you'd never see under white light. The setup is conceptually simple: illuminate the sample with UV, block the excitation from reaching the spectrometer, collect the fluorescence emission.

In practice the first problem appeared before I even built anything.

## TOSLINK fluoresces

I've been routing light from sources to samples via TOSLINK plastic optical fiber — it's cheap, has a standardized 1mm aperture, and the connectors are everywhere. The plan was to use the same fiber for UV excitation delivery.

Quick test first: take a handful of TOSLINK fiber cuts, put them under the UV lamp, see if they transmit cleanly.

![TOSLINK fiber cuts under UV illumination, glowing faint blue-white](toslink-uv-glow.jpg)

They glow. Not faintly — visibly, unmistakably. The PMMA (polymethyl methacrylate) plastic that TOSLINK fiber is made from fluoresces under UV excitation, emitting in the blue-green region (~440–520nm). The very wavelengths a fluorescence experiment needs to measure clearly.

This is a fundamental material property, not a brand issue. All PMMA fiber will do this. The fiber itself becomes a broadband background emitter, and any fluorescence signal from the sample is riding on top of fiber-generated noise across the entire visible range.

TOSLINK is out as a UV excitation carrier. It works fine for visible-range transmission spectroscopy where the source wavelength is above ~450nm, but for anything involving UV excitation and visible emission collection, it contaminates the measurement.

The replacement — properly, at least — is silica (quartz) fiber, which doesn't fluoresce in the UV. That's a separate project. For now I could still use TOSLINK for the collection side, as long as UV light never enters it.

## What actually glows

With that sorted, the immediate experiment was qualitative: what fluoresces, and what color?

The short answer: more than you'd expect.

![Green fluorescent bead necklace under UV — vivid green emission from the plastic](jewelry-green-uv.jpg)

These plastic beads emit a strong vivid green. The dye is almost certainly a UV-reactive fluorescent polymer additive — the kind used deliberately in costume jewelry for black-light effects. Emission is so intense the beads look like they're self-illuminating.

![Blue-white fluorescent bead necklace under UV](jewelry-blue-uv.jpg)

A different bead necklace — softer blue-white emission. The beads have a milky translucent appearance in daylight. Under UV they emit broad-spectrum blue-white, consistent with the kind of fluorescence you see in certain glass types or synthetic opalite. Cooler and more diffuse than the green bead emission.

![Red bead necklace under UV — warm orange-red fluorescence](jewelry-red-uv.jpg)

Red/pink beads with a warm orange-red emission. The emission color differs from the apparent daylight color — under white light these are red, under UV they fluoresce orange-red with a slightly different peak. The emission is less intense than the green beads, but clearly separate from the excitation.

None of these are scientifically interesting as samples — they're doped plastics engineered to fluoresce — but they're useful for checking that the setup works and that the different emission colors are distinguishable.

The more interesting sample:

![Vitamin B solution in quartz cuvette mounted in 3D-printed holder, glowing cyan-blue under UV](vitamin-b-cuvette.jpg)

Riboflavin (vitamin B₂) dissolved in water fluoresces strongly in the blue-green region under UV excitation — peak emission around 520nm with broad shoulders. The glow in the cuvette is cyan rather than the green you might expect because the camera's white balance is pulling toward the UV lamp's purple-blue ambient. Under the spectrometer this would show up as a clean emission band.

## The cuvette holder

Putting samples under a handheld UV lamp and photographing them is informative but not spectroscopy. What's needed is controlled coupling: UV in through one port, sample in the center, fluorescence emission out through another port, with the UV excitation blocked from reaching the collection fiber.

![3D-printed cuvette holder — top view showing four collimated TOSLINK ports and center cuvette aperture](cuvette-holder-top.jpg)

The holder is 3D-printed with four ports arranged symmetrically around a central cuvette aperture. Each port takes a TOSLINK connector directly. The geometry gives four independent channels: two for excitation/reference, two for collection — or in practice, one excitation input, one primary collection output, and the remaining two for filter insertion or secondary reference collection.

Each port has a short collimating tube molded in. Collimation with TOSLINK is forgiving: the 1mm fiber aperture accepts a reasonable cone angle without needing precision alignment. A sheet of paper as an alignment guide gets the fiber close enough for TOSLINK — the 1mm aperture is large enough that small angular errors don't kill throughput.

This is not true for small-core fiber. I tried the same 3D-print approach with 50µm multimode fiber: the alignment tolerance at that scale is tighter than a 3D printer can reliably produce. The coupling efficiency was terrible and not recoverable by hand-tweaking the mounts. TOSLINK's thick core is genuinely useful here — the whole setup becomes mechanically tolerant in a way that wouldn't work with laboratory fiber.

A small brass column in the center holds the cuvette. Standard 10mm × 45mm quartz cuvettes fit directly; the holder keeps the optical path centered on the cuvette's mid-height.

## The box

All of the above needs to happen in the dark. UV fluorescence experiments with ambient light leaking in are useless — the scattered UV and visible room light overwhelm the weak emission signal.

![Interior of the UV fluorescence measurement box — foam-lined project enclosure with UV LED, cuvette holder assembly, and fiber cables](uv-box-interior.jpg)

The enclosure is a standard ABS project box with custom-cut black foam padding that holds the cuvette holder assembly rigidly in place. The UV LED is mounted and wired inside — yellow and orange wires for the LED drive, red for a separate indicator or reference sensor. Fiber cables exit through the box sides via TOSLINK connectors in the box walls.

The foam cutouts matter: everything stays in position when the lid goes on. The cuvette can be swapped without realigning the fiber ports, which is the whole point of a standardized holder geometry.

A TOSLINK switcher on the input side allows switching between measurement modes without opening the box:

- **Transmission**: UV source on one side, collection on the opposite side, direct transmission through the sample
- **Scattering**: 90° collection geometry, collection fiber at right angles to excitation
- **Fluorescence with UV cut filter**: excitation in, longpass filter on the collection port to block any scattered UV, collect only the Stokes-shifted emission

The switcher is a simple mechanical fiber switch — TOSLINK connectors on a rotating selector. No electronics, no software. Crude but reliable.

## What this setup actually measures

The output of the collection fiber goes to the jewel spectroscope + OV9281 camera described in the previous posts. With PySpectrometer3 running in measurement mode, a fluorescence spectrum from the riboflavin solution takes about 10 seconds: set up the sample, switch to fluorescence mode, trigger acquisition, get a wavelength-calibrated emission spectrum.

The TOSLINK limitation still applies to the collection side in a narrow sense: if UV somehow backscatters into the collection fiber, the PMMA would re-emit in the visible, adding a flat background. The longpass UV-cut filter on the collection port prevents this. With the filter in place, only wavelengths above ~400nm reach the fiber, so PMMA fluorescence from the collection fiber (which would require UV excitation) can't occur.

The setup isn't publishing-quality — the cuvette holder geometry isn't optimized for solid angle, the UV LED spectrum isn't calibrated, and the sensitivity correction assumes visible illumination rather than UV. But it's functional enough to identify which samples fluoresce, characterize emission color and rough peak position, and distinguish overlapping emission bands in mixtures.

## What's next

This first version works but it's cramped and hard to reconfigure. A second modular version is in progress — separate excitation, sample, and collection blocks that can be rearranged without reprinting the whole holder.

And there's a different kind of coupling problem that showed up while working on the fiber alignment question. A 200µm multimode fiber, an old microscope, a 532nm laser, and a crude 3D-printed backscattering coupler:

![Backscattering Raman experiment — 532nm beam through beam splitter, 200µm fiber coupling, microscope objective as collection/focusing optic](raman-teaser.jpg)

That's for the next post.
