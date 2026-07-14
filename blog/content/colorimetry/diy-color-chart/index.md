---
title: "DIY Color Calibration Chart: From Sublimation Printer to Acrylic Paint"
date: 2025-10-26T18:00:00+02:00
description: "Building a DIY color calibration chart using ArgyllCMS, a sublimation printer, and acrylic paint — to feed into Darktable's color calibration workflow"
thumbnail: "sublimation-charts.jpg"
author: admin
categories:
  - Color Science
  - Photography
tags:
  - Color Science
  - Darktable
  - ArgyllCMS
  - Calibration
  - DIY
series:
  - Color Science
---

In the [previous post](/colorimetry/reverse-engineering-cr30) I mentioned wanting to build a DIY color chart that could serve as a reference for Darktable's color calibration module. That article teased this as "a topic for another time." This is that time.

The idea is simple in theory: print a set of known color patches, measure each one with the CR30, and use ArgyllCMS to build a correction profile. In practice it took two completely different approaches before I got anything usable.

## What ArgyllCMS expects

ArgyllCMS can generate a target chart — a grid of color patches with known Lab values — and then, after you measure each patch with a colorimeter, compute the delta between what you printed and what you intended. From that it builds an ICC profile or a Darktable color checker correction.

The generated chart is just a TIFF with an accompanying `.cht` descriptor file. You print it, measure it, done. Except "print it accurately" turned out to be the hard part.

## Attempt 1: Sublimation printer

I have an old sublimation printer. CMY only — no black, no extended gamut. The prints look decent to the eye, but CMY-only puts a hard ceiling on how accurately you can hit saturated reds and deep blues.

![ArgyllCMS-generated charts next to a SpyderChecker 24](sublimation-charts.jpg)

The left chart is a ColorChecker24-sized target generated with ArgyllCMS. The right one is a denser 96-patch target — also ArgyllCMS, different randomized layout. The real SpyderChecker 24 sits in the back for comparison.

![Two ArgyllCMS charts side by side with the sublimation printer visible](sublimation-charts-2.jpg)

The sublimation prints aren't terrible. For Darktable's built-in color calibration the 24-patch chart is actually fine. But measuring the printed patches with the CR30 revealed the issue: the CMY gamut clips hard in certain reds and yellows. The ΔE on those patches is large enough to make the resulting profile unreliable for anything serious.

Good enough for a rough correction, not good enough to validate the CR30 software's accuracy.

## Attempt 2: Acrylic paint

If the printer can't hit the colors accurately, paint them by hand. Acrylic paint gives you access to real primaries — including colors a CMY printer can't reproduce — and the CR30 can measure each patch directly on the painted surface.

The substrate: small squares cut from a rigid white plastic sheet, painted individually so each patch can be remeasured and recoated without redoing the whole chart.

![Acrylic paint palette with primary paint chips — dry](acrylic-palette-dry.jpg)

Primary colors: blue, green, black, yellow, red, gray, white. The chips in the tray are the dried swatches cut from the plastic sheet.

![Wet acrylic paint mixed in a palette tray](acrylic-palette-wet.jpg)

Mixing in progress. The top half of the tray holds the base primaries; the bottom half is where I mixed secondaries and adjusted lightness with white.

![Work surface with painted patches and palette](acrylic-work.jpg)

The acrylic approach opened up a whole different problem: batch-to-batch consistency. Acrylic paint dries slightly darker than it looks wet, and thin coats transmit the white substrate, shifting the measured color toward white. Two coats minimum, ideally three, with the CR30 measuring after each coat to confirm the color stabilized.

The upside: the measured colors are genuinely where they need to be. Mixing ultramarine blue gives you a real blue — not the printer's nearest approximation. Same for yellow, red, and the neutrals.

## Why this matters for Darktable

Darktable's color calibration module works by taking a shot of a known color reference under the same light as your subject, then computing a 3×3 matrix (plus offsets) that maps the camera's raw response to the reference Lab values. Normally you'd buy a ColorChecker or SpyderChecker. These DIY charts are a cheaper alternative — provided the reference values you feed Darktable actually match what's on the physical chart.

That's exactly what the CR30 provides: measured Lab values for each patch under D65 illuminant. Feed those into ArgyllCMS alongside the photo measurements and you get a correction that's grounded in actual measurement rather than a factory spec sheet.

Whether the DIY chart is *accurate enough* is still an open question. The acrylic patches are matte and reasonably uniform, but they're nowhere near as spectrally flat or precisely controlled as a professional target. The CR30's ΔE on the painted patches is good — mostly under 3–4 on saturated colors, under 1 on neutrals — but I'd like to validate this against the SpyderChecker 24 before claiming the approach works end-to-end.

That validation is the next step.

## What's next

- Photograph both charts under controlled light, run through ArgyllCMS, compare the resulting corrections
- Validate the CR30's measured values against the SpyderChecker 24's known Lab values
- If the acrylic chart holds up, document the full workflow: chart generation → painting → measurement → Darktable profile

The sublimation-printed charts are useful as a quick baseline. The acrylic chart is the one I'm betting on for actual calibration work.
