---
title: "KV and Prop Matching — Tip Speed, Load, and Selection"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "motor", "kv", "props", "tip-speed", "efficiency", "thrust", "matching"]
---

Matching motor KV to prop size is the single most important factor in build efficiency and motor longevity. The goal: keep tip speed in the efficient range and keep motor temperature reasonable.

---

## Prop Tip Speed

Propeller tip speed is the velocity at which the blade tip moves through the air. As tip speed approaches the speed of sound (~343 m/s), efficiency drops sharply and noise increases dramatically.

**Practical efficient range: 100–150 m/s tip speed at hover/cruise throttle.**

### Calculation

```
Tip Speed (m/s) = (π × Prop Diameter [m] × RPM) / 60
```

Example — 5" prop (0.127 m diameter) at 20,000 RPM:
```
Tip Speed = (π × 0.127 × 20,000) / 60
           = (3.1416 × 0.127 × 20,000) / 60
           ≈ 7,980 / 60
           ≈ 133 m/s
```
→ 133 m/s is within the efficient range.

---

## Quick Reference Table

| Prop diameter | Max efficient RPM (150 m/s) | Typical KV on 4S (14.8V) |
|---------------|----------------------------|--------------------------|
| 3" (76 mm)    | ~37,600 RPM                | ~3,500–4,500 KV          |
| 4" (102 mm)   | ~28,100 RPM                | ~2,600–3,200 KV          |
| 5" (127 mm)   | ~22,500 RPM                | ~2,000–2,500 KV          |
| 5" (127 mm)   | ~22,500 RPM                | ~1,500–1,800 KV on 6S    |
| 7" (178 mm)   | ~16,100 RPM                | ~1,300–1,600 KV on 4S–6S |
| 10" (254 mm)  | ~11,300 RPM                | ~700–900 KV on 6S        |

---

## Motor Load Index (Thrust-to-Weight)

A useful sanity-check at hover: each motor carries a quarter of the all-up weight.

**Hover thrust per motor:**
```
Hover thrust per motor = AUW / 4        (for a quad)
```

For a ~500–700 g 5" quad that is ~125–175 g per motor, which usually lands around 40–50% throttle on a healthy build.

**Thrust-to-weight ratio (TWR)** compares *total full-throttle* thrust to AUW:
```
TWR = (4 × max thrust per motor) / AUW
```

A TWR of 4:1 is typical for freestyle, 3:1 is fine for cinematic, and racing wants 6:1+. A 4:1 quad lifts its own weight using only a quarter of its available thrust — the rest is headroom for punch-outs.

---

## Prop Pitch and Motor Selection

Pitch is the theoretical distance a prop advances per revolution. Higher pitch = more aggressive bite = more speed but more drag = motor works harder.

Unroll a single blade element over one revolution and pitch is easy to see: the base is the circle the element travels (`2πr`), and the *rise* is the pitch — how far it would screw forward through solid air. A steeper blade has more pitch. In the air the prop never advances the full geometric pitch (the shortfall is **slip**), and the angle between the blade's chord and the air it actually meets is the **angle of attack**:

```p5js
const p = sketch;
// Prop pitch = how far a blade would screw forward in one turn.
// Unroll one revolution of a blade element: base = circumference (2*pi*r),
// rise = geometric pitch. At a FIXED airspeed (fixed actual advance), more
// pitch means a steeper blade angle and a bigger angle of attack -> stall.
let W = 560, H = 360;
let ox = 90, oy = H - 64, baseLen = 380;
const C = 11.0;         // circumference at 0.7R of a 5" prop: 2*pi*1.75" (inches)
const peFixed = 3.0;    // actual advance per rev, set by airspeed (held fixed)
const pitches = [3.0, 4.5, 6.0];
let idx = 0, curP = 3.0, timer = 0, dotT = 0;

p.setup = function () {
  p.createCanvas(W, H);
  p.textFont('monospace');
};

p.draw = function () {
  p.background(17, 17, 17);
  timer++;
  if (timer > 210) { timer = 0; idx = (idx + 1) % pitches.length; }
  curP = p.lerp(curP, pitches[idx], 0.06);
  dotT = (dotT + 0.01) % 1;

  const hGeo = baseLen * (curP / C);
  const hEff = baseLen * (peFixed / C);
  const tipX = ox + baseLen;
  const geoTipY = oy - hGeo;
  const effTipY = oy - hEff;
  const thetaDeg = p.degrees(Math.atan2(hGeo, baseLen));
  const phiDeg = p.degrees(Math.atan2(hEff, baseLen));
  const aoa = thetaDeg - phiDeg;

  p.stroke(120, 130, 145); p.strokeWeight(2);
  p.line(ox, oy, tipX, oy);
  p.stroke(90, 110, 130); p.strokeWeight(1);
  dashLine(tipX, oy, tipX, geoTipY);

  p.noStroke(); p.fill(255, 140, 40, 55);
  p.triangle(ox, oy, tipX, geoTipY, tipX, effTipY);

  p.stroke(70, 200, 120); p.strokeWeight(2.5);
  p.line(ox, oy, tipX, effTipY);
  p.stroke(90, 170, 255); p.strokeWeight(3);
  p.line(ox, oy, tipX, geoTipY);

  const mx = ox + baseLen * 0.5, my = oy - hGeo * 0.5;
  p.push(); p.translate(mx, my); p.rotate(-Math.atan2(hGeo, baseLen));
  p.noStroke(); p.fill(205, 218, 232);
  p.ellipse(0, 0, 80, 15);
  p.pop();

  const dx = ox + baseLen * dotT, dy = oy - hGeo * dotT;
  p.noStroke(); p.fill(255, 220, 90); p.ellipse(dx, dy, 9, 9);

  p.textSize(12); p.textAlign(p.LEFT);
  p.fill(150, 190, 255); p.text("geometric pitch  P = " + p.nf(curP, 1, 1) + "\"  (blade angle " + p.nf(thetaDeg, 0, 0) + "\u00B0)", 14, 22);
  p.fill(90, 220, 140); p.text("actual advance / rev (airspeed) = " + p.nf(peFixed, 1, 1) + "\"", 14, 40);
  p.fill(255, 170, 80); p.text("angle of attack = " + p.nf(aoa, 0, 0) + "\u00B0" + (aoa > 12 ? "  near stall" : ""), 14, 58);
  p.fill(140, 150, 165); p.textAlign(p.CENTER);
  p.text("one revolution  =  2\u03C0r  (circumference at this radius)", ox + baseLen / 2, oy + 26);
};

function dashLine(x1, y1, x2, y2) {
  const n = 14;
  for (let i = 0; i < n; i += 2) {
    p.line(p.lerp(x1, x2, i / n), p.lerp(y1, y2, i / n),
           p.lerp(x1, x2, (i + 1) / n), p.lerp(y1, y2, (i + 1) / n));
  }
}
```

Hold airspeed fixed and crank the pitch up: the blade angle steepens, the angle of attack grows, and a high-pitch prop reaches its stall angle sooner. That is the direct link to [propwash](../../aerodynamics/propwash/) — high-pitch props bite hard but stall harder in disturbed inflow, so they lean on higher RPM (and dynamic idle) to stay attached.

| Use case         | Pitch recommendation         |
|------------------|------------------------------|
| Efficiency / long range | Low pitch (3.8"–4.3") |
| Freestyle        | Moderate (4.8"–5.1")         |
| Racing / top speed | Higher pitch (5.1"–6.0")  |

**Higher pitch requires more torque → lower KV motor on higher voltage** for the same efficiency.

---

## Matching Workflow

1. **Choose frame size** → sets prop diameter range
2. **Choose battery voltage** → sets voltage input to motor
3. **Choose use case** → sets target RPM range and pitch preference
4. **Calculate required KV:**
   ```
   KV = Target Cruise RPM ÷ (Voltage × 0.75)
   ```
5. **Verify tip speed** at estimated max RPM:
   ```
   Max RPM = KV × Max Voltage
   Tip Speed = (π × Diameter_m × MaxRPM) / 60
   ```
   Should stay below ~170 m/s; ideally below 150 m/s.
6. **Check stator size** — larger stator (e.g. 2306 vs 2204) handles more heat at a given KV, so heavier prop combinations need a bigger stator.

---

## Worked Example — 5" Freestyle 4S

- Frame: 5" (0.127 m), battery: 4S (16.8 V full charge)
- Target full-throttle *loaded* RPM ≈ 24,000–28,000 RPM

Work backwards from the loaded max RPM to KV (loaded ≈ 75% of no-load `KV × Voltage`):
```
KV = Max loaded RPM ÷ (Voltage × 0.75)
KV = 26,000 ÷ (16.8 × 0.75) = 26,000 ÷ 12.6 ≈ 2,060 KV
```
→ Tip speed at that loaded max: `π × 0.127 × 26,000 / 60 ≈ 173 m/s`. That is above the 150 m/s efficient ceiling — normal for freestyle, which trades some efficiency for punch (racing runs higher still).

In practice, **2000–2450 KV** motors are the established sweet spot for 5" on 4S, matching the quick-reference table above. (A designation like "2306" is the *stator size* — 23 mm × 6 mm — not a KV value.)

---

## Prop Selection Rules of Thumb

- **Diameter** — determined by frame arm length and motor mount spacing. Don't exceed frame limits.
- **Pitch** — higher pitch for speed and responsiveness; lower pitch for efficiency and hover time.
- **Blade count** — 3-blade: efficiency and handling balance. 4/5-blade: more thrust in same diameter, more noise and less efficiency.
- **Tip speed** — always check your calculated tip speed at max throttle. If it exceeds 180 m/s you're leaving efficiency on the table and generating unnecessary noise.

---

## Notes

- These calculations give theoretical RPM. Real loaded RPM with a prop is typically 70–80% of no-load KV × voltage.
- Motor thrust tables from manufacturers are measured at specific voltages on specific props — always cross-reference for your exact combination.
- Small differences in prop manufacturer (HQ, Gemfan, DAL) affect actual RPM, thrust, and efficiency even on nominally identical props.
