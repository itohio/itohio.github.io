---
title: "ESC PWM dažnis — 24 vs 48 vs 96 kHz"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "esc", "betaflight", "blheli", "dshot", "pwm", "frequency", "tune"]
---

ESC PWM dažnis valdo, kaip dažnai ESC atnaujina galią, paduodamą motorui. Aukštesnis dažnis = sklandesnis galios padavimas, žemesnis dažnis = mažiau karščio, bet šiurkštesnė kontrolė.

---

## Ką PWM dažnis daro

ESC naudoja PWM, kad moduliuotų įtampą, paduodamą kiekvienai motoro fazei. Dažnis nulemia, kiek kartų per sekundę vyksta šis perjungimas. Esant *tam pačiam* duty cycle (ta pati vidutinė galia), aukštesnis dažnis supakuoja daugiau perjungimo ciklų į tą patį langą — smulkesnė kontrolė, bet daugiau perjungimo įvykių per sekundę:

```wave
{ signal: [
  { name: "24 kHz", wave: "1...0...1...0..." },
  { name: "48 kHz", wave: "1.0.1.0.1.0.1.0." },
  { name: "96 kHz", wave: "1010101010101010" }
],
  head: { text: "Same ~50% duty, higher frequency = more cycles per unit time" }
}
```

- **24 kHz** — šiurkštesnis perjungimas, mažesni perjungimo nuostoliai, žemesnė ESC temperatūra
- **48 kHz** — subalansuotas; default ant daugumos modernių ESC; sklandi kontrolė, vidutinis karštis
- **96 kHz** — smulkiausias perjungimas, geriausias motoro sklandumas, aukštesnis ESC karštis

Aukštesnis dažnis sumažina „zvimbimo“ jausmą motoruose per hover ir duoda skrydžio kontroleriui detalesnę motoro kontrolę, o tai padeda filtravimui ir PID stabilumui.

---

## Kompromisai

| Dažnis    | Sklandumas | ESC karštis | Motoro efektyvumas | Desync rizika |
|-----------|------------|-------------|--------------------|---------------|
| 24 kHz    | Mažesnis   | Žemas       | Šiek tiek didesnis | Mažesnė       |
| 48 kHz    | Geras      | Vidutinis   | Geras              | Maža          |
| 96 kHz    | Geriausias | Aukštesnis  | Šiek tiek mažesnis | Didesnė (kai kurie ESC) |

- **24 kHz** labiau tinka long-range / efektyvumo build'ams, kur svarbus skrydžio laikas, o tune yra konservatyvus.
- **48 kHz** yra default daugumai freestyle ir 5" racing kvadrų.
- **96 kHz** populiarus 5" freestyle, kur sklandesnis motoro atsakas padeda su propwash, bet įsitikink, kad tavo ESC jį palaiko (ne visi BLHeli_32 ESC stabilūs ant 96 kHz prie full throttle).

---

## PWM dažnio nustatymas

### Per BLHeli Configurator / ESC-Configurator

Prisijunk per passthrough ar tiesioginį USB, atsidaryk ESC-Configurator ar BLHeli Configurator ir nustatyk **PWM Frequency** kiekvienam ESC (arba visiems iškart).

### Per Betaflight CLI (BLHeli_32 su telemetrija)

Kai kurie setup'ai leidžia nustatyti per Betaflight:
```
# Check current ESC settings (BLHeli_32 with telemetry)
# Usually done through BLHeli Suite / ESC Configurator, not BF CLI
```

### Per AM32 ESC

AM32 (atviro kodo alternatyva BLHeli_32) atveria PWM dažnį tiesiogiai:
```
# In AM32 configurator
# Set "Input PWM Frequency" to 24 / 48 / 96 kHz
```

---

## Su RPM filter (DSHOT bidirectional)

Jei naudoji RPM filter, ESC turi palaikyti bidirectional DSHOT. Dauguma BLHeli_32 ESC palaiko.

**RPM filter + 48 kHz** yra standartinė pora moderniems freestyle build'ams. Ėjimas į 96 kHz su RPM filter duoda tik nežymų papildomą pagerinimą ir didina ESC karštį — ne visada vertas (skirtumo pirštais tikrai nepajusi, o ESC — pajus).

Betaflight CLI:
```
set dshot_bidir = ON
set motor_pwm_protocol = DSHOT300  # or DSHOT600
save
```

Įjungęs bidirectional DSHOT, patikrink Motors tab'e, kad RPM telemetrija matoma (kiekvienas motoras rodo RPM rodmenį).

---

## Pastabos

- Kai kurie ESC teigia palaikantys 96 kHz, bet tampa nestabilūs prie didelės apkrovos. Testuok su agresyviais throttle punch'ais prieš skrisdamas virš ko nors brangaus (arba, sakykim, virš kaimyno automobilio — klausiu ne šiaip sau).
- PWM dažnio keitimas ant daugumos ESC reikalauja power cycle.
- PWM dažnis yra atskiras nuo DSHOT protokolo — DSHOT yra skaitmeninis protokolas; PWM dažnio nustatymas veikia analoginį fazių perjungimą motoro apvijose.
