---
title: "Rate preset'ai — 733 / 633 / 533 Betaflight'e ir Actual'e"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "actual-rates", "presets", "cli", "tuning", "733", "633", "533"]
---

`533`, `633`, `733` — tai bendruomenės trumpiniai trims populiariems freestyle rate profiliams, o skaičius **ir yra maksimalus sukimosi greitis °/s** — `733` viršija apie 730 °/s, `533` — apie 530 °/s. Visi jie sukurti vienodai legacy Betaflight sistemoje: fiksuotas **Super Rate 0.70** su laipsniškai keliamu **RC Rate**. Šis snippet'as pateikia kiekvieną abiejose rate sistemose (legacy Betaflight *ir* Actual) su kopijuojamu CLI ir realia kreive. Formules už šių sistemų rasi čia: [Rate Modes](../rate-modes/); centro/vidurio/krašto zonas — [Rates Deep Dive](../rates-deep-dive/).

---

## Kodėl skaičius ir yra maks. rate

Legacy Betaflight sistemoje lubos ties pilnu stick'u yra:

```
maxRate = 200 × RC Rate / (1 − Super Rate)
```

Kai Super Rate fiksuotas ties 0.70, vardiklis lygus `0.30`, tad `maxRate = 666.7 × RC Rate`:

| Profilis | RC Rate | Super Rate | 200 × RC ÷ (1 − SR) | Maks. rate |
|---------|---------|-----------|---------------------|----------|
| **533** | 0.80    | 0.70      | 200 × 0.80 ÷ 0.30   | **533 °/s** |
| **633** | 0.95    | 0.70      | 200 × 0.95 ÷ 0.30   | **633 °/s** |
| **733** | 1.10    | 0.70      | 200 × 1.10 ÷ 0.30   | **733 °/s** |

RC Rate kėlimas pakelia visą kreivę (ir lubas); bendras Super Rate 0.70 suteikia joms tą pačią „minkštokas centras, kietas kraštas“ formą. Expo **nėra** trumpinio dalis — tai asmeninis priedas (dažniausiai 0.10–0.20), kuris suminkština centrą nekeisdamas maksimumo.

> Tai sąžiningi freestyle rates. Palyginimui: racing paprastai laikosi plokštesnio ~550–650 °/s, bendras freestyle ~650–900 °/s, o agresyvus/pro freestyle 900–1200 °/s. `533` — ramusis galas, `733` — tvirtai punchy.

---

## Trumpai

| Profilis | Pojūtis                 | Legacy BF (RC · SR · Expo) | Actual (center / max / expo) | Maks. rate | Kam tinka                         |
|---------|-------------------------|----------------------------|------------------------------|----------|-----------------------------------|
| **733** | Punchy, greiti flip'ai  | `1.10 · 0.70 · 0`          | `220 / 730 / 55`             | ~733 °/s | Užtikrintas freestyle, snappy triukai |
| **633** | Subalansuotas visapusis | `0.95 · 0.70 · 0`          | `190 / 630 / 55`             | ~633 °/s | Bendras freestyle / kruizavimas    |
| **533** | Ramus, sklandus, tiesinis | `0.80 · 0.70 · 0`        | `160 / 530 / 54`             | ~533 °/s | Acro mokymasis, racing linijos, cine |

Actual stulpeliai atkuria tas pačias kreives ~0.5 % tikslumu (patikrinta su Betaflight `applyBetaflightRates` / `applyActualRates`).

---

## Trys kreivės

```chart
{
  "type": "line",
  "data": {
    "labels": ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
    "datasets": [
      {
        "label": "533 — center 160, max 533 °/s",
        "data": [0, 17.2, 37.2, 60.8, 88.9, 123.1, 165.5, 219.6, 290.9, 389.2, 533.3],
        "borderColor": "rgba(34,197,94,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5, "tension": 0.3, "pointRadius": 3
      },
      {
        "label": "633 — center 190, max 633 °/s",
        "data": [0, 20.4, 44.2, 72.2, 105.6, 146.2, 196.6, 260.8, 345.5, 462.2, 633.3],
        "borderColor": "rgba(59,130,246,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5, "tension": 0.3, "pointRadius": 3
      },
      {
        "label": "733 — center 220, max 733 °/s",
        "data": [0, 23.7, 51.2, 83.5, 122.2, 169.2, 227.6, 302.0, 400.0, 535.1, 733.3],
        "borderColor": "rgba(249,115,22,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5, "tension": 0.3, "pointRadius": 3
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": { "display": true, "text": "733 / 633 / 533 — commanded °/s vs stick deflection" },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": { "title": { "display": true, "text": "Stick deflection" } },
      "y": { "beginAtZero": true, "max": 760, "title": { "display": true, "text": "Rotation rate (°/s)" } }
    }
  }
}
```

Ta pati forma, trys aukščiai. Super Rate 0.70 laiko pirmuosius ~60 % stick'o gana ramius, o tada kreivė smarkiai kyla link pilno nuspaudimo — būtent ten gyvena „snap'as“ flip'ams ir rolliams. Didesnis RC Rate pakelia ir centro gyvumą, ir lubas. Taigi jei tau atrodo, kad dronas „nervingas“ — dažniausiai kaltas ne maks. rate, o pernelyg gyvas centras.

---

## 733 — punchy

Centro jautrumas ~220 °/s ir lubos ~733 °/s. Pilno stick'o flip'ai ir rolliai ateina greitai, o Super-Rate kreivė laiko vidurio stick'ą valdomą linijų darbui. Populiarus „dabar tikrai galiu freestyle'inti“ profilis. Agresyvūs pilotai kelia RC Rate dar aukščiau (833/933+).

**Betaflight (legacy):**
```
set rates_type = BETAFLIGHT
set roll_rc_rate = 110
set pitch_rc_rate = 110
set yaw_rc_rate = 110
set roll_srate = 70
set pitch_srate = 70
set yaw_srate = 70
set roll_expo = 0
set pitch_expo = 0
set yaw_expo = 0
save
```

**Actual:**
```
set rates_type = ACTUAL
set roll_rc_rate = 22
set pitch_rc_rate = 22
set yaw_rc_rate = 22
set roll_srate = 73
set pitch_srate = 73
set yaw_srate = 73
set roll_expo = 55
set pitch_expo = 55
set yaw_expo = 55
save
```

---

## 633 — subalansuotas

Centro jautrumas ~190 °/s, maks. ~633 °/s. Vidurinis freestyle profilis — pakankamai jautrus triukams, pakankamai ramus ilgoms kruizavimo linijoms. Puikus numatytasis, kol sprendi, ar nori daugiau (733), ar mažiau (533).

**Betaflight (legacy):**
```
set rates_type = BETAFLIGHT
set roll_rc_rate = 95
set pitch_rc_rate = 95
set yaw_rc_rate = 95
set roll_srate = 70
set pitch_srate = 70
set yaw_srate = 70
set roll_expo = 0
set pitch_expo = 0
set yaw_expo = 0
save
```

**Actual:**
```
set rates_type = ACTUAL
set roll_rc_rate = 19
set pitch_rc_rate = 19
set yaw_rc_rate = 19
set roll_srate = 63
set pitch_srate = 63
set yaw_srate = 63
set roll_expo = 55
set pitch_expo = 55
set yaw_expo = 55
save
```

---

## 533 — ramus

Centro jautrumas ~160 °/s, maks. ~533 °/s. Švelniausias iš trijų: lėtesnis sukimasis ir ramesnis centras daro jį atlaidų mokantis acro, o plokštesnis pojūtis tinka racing stiliaus linijoms ir sklandžiam cinematic skraidymui. Vis tiek tikras freestyle rate, tik atsipalaidavęs.

**Betaflight (legacy):**
```
set rates_type = BETAFLIGHT
set roll_rc_rate = 80
set pitch_rc_rate = 80
set yaw_rc_rate = 80
set roll_srate = 70
set pitch_srate = 70
set yaw_srate = 70
set roll_expo = 0
set pitch_expo = 0
set yaw_expo = 0
save
```

**Actual:**
```
set rates_type = ACTUAL
set roll_rc_rate = 16
set pitch_rc_rate = 16
set yaw_rc_rate = 16
set roll_srate = 53
set pitch_srate = 53
set yaw_srate = 53
set roll_expo = 54
set pitch_expo = 54
set yaw_expo = 54
save
```

---

## Legacy ir Actual aprašo tą pačią kreivę

Abi rate sistemos saugo į *tuos pačius* CLI laukus (`rc_rate`, `srate`, `expo`) — `rates_type` tik keičia, kaip tie skaičiai interpretuojami:

| Laukas      | Legacy Betaflight            | Actual                          |
|-------------|------------------------------|---------------------------------|
| `rc_rate`   | RC Rate × 100 (daugiklis)    | Centro jautrumas ÷ 10 (°/s)     |
| `srate`     | Super Rate × 100             | Maks. rate ÷ 10 (°/s)           |
| `expo`      | Expo × 100                   | Expo × 100 (perstumia lūžį)     |

Kadangi pamatinė matematika skiriasi, **expo skaičiai nesutampa** tarp sistemų net ir identiškai kreivei — todėl 733 `expo=0` legacy sistemoje tampa `expo=55` Actual'e (Actual expo atkuria formą, kurią legacy modelyje sukuria Super Rate). Pačios kreivės persidengia:

```chart
{
  "type": "line",
  "data": {
    "labels": ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
    "datasets": [
      {
        "label": "733 — Betaflight (RC 1.10, SR 0.70, E 0)",
        "data": [0, 23.7, 51.2, 83.5, 122.2, 169.2, 227.6, 302.0, 400.0, 535.1, 733.3],
        "borderColor": "rgba(249,115,22,1)",
        "backgroundColor": "transparent",
        "borderWidth": 3, "tension": 0.3, "pointRadius": 3
      },
      {
        "label": "733 — Actual (center 220, max 730, expo 55)",
        "data": [0, 24.3, 53.2, 86.9, 125.9, 171.8, 227.7, 299.5, 396.4, 533.0, 730.0],
        "borderColor": "rgba(99,102,241,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2, "borderDash": [6,3], "tension": 0.3, "pointRadius": 2
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": { "display": true, "text": "Same feel, two rate systems — 733 legacy vs Actual" },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": { "title": { "display": true, "text": "Stick deflection" } },
      "y": { "beginAtZero": true, "max": 760, "title": { "display": true, "text": "Rotation rate (°/s)" } }
    }
  }
}
```

---

## Pastabos

- **Expo** nėra trumpinio dalis. Pridėk `expo` (0.10–0.20 legacy, ~30–50 Actual'e), kad suminkštintum centrą smulkesnei hover/linijų kontrolei — jis nekeičia maks. rate.
- **Yaw** parodytas lygus roll/pitch švariam kopijavimui. Daug pilotų nuleidžia yaw šiek tiek, kad spin'ai būtų švaresni — sumažink `yaw_*` reikšmes pagal skonį.
- Įklijavęs atsidaryk **Rates** skirtuką Betaflight Configurator'e ir prieš skrisdamas patikrink, ar gyva kreivė ir maks. rate rodmuo atitinka aukščiau esančią lentelę (klausk manęs, iš kur žinau, kad verta pasitikrinti prieš, o ne po pirmo flip'o).
- `rates_type` perjungimas neištrina kitos sistemos reikšmių — Betaflight laiko jas kiekvienam profiliui, tad gali laksčioti tarp BETAFLIGHT ir ACTUAL palyginimui.

---

## Susiję

- [Betaflight Rates paaiškinti](../rates/) — ką daro RC Rate, Super Rate ir Expo
- [Rate Modes — formulės ir konvertavimas](../rate-modes/) — visų penkių rate sistemų matematika
- [Rates Deep Dive](../rates-deep-dive/) — centro/vidurio/krašto zonos ir throttle expo
- [FPV Terminology](../../reference/fpv-terminology/) — žodynėlis, apimantis rates, expo, Acro
