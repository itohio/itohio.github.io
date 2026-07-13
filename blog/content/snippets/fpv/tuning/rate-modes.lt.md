---
title: "Betaflight Rate režimai — formulės, palyginimas ir konvertavimas"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "actual-rates", "kiss-rates", "quickrates", "rc-rate", "expo", "tuning"]
---

Betaflight palaiko keturias rate režimų formules. Visos daro tą patį darbą — susieja stick'o nuspaudimą (0 iki ±1) su komanduojamu sukimosi greičiu (°/s) — bet kreivę parametrizuoja skirtingai. Režimo perjungimas keičia tai, ką reiškia slankikliai, o ne tai, kaip dronas jaučiasi, jei nustatysi ekvivalentiškus parametrus. Na, o jei tave nuo mažens gąsdina matematika — ramiai, apačioje yra grafikas, kuris viską pasako be formulių.

---

## Apžvalga

| Režimas | CLI reikšmė | Parametrai | Kam geriausia |
|------|-----------|-----------|---------|
| **Betaflight** (legacy) | `BETAFLIGHT` | RC Rate, Super Rate, Expo | Numatytasis; dauguma bendruomenės preset'ų naudoja šitą |
| **Actual** | `ACTUAL` | Center Sensitivity, Max Rate, Expo | Tiesioginė konkrečių pojūčio metrikų kontrolė |
| **KISS** | `KISS` | Rate | Vieno „ratuko“ paprastumas |
| **Quickrates** | `QUICK` | Max Rate | Arčiausiai tiesinio; švariausi duomenų skrydžiai |

**Režimo perjungimas per CLI:**
```
set rates_type = ACTUAL    # or BETAFLIGHT, RACEFLIGHT, KISS, QUICK
save
```

Betaflight įsimena paskutines naudotas reikšmes kiekvienam režimui. Režimo perjungimas neperrašo kito režimo išsaugotų reikšmių.

---

## Betaflight (Legacy) rates

**Parametrai:** RC Rate, Super Rate, Expo

**Formulė:**
```
stick = |rcCommand|       // 0 to 1

// Expo pre-shaping (reduces center sensitivity):
if expo > 0:
    stick = stick × (expo × |stick|³ + (1 − expo))

// Rate with super rate denominator (hockey-stick at extremes):
output = sign(rcCommand) × stick × rcRate × 200 / (1 − superRate × stick)
```

Būtent vardiklis `(1 − superRate × stick)` sukuria tą būdingą „hockey-stick“ formą — ties centru (stick≈0) poveikis minimalus, bet artėjant prie pilno nuspaudimo (stick→1) vardiklis artėja prie `(1 − superRate)` ir smarkiai sustiprina išvestį.

| Parametras | Tipiškas diapazonas | Poveikis |
|-----------|--------------|--------|
| RC Rate | 0.8–1.8 | Skaliuoja visą kreivę; 1.0 ≈ 200 °/s ties centru kai SR=0 |
| Super Rate | 0.50–0.75 | Nusako pagreitį ties kraštais; didesnis = daugiau hockey-stick |
| Expo | 0–0.5 | Suminkština centro pojūtį; dauguma pilotų naudoja 0 arba žemas reikšmes |

**Apribojimas:** RC Rate ir Super Rate sąveikauja — keisdamas Super Rate keiti ir centro pojūtį, ir atvirkščiai. Nėra vieno parametro, kuris valdytų tik centro jautrumą, ir kito, kuris valdytų tik maks. rate.

---

## Actual Rates

**Parametrai:** Center Sensitivity, Max Rate, Expo

**Formulė (expo = 0):**
```
output = sign(rcCommand) × ((max_rate − center) × |rcCommand|² + center × |rcCommand|)
```

Tai vienu metu tenkina dvi nepriklausomas sąlygas:
- Nuolydis ties stick = 0: lygus `center` (centro jautrumas, °/s vienetui)
- Reikšmė ties stick = 1: lygi `max_rate`

Kai expo > 0, vidurinė kreivės dalis palinksta link sinuso formos kreivės, sukurdama švelnesnį vidurio diapazono perėjimą.

**Privalumas:** Kiekvienas parametras valdo tiksliai vieną dalyką. Gali keisti centro pojūtį neliesdamas maks. rate, ir atvirkščiai. Todėl Actual rates lengviau derinti iki konkretaus pojūčio.

| Parametras | Poveikis |
|-----------|--------|
| Center Sensitivity | Laipsniai/s vienam stick'o vienetui ties centru. 70–120 įprasta. |
| Max Rate | Kietoji lubos ties pilnu nuspaudimu. 600–900 °/s tipiška freestyle. |
| Expo | 0 = gryna kvadratinė kreivė; didesnis = švelnesnis vidurio perėjimas |

---

## KISS Rates

**Parametrai:** Rate (vienas slankiklis)

**Formulė:**
```
output = sign(rcCommand) × kissRate × 1998 × |rcCommand| / (1 − kissRate × |rcCommand|)
```

Struktūriškai identiška BF legacy formulei, tik su vienu jungtiniu rate parametru (be atskiro RC Rate / Super Rate skaidymo). Paprasčiau suprasti: vienas skaičius valdo bendrą agresyvumą.

- **kissRate = 0.26** → ~700 °/s maks., saikingas centro pojūtis
- **kissRate = 0.32** → ~930 °/s maks., agresyvesnis
- **kissRate < 0.20** → maks. žemiau 400 °/s, pradedančiojo/cinematic pojūtis

KISS rates pavadinti pagal KISS ESC/FC ekosistemą, kur ši formulė ir atsirado. Betaflight realizacija artimai atitinka originalą.

---

## Quickrates

**Parametrai:** Max Rate (vienas slankiklis)

**Formulė:**
```
output = sign(rcCommand) × maxRate × |rcCommand|
```

Idealiai tiesinis. Vienas parametras, viena kreivė. Quickrates yra arčiausiai, kiek Betaflight gali priartėti prie tiesioginio „stick'o pozicija → sukimosi greitis“ susiejimo be jokio formavimo.

**Kada naudoti:**
- Kinematografijai ir slow-motion buildams, kur nori nuspėjamos, tiesinės kontrolės
- Derinimo duomenų rinkimo skrydžiams, kur nori, kad setpoint signalas būtų švarus ir tiesinis (step response analizė tampa kiek švaresnė)
- Naujiems pilotams, besimokantiems acro — tiesinis pojūtis iš pradžių lengviau suprantamas

---

## Rate palyginimo grafikas

Tipiniai nustatymai, suderinti maždaug panašiam pojūčiui:

```chart
{
  "type": "line",
  "data": {
    "labels": ["0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
    "datasets": [
      {
        "label": "Betaflight (rcRate=1.2, SR=0.70) — max 800°/s",
        "data": [0,26,56,91,133,185,248,329,436,584,800],
        "borderColor": "rgba(239,68,68,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "tension": 0.2,
        "pointRadius": 2
      },
      {
        "label": "Actual (Center=100, Max=700, Expo=0)",
        "data": [0,16,44,84,136,200,276,364,464,576,700],
        "borderColor": "rgba(34,197,94,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "tension": 0.2,
        "pointRadius": 2
      },
      {
        "label": "KISS (rate=0.26) — max 702°/s",
        "data": [0,53,110,169,232,299,369,445,525,610,702],
        "borderColor": "rgba(99,102,241,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2,
        "tension": 0.2,
        "pointRadius": 2
      },
      {
        "label": "Quickrates (max=700) — linear",
        "data": [0,70,140,210,280,350,420,490,560,630,700],
        "borderColor": "rgba(249,115,22,1)",
        "backgroundColor": "transparent",
        "borderWidth": 2,
        "borderDash": [5,3],
        "tension": 0,
        "pointRadius": 2
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": { "display": true, "text": "Rate mode comparison — commanded °/s vs stick deflection (0 to 1)" },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": { "title": { "display": true, "text": "Stick deflection (0 = center, 1 = full throw)" } },
      "y": {
        "min": 0,
        "max": 850,
        "title": { "display": true, "text": "Commanded rotation rate (°/s)" }
      }
    }
  }
}
```

**Pagrindiniai pastebėjimai:**
- **BF (raudona)**: prasideda švelniai (žemas centro jautrumas su šiais nustatymais), bet stipriai pagreitėja ties kraštais — hockey stick matoma virš 0.7 stick'o
- **Actual (žalia)**: apibrėžta kvadratinė — švarus, nuspėjamas nuolydis nuo centro iki maks.
- **KISS (violetinė)**: forma panaši į BF, bet parametrizuota vienu skaičiumi; vidurio stick'o pojūtis kiek tiesiškesnis nei BF
- **Quickrates (oranžinė punktyrinė)**: tikrai tiesinis — vienodi tarpai tarp visų taškų

---

## BF ↔ Actual konvertavimas

Abi kreivės skirtingų formų, tad tobulo konvertavimo nėra. Šios formulės duoda artimą atspirties tašką:

### BF → Actual

```
Actual.center    ≈ BF.rcRate × 200
Actual.max_rate  = read from BF configurator (full-stick output value)
Actual.expo      ≈ BF.superRate / 2.0   (rough starting point — tune by feel)
```

**Pavyzdys:** BF rcRate=1.5, superRate=0.70, expo=0
- `Actual.center = 1.5 × 200 = 300`
- Nuskaitai konfigūratoriuje: maks. išvestis ≈ 900°/s → `Actual.max_rate = 900`
- `Actual.expo = 0.70 / 2.0 = 0.35`

### Actual → BF

```
BF.rcRate      ≈ Actual.center / 200
BF.superRate   ≈ Actual.expo × 2.0   (starting point)
```

Suskaičiavęs atsidaryk konfigūratorių ir vizualiai palygink abi kreives Rate grafike. Koreguok `superRate`, kol pilno stick'o išvestis atitiks norimą maks. rate. Kreivės nebus identiškos — rinkis tą, kuri atitinka norimą pojūtį ties vidurio stick'u.

---

## Ekvivalentiškas pojūtis — pavyzdiniai preset'ai

| Pojūtis | Betaflight | Actual | KISS |
|------|-----------|--------|------|
| Freestyle vidutinis | RC=1.4, SR=0.70, E=0 | Center=100, Max=750, E=0.30 | Rate=0.28 |
| Racing / aštrus | RC=1.8, SR=0.75, E=0 | Center=140, Max=900, E=0.25 | Rate=0.32 |
| Sklandus / cinematic | RC=0.8, SR=0.40, E=0.20 | Center=50, Max=350, E=0.40 | Rate=0.15 |
| 2" ripper | RC=1.2, SR=0.65, E=0 | Center=90, Max=650, E=0.30 | Rate=0.24 |

Tai atspirties taškai — koreguok pagal skonį (raktinis žodis čia — *tavo* skonį, ne mano). Konfigūratoriaus Rate grafikas atsinaujina gyvai, tad palygink vizualiai, o ne pasikliauk skaičiais tiksliai.

---

## Kurį režimą man naudoti?

**Naudok Betaflight (legacy)**, jei kelí bendruomenės preset'us ar seki tune gidą, kuris nurodo RC Rate ir Super Rate reikšmes. Didžioji dauguma paskelbtų preset'ų remiasi šiuo režimu.

**Naudok Actual**, jei nori nepriklausomai valdyti „koks aštrus centras“ ir „kaip greitai eina pilnas stick'as“. Ypač naudinga perkeliant ankstesnio buildo pojūtį į naują su kitokiais varikliais/propeleriais — gali suderinti centro jautrumą ir maks. rate nepriklausomai.

**Naudok KISS**, jei mėgsti vieną slankiklį, skraidai vien iš raumenų atminties ir nenori galvoti apie RC Rate ir Super Rate sąveiką.

**Naudok Quickrates** kinematografijai arba derinimo duomenų rinkimo skrydžiams (tiesinis setpoint = švariausi step response duomenys).

> Visi keturi režimai gali sukurti identišką realų skraidymo pojūtį, jei tinkamai parametrizuoti. Režimo pasirinkimas — tai klausimas, kaip tau patogiau galvoti apie kreivę ir ją koreguoti, o ne kuris „geresnis“.

---

## Susiję

- [Betaflight Tuning Math](../betaflight-tuning-math/) — kas nutinka rates PID kilpos viduje
- [Wobble-Test PID Protocol](../pid-tuning-wobble-test/) — derinimo eiga, kuri remiasi šiais rate nustatymais
- [Tuning Flight Protocol](../tuning-flight-protocol/) — kodėl Quickrates ar FF=0 svarbu švariems BBL duomenims
- [FPV Terminology](../../reference/fpv-terminology/) — žodynėlis, apimantis Acro režimą, rates, PID
- **Rylo** — rate setup pagalba ir PID derinimo patarimai → [app.sintra.ai/community/helpers/rylo](https://app.sintra.ai/community/helpers/rylo)
