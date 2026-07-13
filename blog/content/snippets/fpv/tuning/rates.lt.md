---
title: "Betaflight Rates paaiškinti"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "tuning", "freestyle"]
---

Rates nusako, kaip greitai dronas sukasi reaguodamas į stick'ų judesį. Didesni rates = greitesnis sukimasis = agresyvesnis pojūtis. Rate profilį formuoja trys parametrai — **RC Rate**, **Super Rate** ir **Expo** — plius pasirinkta rate *sistema* (Betaflight, Actual ir t. t.). Beje, nesikankink versdamas viską mintinai — aš pats tiesiog atsidarau Rates Preview skirtuką ir stumdau slankiklius, kol pojūtis pataiko.

---

## Populiarūs trumpiniai profiliams

Trumpinys `533 / 633 / 733` įvardija profilį pagal jo **maksimalų sukimosi greitį °/s** — `733` pasiekia beveik 730 °/s, `533` — apie 530 °/s:

| Profilis  | Maks. rate | Pojūtis          | Kam tinka                             |
|-----------|-----------|------------------|---------------------------------------|
| **733**   | ~733 °/s  | Punchy           | Freestyle, flip'ai, užtikrinti pilotai |
| **633**   | ~633 °/s  | Subalansuotas    | Bendras freestyle / kruizavimas       |
| **533**   | ~533 °/s  | Ramus            | Acro mokymasis, racing linijos, cinematic |

Visi trys naudoja fiksuotą **Super Rate 0.70**, o RC Rate keliamas laipsniškai (0.80 / 0.95 / 1.10). Tikslias reikšmes abiejose rate sistemose, su kopijuojamu CLI ir grafikais, rasi čia: [Rate Presets](../rate-presets/).

---

## Ką daro kiekvienas parametras

**RC Rate** — bazinis jautrumas ties stick'o centru. Didesnės reikšmės daro visą stick'ą jautresnį.

**Super Rate** — prideda papildomo sukimosi greičio artėjant prie stick'o kraštų. Tai tas „viršutinis“ greitis, kurį pasieki visiškai nuspaudęs. Diapazonas: 0.0 – 1.0 (0 = tiesinis, didesnis = daugiau kreivės link kraštų).

**Expo** — suminkština stick'o centrinę zoną, suteikdamas smulkesnę kontrolę ties hover/centru, o pilnas nuspaudimas išlieka greitas. Diapazonas: 0.0 – 1.0.

---

## Maksimalaus sukimosi greičio formulė

Numatytajam Betaflight rate stiliui:

```
maxRate = (RC Rate × 200) × (1 / (1 - Super Rate))
```

Pavyzdys — profilis **733** (RC Rate 1.1, Super Rate 0.7):
```
maxRate = (1.1 × 200) × (1 / (1 - 0.7))
        = 220 × 3.33
        ≈ 733 °/s
```

Štai kodėl trumpinio skaičius *ir yra* maksimalus rate: kai Super Rate fiksuotas ties 0.70, vardiklis lygus 0.30, tad `maxRate = 666.7 × RC Rate`.

> **Expo nekeičia maksimumo.** Expo tik suminkština kreivę tarp centro ir pilno stick'o; ties pilnu nuspaudimu išvestis grįžta į tą patį `maxRate`. Kad skristum greičiau, kelk RC Rate arba Super Rate, o ne Expo. Naudok Rates Preview skirtuką Betaflight Configurator'e, kad pamatytum realią kreivę bet kokioms reikšmėms.

---

## Rate stiliai

Betaflight palaiko kelis rate stilius — kiekvienas naudoja tuos pačius tris slankiklius, tik pritaiko juos skirtingai:

| Stilius      | Charakteristika                                    |
|--------------|----------------------------------------------------|
| Betaflight   | Numatytasis. Nuspėjamas, plačiai naudojamas.       |
| Actual       | RC Rate tiesiogiai atitinka maks. °/s. Intuityvus. |
| Quickrates   | Supaprastinta dviejų parametrų kreivė.             |
| Kiss         | Klasikinis Kiss FC stilius.                        |

**Actual Rates** vis populiaresni, nes RC Rate tiesiogiai lygus maksimaliam sukimosi greičiui °/s — jokios matematikos nereikia.

---

## Praktiniai patarimai

- Mokydamasis pradėk nuo **533**; žemesnis maks. rate laiko sukimąsi lėtą ir kontroliuojamą.
- Pereik prie **633** ar **733**, kai nori greitesnių flip'ų ir gyvesnio centro.
- Pridėk truputį **Expo** (0.10–0.20), jei centras jaučiasi trūkčiojantis — jis suminkština smulkią kontrolę nekeisdamas maks. rate. Virš ~0.5 pradeda jaustis „lagas“ ties centru.
- Rates nustatomi kiekvienai ašiai atskirai. Dauguma pilotų kopijuoja tą patį profilį Roll, Pitch ir Yaw — bet yaw dažnai nustatomas šiek tiek žemesnis, kad spin'ai būtų švaresni (aš yaw laikau ramesnį — greitesnis man tik verčia galvą sukti :)).

---

## CLI

```
# View current rates
rates

# Set a 733 profile (legacy Betaflight style) on Roll
set roll_rc_rate = 110
set roll_srate = 70
set roll_expo = 0

# Copy same values to Pitch
set pitch_rc_rate = 110
set pitch_srate = 70
set pitch_expo = 0

# Lower yaw slightly for cleaner spins
set yaw_rc_rate = 95
set yaw_srate = 70
set yaw_expo = 0

save
```
