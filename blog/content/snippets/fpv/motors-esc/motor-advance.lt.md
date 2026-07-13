---
title: "Motor Timing Advance — 15° vs 22°"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "esc", "motor", "timing", "advance", "efficiency", "blheli"]
---

Motor timing advance pastumia ESC komutacijos tašką pirmyn elektriniame cikle motoro rotoriaus pozicijos atžvilgiu. Aukštesnis advance = daugiau galios efektyvumo ir karščio sąskaita. Na, o jei degimo timing degimo variklyje kada nors tau atrodė juoda magija — čia lygiai tas pats, tik be alyvos kvapo.

---

## Ką timing advance daro

Brushless motoro ESC perjungia fazių galią pagal rotoriaus poziciją. „On time“ — perjungimas tiksliai tada, kai rotorius susilygiuoja — yra 0° advance. Perjungimo taško pastūmimas anticipuoja rotoriaus judėjimą, didindamas torque power band'e, bet už tai kaista labiau.

Įsivaizduok tai kaip uždegimo timing vidaus degimo variklyje — pastūmi per toli ir gauni knock bei karštį; nustatai teisingai ir gauni piko galią; atitrauki ir prarandi galią, bet suki vėsiai.

---

## 15° vs 22° (dažni ESC nustatymai)

| Nustatymas | Efektyvumas | Galia  | Karštis | Naudojimo atvejis              |
|---------|------------|--------|-------|--------------------------------|
| **15°** | Aukštesnis | Mažesnė | Žemesnis | Efektyvumas, long range, cruising |
| **22°** | Vidutinis  | Didesnė | Aukštesnis | Freestyle, racing, punchy build'ai |

**15° advance** yra termiškai efektyvesnis — ESC perjungia taške, kuris minimizuoja geležies nuostolius motoro stator. Mažiau karščio reiškia mažiau iššvaistytos energijos, o tai tiesiogiai virsta ilgesniais skrydžio laikais.

**22°** (ir aukščiau, iki ~30°) išspaudžia daugiau galios ir RPM iš tos pačios motoro/prop kombinacijos, bet didina motoro ir ESC temperatūrą, ypač prie ilgai laikomo aukšto throttle.

---

## Kada naudoti 15°

- Long-range build'ai, kur svarbus skrydžio laikas
- Efektyvumui orientuoti 5" cruising setup'ai
- Motorai su griežtomis tolerancijomis (per dideli stator'iai), kurie kaista ant 22°
- Bet koks build'as, kur motoro temperatūra nusileidus atrodo aukšta

**Nykščio taisyklė:** jei tavo motorai šilti liesti (gali laikyti pirštą > 3 sekundes) po full-throttle punch sesijos, žemesnis timing advance yra viena svirtis karščiui sumažinti prieš griebiant didesnį ESC.

---

## Kada naudoti 22°

- Freestyle ir racing, kur piko galia svarbesnė už efektyvumą
- Build'ai su pakankamu aušinimu (atviri rėmai, didesni ESC)
- Kai išbandei 15° ir piko throttle jausmas pastebimai vangus

---

## Timing advance nustatymas

BLHeli_32 / AM32 configurator'yje:

| ESC Firmware | Nustatymo pavadinimas | Vertės                |
|--------------|--------------------|-----------------------|
| BLHeli_32    | Motor Timing       | Low / MedLow / Med / MedHigh / High |
| AM32         | Motor Timing (deg) | Tiesioginė laipsnių įvestis |

BLHeli_32 „Motor Timing“ apytikslis mapping:
- Low ≈ 15°
- MedLow ≈ 18°
- Medium ≈ 22°
- MedHigh ≈ 25°
- High ≈ 30°

Efektyvumui: rink **Low**. Freestyle: **Medium**.

---

## Sąveika su RPM ir demag

BLHeli_32 ir AM32 taip pat turi **Demag Compensation** nustatymą. Demag tvarko back-EMF šuolį, kai fazė išsijungia. Aukštesnė demag kompensacija gali padėti išvengti desync prie staigių RPM pokyčių.

Jei suki 15° timing dėl efektyvumo, poruok jį su normaliais ar vidutiniais demag nustatymais. Agresyvus demag su žemu timing kartais gali sukelti dvejonę prie greito throttle-up.

---

## Pastabos

- Timing advance sąveikauja su motoro KV ir prop apkrova. High-KV motoras ant mažo prop'o, dažnai pasiekiantis piko RPM, gauna mažiau naudos iš advance timing nei mid-KV motoras ant didesnio prop'o.
- Visada paleisk full-throttle punch sesiją ir patikrink motorų temperatūras pakeitęs timing. Pagauk šilumines problemas, kol jos neiškepė motoro apvijų — pirštas ant variklio nusileidus kainuoja nieko, naujas motoras kainuoja gerokai daugiau.
