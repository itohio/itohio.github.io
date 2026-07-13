---
title: "FPV vaizdo sistemos — analog vs digital"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "video", "analog", "digital", "dji", "walksnail", "hdzero", "vtx", "latency"]
---

Vaizdo sistemos pasirinkimas nulemia vaizdo kokybę, latency, atstumą ir kainą. Universalaus geriausio nėra — kiekviena sistema turi savo vietą (ir kiekvienas pilotas turi nuomonę, kurią gina kaip futbolo komandą).

---

## Analog

**Kaip veikia:** kamera išveda composite vaizdą; VTX (vaizdo siųstuvas) transliuoja jį 5,8 GHz dažniu. Imtuvas dekoduoja analog RF tiesiogiai.

**Privalumai:**
- Mažiausias latency (sub-ms RF link'as; ~3–7 ms glass-to-glass)
- Trukdžius atlaiko gražiai — vaizdas blogėja palaipsniui prieš dingdamas („snow“ prieš drop'ą)
- Pigiausi komponentai
- Plačiausias dažnių ir kanalų pasirinkimas

**Trūkumai:**
- Prasta vaizdo kokybė (720×480 NTSC arba 720×576 PAL)
- Nėra įrašyto HD vaizdo be atskiros HD kameros
- Dažnių grūstis, kai skraidote grupėje

**Tipiniai komponentai:** Foxeer/Runcam kamera, RushFPV/Hglrc/Tramp VTX, Furious FPV/ImmersionRC VRx, Fatshark/Skyzone goggle'ai su analog moduliu.

---

## DJI O3 / O4 (digital HD)

**Kaip veikia:** DJI nuosavas digital link'as; air unit'as užsiima enkodavimu, goggle'ai dekoduoja.

**Privalumai:**
- HD vaizdas (1080p/60fps įrašas, ~810p live feed'as)
- Integruotas DVR ir OSD
- Švarus, lag'ą atlaikantis link'as švariose RF aplinkose
- Puikus atstumas prie 700 mW

**Trūkumai:**
- Didesnis latency (~22–28 ms vs analog ~3–7 ms glass-to-glass — daugumai nepastebima)
- Brangus ekosistemos lock-in
- Sunkesnis air unit'as
- Link'as blogėja kitaip nei analog — „digital wall“: staigus nutrūkimas vietoj palaipsnio „snow“

**Latency:** ~22 ms Normal režime, ~28 ms High Quality režime. Priimtina freestyle ir kinematografijai; racing'o konkurentai kartais renkasi analog.

---

## Walksnail Avatar (digital HD)

**Kaip veikia:** Betaflight suderinama Walksnail digital sistema; architektūra panaši į DJI.

**Privalumai:**
- Gera HD vaizdo kokybė
- Atviresnė ekosistema nei DJI
- Palaikomi custom goggle'ų displėjai

**Trūkumai:**
- Šiek tiek prastesnė vaizdo kokybė nei DJI O3
- Mažesnė ekosistema / mažiau priedų

---

## HDZero (digital HD)

**Kaip veikia:** MIPI paremtas digital vaizdas su fokusu į ultra žemą latency.

**Privalumai:**
- Labai žemas digital latency (~8–10 ms)
- Atvira ekosistema, keli suderinami goggle'ai
- Gera vaizdo kokybė

**Trūkumai:**
- Trumpesnis atstumas nei DJI O3 prie tos pačios galios
- Mažesnė bendruomenė

**Geriausias panaudojimas:** racing'o pilotai, kuriems reikia digital kokybės, bet netoleruoja DJI O3 latency.

---

## Palyginimo santrauka

| Sistema        | Latency     | Vaizdo kokybė | Atstumas | Kaina     | Lock-in   |
|----------------|-------------|---------------|----------|-----------|-----------|
| Analog         | ~3–7 ms     | Prasta        | Didelis  | Maža      | Nėra      |
| DJI O3         | 22–28 ms    | Puiki         | Labai didelis| Didelė | Didelis   |
| Walksnail Avatar| 15–25 ms   | Gera          | Didelis  | Vidutinė  | Vidutinis |
| HDZero         | 8–10 ms     | Gera          | Vidutinis| Vidutinė  | Mažas     |

---

## Dažnių kanalai (analog)

Standartinis 5,8 GHz analog vaizdas veikia per 40 kanalų 8 juostose (A/B/E/F/R/L/ir t. t.). Dažniausi:

- **Raceband (R)** — sukurta minimaliems trukdžiams multi-pilot sesijose
- **Fatshark juosta (F)** — dažna mėgėjiškam skraidymui
- **1–8 kanalas bet kurioje juostoje** — susiderink su kitais pilotais prieš skrydį

Visada paskelbk savo kanalą prieš įjungdamas VTX šalia kitų.

---

## OSD integracija

Visos digital sistemos (DJI O3, Walksnail, HDZero) palaiko MSP OSD — Betaflight siunčia OSD duomenis į air unit'ą, kuris juos atvaizduoja ant vaizdo feed'o. Konfigūruok Betaflight'e lygiai kaip analog atveju, tik nustatyk display port'ą:

```
set osd_displayport_device = MSP
```

Žiūrėk [OSD nustatymas](../osd-setup/) dėl elementų parinkimo.
