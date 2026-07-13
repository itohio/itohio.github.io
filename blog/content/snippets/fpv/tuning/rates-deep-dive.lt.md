---
title: "Rates Deep Dive — Expo kreivė, zonos ir throttle"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "expo", "throttle", "tuning", "freestyle", "cinematic"]
---

Rates — tai ne tik „kaip greitai sukasi“ nustatymas. Jie nusako, *kurioje stick'o vietoje* gyvena kontrolė — o kreivės supratimas leidžia derinti tiksliam hover'iui, natūraliam kruizavimui ir sprogstamiems triukams, viską tame pačiame profilyje. Man tai buvo tas momentas, kai rates nustojo būti „vienas skaičius“ ir tapo kreive, kurią pagaliau supratau.

---

## Rate kreivė

Gimbal stick'o judinimas nuo centro iki pilno nuspaudimo nesukelia tiesinio sukimosi greičio augimo. Rate kreivė nusako tą sąryšį. Štai trys populiarūs profiliai nubraižyti:

```chart
{
  "type": "line",
  "data": {
    "labels": ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
    "datasets": [
      {
        "label": "Stability Zone (0–20%)",
        "data": [760, 760, 760, null, null, null, null, null, null, null, null],
        "backgroundColor": "rgba(34, 197, 94, 0.13)",
        "borderColor": "transparent",
        "borderWidth": 0,
        "pointRadius": 0,
        "fill": "origin",
        "order": 10
      },
      {
        "label": "Normal Flying (20–70%)",
        "data": [null, null, 760, 760, 760, 760, 760, 760, null, null, null],
        "backgroundColor": "rgba(59, 130, 246, 0.10)",
        "borderColor": "transparent",
        "borderWidth": 0,
        "pointRadius": 0,
        "fill": "origin",
        "order": 10
      },
      {
        "label": "Tricks & Flicks (70–100%)",
        "data": [null, null, null, null, null, null, null, 760, 760, 760, 760],
        "backgroundColor": "rgba(249, 115, 22, 0.13)",
        "borderColor": "transparent",
        "borderWidth": 0,
        "pointRadius": 0,
        "fill": "origin",
        "order": 10
      },
      {
        "label": "533 — Mellow / Learning (max 533°/s)",
        "data": [0, 17.2, 37.2, 60.8, 88.9, 123.1, 165.5, 219.6, 290.9, 389.2, 533.3],
        "borderColor": "rgba(34, 197, 94, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.35,
        "fill": false,
        "order": 1
      },
      {
        "label": "633 — All-round Freestyle (max 633°/s)",
        "data": [0, 20.4, 44.2, 72.2, 105.6, 146.2, 196.6, 260.8, 345.5, 462.2, 633.3],
        "borderColor": "rgba(59, 130, 246, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.35,
        "fill": false,
        "order": 1
      },
      {
        "label": "733 — Punchy Freestyle (max 733°/s)",
        "data": [0, 23.7, 51.2, 83.5, 122.2, 169.2, 227.6, 302.0, 400.0, 535.1, 733.3],
        "borderColor": "rgba(249, 115, 22, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.35,
        "fill": false,
        "order": 1
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": {
        "display": true,
        "text": "Rotation Rate vs Stick Position — 533 / 633 / 733 (Super Rate 0.70, Expo 0)"
      },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": {
        "title": { "display": true, "text": "Stick deflection" }
      },
      "y": {
        "beginAtZero": true,
        "max": 760,
        "title": { "display": true, "text": "Rotation rate (°/s)" }
      }
    }
  }
}
```

---

## Trys zonos

### 1 zona — Stabilumas (0–20% stick'o, žalia juosta)

Pati stick'o eigos pradžia yra plokščiausia kreivės dalis. Super Rate 0.70 palieka statų atsaką stick'o kraštams, tad centras lieka švelnus ir atlaidus (šiek tiek Expo suplokština jį dar labiau).

Ties 5% stick'o nuspaudimu 533 profilyje sukimosi greitis yra vos **8.3 °/s** — apie **1.6 °/s vienam 1% stick'o judesio**. 733 profilyje šioje zonoje tai **2.2 °/s vienam 1%**. Šiaip ar taip, maži netyčiniai stick'o judesiai sukelia mažai sukimosi: rankos virpėjimas, vėjo blaškymas grąžinant droną prie stick'ų ar ne visai atsipalaidavusi ranka — didžia dalimi sugeriami čia.

**Tai tavo hover zona.** Vietoje kabantis dronas yra kažkur šiame diapazone. Kuo mažesnis jautrumas čia, tuo lengviau laikyti stabilią poziciją ar daryti lėtus, cinematic panoraminius kadrus. Pradedantieji laimi labiausiai — kiekvienas treniruotės skrydis praleidžiamas šioje zonoje.

### 2 zona — Normalus skraidymas (20–70% stick'o, mėlyna juosta)

Vidurinė kreivės dalis — kur praleidi daugumą skrydžio laiko. Šis regionas turėtų jaustis **maždaug tiesinis** — nuoseklus atsakas į stick'o judesį, kad dronas skristų ten, kur rodai, be staigmenų.

Ties 50% stick'u 533 profilyje rate yra 123 °/s; ties 50% 733 profilyje — 169 °/s. Skirtumas tarp profilių čia tampa labiau juntamas — 533 jaučiasi švelnus ir sklandus, 733 — punchy ir gyvas.

**Kodėl „maždaug tiesinis“ svarbu:** jei vidurinėje dalyje per daug kreivumo, perėjimai iš hover'io į vidutinio greičio skrydį jaučiasi trūkčiojantys — stick'as staiga tampa daug jautresnis, kai išeini iš centro zonos. Geras tune'as išlaiko perėjimą pakankamai sklandų, kad jo nepastebėtum. Padeda saikinga Expo (0.10–0.20); Expo virš ~0.5 pradeda kurti pastebimą „negyvas centras → staiga gyvas“ pojūtį.

### 3 zona — Triukai ir flick'ai (70–100% stick'o, oranžinė juosta)

Pilnas stick'o nuspaudimas — kur Super Rate įsijungia stipriausiai. Kreivė pastebimai statėja — 733 profilyje paskutiniai 5% stick'o eigos (95%→100%) prideda **~22 °/s vienam 1%**, palyginti su **~2.2 °/s vienam 1%** ties centru — maždaug 10× jautrumo pokytis nuo centro iki krašto.

Ši zona suteikia „snap'ą“ split-S manevrams, power loop'ams ir flip'ams. Skrydžio metu čia praleidi nedaug laiko — tai momentiniai pilno nuspaudimo judesiai. Aukštas rate šioje zonoje reiškia, kad triukai yra snappy ir greiti, o visas stick'as nesijaučia agresyvus.

**Praktinė išvada:** Super Rate sukoncentruoja papildomą greitį kraštuose, tad aukštas maks. rate neprivalo reikšti trūkčiojančio centro. Būtent todėl įgudęs pilotas gali skristi su 733 (ar aukštesniu) ir vis tiek sklandžiai nusileisti — tikslumas gyvena 1 zonoje, greitis — 3 zonoje.

---

## Jautrumo palyginimas ties kraštais

| Profilis | Centro jautrumas (°/s vienam 1% stick'o) | Pilno nuspaudimo jautrumas |
|---------|---------------------------------------|-----------------------------|
| 533     | 1.6 °/s / 1%                          | 15.9 °/s / 1%               |
| 633     | 1.9 °/s / 1%                          | 18.9 °/s / 1%               |
| 733     | 2.2 °/s / 1%                          | 21.9 °/s / 1%               |

733 piloto centro stick'as yra tik ~38% jautresnis nei 533 piloto, ir pilnas nuspaudimas ~38% greitesnis. Kadangi keičiasi tik RC Rate, visa kreivė skaliuojasi kartu — centro-krašto *santykis* (~10×) yra identiškas visuose trijuose profiliuose.

---

## Throttle Expo

Sukimosi greitis nėra vienintelė svarbi kreivė. Throttle kanalą — pagal nutylėjimą visiškai tiesinį — irgi galima formuoti. Betaflight tai vadina **Throttle Mid** ir **Throttle Expo**.

```chart
{
  "type": "line",
  "data": {
    "labels": ["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
    "datasets": [
      {
        "label": "Linear (no expo)",
        "data": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "borderColor": "rgba(156, 163, 175, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2,
        "borderDash": [6, 3],
        "pointRadius": 3,
        "tension": 0,
        "fill": false
      },
      {
        "label": "Expo 0.3 (mild curve)",
        "data": [0, 7.0, 14.2, 21.8, 29.9, 38.7, 48.5, 59.3, 71.4, 84.9, 100],
        "borderColor": "rgba(59, 130, 246, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.3,
        "fill": false
      },
      {
        "label": "Expo 0.5 (aggressive curve)",
        "data": [0, 5.1, 10.4, 16.3, 23.2, 31.2, 40.8, 52.1, 65.6, 81.5, 100],
        "borderColor": "rgba(249, 115, 22, 1)",
        "backgroundColor": "transparent",
        "borderWidth": 2.5,
        "pointRadius": 3,
        "tension": 0.3,
        "fill": false
      }
    ]
  },
  "options": {
    "responsive": true,
    "interaction": { "mode": "index", "intersect": false },
    "plugins": {
      "title": {
        "display": true,
        "text": "Throttle Output vs Stick Position — Effect of Expo"
      },
      "legend": { "position": "bottom" }
    },
    "scales": {
      "x": {
        "title": { "display": true, "text": "Throttle stick position" }
      },
      "y": {
        "beginAtZero": true,
        "max": 100,
        "title": { "display": true, "text": "Motor output (%)" }
      }
    }
  }
}
```

Su tiesiniu throttle judant nuo 40% iki 60% stick'o variklio išvestis pakyla 20%. Su Expo 0.5 tas pats judesys pakelia išvestį nuo 23% iki 41% — 18% pokytis paskirstytas per platesnį stick'o diapazoną, suteikiant smulkesnį žingsnį hover zonoje.

### Throttle Mid

**Throttle Mid** nustato stick'o poziciją, kuri duoda 50% variklio išvesties. Pagal nutylėjimą tai 50% (centrinis stick'as = pusė galios). Ši viena reikšmė turi didžiausią praktinį poveikį tam, kaip jaučiasi naujas build'as.

Dauguma freestyle dronų kabo kažkur tarp 30–50% throttle, priklausomai nuo AUW, propelerio dydžio ir baterijos įtampos. Jei tavo hover taškas yra ties 40% stick'o, Throttle Mid 0.5 reiškia, kad viršutinė throttle stick'o pusė padengia tik 10% diapazoną — 40% iki 50% iki 100% — todėl aukšto throttle manevrai tampa trūkčiojantys. Perstumk Throttle Mid link hover taško ir stick'o diapazonas išsiplečia simetriškai aplink jį.

---

## Baterijos išsikrovimas perstumia hover tašką

Pilnai įkrautas 4S pakas yra ~16.8 V. Baterijai išsikraunant įtampa krenta — ir ta pati variklio + propelerio kombinacija sukuria mažiau traukos vienam throttle vienetui. Kad išlaikytum aukštį, stumi stick'ą aukščiau.

```chart
{
  "type": "bar",
  "data": {
    "labels": ["Full pack (16.8V)", "75% capacity (15.8V)", "50% capacity (15.0V)", "20% capacity (14.2V)"],
    "datasets": [
      {
        "label": "Required hover throttle (%)",
        "data": [38, 43, 49, 57],
        "backgroundColor": [
          "rgba(34, 197, 94, 0.7)",
          "rgba(59, 130, 246, 0.7)",
          "rgba(249, 115, 22, 0.7)",
          "rgba(239, 68, 68, 0.7)"
        ],
        "borderColor": [
          "rgba(34, 197, 94, 1)",
          "rgba(59, 130, 246, 1)",
          "rgba(249, 115, 22, 1)",
          "rgba(239, 68, 68, 1)"
        ],
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "responsive": true,
    "plugins": {
      "title": {
        "display": true,
        "text": "Hover Throttle Shift vs Battery State — Typical 5\" 4S Build"
      },
      "legend": { "display": false }
    },
    "scales": {
      "y": {
        "beginAtZero": true,
        "max": 80,
        "title": { "display": true, "text": "Throttle stick position (%)" }
      }
    }
  }
}
```

5" freestyle drone hover taškas per visą paką pasislenka ~15–20 procentinių punktų. Tai turi dvi pasekmes:

1. **Throttle pojūtis keičiasi per skrydį.** Šviežiam pakui suderintas Throttle Mid jausis netinkamas ties senku paku ir atvirkščiai.
2. **Perėjimas iš hover'io į kilimą tampa staigesnis**, baterijai išsikraunant, nes hover taškas priartėja prie aukštos išvesties kreivės zonos.

Throttle Mid nustatyti į **vidutinį** hover tašką per visą paką (maždaug 46–48% tipiniam 5" 4S) yra praktinis kompromisas. Saikinga expo (0.3) aplink jį suteikia pakankamai centro minkštumo, kad sugertų poslinkį be perderinimo vidury sesijos.

---

## Optimalaus Throttle Mid radimas

### 1 metodas — OSD įrašymas

Įjunk throttle OSD elementą ir įrašyk skrydį. Rask stabilaus hover'io atkarpą įraše ir nuskaityk ekrane rodomą throttle procentą. Padaryk tai pako pradžioje, viduryje ir pabaigoje. Suvidurkink tris reikšmes — tai tavo Throttle Mid atspirties taškas.

Configurator → OSD skirtuke įjunk **Throttle Position** (rodoma procentais) ir padėk jį kampe, kuris matomas tavo įrašuose.

### 2 metodas — Blackbox analizė

Su įjungtu blackbox logginimu, ištrauk logą iš viso skrydžio ir nubraižyk `rcCommand[3]` (throttle) kanalą prieš laiką. Identifikuok stabilaus hover'io atkarpas (mažas svyravimas, pastovus aukštis) ir apskaičiuok vidutinę throttle reikšmę tuose languose.

Betaflight Configurator Blackbox Explorer'yje naudok ašies filtrą, kad rodytų tik throttle. Rask 5–10 sekundžių lygaus skrydžio langą ir nuskaityk vidutinę rcCommand reikšmę. Perskaičiuok į procentus: rcCommand throttle diapazonas paprastai yra 1000–2000 µs; atimk 1000 ir padalink iš 10, kad gautum procentus.

Tai duoda vidurkį vienam skrydžiui. Po 3–5 skrydžių turėsi stabilų įvertį per šviežius ir išsekusius pakus.

---

## Viską sudėjus

| Nustatymas      | Į ką nustatyti                             |
|-----------------|--------------------------------------------|
| RC Rate         | 0.80–1.10 533–733 šeimai (nustato maks. rate) |
| Super Rate      | 0.70 533/633/733 šeimai; didesnis = kietesnis kraštas |
| Expo            | 0 pagal nutylėjimą; pridėk 0.10–0.20 centrui suminkštinti |
| Throttle Expo   | 0.3 daugumai build'ų                       |
| Throttle Mid    | Išmatuotas hover taškas (metodas aukščiau) ≈ 0.45–0.55 |

Rates ir expo pirmiausia derink simuliatoriuje, kur avarijos nieko nekainuoja (o propeleriai — kainuoja, patikėk). Tada eik į lauką ir naudok OSD/blackbox throttle mid tiksliai suderinti.

Taip pat žiūrėk: [Rate Presets (733 / 633 / 533)](../rate-presets/) — tikslias reikšmes ir kopijuojamą CLI abiejose rate sistemose.
