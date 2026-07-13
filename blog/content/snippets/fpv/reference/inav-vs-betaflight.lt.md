---
title: "INAV vs Betaflight — kada kurį naudoti"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "inav", "betaflight", "gps", "autonomous", "firmware", "navigation", "pavo20"]
---

Betaflight ir INAV — abu atviro kodo skrydžio kontrolerio firmware. Jie turi bendrą kodo istoriją, bet išsišakojo į skirtingus įrankius su skirtingomis stiprybėmis. Netinkamas pasirinkimas GPS dronui baigiasi arba erzinančia patirtimi, arba nepanaudotomis galimybėmis. (Spėkit, iš kurios pusės aš mokiausi.)

---

## Pagrindinė filosofija

```mermaid
flowchart LR
    BF[Betaflight] -->|Optimised for| M1[Manual acro flying<br/>Low latency PID loop<br/>Freestyle & racing]
    BF -->|GPS added as| M2[Emergency recovery only<br/>GPS Rescue = last resort]
    
    INAV[INAV] -->|Optimised for| N1[Navigation-first<br/>Autonomous missions<br/>Position hold, waypoints, RTH]
    INAV -->|Manual flying| N2[Supported but PID loop<br/>not as tuned as BF]
```

Betaflight į GPS žiūri kaip į saugumo funkciją. INAV į GPS žiūri kaip į pagrindinį naudojimo scenarijų.

---

## Funkcijų palyginimas

| Funkcija                      | Betaflight        | INAV               |
|-------------------------------|-------------------|--------------------|
| Rankinis acro skraidymas      | Puikus            | Geras              |
| PID kilpos vėlinimas          | ~1 ms tikslas     | ~2–4 ms tipiškai   |
| GPS Rescue (RTH)              | Bazinis — tik avariniam | Pilnas RTH su stabdymu, hold, misija |
| Position Hold                 | Nėra              | Taip — POSHOLD režimas |
| Waypoint misijos              | Nėra              | Taip — autonominiai maršrutai |
| Aukščio išlaikymas            | Nėra              | Taip — ALTHOLD režimas |
| Fiksuoto sparno palaikymas    | Ne                | Taip — pilnas palaikymas |
| Blackbox / derinimo įrankiai  | Puikūs            | Geri               |
| OSD integracija               | Puiki             | Gera (rodo daugiau GPS duomenų) |
| Bendruomenė / forumai         | Didesnė           | Aktyvi, bet mažesnė |
| Konfigūratoriaus patirtis     | Subrendusi        | Subrendusi, sudėtingesnė |
| Failsafe                      | Stage 1/2, GPS Rescue | RTH su stabdymu, EMERG nusileidimas |

---

## Kada naudoti Betaflight

- **Freestyle ar racing dronai**, kur PID kilpos kokybė yra prioritetas
- **Cinewhoop'ai ir proximity dronai**, kur sklandus atsakas svarbesnis už navigaciją
- **Dronai, kuriems GPS reikia tik kaip „saugos tinklo“** — GPS Rescue beveik niekada nespaudi, bet jis yra, jei kas nutiktų
- **Bet koks standartinis 5" freestyle rėmas** — bendruomenės derinimo resursai (presetai, Betaflight presetų bazė) čia gerokai pranašesni

Betaflight GPS Rescue veikia ir per 4.3/4.4 versijas ženkliai pagerėjo — bet jis neskirtas patikimai autonominei navigacijai. Tai funkcija „parvaryk droną namo, kol baterija dar neišsikrovė“.

---

## Kada naudoti INAV

- **GPS tyrinėtojams / long-range dronams**, kur nori, kad dronas realiai navigatų pats
- **Waypoint misijoms** — INAV gali skristi iš anksto suprogramuotu maršrutu, laikyti aukštį ir grįžti namo be RC valdymo
- **Fiksuoto sparno hibridams** — INAV palaiko stabilizuotą fiksuoto sparno skrydį ir maišymą (mixing)
- **Dronams, kur nori POSHOLD** — galimybė paleisti stick'us ir kad dronas kabotų vietoje 3D erdvėje be dreifo
- **Kinematografijai su gimbalu** — INAV pozicijos/aukščio išlaikymas leidžia sklandžius „dolly“ kadrus be nuolatinio koregavimo

---

## Pavo20 GPS problema

Pavo20 — tai whoop'as su GPS. Betaflight'e GPS Rescue whoop klasės drone susiduria su keliais iššūkiais:

```mermaid
flowchart TD
    P1[Small frame size<br/><250g AUW] -->|Low inertia| C1[GPS Rescue corrections<br/>overshoot and oscillate]
    P2[Whoop ducted props<br/>Higher drag] -->|Slower response<br/>to GPS commands| C2[Rescue turns are sluggish<br/>not crisp]
    P3[Short antenna<br/>Internal GPS module] -->|Slower fix<br/>Weaker signal| C3[Poor position accuracy<br/>in GPS Rescue mode]
    P4[BF GPS Rescue<br/>not tuned for micro quads] -->|Default gains<br/>too aggressive for small builds| C4[Oscillation or crash<br/>on rescue activation]
```

INAV navigacijos „stackas“ tokias situacijas tvarko geriau, nes naudoja tikrą pozicijos kontrolerį (o ne grubų avarinį režimą), o jo RTH seka apima lėtėjimą ir stabdymą. INAV taip pat turi geresnę barometro integraciją aukščiui laikyti dronuose be GPS aukščio fiksacijos.

**Pavo20 migravimas į INAV:**
- INAV palaiko daugumą populiarių FC (patikrink INAV aparatūros sąrašą dėl suderinamumo)
- Pavo20 numatytasis FC turi palaikyti INAV — patikrink [INAV target sąraše](https://github.com/iNavFlight/inav/blob/master/docs/Boards.md)
- Tikėkis, kad PID reikės derinti iš naujo — INAV numatytieji nustatymai suderinti sunkesniems GPS dronams

---

## Signalo kokybė veikia abu firmware

Nepriklausomai nuo firmware, GPS veikimas mažuose dronuose kenčia dėl:

```chart
{
  "type": "bar",
  "data": {
    "labels": ["Clear sky\nopen field", "Suburban area\ntrees + buildings", "Under canopy\nor indoors", "Carbon frame\nshadowing GPS", "GPS near VTX\n5.8GHz interference"],
    "datasets": [{
      "label": "Typical GPS fix quality (1=terrible, 10=excellent)",
      "data": [9, 6, 2, 4, 3],
      "backgroundColor": [
        "rgba(34,197,94,0.7)",
        "rgba(132,204,22,0.7)",
        "rgba(239,68,68,0.7)",
        "rgba(249,115,22,0.7)",
        "rgba(239,68,68,0.7)"
      ],
      "borderWidth": 1
    }]
  },
  "options": {
    "indexAxis": "y",
    "responsive": true,
    "plugins": {
      "title": { "display": true, "text": "GPS Fix Quality by Environment (approximate)" },
      "legend": { "display": false }
    },
    "scales": {
      "x": { "beginAtZero": true, "max": 10 }
    }
  }
}
```

VTX trukdžių problema ypač dažna micro dronuose: 5,8 GHz vaizdo perdavimas gali „apkurtinti“ GPS modulius, esančius ant tos pačios PCB. Pavo20 atveju VTX ir GPS yra visai šalia — tai žinomas aparatūros apribojimas, kurio joks firmware pilnai neišspręs.

**Aparatūriniai sprendimai:**
- Laikyk GPS anteną kuo toliau nuo VTX antenos, kiek fiziškai įmanoma
- Naudok ekranuotą GPS modulį (metalinis „dangtelis“ virš modulio)
- Prieš tikrindamas GPS fix ant žemės, perjunk VTX į mažesnę galią (25 mW ar 0 mW)
- Palauk GPS fix su išjungtu VTX, tada įjunk jį skrydžiui

---

## Sprendimo santrauka

```mermaid
flowchart TD
    Q1{Primary goal?} -->|Fly fast / smooth<br/>acro / racing / freestyle| BF[Use Betaflight]
    Q1 -->|Autonomous<br/>navigation / missions| INAV[Use INAV]
    Q1 -->|GPS safety net only<br/>still flying manually| Q2{Build size?}
    Q2 -->|5 inch or larger| BF
    Q2 -->|Sub-250g micro| Q3{GPS Rescue<br/>actually important?}
    Q3 -->|Nice to have| BF[Betaflight BF GPS Rescue<br/>works well enough]
    Q3 -->|Critical for safe recovery| INAV[INAV RTH is more reliable<br/>on small GPS builds]
```

Daugumai freestyle ir racing dronų: **Betaflight**.  
GPS priklausomai navigacijai, long-range ar autonominėms misijoms: **INAV**.  
Pavo20 atveju, kai svarbus patikimas GPS parskridimas: verta apsvarstyti **INAV**, susitaikant su prastesniu rankiniu skraidymu.
