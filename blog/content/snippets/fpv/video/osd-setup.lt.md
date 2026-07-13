---
title: "OSD nustatymas — analog, digital, GPS"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "osd", "analog", "digital", "gps", "dji", "walksnail"]
---

Švarus OSD duoda tau esminius skrydžio duomenis ir neužgriozdina vaizdo, kai peržiūri įrašą. Tikslas: rodyti tik tai, ko realiai prireiks nusileidus.

---

## Analog OSD (MAX7456)

MAX7456 čipas, esantis daugumoje FC stack'ų, tvarko OSD overlay ant analog vaizdo.

**Minimalus naudingas rinkinys įrašo analizei:**

| Elementas          | Kodėl                                         |
|--------------------|-----------------------------------------------|
| Baterijos įtampa   | Susieti įtampos sag'ą su elgesiu              |
| Srovė (mA)         | Pastebėti motorų ar ESC apkrovą               |
| Sunaudoti mAh      | Žinoti, kada perėjai 80 % paketo talpos       |
| Arming flag'ai     | Suprasti, kodėl arm'as buvo atmestas          |
| Timeris (on / total) | Laiko juosta klipo peržiūrai                |
| RSSI / Link quality| Pastebėti RF įvykius įraše                     |

GPS elementus dėk tik tada, jei sumontuotas GPS modulis.

**Nesigriozdink su:** dirbtinis horizontas, crosshair'ai, throttle juosta, G-force, heading juosta — jie retai padeda po skrydžio analizei ir tik ryja ekrano vietą. (Taip, pradžioje irgi buvau prikabinęs viską — atrodė kaip lėktuvo kokpitas, o naudos jokios.)

Configurator → **OSD** tab'as → tempk elementus į kampus. Centrinį kadrą laikyk švarų.

---

## Digital OSD (DJI O3 / Walksnail Avatar / HDZero)

Digital sistemos OSD uždeda kitaip:

- **DJI O3 / O4**: OSD renderinamas goggle'uose. Betaflight vis tiek siunčia MSP OSD duomenis; DJI juos atvaizduoja. Konfigūruok Betaflight OSD tab'e kaip įprasta.
- **Walksnail Avatar**: tas pats MSP OSD kelias. Pilnas custom elementų pozicionavimas.
- **HDZero**: MSP OSD. Tikslus valdymas per HDZero goggle'ų meniu.

Galioja tas pats minimalistinis elementų sąrašas. Su digital turi daugiau ekrano vietos, bet centrą vis tiek laikyk švarų.

DJI atveju įjunk MSP OSD per CLI:
```
set osd_displayport_device = MSP
set displayport_msp_serial = <serial port number of DJI air unit>
save
```

---

## Su GPS

GPS prideda prasmingų elementų analizei:

| Elementas          | Kodėl                                         |
|--------------------|-----------------------------------------------|
| GPS greitis        | Palyginti max greitį blackbox'e ir OSD        |
| Aukštis            | Legalaus aukščio patikra, aukščio šuoliai     |
| GPS koordinatės    | Kur įvyko kritimas                            |
| Home rodyklė + atstumas | Return-to-home atsargos patikra          |
| GPS fix / satelitai | Pastebėti silpną GPS fiksaciją prieš skrydį  |

GPS elementus dėk prie ekrano kraštų (viršuje ar apačioje). Aukštį ir greitį laikyk kampe, ne centre.

---

## Be GPS (minimalus)

Jei GPS nėra: pašalink visus su GPS susijusius elementus. Palik:
- Įtampą
- mAh
- Timerį
- Link quality / RSSI

Keturi elementai dviejuose kampuose. Neužgriozdinta, viskas aktualu.

---

## Kinematografinis analog OSD

Kinematografiniuose setup'uose prioritetas — švarus vaizdas, tad OSD turi beveik dingti.

**Rekomenduojamas išdėstymas:**
- Baterijos įtampa: apatinis kairysis kampas, mažas
- Sunaudoti mAh: apatinis dešinysis kampas, mažas
- Timeris: viršuje dešinėje, mažas
- Nieko daugiau

Išjunk visus warning'us (low battery teksto pop-up'ą gali palikti, bet nustatyk konservatyvią ribą, kad neiššoktų per patį kadrą).

CLI:
```
# Reduce OSD element clutter
set osd_vbat_pos = 2400       # bottom left
set osd_mah_drawn_pos = 2424  # bottom right
set osd_warnings_pos = 14731  # off-screen or disabled
save
```

> Pozicijų reikšmės koduoja eilutę/stulpelį kaip `row*32 + col + 0x800` dėl blink bito. Naudok Configurator drag-and-drop, o ne skaičiuok ranka.

---

## Stats ekranas (po disarm)

Stats ekranas rodomas po disarm, jei `stats = ON`. Naudingi elementai:

```
set stats = ON
set stats_min_voltage = ON
set stats_max_current = ON
set stats_used_mah = ON
set stats_max_speed = ON    # requires GPS
set stats_max_altitude = ON # requires GPS or baro
set stats_total_flights = ON
save
```

Taip pat žiūrėk: [Rate profiliai ir persistent stats](../../tuning/rate-profiles-persistent-stats/) — kaip rate profilio perjungimas sąveikauja su statistikos išsaugojimu.
