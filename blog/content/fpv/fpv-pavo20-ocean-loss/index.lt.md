---
title: "Prarasta Baltijoje: ką sako telemetrija"
date: 2026-08-28
description: "Pavo20 Pro II praradimo Baltijos jūroje analizė: išmatuota telemetrija, dvispindulinis atspindys virš vandens, kodėl nutilo visi perspėjimai, ir kokius pakeitimus darau."
draft: false
toc: true
categories:
  - FPV
tags:
  - fpv
  - edgetx
  - telemetrija
  - gps
  - baterija
  - ilgas-nuotolis
  - perspėjimai
  - rizikos-valdymas
keywords: ["fpv baterijos perspėjimai", "edgetx telemetrija", "gps rescue", "grąžos taškas", "ilgas nuotolis fpv", "pavo20 pro ii praradimas", "fpv avarija jūroje", "dvispindulinis atspindys"]
thumbnail: "https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/39e00838-047a-453b-b5b8-443a54420a5e/ocean_flight_analysis.png"
---

## Kas nutiko

Nuskridau su Pavo20 Pro II 2,47 km virš Baltijos jūros, norėdamas pamatyti,
kaip atrodo ELRS ryšys atviru horizontu be jokių kliūčių. Protingas
eksperimentas. Rezultatas labai informatyvus.

Dronas dabar jūros dugne.

Telemetrija dingo prie t = 79 s, aukštis 0 m, maždaug 920 m nuo starto.
Atsinaujino prie t = 230 s, kai dronas grįžtamuoju maršrutu pakilo iki 75 m.
Dvispindulinis atspindys nuo jūros paviršiaus ties nulinio kampo sklidimo
kryptimi: jūra yra beveik tobulas RF reflektorius, o esant mažam aukščiui
tiesioginė ir atspindėta banga ateina vienodo amplitudės ir priešingos fazės.
Pakilimas nutraukia slopinimą. Telemetrijai atsinaujinus buvo sunaudota
555 mAh. Liko 266 mAh. Grąžos taškas buvo 1323 m nuo namų. Dronas buvo
1946 m nuo namų. Jau 623 m už jo.

## Skrydžio analizė

![Penki grafikų blokai iš EdgeTX telemetrijos. Viršuje: GPS ir nuspėtas skrydžio kelias. Viduryje: sunaudota talpa ir srovė pagal atstumą. Apačioje: sunaudota talpa pagal laiką ir RSSI / ryšio kokybė pagal atstumą. Pilki tarpai žymi telemetrijos tamsą.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/39e00838-047a-453b-b5b8-443a54420a5e/ocean_flight_analysis.png)

## Ką sako skaičiai

Baterija buvo LAVA II 680 mAh 3S LiHV. Skrydžio valdiklis ją užregistravo kaip
821 mAh, kas per 20% perdeda talpą ir tyliai išpučia kiekvieną SoC rodmenį
telemetrijoje. Baterijos SoC rodė 85% likusių, kai ryšys pirmą kartą dingo.

Išvykimo kursas — 126 mAh/km. Dronas turėjo maždaug 10 km/h pastiprą vėją
prie ~50 km/h oro greičio, tad tai geriausias rezultatas, kurio galima tikėtis.
Grįžtamasis maršrutas, tas pats oro greitis prieš tą patį vėją, kainavo
183 mAh/km: 1,45 karto brangiau kilometrui. Telemetrijos tamsa iš viso:
171 s iš 350 s. 49% skrydžio radio nieko nematė.

RSSI starte buvo −36 dBm. Po 60 m nuo starto, vis dar jūros lygyje, jau
−84 dBm. 48 dB nuostolis per 60 horizontalių metrų, kol ryšio kokybės
rodmuo rodė 100%. RSSI buvo rampa; LQ buvo skardis.

Mentalinis modelio gedimas: tą pačią dieną anksčiau nuskridau 2 km apvalų
reisą stipriame lauko vėjyje ir nusileisdau su 20% likusių. Nusprendžiau, kad
virš atviro vandens bus lengviau. Ko neįskaičiavau: grįžtamasis maršrutas
kainuoja 1,45× daugiau nei išvykimas tokiu pat oro greičiu; vėjas pasikeitė
kryptimi kai pasukau; ir skridau 0 m aukštyje virš beveik tobulo RF reflektoriaus.

## Ką darė ir nedarė perspėjimai

Garsinė komanda `rth` pasiekė radiją prie t = 229,8 s, o tai buvo 150 s po to,
kai baterija peržengė 3,8 V ant celės ribą. EdgeTX loginiai jungikliai, susieti
su telemetrijos jutikliais, telemetrijos tamsoje priverstinai išjungiami.
Lyginimo jungiklis negali suveikti kol ryšys nutrūkęs. Visą tamsos laikotarpį
perspėjimas tylėjo.

RSSI dBm aliarmas, nustatytas ties −85 dBm, būtų suveikęs prie t = 38 s,
maždaug 235 m nuo starto. Tai teisinga šiam skrydžiui: ryšys jau nuo trumpo
atstumo buvo neįprastai prastas dėl jūros paviršiaus geometrijos. Perspėjimas
ties 235 m būtų pakeitęs skrydį. Radijuje `rssiSource` buvo nustatytas į
`none`. Jokio RF aliarmo iš viso nebuvo sujungta.

RQly rodė 100% iki visiško ryšio nutrūkimo. Šiame skrydyje ryšio kokybė man
nieko nepasakė. RSSI pasakojo istoriją nuo pirmų 40 sekundžių.

## Vaizdo medžiaga iš akinių

![Ankstyviausias išsaugotas kadras. 2,47 km nuo starto, 66 km/h, baterija 11,0 V.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/f031f1b9-00b9-420e-899f-1e9ee2405f31/vlcsnap-2026-08-28-00h18m36s708.png)

![Sukamas atgal. 1,89 km nuo starto, 10,9 V.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/355d3028-6331-4409-8c1e-9fd772811eac/vlcsnap-2026-08-28-00h18m53s493.png)

![Grįžtama. 1,33 km. OSD rodo LOW BATTERY.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/0d4f62f8-43d0-480a-8bc0-14b69a3198e1/vlcsnap-2026-08-28-00h19m11s482.png)

![1,04 km nuo namų. OSD rodo LAND NOW. 6,1 V iš viso, 2,03 V ant celės.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/4a54fb8f-71fe-4ecf-a946-08cb51a7d5c2/vlcsnap-2026-08-28-00h19m25s148.png)

![Paskutinis kadras prieš vandenį. 1,04 km nuo namų, leidžiasi.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/ab568c4f-a08e-41ba-9341-6bdb2544a9ec/vlcsnap-2026-08-28-00h19m56s313.png)

## Ką keičiu

Žemiau pateikta lentelė yra strategija ir projektavimo sprendimai. Kiekviena
eilutė yra gedimo režimas, kurį atskleidė telemetrija. Įgyvendinimas — atskirame
straipsnyje.

| Sąlyga | Senasis elgesys | Naujasis elgesys |
|---|---|---|
| Baterija kerta 4,2 / 4,0 / 3,8 V/celę leidžiantis | Vienas lyginimo loginis jungiklis, suveikia vieną kartą prie pirmo kirtimo, valdomas baterijos mygtuku | Ištariamas skaičius ("vienas", "du", "trys") prie kiekvieno 0,1 V kirtimo žemiau 3,8 V; be pyptelėjimo (pyptelėjimo garsas šiame radijuje 0) |
| Baterija < 3,6 V/celę | `lowbat` takeliu per loginį jungiklį | Tas pats, bet dabar ir iš fono scenarijaus — pirmoji gynybos linija net per telemetrijos tamsą |
| Grįžtamasis maršrutas kainuoja daugiau nei 1,3× išvykimo maršrutą km | Jokio perspėjimo nebuvo | Vėjo asimetrijos perspėjimas po ≥30 s išvykimo + ≥15 s grįžimo: "warning close {santykis}%" |
| Specifinė galia padidėjusi + specifinis greitis normalus (vidinis gedimas: variklis, sraigtas, guolis) | Jokio perspėjimo nebuvo | "warning power {santykis}%" vieną kartą, nustačius kruizinę bazinę vertę |
| Specifinė galia padidėjusi + specifinis greitis sumažėjęs (išorinė: priešvėjis, pasipriešinimas) | Jokio perspėjimo nebuvo | "warning speed {santykis}%" vieną kartą |
| Ryšio RSSI žemiau −92 dBm | `rssiSource: none` — jokio RF aliarmo iš viso nesujungta. RQly rodė 100% iki visiško praradimo | `siglow` + dBm reikšmė; `sigcrt` ties −100 dBm. RSSI yra rampa; LQ yra skardis |
| RSSI blogėja + aukštis < 25 m | Nieko | "tolow" — kiltis. Išmatuota: ryšys dingo ties 0 m, atsinaujino ties 75 m, dvispindulinis atspindys virš vandens |
| Telemetrija tamsi ilgiau nei 4 s | Loginiai jungikliai priverstinai išjungti; visi baterijos perspėjimai nutyla | Fono Lua scenarijus toliau skaičiuoja atstumą ir SoC, skelbia būseną atsinaujinus ryšiui |
| Grąžos taškas artėja (išvykimas, GPS veikia) | Nieko | "close {metrų}" skaičiuojant mažyn, kol PNR − dHome < 400 m |
| Ginkluojamas su neveikiančiu GPS rescue (FC sako "?") | Palydovų skaičiaus palyginimas su scenariuje užkoduota riba | FC sprendimas: Betaflight prideda `?` prie CRSF skrydžio režimo eilutės kai `numSat < gps_rescue_min_sats`. Scenarijus skaito tai tiesiogiai. |
| Impedansas didėja labiau nei galima paaiškinti iškrovimu (perteklius ≥ 1,4×) | Jokio perspėjimo | "warning bad {mΩ}" vieną kartą, kol SoC > 25% |

## Pabaiga

Dronas jūros dugne. Telemetrija ne. Viena iš tų dviejų labiau naudinga
nekartojandam šio eksperimento.

Scenarijai ir YAML — atskirame straipsnyje. Šis yra apie skrydžio duomenis
ir sprendimus, kuriuos jie privertė priimti. Vat taip vat.
