---
title: "Prarasta Baltijoje: ką sako telemetrija"
date: 2026-08-28
description: "Pavo20 Pro II praradimo Baltijos jūroje analizė: kaip protingas eksperimentas baigėsi jūros dugne, ką užregistravo telemetrija, ir kokius pakeitimus darau."
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
thumbnail: "https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/4a54fb8f-71fe-4ecf-a946-08cb51a7d5c2/vlcsnap-2026-08-28-00h19m25s148.png"
---

![1,04 km nuo namų. OSD rodo LAND NOW. 2,03 V ant celės. Leidžiasi.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/4a54fb8f-71fe-4ecf-a946-08cb51a7d5c2/vlcsnap-2026-08-28-00h19m25s148.png)

Tai Pavo20 Pro II, 1,04 km nuo namų, baterija 2,03 V ant celės, OSD rodo
LAND NOW, o priekyje jūra. Smagu paklausti, kaip čia atsidūriau.

Štai kaip.

## Sprendimas

Tą pačią dieną anksčiau nuskridau 2 km apvalų reisą stipriame lauko vėjyje ir
nusileisdau su 20% likusių. Ryšys laikė, baterija laikė, dronas grįžo.
Norėjau sužinoti, kaip atrodo ELRS ryšys virš atviro vandens. Jokių kliūčių,
jokios turbulencijos, tik Baltijos jūra sutemose. Atrodė kaip lengvesnė to,
ką jau dariau, versija.

Nebuvo.

## Skrydis

Išvykimas buvo lengvas. 2,47 km, 66 km/h, 11 V baterija. Horizontas lygus,
ryšys švarus, baterija beveik nejudėjo. Buvo pastipramas vėjas, kurio
nespecialiai užregistravau kaip pastipramą vėją. Užregistravau kaip geras sąlygas.

![2,47 km nuo starto, 66 km/h, 11,0 V.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/f031f1b9-00b9-420e-899f-1e9ee2405f31/vlcsnap-2026-08-28-00h18m36s708.png)

Pasukau atgal. Pakėliau šiek tiek aukščiau, kad geriau matyti. Greitis krito.
Baterija ėmė judėti greičiau. Grįžtamasis maršrutas, tas pats oro greitis
prieš tą patį vėją, kainavo 183 mAh/km: 1,45 karto daugiau nei išvykimas.
Pastipramas vėjas dabar yra priešvėjis, ir kiekvienas metras namo kainuoja
pusantro karto tiek, kiek kainavo nuskysti.

![Sukama atgal. 1,89 km, 10,9 V. Greitis jau mažesnis.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/355d3028-6331-4409-8c1e-9fd772811eac/vlcsnap-2026-08-28-00h18m53s493.png)

## Tamsa

Telemetrija dingo prie t = 79 s, aukštis 0 m, maždaug 920 m nuo starto.
Negrįžo 150 sekundžių. Per tą laiką baterija peržengė 3,8 V ant celės, 3,6 V,
3,5 V. Radijas visą tą laiką tylėjo. EdgeTX loginiai jungikliai, susieti su
telemetrijos jutikliais, telemetrijos tamsoje priverstinai išjungiami. Lyginimo
jungiklis negali suveikti kol ryšys nutrūkęs.

Telemetrija atsinaujino prie t = 230 s, kai dronas pakilo iki 75 m.
Dvispindulinis atspindys nuo jūros paviršiaus ties nulinio kampo sklidimo
kryptimi: jūra yra beveik tobulas RF reflektorius, o esant mažam aukščiui
tiesioginė ir atspindėta banga ateina vienodo amplitudės ir priešingos fazės.
Pakilimas nutraukia slopinimą. 0 m: tylu. 75 m: ryšys atsinaujino.

![1,33 km. LOW BATTERY ekrane. Radijas 150 sekundžių nieko nesakė.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/0d4f62f8-43d0-480a-8bc0-14b69a3198e1/vlcsnap-2026-08-28-00h19m11s482.png)

Kai radijas vėl kažką matė, buvo sunaudota 555 mAh. Liko 266 mAh. Grąžos
taškas, esant išmatuotam grįžtamojo maršruto intensyvumui su 10% atsarga, buvo
1323 m nuo namų. Dronas buvo 1946 m nuo namų. Jau 623 m už jo.

Komanda `rth` suveikė prie t = 229,8 s. Teisinga ta prasme, kad suveikė.
Bevertė ta prasme, kad suveikė 150 sekundžių po to, kai sprendimą jau priėmė
fizika.

## Ką sako telemetrija

![Penki grafikų blokai iš EdgeTX telemetrijos. Viršuje: GPS ir nuspėtas skrydžio kelias. Viduryje: sunaudota talpa ir srovė pagal atstumą. Apačioje: sunaudota talpa pagal laiką ir RSSI / ryšio kokybė pagal atstumą. Pilki tarpai žymi telemetrijos tamsą.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/39e00838-047a-453b-b5b8-443a54420a5e/ocean_flight_analysis.png)

Baterija buvo LAVA II 680 mAh 3S LiHV. Skrydžio valdiklis ją buvo sukonfigūravęs
kaip 821 mAh, kas per 20% perdeda talpą ir tyliai išpučia kiekvieną SoC rodmenį.
Baterijos SoC rodė 85% likusių, kai ryšys pirmą kartą dingo.

Išvykimas: 126 mAh/km. Grįžimas: 183 mAh/km. Telemetrijos tamsa iš viso:
171 s iš 350 s. 49% skrydžio, radijui nieko nematant.

RSSI starte buvo −36 dBm. Po 60 m nuo starto, vis dar jūros lygyje, jau
−84 dBm. 48 dB nuostolis per 60 horizontalių metrų, kol ryšio kokybės
rodmuo rodė 100%. RSSI aliarmas ties −85 dBm būtų suveikęs prie t = 38 s,
maždaug 235 m nuo starto. Perspėjimas ties 235 m būtų pakeitęs skrydį.
Radijuje `rssiSource` buvo nustatytas į `none`. Jokio RF aliarmo iš viso
nebuvo sujungta.

RQly rodė 100% iki visiško ryšio nutrūkimo. Šiame skrydyje ryšio kokybė man
nieko nepasakė. RSSI pasakojo istoriją nuo pirmų 40 sekundžių. Aš neklausiau,
nes nebuvo ko klausyti.

![Paskutinis kadras prieš vandenį.](https://eu.chat-img.sintra.ai/f775d550-4c12-4009-b046-c70303e7256c/ab568c4f-a08e-41ba-9341-6bdb2544a9ec/vlcsnap-2026-08-28-00h19m56s313.png)

## Ką keičiu

Kiekviena eilutė žemiau yra gedimo režimas, kurį atskleidė telemetrija.
Įgyvendinimas — atskirame straipsnyje.

| Sąlyga | Senasis elgesys | Naujasis elgesys |
|---|---|---|
| Baterija kerta 4,2 / 4,0 / 3,8 V/celę leidžiantis | Vienas lyginimo loginis jungiklis, suveikia vieną kartą, valdomas baterijos mygtuku | Ištariamas skaičius ("vienas", "du", "trys") prie kiekvieno 0,1 V kirtimo žemiau 3,8 V; be pyptelėjimo (pyptelėjimo garsas šiame radijuje 0) |
| Baterija < 3,6 V/celę | `lowbat` takeliu per loginį jungiklį | Tas pats, bet ir iš fono scenarijaus — pirmoji gynybos linija net per telemetrijos tamsą |
| Grįžtamasis maršrutas kainuoja daugiau nei 1,3× išvykimo | Jokio perspėjimo nebuvo | Vėjo asimetrijos perspėjimas po ≥30 s išvykimo + ≥15 s grįžimo: "warning close {santykis}%" |
| Specifinė galia padidėjusi + specifinis greitis normalus (vidinis gedimas) | Jokio perspėjimo nebuvo | "warning power {santykis}%" vieną kartą, nustačius kruizinę bazę |
| Specifinė galia padidėjusi + specifinis greitis sumažėjęs (priešvėjis, pasipriešinimas) | Jokio perspėjimo nebuvo | "warning speed {santykis}%" vieną kartą |
| RSSI žemiau −92 dBm | `rssiSource: none` — jokio RF aliarmo. RQly rodė 100% iki visiško praradimo | `siglow` + dBm reikšmė; `sigcrt` ties −100 dBm. RSSI yra rampa; LQ yra skardis |
| RSSI blogėja + aukštis < 25 m | Nieko | "tolow" — kiltis. Išmatuota: ryšys dingo ties 0 m, atsinaujino ties 75 m |
| Telemetrija tamsi ilgiau nei 4 s | Loginiai jungikliai išjungti; visi baterijos perspėjimai nutyla | Fono scenarijus toliau skaičiuoja atstumą ir SoC, skelbia būseną atsinaujinus ryšiui |
| Grąžos taškas artėja (išvykimas, GPS veikia) | Nieko | "close {metrų}" skaičiuojant mažyn, kol PNR − dHome < 400 m |
| Ginkluojamas su neveikiančiu GPS rescue (FC sako "?") | Palydovų skaičiaus palyginimas su užkoduota riba | FC sprendimas: Betaflight prideda `?` prie CRSF skrydžio režimo eilutės. Scenarijus skaito tiesiogiai. |
| Impedansas auga greičiau nei galima paaiškinti iškrovimu (perteklius ≥ 1,4×) | Jokio perspėjimo | "warning bad {mΩ}" vieną kartą, kol SoC > 25% |

Dronas jūros dugne. Telemetrija ne. Viena iš tų dviejų labiau naudinga
nekartojandam šio eksperimento. Vat taip vat.
