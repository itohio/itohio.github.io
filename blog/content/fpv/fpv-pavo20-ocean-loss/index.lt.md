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
thumbnail: "landnow.jpg"
---

![1,04 km nuo namų. OSD rodo LAND NOW. 2,03 V ant celės. Leidžiasi.](landnow.jpg)

Tai Pavo20 Pro II, 1,04 km nuo namų, baterija 2,03 V ant celės, OSD rodo
LAND NOW, o priekyje jūra. Smagu paklausti, kaip čia atsidūriau.

Štai kaip.

## Sprendimas

Kelias savaites anksčiau nuskridau 2 km apvalų reisą stipriame lauko vėjyje ir
nusileisdau su 20% likusių. Ryšys laikė, baterija laikė, dronas grįžo.
Norėjau sužinoti, kaip atrodo ELRS ryšys virš atviro vandens. Jokių kliūčių,
jokios turbulencijos, tik Baltijos jūra sutemose. Atrodė kaip lengvesnė to,
ką jau dariau, versija.

Nebuvo.

## Skrydis

Išvykimas buvo lengvas. 2,47 km, 66 km/h, 11 V baterija. Horizontas lygus,
ryšys švarus, baterija beveik nejudėjo. Buvo pastipramas vėjas, kurio
nespecialiai užregistravau kaip pastipramą vėją. Užregistravau kaip geras sąlygas.

![2,47 km nuo starto, 66 km/h, 11,0 V.](outbound.jpg)

Pasukau atgal. Pakėliau šiek tiek aukščiau, kad geriau matyti. Greitis krito.
Baterija ėmė judėti greičiau. Grįžtamasis maršrutas, tas pats oro greitis
prieš tą patį vėją, kainavo 183 mAh/km: 1,45 karto daugiau nei išvykimas.
Pastipramas vėjas dabar yra priešvėjis, ir kiekvienas metras namo kainuoja
pusantro karto tiek, kiek kainavo nuskysti.

![Sukama atgal. 1,89 km, 10,9 V. Greitis jau mažesnis.](turning.jpg)

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

![1,33 km. LOW BATTERY ekrane. Radijas 150 sekundžių nieko nesakė.](lowbat.jpg)

Kai radijas vėl kažką matė, buvo sunaudota 555 mAh. Liko 266 mAh. Grąžos
taškas, esant išmatuotam grįžtamojo maršruto intensyvumui su 10% atsarga, buvo
1323 m nuo namų. Dronas buvo 1946 m nuo namų. Jau 623 m už jo.

Komanda `rth` suveikė prie t = 229,8 s. Teisinga ta prasme, kad suveikė.
Bevertė ta prasme, kad suveikė 150 sekundžių po to, kai sprendimą jau priėmė
fizika.

## Ką sako telemetrija


```chart
{"type":"scatter","data":{"datasets":[{"label":"Telemetry OK","data":[{"x":0.0,"y":0.0},{"x":6.0,"y":1.0},{"x":6.0,"y":1.0},{"x":6.0,"y":1.0},{"x":12.0,"y":9.0},{"x":12.0,"y":9.0},{"x":15.0,"y":12.0},{"x":15.0,"y":12.0},{"x":15.0,"y":12.0},{"x":16.0,"y":11.0},{"x":16.0,"y":11.0},{"x":7.0,"y":11.0},{"x":7.0,"y":11.0},{"x":7.0,"y":11.0},{"x":-18.0,"y":14.0},{"x":-18.0,"y":14.0},{"x":-18.0,"y":14.0},{"x":-54.0,"y":28.0},{"x":-54.0,"y":28.0},{"x":-105.0,"y":46.0},{"x":-105.0,"y":46.0},{"x":-105.0,"y":46.0},{"x":-157.0,"y":61.0},{"x":-157.0,"y":61.0},{"x":-157.0,"y":61.0},{"x":-222.0,"y":79.0},{"x":-222.0,"y":79.0},{"x":-282.0,"y":96.0},{"x":-282.0,"y":96.0},{"x":-282.0,"y":96.0},{"x":-355.0,"y":113.0},{"x":-355.0,"y":113.0},{"x":-355.0,"y":113.0},{"x":-355.0,"y":113.0},{"x":-425.0,"y":133.0},{"x":-425.0,"y":133.0},{"x":-425.0,"y":133.0},{"x":-425.0,"y":133.0},{"x":-544.0,"y":163.0},{"x":-544.0,"y":163.0},{"x":-544.0,"y":163.0},{"x":-544.0,"y":163.0},{"x":-652.0,"y":189.0},{"x":-652.0,"y":189.0},{"x":-652.0,"y":189.0},{"x":-652.0,"y":189.0},{"x":-652.0,"y":189.0},{"x":-758.0,"y":216.0},{"x":-758.0,"y":216.0},{"x":-758.0,"y":216.0},{"x":-758.0,"y":216.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-1818.0,"y":694.0},{"x":-1770.0,"y":699.0},{"x":-1770.0,"y":699.0},{"x":-1770.0,"y":699.0},{"x":-1721.0,"y":696.0},{"x":-1721.0,"y":696.0},{"x":-1673.0,"y":692.0},{"x":-1673.0,"y":692.0},{"x":-1673.0,"y":692.0},{"x":-1626.0,"y":685.0},{"x":-1626.0,"y":685.0},{"x":-1626.0,"y":685.0},{"x":-1580.0,"y":679.0},{"x":-1580.0,"y":679.0},{"x":-1531.0,"y":673.0},{"x":-1531.0,"y":673.0},{"x":-1531.0,"y":673.0},{"x":-1476.0,"y":664.0},{"x":-1476.0,"y":664.0},{"x":-1476.0,"y":664.0},{"x":-1426.0,"y":670.0},{"x":-1426.0,"y":670.0},{"x":-1426.0,"y":670.0},{"x":-1381.0,"y":674.0},{"x":-1381.0,"y":674.0},{"x":-1335.0,"y":679.0},{"x":-1335.0,"y":679.0},{"x":-1335.0,"y":679.0},{"x":-1286.0,"y":680.0},{"x":-1286.0,"y":680.0},{"x":-1286.0,"y":680.0},{"x":-1235.0,"y":678.0},{"x":-1235.0,"y":678.0},{"x":-1235.0,"y":678.0},{"x":-1193.0,"y":676.0},{"x":-1193.0,"y":676.0},{"x":-1147.0,"y":672.0},{"x":-1147.0,"y":672.0},{"x":-1147.0,"y":672.0},{"x":-1107.0,"y":662.0},{"x":-1107.0,"y":662.0},{"x":-1107.0,"y":662.0},{"x":-1067.0,"y":651.0},{"x":-1067.0,"y":651.0},{"x":-1031.0,"y":641.0},{"x":-1031.0,"y":641.0},{"x":-1031.0,"y":641.0},{"x":-989.0,"y":628.0},{"x":-989.0,"y":628.0},{"x":-989.0,"y":628.0},{"x":-956.0,"y":618.0},{"x":-956.0,"y":618.0},{"x":-924.0,"y":604.0},{"x":-924.0,"y":604.0},{"x":-924.0,"y":604.0},{"x":-890.0,"y":589.0},{"x":-890.0,"y":589.0},{"x":-890.0,"y":589.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0}],"backgroundColor":"rgba(41,128,185,0.7)","pointRadius":3,"showLine":true,"borderColor":"rgba(41,128,185,0.5)","borderWidth":1},{"label":"Dark / dead-reckoned","data":[{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1818.0,"y":694.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0}],"backgroundColor":"rgba(192,57,43,0.5)","pointRadius":3,"showLine":true,"borderColor":"rgba(192,57,43,0.4)","borderWidth":1}]},"options":{"responsive":true,"maintainAspectRatio":true,"aspectRatio":2.2,"plugins":{"title":{"display":true,"text":"Flight path  (GPS + dead-reckoned)"},"legend":{"position":"bottom"}},"scales":{"x":{"title":{"display":true,"text":"East \u2190 0 \u2192 West  [m]"},"grid":{"color":"rgba(0,0,0,0.08)"}},"y":{"title":{"display":true,"text":"South \u2190 0 \u2192 North  [m]"},"grid":{"color":"rgba(0,0,0,0.08)"}}}}}
```

Skrydžio kelias. Mėlyna: GPS ryšys. Raudona: telemetrija tamsi, pozicija atkurta spėjimu.

```chart
{"type":"scatter","data":{"datasets":[{"label":"mAh/km  (150 m rolling window)","data":[{"x":0.021,"y":47.0},{"x":0.021,"y":281.0},{"x":0.031,"y":194.0},{"x":0.031,"y":194.0},{"x":0.031,"y":356.0},{"x":0.056,"y":197.0},{"x":0.056,"y":287.0},{"x":0.056,"y":287.0},{"x":0.094,"y":170.0},{"x":0.094,"y":223.0},{"x":0.148,"y":142.0},{"x":0.148,"y":182.0},{"x":0.148,"y":182.0},{"x":0.202,"y":93.0},{"x":0.202,"y":134.0},{"x":0.202,"y":134.0},{"x":0.27,"y":74.0},{"x":0.27,"y":114.0},{"x":0.333,"y":76.0},{"x":0.333,"y":76.0},{"x":0.333,"y":119.0},{"x":0.408,"y":73.0},{"x":0.408,"y":122.0},{"x":0.408,"y":122.0},{"x":0.408,"y":122.0},{"x":0.48,"y":86.0},{"x":0.48,"y":86.0},{"x":0.48,"y":157.0},{"x":0.48,"y":157.0},{"x":0.603,"y":77.0},{"x":0.603,"y":77.0},{"x":0.603,"y":159.0},{"x":0.603,"y":159.0},{"x":0.714,"y":68.0},{"x":0.714,"y":128.0},{"x":0.714,"y":128.0},{"x":0.714,"y":128.0},{"x":0.714,"y":128.0},{"x":0.824,"y":63.0},{"x":0.824,"y":63.0},{"x":0.824,"y":154.0},{"x":0.824,"y":154.0},{"x":0.955,"y":83.0},{"x":0.955,"y":83.0},{"x":0.955,"y":207.0},{"x":0.955,"y":207.0},{"x":0.955,"y":207.0},{"x":2.194,"y":124.0},{"x":2.194,"y":171.0},{"x":2.194,"y":171.0},{"x":2.241,"y":131.0},{"x":2.241,"y":173.0},{"x":2.291,"y":131.0},{"x":2.291,"y":172.0},{"x":2.291,"y":172.0},{"x":2.346,"y":105.0},{"x":2.346,"y":105.0},{"x":2.346,"y":171.0},{"x":2.397,"y":116.0},{"x":2.397,"y":161.0},{"x":2.397,"y":161.0},{"x":2.441,"y":113.0},{"x":2.441,"y":172.0},{"x":2.488,"y":131.0},{"x":2.488,"y":131.0},{"x":2.488,"y":177.0},{"x":2.537,"y":131.0},{"x":2.537,"y":178.0},{"x":2.537,"y":178.0},{"x":2.588,"y":141.0},{"x":2.588,"y":178.0},{"x":2.588,"y":178.0},{"x":2.63,"y":132.0},{"x":2.63,"y":169.0},{"x":2.676,"y":123.0},{"x":2.676,"y":155.0},{"x":2.676,"y":155.0},{"x":2.717,"y":111.0},{"x":2.717,"y":156.0},{"x":2.717,"y":156.0},{"x":2.759,"y":123.0},{"x":2.759,"y":158.0},{"x":2.797,"y":120.0},{"x":2.797,"y":120.0},{"x":2.797,"y":168.0},{"x":2.84,"y":134.0},{"x":2.84,"y":171.0},{"x":2.84,"y":171.0},{"x":2.875,"y":127.0},{"x":2.875,"y":171.0},{"x":2.909,"y":139.0},{"x":2.909,"y":179.0},{"x":2.909,"y":179.0},{"x":2.947,"y":127.0},{"x":2.947,"y":167.0},{"x":2.947,"y":167.0},{"x":2.982,"y":135.0},{"x":2.982,"y":173.0},{"x":2.982,"y":173.0},{"x":2.982,"y":173.0}],"backgroundColor":"rgba(41,128,185,0.7)","pointRadius":3,"showLine":true,"borderColor":"rgba(41,128,185,0.5)","borderWidth":1}]},"options":{"responsive":true,"maintainAspectRatio":true,"aspectRatio":2.2,"plugins":{"title":{"display":true,"text":"Consumption rate vs distance flown"},"legend":{"position":"bottom"}},"scales":{"x":{"title":{"display":true,"text":"Cumulative distance flown  [km]"},"grid":{"color":"rgba(0,0,0,0.08)"}},"y":{"title":{"display":true,"text":"Consumption  [mAh/km]"},"min":0,"grid":{"color":"rgba(0,0,0,0.08)"}}}}}
```

Sunaudojimo intensyvumas mAh/km, skaičiuotas per slenkantį 150 m langą.
Išvykimas (kairė pusė): ~126 mAh/km su pastipramu vėju. Grįžimas (dešinė,
po apsisukimo ties ~2,5 km): ~183 mAh/km prieš tą patį vėją. Tarpas tarp ~1 ir
~2,5 km — tai 150 s tamsa, nėra matavimų.

```chart
{"type":"scatter","data":{"datasets":[{"label":"Current  [A]","data":[{"x":0.0,"y":0.3},{"x":0.006,"y":0.3},{"x":0.006,"y":0.2},{"x":0.006,"y":0.2},{"x":0.016,"y":0.2},{"x":0.016,"y":0.5},{"x":0.02,"y":0.5},{"x":0.02,"y":0.8},{"x":0.02,"y":0.8},{"x":0.021,"y":0.8},{"x":0.021,"y":3.0},{"x":0.031,"y":3.0},{"x":0.031,"y":3.0},{"x":0.031,"y":4.4},{"x":0.056,"y":4.4},{"x":0.056,"y":3.5},{"x":0.056,"y":3.5},{"x":0.094,"y":3.5},{"x":0.094,"y":6.2},{"x":0.148,"y":6.2},{"x":0.148,"y":5.9},{"x":0.148,"y":5.9},{"x":0.202,"y":5.9},{"x":0.202,"y":6.0},{"x":0.202,"y":6.0},{"x":0.27,"y":6.0},{"x":0.27,"y":5.7},{"x":0.333,"y":5.7},{"x":0.333,"y":5.7},{"x":0.333,"y":7.2},{"x":0.408,"y":7.2},{"x":0.408,"y":7.9},{"x":0.408,"y":7.9},{"x":0.408,"y":7.9},{"x":0.48,"y":7.9},{"x":0.48,"y":7.9},{"x":0.48,"y":9.5},{"x":0.48,"y":9.5},{"x":0.603,"y":9.5},{"x":0.603,"y":9.5},{"x":0.603,"y":10.0},{"x":0.603,"y":10.0},{"x":0.714,"y":10.0},{"x":0.714,"y":9.0},{"x":0.714,"y":9.0},{"x":0.714,"y":9.0},{"x":0.714,"y":9.0},{"x":0.824,"y":9.0},{"x":0.824,"y":9.0},{"x":0.824,"y":9.2},{"x":0.824,"y":9.2},{"x":0.955,"y":9.2},{"x":0.955,"y":9.2},{"x":0.955,"y":9.2},{"x":0.955,"y":9.2},{"x":0.955,"y":9.2},{"x":2.001,"y":6.7},{"x":2.05,"y":6.7},{"x":2.05,"y":6.7},{"x":2.05,"y":6.7},{"x":2.099,"y":6.7},{"x":2.099,"y":7.7},{"x":2.148,"y":7.7},{"x":2.148,"y":6.7},{"x":2.148,"y":6.7},{"x":2.194,"y":6.7},{"x":2.194,"y":7.3},{"x":2.194,"y":7.3},{"x":2.241,"y":7.3},{"x":2.241,"y":6.3},{"x":2.291,"y":6.3},{"x":2.291,"y":8.2},{"x":2.291,"y":8.2},{"x":2.346,"y":8.2},{"x":2.346,"y":8.2},{"x":2.346,"y":6.7},{"x":2.397,"y":6.7},{"x":2.397,"y":8.0},{"x":2.397,"y":8.0},{"x":2.441,"y":8.0},{"x":2.441,"y":8.4},{"x":2.488,"y":8.4},{"x":2.488,"y":8.4},{"x":2.488,"y":8.7},{"x":2.537,"y":8.7},{"x":2.537,"y":7.6},{"x":2.537,"y":7.6},{"x":2.588,"y":7.6},{"x":2.588,"y":6.0},{"x":2.588,"y":6.0},{"x":2.63,"y":6.0},{"x":2.63,"y":5.5},{"x":2.676,"y":5.5},{"x":2.676,"y":5.7},{"x":2.676,"y":5.7},{"x":2.717,"y":5.7},{"x":2.717,"y":5.8},{"x":2.717,"y":5.8},{"x":2.759,"y":5.8},{"x":2.759,"y":6.8},{"x":2.797,"y":6.8},{"x":2.797,"y":6.8},{"x":2.797,"y":6.4},{"x":2.84,"y":6.4},{"x":2.84,"y":6.2},{"x":2.84,"y":6.2},{"x":2.875,"y":6.2},{"x":2.875,"y":5.8},{"x":2.909,"y":5.8},{"x":2.909,"y":7.1},{"x":2.909,"y":7.1},{"x":2.947,"y":7.1},{"x":2.947,"y":4.9},{"x":2.947,"y":4.9},{"x":2.982,"y":4.9},{"x":2.982,"y":7.2},{"x":2.982,"y":7.2},{"x":2.982,"y":7.2}],"backgroundColor":"rgba(39,174,96,0.7)","pointRadius":3,"showLine":true,"borderColor":"rgba(39,174,96,0.5)","borderWidth":1}]},"options":{"responsive":true,"maintainAspectRatio":true,"aspectRatio":2.2,"plugins":{"title":{"display":true,"text":"Current draw vs distance flown"},"legend":{"position":"bottom"}},"scales":{"x":{"title":{"display":true,"text":"Cumulative distance flown  [km]"},"grid":{"color":"rgba(0,0,0,0.08)"}},"y":{"title":{"display":true,"text":"Current  [A]"},"min":0,"grid":{"color":"rgba(0,0,0,0.08)"}}}}}
```

Srovė skrydžio metu. Grįžtamuoju maršrutu didesnė ir labiau išsibarsčiusi —
dronas sunkiau dirba prieš vėją.

```chart
{"type":"scatter","data":{"datasets":[{"label":"1RSS  [dBm]","data":[{"x":0.0,"y":-36.0},{"x":6.0,"y":-38.0},{"x":6.0,"y":-36.0},{"x":6.0,"y":-38.0},{"x":15.0,"y":-40.0},{"x":15.0,"y":-40.0},{"x":19.0,"y":-42.0},{"x":19.0,"y":-42.0},{"x":19.0,"y":-34.0},{"x":20.0,"y":-50.0},{"x":20.0,"y":-57.0},{"x":13.0,"y":-67.0},{"x":13.0,"y":-75.0},{"x":13.0,"y":-77.0},{"x":23.0,"y":-81.0},{"x":23.0,"y":-75.0},{"x":23.0,"y":-84.0},{"x":61.0,"y":-84.0},{"x":61.0,"y":-92.0},{"x":114.0,"y":-85.0},{"x":114.0,"y":-82.0},{"x":114.0,"y":-82.0},{"x":168.0,"y":-83.0},{"x":168.0,"y":-83.0},{"x":168.0,"y":-84.0},{"x":235.0,"y":-85.0},{"x":235.0,"y":-87.0},{"x":298.0,"y":-89.0},{"x":298.0,"y":-90.0},{"x":298.0,"y":-89.0},{"x":373.0,"y":-89.0},{"x":373.0,"y":-91.0},{"x":373.0,"y":-93.0},{"x":373.0,"y":-98.0},{"x":445.0,"y":-94.0},{"x":445.0,"y":-92.0},{"x":445.0,"y":-91.0},{"x":445.0,"y":-91.0},{"x":568.0,"y":-93.0},{"x":568.0,"y":-90.0},{"x":568.0,"y":-91.0},{"x":568.0,"y":-92.0},{"x":679.0,"y":-91.0},{"x":679.0,"y":-92.0},{"x":679.0,"y":-92.0},{"x":679.0,"y":-90.0},{"x":679.0,"y":-93.0},{"x":789.0,"y":-90.0},{"x":789.0,"y":-90.0},{"x":789.0,"y":-93.0},{"x":789.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":1946.0,"y":-94.0},{"x":1903.0,"y":-93.0},{"x":1903.0,"y":-92.0},{"x":1903.0,"y":-92.0},{"x":1857.0,"y":-92.0},{"x":1857.0,"y":-92.0},{"x":1810.0,"y":-91.0},{"x":1810.0,"y":-91.0},{"x":1810.0,"y":-91.0},{"x":1765.0,"y":-91.0},{"x":1765.0,"y":-92.0},{"x":1765.0,"y":-92.0},{"x":1720.0,"y":-91.0},{"x":1720.0,"y":-90.0},{"x":1672.0,"y":-91.0},{"x":1672.0,"y":-90.0},{"x":1672.0,"y":-90.0},{"x":1618.0,"y":-93.0},{"x":1618.0,"y":-92.0},{"x":1618.0,"y":-92.0},{"x":1576.0,"y":-91.0},{"x":1576.0,"y":-91.0},{"x":1576.0,"y":-91.0},{"x":1537.0,"y":-90.0},{"x":1537.0,"y":-90.0},{"x":1497.0,"y":-90.0},{"x":1497.0,"y":-90.0},{"x":1497.0,"y":-90.0},{"x":1455.0,"y":-90.0},{"x":1455.0,"y":-90.0},{"x":1455.0,"y":-89.0},{"x":1409.0,"y":-89.0},{"x":1409.0,"y":-89.0},{"x":1409.0,"y":-89.0},{"x":1371.0,"y":-89.0},{"x":1371.0,"y":-88.0},{"x":1330.0,"y":-88.0},{"x":1330.0,"y":-89.0},{"x":1330.0,"y":-89.0},{"x":1290.0,"y":-88.0},{"x":1290.0,"y":-88.0},{"x":1290.0,"y":-87.0},{"x":1250.0,"y":-87.0},{"x":1250.0,"y":-87.0},{"x":1214.0,"y":-87.0},{"x":1214.0,"y":-87.0},{"x":1214.0,"y":-87.0},{"x":1172.0,"y":-87.0},{"x":1172.0,"y":-87.0},{"x":1172.0,"y":-87.0},{"x":1138.0,"y":-86.0},{"x":1138.0,"y":-87.0},{"x":1104.0,"y":-87.0},{"x":1104.0,"y":-87.0},{"x":1104.0,"y":-87.0},{"x":1068.0,"y":-87.0},{"x":1068.0,"y":-89.0},{"x":1068.0,"y":-92.0},{"x":1033.0,"y":-90.0},{"x":1033.0,"y":-93.0},{"x":1033.0,"y":-93.0},{"x":1033.0,"y":-93.0}],"backgroundColor":"rgba(41,128,185,0.65)","pointRadius":3,"showLine":true,"borderColor":"rgba(41,128,185,0.4)","borderWidth":1,"yAxisID":"y"},{"label":"Link Quality  [%]","data":[{"x":0.0,"y":100.0},{"x":6.0,"y":100.0},{"x":6.0,"y":100.0},{"x":6.0,"y":100.0},{"x":15.0,"y":100.0},{"x":15.0,"y":100.0},{"x":19.0,"y":100.0},{"x":19.0,"y":100.0},{"x":19.0,"y":100.0},{"x":20.0,"y":99.0},{"x":20.0,"y":100.0},{"x":13.0,"y":99.0},{"x":13.0,"y":100.0},{"x":13.0,"y":100.0},{"x":23.0,"y":99.0},{"x":23.0,"y":100.0},{"x":23.0,"y":100.0},{"x":61.0,"y":100.0},{"x":61.0,"y":100.0},{"x":114.0,"y":100.0},{"x":114.0,"y":100.0},{"x":114.0,"y":99.0},{"x":168.0,"y":99.0},{"x":168.0,"y":100.0},{"x":168.0,"y":100.0},{"x":235.0,"y":100.0},{"x":235.0,"y":100.0},{"x":298.0,"y":100.0},{"x":298.0,"y":99.0},{"x":298.0,"y":100.0},{"x":373.0,"y":100.0},{"x":373.0,"y":99.0},{"x":373.0,"y":100.0},{"x":373.0,"y":100.0},{"x":445.0,"y":100.0},{"x":445.0,"y":100.0},{"x":445.0,"y":100.0},{"x":445.0,"y":100.0},{"x":568.0,"y":100.0},{"x":568.0,"y":100.0},{"x":568.0,"y":99.0},{"x":568.0,"y":100.0},{"x":679.0,"y":100.0},{"x":679.0,"y":99.0},{"x":679.0,"y":100.0},{"x":679.0,"y":99.0},{"x":679.0,"y":100.0},{"x":789.0,"y":100.0},{"x":789.0,"y":100.0},{"x":789.0,"y":100.0},{"x":789.0,"y":98.0},{"x":920.0,"y":98.0},{"x":920.0,"y":98.0},{"x":920.0,"y":100.0},{"x":920.0,"y":100.0},{"x":920.0,"y":100.0},{"x":1946.0,"y":100.0},{"x":1903.0,"y":100.0},{"x":1903.0,"y":99.0},{"x":1903.0,"y":100.0},{"x":1857.0,"y":100.0},{"x":1857.0,"y":100.0},{"x":1810.0,"y":100.0},{"x":1810.0,"y":100.0},{"x":1810.0,"y":100.0},{"x":1765.0,"y":100.0},{"x":1765.0,"y":100.0},{"x":1765.0,"y":100.0},{"x":1720.0,"y":100.0},{"x":1720.0,"y":100.0},{"x":1672.0,"y":100.0},{"x":1672.0,"y":97.0},{"x":1672.0,"y":100.0},{"x":1618.0,"y":100.0},{"x":1618.0,"y":100.0},{"x":1618.0,"y":99.0},{"x":1576.0,"y":99.0},{"x":1576.0,"y":100.0},{"x":1576.0,"y":100.0},{"x":1537.0,"y":100.0},{"x":1537.0,"y":100.0},{"x":1497.0,"y":100.0},{"x":1497.0,"y":100.0},{"x":1497.0,"y":100.0},{"x":1455.0,"y":100.0},{"x":1455.0,"y":100.0},{"x":1455.0,"y":100.0},{"x":1409.0,"y":100.0},{"x":1409.0,"y":99.0},{"x":1409.0,"y":100.0},{"x":1371.0,"y":100.0},{"x":1371.0,"y":100.0},{"x":1330.0,"y":98.0},{"x":1330.0,"y":98.0},{"x":1330.0,"y":100.0},{"x":1290.0,"y":100.0},{"x":1290.0,"y":100.0},{"x":1290.0,"y":100.0},{"x":1250.0,"y":100.0},{"x":1250.0,"y":99.0},{"x":1214.0,"y":100.0},{"x":1214.0,"y":100.0},{"x":1214.0,"y":100.0},{"x":1172.0,"y":100.0},{"x":1172.0,"y":100.0},{"x":1172.0,"y":100.0},{"x":1138.0,"y":100.0},{"x":1138.0,"y":100.0},{"x":1104.0,"y":100.0},{"x":1104.0,"y":100.0},{"x":1104.0,"y":99.0},{"x":1068.0,"y":100.0},{"x":1068.0,"y":100.0},{"x":1068.0,"y":100.0},{"x":1033.0,"y":100.0},{"x":1033.0,"y":100.0},{"x":1033.0,"y":100.0},{"x":1033.0,"y":100.0}],"backgroundColor":"rgba(39,174,96,0.5)","pointRadius":3,"showLine":true,"borderColor":"rgba(39,174,96,0.3)","borderWidth":1,"yAxisID":"y2"}]},"options":{"responsive":true,"maintainAspectRatio":true,"aspectRatio":2.2,"plugins":{"title":{"display":true,"text":"1RSS (dBm) and link quality vs distance from home"},"legend":{"position":"bottom"}},"scales":{"x":{"title":{"display":true,"text":"Distance from home  [m]"},"grid":{"color":"rgba(0,0,0,0.08)"}},"y":{"title":{"display":true,"text":"1RSS  [dBm]"},"position":"left","grid":{"color":"rgba(0,0,0,0.08)"}},"y2":{"title":{"display":true,"text":"Link Quality  [%]"},"position":"right","min":0,"max":110,"grid":{"drawOnChartArea":false}}}}}
```

1RSS ir ryšio kokybė pagal atstumą nuo namų. RSSI jau buvo −84 dBm ties 60 m
nuo starto — 48 dB žemiau pradžios reikšmės. RQly rodė 100% iki visiško
ryšio nutrūkimo. Jokio RF aliarmo nebuvo sujungta.


Baterija buvo LAVA II 680 mAh 3S LiHV. Skrydžio valdiklis ją buvo sukonfigūravęs
kaip 821 mAh — 20% per daug, ir kiekvienas SoC rodmuo buvo per optimistiškas.
Baterijos SoC rodė 85% likusių, kai ryšys pirmą kartą dingo.

Išvykimas: 126 mAh/km. Grįžimas: 183 mAh/km. Telemetrijos tamsa iš viso:
171 s iš 350 s. 49% skrydžio, radijui nieko nematant.

![Paskutinis kadras prieš vandenį.](lastframe.jpg)

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
