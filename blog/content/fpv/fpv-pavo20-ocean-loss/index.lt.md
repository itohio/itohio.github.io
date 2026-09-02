---
title: "Saulėlydžio nardymas prie Olando Kepurės: už ką mėgstu telemetriją mano radijuje"
date: 2026-08-28
description: "Pavo20 Pro II praradimo analizė: kaip saulėlydžio skrydis prie Olando Kepurės baigėsi neplanuotu nardymu, ką užfiksavo telemetrija ir kokius pakeitimus po to darau."
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
keywords: ["fpv baterijos perspėjimai", "edgetx telemetrija", "gps rescue", "negrįžimo taškas", "ilgo nuotolio fpv", "pavo20 pro ii praradimas", "olando kepurė", "karkle pakrantė", "dviejų spindulių interferencija"]
thumbnail: "landnow.jpg"
---

![1,04 km nuo namų. OSD rodo LAND NOW. 2,03 V vienai celei. Leidžiasi.](landnow.jpg)

Štai Pavo20 Pro II: 1,04 km nuo namų, baterija 2,03 V vienai celei, OSD rodo
LAND NOW, o priekyje jūra. Natūralu paklausti, kaip aš čia atsidūriau.

Štai kaip.

## Sprendimas

Prieš kelias savaites, per stiprų vėją atvirame lauke, nuskridau 2 km tolyn
ir atgal, o nusileidau turėdamas 20 % likutį. Ryšys laikė, baterija laikė,
dronas grįžo. Tas pats dronas, ta pati baterija, šešios minutės kruizinio
skridimo vidutiniame vėjyje, ir įtampa nė karto nenukrito žemiau 3,56 V.
Norėjau pamatyti, koks yra ELRS nuotolis virš atviro vandens, kai horizontas
visiškai laisvas. Jokių kliūčių, jokios turbulencijos, tik Baltija prie Olando Kepurės
sutemose. Atrodė kaip lengvesnė versija to, ką jau buvau padaręs.

Nebuvo.

## Skrydis

Atkarpa tolyn buvo lengva. 2,47 km, 66 km/h, 11 V baterijos įtampa. Horizontas
lygus, ryšys švarus, baterijos rodmuo beveik nekrito. Pūtė pavėjys, kurio kaip
pavėjo neįvertinau. Įvertinau kaip geras sąlygas.

![2,47 km nuo starto, 66 km/h, 11,0 V.](outbound.jpg)

Apsisukau. Pakilau šiek tiek aukščiau, kad geriau matyčiau. Greitis nukrito.
Baterijos rodmuo ėmė kristi greičiau. Atkarpa atgal, tas pats oro greitis prieš
tą patį vėją, kainavo 183 mAh/km, tai yra 1,45 karto daugiau už kilometrą nei
kelias tolyn. Pavėjys dabar yra priešvėjis, ir kiekvienas metras namo kainuoja
pusantro karto daugiau, nei kainavo nuskristi.

![Apsisukimas. 1,89 km, 10,9 V. Greitis jau mažesnis.](turning.jpg)

## Telemetrijos tamsa

Telemetrija dingo ties t = 79 s, aukštis 0 m, maždaug 920 m nuo starto. Negrįžo
150 sekundžių. Per tą laiką baterija nukrito žemiau 3,8 V vienai celei, tada
žemiau 3,6 V, tada žemiau 3,5 V. Radijas visą tą laiką tylėjo. EdgeTX loginiai
jungikliai, kurių šaltinis yra telemetrijos jutiklis, tamsos metu priverstinai
laikomi FALSE. Įtampos slenksčio jungiklis negali suveikti, kol ryšio nėra.

Telemetrija atsistatė ties t = 230 s, kai dronas jau buvo pakilęs į 75 m.
Priežastis yra dviejų spindulių interferencija virš jūros esant beveik
slystančiam kritimo kampui. Jūros paviršius yra beveik idealus radijo bangų atspindėtojas, o
mažame aukštyje tiesioginė ir atspindėta banga atkeliauja beveik vienodos
amplitudės ir priešingų fazių. Pakilus aukščiau ta fazių kompensacija suyra. 0 m: tyla.
75 m: ryšys atgal.

![1,33 km. LOW BATTERY ekrane. Radijas 150 sekundžių nesakė nieko.](lowbat.jpg)

Kai radijas vėl ką nors pamatė, valdiklis rodė, kad sunaudota 555 mAh ir liko 34 % SoC. Realioje 680 mAh baterijoje tai yra apie 231 mAh. Negrįžimo taškas,
skaičiuojant pagal išmatuotas grįžimo sąnaudas su 10 % atsarga, buvo 1147 m nuo
namų. Dronas buvo 1946 m nuo namų, tai yra jau 799 m už jo.

Pranešimas `rth` suveikė ties t = 229,8 s. Teisingai ta prasme, kad suveikė. Be
jokios vertės ta prasme, kad suveikė 150 sekundžių po to, kai sprendimą jau buvo
priėmusi fizika.

## Ką sako telemetrija

```chart
{"type":"scatter","data":{"datasets":[{"label":"Telemetrija veikia","data":[{"x":0.0,"y":0.0},{"x":6.0,"y":1.0},{"x":6.0,"y":1.0},{"x":6.0,"y":1.0},{"x":12.0,"y":9.0},{"x":12.0,"y":9.0},{"x":15.0,"y":12.0},{"x":15.0,"y":12.0},{"x":15.0,"y":12.0},{"x":16.0,"y":11.0},{"x":16.0,"y":11.0},{"x":7.0,"y":11.0},{"x":7.0,"y":11.0},{"x":7.0,"y":11.0},{"x":-18.0,"y":14.0},{"x":-18.0,"y":14.0},{"x":-18.0,"y":14.0},{"x":-54.0,"y":28.0},{"x":-54.0,"y":28.0},{"x":-105.0,"y":46.0},{"x":-105.0,"y":46.0},{"x":-105.0,"y":46.0},{"x":-157.0,"y":61.0},{"x":-157.0,"y":61.0},{"x":-157.0,"y":61.0},{"x":-222.0,"y":79.0},{"x":-222.0,"y":79.0},{"x":-282.0,"y":96.0},{"x":-282.0,"y":96.0},{"x":-282.0,"y":96.0},{"x":-355.0,"y":113.0},{"x":-355.0,"y":113.0},{"x":-355.0,"y":113.0},{"x":-355.0,"y":113.0},{"x":-425.0,"y":133.0},{"x":-425.0,"y":133.0},{"x":-425.0,"y":133.0},{"x":-425.0,"y":133.0},{"x":-544.0,"y":163.0},{"x":-544.0,"y":163.0},{"x":-544.0,"y":163.0},{"x":-544.0,"y":163.0},{"x":-652.0,"y":189.0},{"x":-652.0,"y":189.0},{"x":-652.0,"y":189.0},{"x":-652.0,"y":189.0},{"x":-652.0,"y":189.0},{"x":-758.0,"y":216.0},{"x":-758.0,"y":216.0},{"x":-758.0,"y":216.0},{"x":-758.0,"y":216.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-1818.0,"y":694.0},{"x":-1770.0,"y":699.0},{"x":-1770.0,"y":699.0},{"x":-1770.0,"y":699.0},{"x":-1721.0,"y":696.0},{"x":-1721.0,"y":696.0},{"x":-1673.0,"y":692.0},{"x":-1673.0,"y":692.0},{"x":-1673.0,"y":692.0},{"x":-1626.0,"y":685.0},{"x":-1626.0,"y":685.0},{"x":-1626.0,"y":685.0},{"x":-1580.0,"y":679.0},{"x":-1580.0,"y":679.0},{"x":-1531.0,"y":673.0},{"x":-1531.0,"y":673.0},{"x":-1531.0,"y":673.0},{"x":-1476.0,"y":664.0},{"x":-1476.0,"y":664.0},{"x":-1476.0,"y":664.0},{"x":-1426.0,"y":670.0},{"x":-1426.0,"y":670.0},{"x":-1426.0,"y":670.0},{"x":-1381.0,"y":674.0},{"x":-1381.0,"y":674.0},{"x":-1335.0,"y":679.0},{"x":-1335.0,"y":679.0},{"x":-1335.0,"y":679.0},{"x":-1286.0,"y":680.0},{"x":-1286.0,"y":680.0},{"x":-1286.0,"y":680.0},{"x":-1235.0,"y":678.0},{"x":-1235.0,"y":678.0},{"x":-1235.0,"y":678.0},{"x":-1193.0,"y":676.0},{"x":-1193.0,"y":676.0},{"x":-1147.0,"y":672.0},{"x":-1147.0,"y":672.0},{"x":-1147.0,"y":672.0},{"x":-1107.0,"y":662.0},{"x":-1107.0,"y":662.0},{"x":-1107.0,"y":662.0},{"x":-1067.0,"y":651.0},{"x":-1067.0,"y":651.0},{"x":-1031.0,"y":641.0},{"x":-1031.0,"y":641.0},{"x":-1031.0,"y":641.0},{"x":-989.0,"y":628.0},{"x":-989.0,"y":628.0},{"x":-989.0,"y":628.0},{"x":-956.0,"y":618.0},{"x":-956.0,"y":618.0},{"x":-924.0,"y":604.0},{"x":-924.0,"y":604.0},{"x":-924.0,"y":604.0},{"x":-890.0,"y":589.0},{"x":-890.0,"y":589.0},{"x":-890.0,"y":589.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0}],"backgroundColor":"rgba(41,128,185,0.7)","pointRadius":3,"showLine":true,"borderColor":"rgba(41,128,185,0.5)","borderWidth":1},{"label":"Tamsa / skaičiuojamoji navigacija","data":[{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-886.0,"y":248.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1275.0,"y":346.0},{"x":-1818.0,"y":694.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0},{"x":-858.0,"y":576.0}],"backgroundColor":"rgba(192,57,43,0.5)","pointRadius":3,"showLine":true,"borderColor":"rgba(192,57,43,0.4)","borderWidth":1}]},"options":{"responsive":true,"maintainAspectRatio":true,"aspectRatio":2.2,"plugins":{"title":{"display":true,"text":"Skrydžio trajektorija (GPS + skaičiuojamoji navigacija)"},"legend":{"position":"bottom"}},"scales":{"x":{"title":{"display":true,"text":"Rytai ← 0 → Vakarai [m]"},"grid":{"color":"rgba(0,0,0,0.08)"}},"y":{"title":{"display":true,"text":"Pietūs ← 0 → Šiaurė [m]"},"grid":{"color":"rgba(0,0,0,0.08)"}}}}}
```

Skrydžio trajektorija. Mėlyna: GPS ryšys yra. Raudona: telemetrijos tamsa,
padėtis atkurta skaičiuojamąja navigacija.

```chart
{"type":"scatter","data":{"datasets":[{"label":"mAh/km (slenkantis 150 m langas)","data":[{"x":0.021,"y":47.0},{"x":0.021,"y":281.0},{"x":0.031,"y":194.0},{"x":0.031,"y":194.0},{"x":0.031,"y":356.0},{"x":0.056,"y":197.0},{"x":0.056,"y":287.0},{"x":0.056,"y":287.0},{"x":0.094,"y":170.0},{"x":0.094,"y":223.0},{"x":0.148,"y":142.0},{"x":0.148,"y":182.0},{"x":0.148,"y":182.0},{"x":0.202,"y":93.0},{"x":0.202,"y":134.0},{"x":0.202,"y":134.0},{"x":0.27,"y":74.0},{"x":0.27,"y":114.0},{"x":0.333,"y":76.0},{"x":0.333,"y":76.0},{"x":0.333,"y":119.0},{"x":0.408,"y":73.0},{"x":0.408,"y":122.0},{"x":0.408,"y":122.0},{"x":0.408,"y":122.0},{"x":0.48,"y":86.0},{"x":0.48,"y":86.0},{"x":0.48,"y":157.0},{"x":0.48,"y":157.0},{"x":0.603,"y":77.0},{"x":0.603,"y":77.0},{"x":0.603,"y":159.0},{"x":0.603,"y":159.0},{"x":0.714,"y":68.0},{"x":0.714,"y":128.0},{"x":0.714,"y":128.0},{"x":0.714,"y":128.0},{"x":0.714,"y":128.0},{"x":0.824,"y":63.0},{"x":0.824,"y":63.0},{"x":0.824,"y":154.0},{"x":0.824,"y":154.0},{"x":0.955,"y":83.0},{"x":0.955,"y":83.0},{"x":0.955,"y":207.0},{"x":0.955,"y":207.0},{"x":0.955,"y":207.0},{"x":2.194,"y":124.0},{"x":2.194,"y":171.0},{"x":2.194,"y":171.0},{"x":2.241,"y":131.0},{"x":2.241,"y":173.0},{"x":2.291,"y":131.0},{"x":2.291,"y":172.0},{"x":2.291,"y":172.0},{"x":2.346,"y":105.0},{"x":2.346,"y":105.0},{"x":2.346,"y":171.0},{"x":2.397,"y":116.0},{"x":2.397,"y":161.0},{"x":2.397,"y":161.0},{"x":2.441,"y":113.0},{"x":2.441,"y":172.0},{"x":2.488,"y":131.0},{"x":2.488,"y":131.0},{"x":2.488,"y":177.0},{"x":2.537,"y":131.0},{"x":2.537,"y":178.0},{"x":2.537,"y":178.0},{"x":2.588,"y":141.0},{"x":2.588,"y":178.0},{"x":2.588,"y":178.0},{"x":2.63,"y":132.0},{"x":2.63,"y":169.0},{"x":2.676,"y":123.0},{"x":2.676,"y":155.0},{"x":2.676,"y":155.0},{"x":2.717,"y":111.0},{"x":2.717,"y":156.0},{"x":2.717,"y":156.0},{"x":2.759,"y":123.0},{"x":2.759,"y":158.0},{"x":2.797,"y":120.0},{"x":2.797,"y":120.0},{"x":2.797,"y":168.0},{"x":2.84,"y":134.0},{"x":2.84,"y":171.0},{"x":2.84,"y":171.0},{"x":2.875,"y":127.0},{"x":2.875,"y":171.0},{"x":2.909,"y":139.0},{"x":2.909,"y":179.0},{"x":2.909,"y":179.0},{"x":2.947,"y":127.0},{"x":2.947,"y":167.0},{"x":2.947,"y":167.0},{"x":2.982,"y":135.0},{"x":2.982,"y":173.0},{"x":2.982,"y":173.0},{"x":2.982,"y":173.0}],"backgroundColor":"rgba(41,128,185,0.7)","pointRadius":3,"showLine":true,"borderColor":"rgba(41,128,185,0.5)","borderWidth":1}]},"options":{"responsive":true,"maintainAspectRatio":true,"aspectRatio":2.2,"plugins":{"title":{"display":true,"text":"Sąnaudų tempas pagal nuskristą atstumą"},"legend":{"position":"bottom"}},"scales":{"x":{"title":{"display":true,"text":"Bendras nuskristas atstumas [km]"},"grid":{"color":"rgba(0,0,0,0.08)"}},"y":{"title":{"display":true,"text":"Sąnaudos [mAh/km]"},"min":0,"grid":{"color":"rgba(0,0,0,0.08)"}}}}}
```

Sąnaudų tempas mAh/km, skaičiuotas slenkančiu 150 m langu. Kelias tolyn (kairė
pusė): apie 126 mAh/km su pavėjumi. Kelias atgal (dešinė pusė, po apsisukimo
ties maždaug 2,5 km odometro): apie 183 mAh/km prieš tą patį vėją. Tarpas tarp
maždaug 1 ir 2,5 km yra ta pati 150 s tamsa: matavimų nėra, taškų nėra.

```chart
{"type":"scatter","data":{"datasets":[{"label":"Srovė [A]","data":[{"x":0.0,"y":0.3},{"x":0.006,"y":0.3},{"x":0.006,"y":0.2},{"x":0.006,"y":0.2},{"x":0.016,"y":0.2},{"x":0.016,"y":0.5},{"x":0.02,"y":0.5},{"x":0.02,"y":0.8},{"x":0.02,"y":0.8},{"x":0.021,"y":0.8},{"x":0.021,"y":3.0},{"x":0.031,"y":3.0},{"x":0.031,"y":3.0},{"x":0.031,"y":4.4},{"x":0.056,"y":4.4},{"x":0.056,"y":3.5},{"x":0.056,"y":3.5},{"x":0.094,"y":3.5},{"x":0.094,"y":6.2},{"x":0.148,"y":6.2},{"x":0.148,"y":5.9},{"x":0.148,"y":5.9},{"x":0.202,"y":5.9},{"x":0.202,"y":6.0},{"x":0.202,"y":6.0},{"x":0.27,"y":6.0},{"x":0.27,"y":5.7},{"x":0.333,"y":5.7},{"x":0.333,"y":5.7},{"x":0.333,"y":7.2},{"x":0.408,"y":7.2},{"x":0.408,"y":7.9},{"x":0.408,"y":7.9},{"x":0.408,"y":7.9},{"x":0.48,"y":7.9},{"x":0.48,"y":7.9},{"x":0.48,"y":9.5},{"x":0.48,"y":9.5},{"x":0.603,"y":9.5},{"x":0.603,"y":9.5},{"x":0.603,"y":10.0},{"x":0.603,"y":10.0},{"x":0.714,"y":10.0},{"x":0.714,"y":9.0},{"x":0.714,"y":9.0},{"x":0.714,"y":9.0},{"x":0.714,"y":9.0},{"x":0.824,"y":9.0},{"x":0.824,"y":9.0},{"x":0.824,"y":9.2},{"x":0.824,"y":9.2},{"x":0.955,"y":9.2},{"x":0.955,"y":9.2},{"x":0.955,"y":9.2},{"x":0.955,"y":9.2},{"x":0.955,"y":9.2},{"x":2.001,"y":6.7},{"x":2.05,"y":6.7},{"x":2.05,"y":6.7},{"x":2.05,"y":6.7},{"x":2.099,"y":6.7},{"x":2.099,"y":7.7},{"x":2.148,"y":7.7},{"x":2.148,"y":6.7},{"x":2.148,"y":6.7},{"x":2.194,"y":6.7},{"x":2.194,"y":7.3},{"x":2.194,"y":7.3},{"x":2.241,"y":7.3},{"x":2.241,"y":6.3},{"x":2.291,"y":6.3},{"x":2.291,"y":8.2},{"x":2.291,"y":8.2},{"x":2.346,"y":8.2},{"x":2.346,"y":8.2},{"x":2.346,"y":6.7},{"x":2.397,"y":6.7},{"x":2.397,"y":8.0},{"x":2.397,"y":8.0},{"x":2.441,"y":8.0},{"x":2.441,"y":8.4},{"x":2.488,"y":8.4},{"x":2.488,"y":8.4},{"x":2.488,"y":8.7},{"x":2.537,"y":8.7},{"x":2.537,"y":7.6},{"x":2.537,"y":7.6},{"x":2.588,"y":7.6},{"x":2.588,"y":6.0},{"x":2.588,"y":6.0},{"x":2.63,"y":6.0},{"x":2.63,"y":5.5},{"x":2.676,"y":5.5},{"x":2.676,"y":5.7},{"x":2.676,"y":5.7},{"x":2.717,"y":5.7},{"x":2.717,"y":5.8},{"x":2.717,"y":5.8},{"x":2.759,"y":5.8},{"x":2.759,"y":6.8},{"x":2.797,"y":6.8},{"x":2.797,"y":6.8},{"x":2.797,"y":6.4},{"x":2.84,"y":6.4},{"x":2.84,"y":6.2},{"x":2.84,"y":6.2},{"x":2.875,"y":6.2},{"x":2.875,"y":5.8},{"x":2.909,"y":5.8},{"x":2.909,"y":7.1},{"x":2.909,"y":7.1},{"x":2.947,"y":7.1},{"x":2.947,"y":4.9},{"x":2.947,"y":4.9},{"x":2.982,"y":4.9},{"x":2.982,"y":7.2},{"x":2.982,"y":7.2},{"x":2.982,"y":7.2}],"backgroundColor":"rgba(39,174,96,0.7)","pointRadius":3,"showLine":true,"borderColor":"rgba(39,174,96,0.5)","borderWidth":1}]},"options":{"responsive":true,"maintainAspectRatio":true,"aspectRatio":2.2,"plugins":{"title":{"display":true,"text":"Srovė pagal nuskristą atstumą"},"legend":{"position":"bottom"}},"scales":{"x":{"title":{"display":true,"text":"Bendras nuskristas atstumas [km]"},"grid":{"color":"rgba(0,0,0,0.08)"}},"y":{"title":{"display":true,"text":"Srovė [A]"},"min":0,"grid":{"color":"rgba(0,0,0,0.08)"}}}}}
```

Srovė viso skrydžio metu. Atkarpoje atgal ji didesnė ir labiau išsibarsčiusi,
nes prieš vėją dronas dirba sunkiau.

```chart
{"type":"scatter","data":{"datasets":[{"label":"1RSS [dBm]","data":[{"x":0.0,"y":-36.0},{"x":6.0,"y":-38.0},{"x":6.0,"y":-36.0},{"x":6.0,"y":-38.0},{"x":15.0,"y":-40.0},{"x":15.0,"y":-40.0},{"x":19.0,"y":-42.0},{"x":19.0,"y":-42.0},{"x":19.0,"y":-34.0},{"x":20.0,"y":-50.0},{"x":20.0,"y":-57.0},{"x":13.0,"y":-67.0},{"x":13.0,"y":-75.0},{"x":13.0,"y":-77.0},{"x":23.0,"y":-81.0},{"x":23.0,"y":-75.0},{"x":23.0,"y":-84.0},{"x":61.0,"y":-84.0},{"x":61.0,"y":-92.0},{"x":114.0,"y":-85.0},{"x":114.0,"y":-82.0},{"x":114.0,"y":-82.0},{"x":168.0,"y":-83.0},{"x":168.0,"y":-83.0},{"x":168.0,"y":-84.0},{"x":235.0,"y":-85.0},{"x":235.0,"y":-87.0},{"x":298.0,"y":-89.0},{"x":298.0,"y":-90.0},{"x":298.0,"y":-89.0},{"x":373.0,"y":-89.0},{"x":373.0,"y":-91.0},{"x":373.0,"y":-93.0},{"x":373.0,"y":-98.0},{"x":445.0,"y":-94.0},{"x":445.0,"y":-92.0},{"x":445.0,"y":-91.0},{"x":445.0,"y":-91.0},{"x":568.0,"y":-93.0},{"x":568.0,"y":-90.0},{"x":568.0,"y":-91.0},{"x":568.0,"y":-92.0},{"x":679.0,"y":-91.0},{"x":679.0,"y":-92.0},{"x":679.0,"y":-92.0},{"x":679.0,"y":-90.0},{"x":679.0,"y":-93.0},{"x":789.0,"y":-90.0},{"x":789.0,"y":-90.0},{"x":789.0,"y":-93.0},{"x":789.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":920.0,"y":-91.0},{"x":1946.0,"y":-94.0},{"x":1903.0,"y":-93.0},{"x":1903.0,"y":-92.0},{"x":1903.0,"y":-92.0},{"x":1857.0,"y":-92.0},{"x":1857.0,"y":-92.0},{"x":1810.0,"y":-91.0},{"x":1810.0,"y":-91.0},{"x":1810.0,"y":-91.0},{"x":1765.0,"y":-91.0},{"x":1765.0,"y":-92.0},{"x":1765.0,"y":-92.0},{"x":1720.0,"y":-91.0},{"x":1720.0,"y":-90.0},{"x":1672.0,"y":-91.0},{"x":1672.0,"y":-90.0},{"x":1672.0,"y":-90.0},{"x":1618.0,"y":-93.0},{"x":1618.0,"y":-92.0},{"x":1618.0,"y":-92.0},{"x":1576.0,"y":-91.0},{"x":1576.0,"y":-91.0},{"x":1576.0,"y":-91.0},{"x":1537.0,"y":-90.0},{"x":1537.0,"y":-90.0},{"x":1497.0,"y":-90.0},{"x":1497.0,"y":-90.0},{"x":1497.0,"y":-90.0},{"x":1455.0,"y":-90.0},{"x":1455.0,"y":-90.0},{"x":1455.0,"y":-89.0},{"x":1409.0,"y":-89.0},{"x":1409.0,"y":-89.0},{"x":1409.0,"y":-89.0},{"x":1371.0,"y":-89.0},{"x":1371.0,"y":-88.0},{"x":1330.0,"y":-88.0},{"x":1330.0,"y":-89.0},{"x":1330.0,"y":-89.0},{"x":1290.0,"y":-88.0},{"x":1290.0,"y":-88.0},{"x":1290.0,"y":-87.0},{"x":1250.0,"y":-87.0},{"x":1250.0,"y":-87.0},{"x":1214.0,"y":-87.0},{"x":1214.0,"y":-87.0},{"x":1214.0,"y":-87.0},{"x":1172.0,"y":-87.0},{"x":1172.0,"y":-87.0},{"x":1172.0,"y":-87.0},{"x":1138.0,"y":-86.0},{"x":1138.0,"y":-87.0},{"x":1104.0,"y":-87.0},{"x":1104.0,"y":-87.0},{"x":1104.0,"y":-87.0},{"x":1068.0,"y":-87.0},{"x":1068.0,"y":-89.0},{"x":1068.0,"y":-92.0},{"x":1033.0,"y":-90.0},{"x":1033.0,"y":-93.0},{"x":1033.0,"y":-93.0},{"x":1033.0,"y":-93.0}],"backgroundColor":"rgba(41,128,185,0.65)","pointRadius":3,"showLine":true,"borderColor":"rgba(41,128,185,0.4)","borderWidth":1,"yAxisID":"y"},{"label":"Ryšio kokybė [%]","data":[{"x":0.0,"y":100.0},{"x":6.0,"y":100.0},{"x":6.0,"y":100.0},{"x":6.0,"y":100.0},{"x":15.0,"y":100.0},{"x":15.0,"y":100.0},{"x":19.0,"y":100.0},{"x":19.0,"y":100.0},{"x":19.0,"y":100.0},{"x":20.0,"y":99.0},{"x":20.0,"y":100.0},{"x":13.0,"y":99.0},{"x":13.0,"y":100.0},{"x":13.0,"y":100.0},{"x":23.0,"y":99.0},{"x":23.0,"y":100.0},{"x":23.0,"y":100.0},{"x":61.0,"y":100.0},{"x":61.0,"y":100.0},{"x":114.0,"y":100.0},{"x":114.0,"y":100.0},{"x":114.0,"y":99.0},{"x":168.0,"y":99.0},{"x":168.0,"y":100.0},{"x":168.0,"y":100.0},{"x":235.0,"y":100.0},{"x":235.0,"y":100.0},{"x":298.0,"y":100.0},{"x":298.0,"y":99.0},{"x":298.0,"y":100.0},{"x":373.0,"y":100.0},{"x":373.0,"y":99.0},{"x":373.0,"y":100.0},{"x":373.0,"y":100.0},{"x":445.0,"y":100.0},{"x":445.0,"y":100.0},{"x":445.0,"y":100.0},{"x":445.0,"y":100.0},{"x":568.0,"y":100.0},{"x":568.0,"y":100.0},{"x":568.0,"y":99.0},{"x":568.0,"y":100.0},{"x":679.0,"y":100.0},{"x":679.0,"y":99.0},{"x":679.0,"y":100.0},{"x":679.0,"y":99.0},{"x":679.0,"y":100.0},{"x":789.0,"y":100.0},{"x":789.0,"y":100.0},{"x":789.0,"y":100.0},{"x":789.0,"y":98.0},{"x":920.0,"y":98.0},{"x":920.0,"y":98.0},{"x":920.0,"y":100.0},{"x":920.0,"y":100.0},{"x":920.0,"y":100.0},{"x":1946.0,"y":100.0},{"x":1903.0,"y":100.0},{"x":1903.0,"y":99.0},{"x":1903.0,"y":100.0},{"x":1857.0,"y":100.0},{"x":1857.0,"y":100.0},{"x":1810.0,"y":100.0},{"x":1810.0,"y":100.0},{"x":1810.0,"y":100.0},{"x":1765.0,"y":100.0},{"x":1765.0,"y":100.0},{"x":1765.0,"y":100.0},{"x":1720.0,"y":100.0},{"x":1720.0,"y":100.0},{"x":1672.0,"y":100.0},{"x":1672.0,"y":97.0},{"x":1672.0,"y":100.0},{"x":1618.0,"y":100.0},{"x":1618.0,"y":100.0},{"x":1618.0,"y":99.0},{"x":1576.0,"y":99.0},{"x":1576.0,"y":100.0},{"x":1576.0,"y":100.0},{"x":1537.0,"y":100.0},{"x":1537.0,"y":100.0},{"x":1497.0,"y":100.0},{"x":1497.0,"y":100.0},{"x":1497.0,"y":100.0},{"x":1455.0,"y":100.0},{"x":1455.0,"y":100.0},{"x":1455.0,"y":100.0},{"x":1409.0,"y":100.0},{"x":1409.0,"y":99.0},{"x":1409.0,"y":100.0},{"x":1371.0,"y":100.0},{"x":1371.0,"y":100.0},{"x":1330.0,"y":98.0},{"x":1330.0,"y":98.0},{"x":1330.0,"y":100.0},{"x":1290.0,"y":100.0},{"x":1290.0,"y":100.0},{"x":1290.0,"y":100.0},{"x":1250.0,"y":100.0},{"x":1250.0,"y":99.0},{"x":1214.0,"y":100.0},{"x":1214.0,"y":100.0},{"x":1214.0,"y":100.0},{"x":1172.0,"y":100.0},{"x":1172.0,"y":100.0},{"x":1172.0,"y":100.0},{"x":1138.0,"y":100.0},{"x":1138.0,"y":100.0},{"x":1104.0,"y":100.0},{"x":1104.0,"y":100.0},{"x":1104.0,"y":99.0},{"x":1068.0,"y":100.0},{"x":1068.0,"y":100.0},{"x":1068.0,"y":100.0},{"x":1033.0,"y":100.0},{"x":1033.0,"y":100.0},{"x":1033.0,"y":100.0},{"x":1033.0,"y":100.0}],"backgroundColor":"rgba(39,174,96,0.5)","pointRadius":3,"showLine":true,"borderColor":"rgba(39,174,96,0.3)","borderWidth":1,"yAxisID":"y2"}]},"options":{"responsive":true,"maintainAspectRatio":true,"aspectRatio":2.2,"plugins":{"title":{"display":true,"text":"1RSS (dBm) ir ryšio kokybė pagal atstumą nuo namų"},"legend":{"position":"bottom"}},"scales":{"x":{"title":{"display":true,"text":"Atstumas nuo namų [m]"},"grid":{"color":"rgba(0,0,0,0.08)"}},"y":{"title":{"display":true,"text":"1RSS [dBm]"},"position":"left","grid":{"color":"rgba(0,0,0,0.08)"}},"y2":{"title":{"display":true,"text":"Ryšio kokybė [%]"},"position":"right","min":0,"max":110,"grid":{"drawOnChartArea":false}}}}}
```

1RSS ir ryšio kokybė pagal atstumą nuo namų. Ties 60 m nuo starto 1RSS jau buvo
−84 dBm, tai yra 48 dB žemiau starto reikšmės. RQly rodė 100 % iki pat visiško
ryšio nutrūkimo. RF pavojaus signalas apskritai nebuvo sukonfigūruotas. Jei būtų
buvęs, jis būtų suveikęs jau 235 m nuo starto.

Baterija buvo LAVA II 680 mAh 3S LiHV. Skrydžio valdiklyje ji buvo aprašyta kaip
821 mAh, o tai talpą padidina 20 % ir tyliai išpučia kiekvieną SoC rodmenį. Kai
ryšys dingo pirmą kartą, SoC rodė 85 % likučio.

Tolyn: 126 mAh/km. Atgal: 183 mAh/km. Iš viso telemetrijos tamsoje: 171 s iš
350 s. 49 % skrydžio radijas nematė nieko.

![Paskutinis kadras prieš vandenį.](lastframe.jpg)

## Kodėl baterija taip greitai išsikrovė

Matoma telemetrija pasakoja nuobodžią istoriją: 7,2 A tolyn, 6,8 A atgal,
beveik tas pats. Vėjas sulėtino grįžimą iki 40 km/h palyginti su 55 km/h tolyn, ir vien
tai per laiką paaiškina 1,45× mAh/km skirtumą. Nieko stebėtino.

Tamsa pasakoja kitą istoriją.

Valdiklio Capa skaitiklis integruoja srovę net tada, kai telemetrijos nėra.
Paskutinis geras rodmuo prieš pagrindinę tamsą: 156 mAh sunaudota. Pirmas geras
rodmuo po jos: 555 mAh. Tai 399 mAh per 139 sekundes. Perskaičiavus:
**vidutinė srovė 10,3 A** per tas 139 sekundes.

OSD kadras 2,47 km atstumu, jau tamsos lange, rodo 10,09 A. Kadras tai
patvirtina. Tamsa uždengė būtent tą skrydžio fazę, kurioje srovė buvo didžiausia:

- apsisukimą ties 2,47 km (galios šuolis pasisukus prieš vėją)
- kilimą iš 0 m į 75 m
- pirmąsias maždaug 70 s grįžimo beveik maksimalia galia

Tos 139 sekundės su 10,3 A vidutine srove sunaudojo 399 mAh, tai yra **59 % visos
baterijos**. Radijas visą tą laiką buvo aklas.

Palyginimas su skrydžiais kieme dabar tampa suprantamas. Esant 6 A vidurkiui
(kruizas, kybojimas ir lėtesni manevrai) 680 mAh baterijos užtenka 6,8 minutės,
o tai atitinka tuos 5–6 minučių rezultatus. Skrydis virš jūros būtų tęsęsis
5,7 minutės, jei visas būtų buvęs 7,2 A kruizas. Bet 139 sekundės su 10,3 A
ištraukė 122 mAh daugiau, nei būtų ištraukęs kruizas, ir sudegino visą papildomą
skrydžio minutę toje vienoje fazėje, kurios niekas negalėjo matyti.

Baterija buvo praktiškai tuščia dar prieš tai, kai telemetrija galėjo apie tai
pranešti.

## Ką keičiu

Kiekviena eilutė žemiau yra gedimo režimas, kurį atskleidė telemetrija.
Įgyvendinimas bus atskirame straipsnyje.

| Sąlyga | Senasis elgesys | Naujasis elgesys |
|---|---|---|
| Įtampa krisdama kerta 4,2 / 4,0 / 3,8 V vienai celei | Vienas įtampos slenksčio loginis jungiklis, suveikia vieną kartą, veikia tik įjungus baterijos mygtuką | Ištariamas skaičius („vienas“, „du“, „trys“) prie kiekvieno 0,1 V kirtimo žemiau 3,8 V; be tono (šiame radijuje pyptelėjimų garsas yra 0) |
| Baterija žemiau 3,6 V vienai celei | `lowbat` takelis per loginį jungiklį | Tas pats, bet ir iš fone veikiančio scenarijaus: pirmoji gynybos linija net per telemetrijos tamsą |
| Kelias atgal kainuoja daugiau nei 1,3× kelio tolyn už kilometrą | Perspėjimo nebuvo | Vėjo asimetrijos perspėjimas po ≥30 s tolyn ir ≥15 s atgal: „warning close {santykis}%“ |
| Savitoji galia padidėjusi, savitasis greitis normalus (vidinis gedimas: variklis, propeleris, guolis) | Perspėjimo nebuvo | „warning power {santykis}%“ vieną kartą, nusistačius kruizo bazinę reikšmę |
| Savitoji galia padidėjusi, savitasis greitis sumažėjęs (išorinė priežastis: priešvėjis, pasipriešinimas) | Perspėjimo nebuvo | „warning speed {santykis}%“ vieną kartą |
| 1RSS žemiau −92 dBm | `rssiSource: none`, RF pavojaus signalas apskritai nebuvo sukonfigūruotas. RQly rodė 100 % iki visiško praradimo | `siglow` ir dBm reikšmė; `sigcrt` ties −100 dBm. RSSI yra nuolydis, LQ yra skardis |
| 1RSS blogėja, aukštis mažiau nei 25 m | Nieko | „tolow“, tai yra kilk. Išmatuota: ryšys nutrūko ties 0 m, atsistatė ties 75 m dėl dviejų spindulių interferencijos virš vandens |
| Telemetrijos nėra ilgiau nei 4 s | Loginiai jungikliai priverstinai FALSE, visi baterijos perspėjimai nutyla | Fone veikiantis Lua scenarijus toliau skaičiuoja atstumą ir SoC skaičiuojamąja navigacija ir, atsistačius ryšiui, būseną paskelbia iš naujo |
| Artėja negrįžimo taškas (skrendant tolyn, veikiant GPS) | Nieko | „close {likę metrai}“ skaičiuojant žemyn, kol PNR − dHome < 400 m |
| Aktyvavimas (arm), kai GPS rescue neparengtas (valdiklis rodo „?“) | Palydovų skaičius lyginamas su scenarijuje įrašyta riba | Sprendžia pats valdiklis: kai `numSat < gps_rescue_min_sats`, Betaflight prie CRSF skrydžio režimo eilutės prideda `?`. Scenarijus skaito tai tiesiogiai. |
| Vidinė varža auga daugiau, nei paaiškina iškrovos gylis (perteklius ≥ 1,4×) | Perspėjimo nebuvo | „warning bad {mΩ}“ vieną kartą, kol SoC > 25 % |

Pavo20 išmoko nardyti. Telemetrija — ne. Vienas iš tų dviejų dalykų labiau padeda
šio eksperimento nepakartoti.

Kasdienio cinewoopinio drono vis dar ieškau. Pavo20 4S ar DeepSpace Stellar25. Dar nežinau, kuriam teks sekanti plaukimo pamoka. Vat taip vat...
