---
title: "Pavo20 Pro II GPS Taisymo Bandymai: BEC Perjungimo Triukšmas 1575 MHz ir Kas Iš Tikrųjų Padėjo"
date: 2026-07-13
weight: 40
description: "Pavo20 Pro II aptinka 3 GPS palydovus ten, kur 1S burbulinis dronas aptinka 20+. TinySA matavimai, BEC harmonikų analizė, feritinių karoliukų ir ekranavimo bandymai — ir kodėl niekas iki galo neišsprendė problemos."
toc: true
categories:
  - FPV
  - Aparatinė įranga
tags:
  - fpv
  - gps
  - pavo20
  - triukšmas
  - rf
  - tinysa
  - betaflight
  - emi
  - bec
  - switching-regulator
  - gps-l1
  - electromagnetic-interference
  - whoop
keywords: ["Pavo20 Pro II GPS", "BEC GPS trukdžiai", "perjungimo reguliatorius GPS triukšmas", "GPS L1 1575 MHz dronas", "FPV whoop GPS taisymas", "TinySA FPV spektras", "elektromagnetiniai trukdžiai FPV"]
series:
  - FPV Builds
thumbnail: "pavo20-gps-module.jpg"
---

Pavo20 Pro II yra gabus 2,5 colio burbulinis dronas — kompaktiškas, galingas, su gera kamera ir nustebinančiai stabiliu vaizdo ryšiu net su tiesinėmis antena-plaktukinėmis antenomis. Norėjau jo kaip kalnų kelionių drono: tilptų į švarko kišenę, nertų tarpekliais ir turėtų GPS gelbėjimą kaip tikrą saugos tinklą. GPS modulis neįtrauktas komplekte — pridėjau pats, sulitavau ir sukonfigūravau. Tada prasidėjo problemos.

Praktiškai GPS beveik nenaudingas daugelyje skrydžio aplinkų: stebiu, kaip palydovų skaičius sustoja ties 2 ar 3, o kitas dronas tame pačiame lauke, tuo pačiu metu, prisijungia prie 20+. Tai istorija apie tai, ką radau ir kur esu dabar.

---

## Simptomas

Pirmas tos dienos skrydis. Laukas atviras, dangus giedras, pastatų nėra. Įjungiu abu dronus ir laukiu:

| Dronas | Laikas iki pirmo fiksavimo | Palydovai fiksuojant |
|--------|---------------------------|---------------------|
| 1S Matrix 3-in-1 skaitmeninis | ~90s | 20–22 |
| Pavo20 Pro II | >5min | 2–4 |

1S dronas taip pat naudoja integruotą FC/ESC/VTX plokštę — BetaFPV Matrix 3-in-1, tą pačią plokštę, kuri naudojama Meteor serijos dronuose. Jis nėra mažiau integruotas nei Pavo20. Skiriasi tai, kad jis veikia su vienu 18650 elementu, todėl BEC mažina įtampą nuo 3,7V, o ne iš kelių elementų akumuliatoriaus. Kitoks BEC darbinis taškas, kitoks harmonikų profilis, kitoks triukšmo lygis GPS juostoje.

Burbulinio drono korpuse beveik nėra vietos tarp papildomai pritvirtinto GPS modulio ir visko, kas generuoja triukšmą.

---

## Pirmas žvilgsnis: fizinė sąranka

Prieš imantis spektro analizatoriaus, atlikau akivaizdžius patikrinimus.

![GPS modulis, sumontuotas ant DJI O4 Pro kameros korpuso viršaus — matoma plytelės antena ir ekranuotas kabelis](pavo20-gps-module.jpg)
*GPS modulis ant O4 Pro kameros viršaus — vienintelė praktiškai tinkama pozicija. Keraminis plytelės antenas turi nekliudomą dangaus vaizdą viršuje. Po kamera — integruota FC/ESC/VTX plokštė su 5V BEC. Net ir esant šiam pakilimui virš steko, BEC artimojo lauko spinduliavimas vis tiek pasiekia LNA.*

GPS modulis montuojamas ant O4 Pro kameros viršaus — keraminis plytelės antenas turi turėti laisvą, nekliudomą dangaus vaizdą, todėl tai vienintelė praktiškai tinkama pozicija šiame rėme. Po kamera yra integruota FC/ESC/VTX plokštė. GPS LNA pakyla virš steko, bet ne pakankamai — BEC artimojo lauko spinduliavimas siekia toliau nei kameros aukštis teikia.

Pridėjau feritinį karoliuką ant GPS VCC linijos ir 1 µF bei 0,1 µF SMD kondensatorius lygiagrečiai prie modulio maitinimo kontaktų. Tai standartinis laidiniam žemo dažnio triukšmui skirtas sprendimas. Jis nepadarė jokio išmatuojamo poveikio palydovų skaičiui.

---

## Spektro analizė — 1S dronas vs Pavo20

Laikas iš tikrųjų išmatuoti triukšmo lygį ten, kur veikia GPS. GPS L1 juosta yra **1575,42 MHz**. Konsteliacijos signalai, pasiekiantys anteną, yra nepaprastai silpni — paprastai apie −130 dBm. Bet koks vietinis trikdis 1,5–1,6 GHz diapazone juos nustelbia.

Prie kiekvieno drono steko prijungiau TinySA su trumpa vielos antena, dronams maitinamus tik iš baterijos — varikliai nestartavo, sraigtų nebuvo. Norint atskirti FC/ESC steko triukšmą nuo VTX, pradinį Pavo20 matavimą atlikau visiškai pašalinęs VTX.

![Pavo20 Pro II be VTX — TinySA trumpa vielos antena prie steko RF triukšmo matavimui](pavo20-no-vtx.jpg)
*Pavo20 be VTX. TinySA trumpa vielos antena šalia FC/ESC steko. Be VTX — bet koks triukšmas čia kilęs tik iš FC, ESC ir GPS modulio.*

![TinySA bazinio triukšmo matuojamumas, 1,2–1,8 GHz — viskas išjungta, etaloninis matavimas prieš prijungiant Pavo20](tinysa-baseline.jpg)
*Bazinis matavimas. TinySA antena pozicijoje, viskas išjungta. Plokščia triukšmo grinda apie −105 dBm visame 1,2–1,8 GHz diapazone — tai etalonas.*

1S drono matavimas tame pačiame diapazone buvo praktiškai lygus — triukšmas ties baziniu lygiu arba žemiau jo, nieko verto ekrano kopijos. Pavo20 rodo visiškai kitą vaizdą:

![TinySA matavimas — Pavo20 Pro II su baterija, be VTX, 1,2–1,8 GHz — FC/ESC triukšmas aiškiai padidėjęs virš bazinio lygio](tinysa-pavo20-initial.jpg)
*Pavo20 su baterija (be VTX). Triukšmo grinda gerokai virš −105 dBm bazinio lygio. Aštrus smaigalys apie 1,34 GHz siekia −89 dBm — 16 dB virš bazinio. GPS juosta ties 1575 MHz jau pastebimai pakelta.*

Kontrastas ryškus. 1S drono GPS juostoje triukšmo grinda švari, matomas tik laukiamas atmosferinis fonas. Pavo20 rodo padidėjusią triukšmo grindą visame 1,2–1,8 GHz diapazone su keliais aiškiais smaigaliais 1,4–1,6 GHz regione.

---

## Perjungimo harmonikų problema

Pagrindinis triukšmo šaltinis čia yra ne tas, kurį dauguma žmonių įtaria. Dronas gulėjo ant stalo — jokie varikliai nesisuko, sraigtų nebuvo, skrydžio nebuvo. Variklio PWM čia nedalyvavo.

Tikrasis kaltininkas yra **5V BEC** (akumuliatoriaus eliminavimo grandinė) integruotoje FC/ESC plokštėje. BEC yra perjungimo reguliatoriai, ir kompaktiškame integruotame steke, kaip Pavo20, jie persijungia keliais MHz. Tai skamba nekenksmingai — keli MHz yra toli nuo 1575 MHz. Tačiau greito krašto perjungimo srovės sukuria harmonines ir intermoduliavimo produktus, spinduliuojamus per plačią spektro juostą. Praktiškai BEC triukšmas bjauriai išsilieja iki kelių GHz, o smaigaliai patenka į neprognozuojamus dažnius, priklausomai nuo konkretaus reguliatoriaus dizaino, PCB išdėstymo ir apkrovos.

GPS moduliui esant virš kameros, kuri pati yra virš FC/ESC steko, LNA vis tiek yra kelių centimetrų atstumu nuo BEC. Ties 1,5 GHz artimojo lauko riba (λ/2π) yra maždaug 3 cm — GPS modulis yra ties ta riba arba jos viduje. Ryšys yra artimojo lauko: jis nekeliaus maitinimo linija — jis tiesiogiai jungiamas iš PCB takelių į GPS LNA.

Taip pat patikrinau VTX kaip kintamąjį: 5,8 GHz siųstuvai gali gaminti subharmonikas ir maišymo produktus plačiame dažnių diapazone. Fiziškai pašalinus VTX iš steko, TinySA triukšmo profilis GPS juostoje nepasikeitė. VTX nėra reikšmingas veiksnys. BEC yra šaltinis.

Tai patvirtinau kitoje aplinkoje — rūsyje, kur aplinkos RF triukšmas mažesnis:

![TinySA MAX HOLD matavimas — Pavo20 Pro II, 1,2–1,8 GHz, akumuliuotas ilgą laiką rūsyje — kelios harmoninių smaigaliai matomi GPS juostos srityje](tinysa-pavo20-maxhold.jpg)
*MAX HOLD skenavimas po kelių minučių akumuliacijos rūsyje. Keli smaigaliai išsklaidyti 1,2–1,6 GHz diapazone. Smaigaliai nėra pastovaus dažnio harmonikos — jie dreifuoja ir keičiasi priklausomai nuo BEC apkrovos ir temperatūros, kas būdinga perjungimo reguliatoriaus intermoduliavimo produktams.*

Lauke, GPS antena nukreipta į atvirą dangų, matomas tikrasis GPS signalo kontekstas:

![TinySA matavimas lauke — 1,2–1,8 GHz diapazonas — GPS L1 ties 1575,42 MHz vos matomas virš Pavo20 triukšmo grindos](tinysa-outside-gps.jpg)
*Matavimas lauke, atviroje erdvėje. GPS L1 signalas ties 1575,42 MHz sukuria plačią plokščiakalniui panašią elevaciją GPS juostoje — visos konsteliacijos signalai vienu metu. Suvestinis GPS signalas yra maždaug 20 dB virš bazinės triukšmo grindos. Pavo20 matavimuose matomi BEC smaigaliai yra 10–15 dB virš tos pačios bazinės grindos — ne tokie dramatiškai absoliučiu lygiu, tačiau pakankami, kad degraduotų SNR, kurį LNA turi atkurti atskirų palydovų signalams.*

Problema ne ta, kad smaigaliai nustelbia suvestinę GPS juostą — problema ta, kad jie pakelia vietinę triukšmo grindą. Atskiri palydovų signalai, kuriuos GPS modulis turi išskirti atskirai, nepereina tokio triukšmo grindos pakilimo. LNA kovoja su pakelta bazine grinda, o ne švariu dangumi.

---

## Ką išbandžiau

### 1. Feritinis karoliukas ir dekupliavimo kondensatoriai ant GPS maitinimo

Feritinis karoliukas ant GPS VCC linijos, 1 µF ir 0,1 µF SMD kondensatoriai lygiagrečiai prie modulio maitinimo kontaktų. Veiksmingi laidiniam triukšmui maitinimo linijoje žemesniuose dažniuose. Jokio poveikio BEC spinduliuojamam RF GPS juostoje.

**Rezultatas: Palydovų skaičius nepagerėjo.**

### 2. VTX pašalinimas

VTX visiškai pašalintas iš steko — ne tik išjungtas, fiziškai pašalintas. Jei VTX subharmonikos ties 1450 MHz būtų pagrindinis šaltinis, tai turėjo parodyti aiškų pagerėjimą.

**Rezultatas: Jokio pagerėjimo.** TinySA triukšmo profilis nepasikeitė pašalinus VTX. BEC yra dominuojantis šaltinis, ne VTX.

### 3. Ekranuotas kabelis ir dekupliavimas ant GPS modulio

Standartinė GPS laido instaliacija pakeista ekranuotu subalansuotu audio kabeliu (4 laidų su pynimo ekranu). Ekranas prijungtas prie FC žemės **tik FC gale** — modulio galo ekranas paliktas neprijungtas (plūduriuojantis). Vidiniai laidai neša maitinimą (VCC ir GND) bei duomenis (RX/TX). 1 µF ir 0,1 µF kondensatoriai prijungti lygiagrečiai tiesiai ant GPS modulio maitinimo kontaktų.

**Rezultatas: Dalinis pagerėjimas.** Palydovų skaičius kartais pasiekia 8, vietoj ankstesnio maksimalaus 5. Fiksavimas vis dar nepatikimas ir kartais visiškai nepavyksta. Geriau, bet neišspręsta.

---

## Pagrindinė priežasties vertinimas

Pavo20 Pro II integruoto steko dizainas teikia pirmenybę kompaktiškumui prieš RF izoliaciją. Tai sąmoningas kompromisas 2,5 colio korpusui — tiesiog nėra vietos atskirumui, kuris padarytų skirtumą.

Trukdžių šaltinis yra **5V BEC** — integruotos FC/ESC plokštės perjungimo reguliatorius. Jis veikia keliais MHz, tačiau greito krašto perjungimo impulsai sukuria harmonikus ir intermoduliavimo produktus, besiskleidžiančius į GHz diapazoną ir patenkančius į GPS juostą. Tai patvirtinta pašalinus visus kitus kintamuosius: VTX fiziškai pašalintas, varikliai neveikia, tik stekas su baterija — triukšmo profilis nepasikeitė.

Feritiniai karoliukai veikia tik laidiniam triukšmui maitinimo linijoje — BEC spinduliuojamam RF jie neturi jokio poveikio. Fizinis atstumas yra vienintelis svertas, iš tikrųjų turintis reikšmės.

GPS modulis yra standartinis nano M10 su metaliniu ekrano dangteliu virš LNA sekcijos. Tai, kad dangtelis neapsaugo nuo šių trukdžių, yra informatyvu: BEC sujungimas yra pakankamai stiprus, kad prasiskverbtų pro modulio paties RF ekranavimą — patekdamas per maitinimo kontaktus ir tiesiogiai jungiamas į antenos angą kameros viršaus aukštyje. Dangtelis yra suprojektuotas ir sureguliuotas tolimojo lauko izoliacijai; artimojo lauko sujungimas kelių centimetrų atstumu jį apeina.

---

## Kur esu dabar

Vis dar ieškau patikimo triukšmo izoliacijos sprendimo. Ekranuotas kabelis ir dekupliavimas davė dalinį pagerėjimą; BEC artimojo lauko spinduliavimas aiškiai siekia GPS LNA net esant kameros viršaus aukštyje virš steko. Didesnis fizinis atstumas yra vienintelis likęs svertas.

Kas dar liko išbandyti:

- **Ilgesnis kabelis iki kojos**: GPS modulio išvedimas iki priekinės ar galinės kojos maksimaliai padidina atstumą nuo FC/ESC plokštės. Kameros viršus yra atskaitos taškas — kojų lygio atstumas artimojo lauko slopinimui būtų visiškai kito masto.
- **Aktyvus GPS kartojimas**: išorinė aktyvi GPS antena su sava LNA, gerai išvesta už orlaivio apvalkalo ribų, prijungta plonu koaksialiniu laidu. Galutinai atsakytų, ar tai artimumu pagrįsta problema. Visiškai per sudėtinga burbuliniams dronams, bet patvirtintų pagrindinę priežastį.

---

### Tuo pačiu metu: ELRS UFL antenos modifikacija

Nesusiję su GPS, bet atlikta tuo pačiu kabelių tvarkymo metu: pridėjau UFL jungtį ELRS imtuvo antenai. Standartinė vidinė antena Pavo20 steke tinkama artimam skrydžiui, bet ne daugiau.

Su UFL modifikacija ir išorine antena ryšys laikėsi stabiliai 1 km atstumu. Ribojantis veiksnys tokiu atstumu buvo baterijos talpa 15 m/s vėjyje — ne radijo ryšys. Ši modifikacija veikė iš pirmo karto ir davė akimirksniui pastebimą rezultatą. Naudingas kontrastas GPS darbui, kuris tokio rezultato nedavė.

---

## Saugos tinklas, kuris iš tikrųjų veikia

GPS gelbėjimas buvo visas tikslas — kalnų dronas, telpa į švarko kišenę, sugrąžina namo tarpeklyje praradus ryšį. Praktiškai jis nė karto nesėkmingai nesuveikė realiomis lauko sąlygomis.

Dėl to jau kelis kartus praradau Pavo20. Aiškiausias atvejis: variklio gedimas skrydžio metu — dronas nukrito maždaug 500 m atstumu. O4 Pro ryšys išliko. Akinėse trumpam pamačiau savo šešėlį ant žolės po krintančiu dronu — to pakako orientuotis prieš nutrūkstant vaizdui. Radau jį su buzeriu. Be ELRS lokalizavimo skriptu ir garsaus buzerio, nė vienas tų dronų nebūtų grįžęs.

GPS gelbėjimas vis dar yra tikslas. ELRS lokalizavimas ir buzeris — atsarginis variantas, kuris iš tikrųjų veikia dabar.

Tinkamo gedimo, buzerio ir lokalizavimo sąranka nusipelno atskiro straipsnio — tikriausiai tada, kai bus ką geresnio pranešti apie patį GPS. Šį straipsnį atnaujinsiu eksperimentams progresuojant.
