---
title: "Pavo20 Pro II GPS Taisymo Bandymai: BEC Perjungimo Triukšmas 1575 MHz ir Kas Iš Tikrųjų Padėjo"
date: 2026-07-13
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

1S dronas yra mažesnis. Jo elektronika, galima sakyti, dar tankiau sugromuota. GPS modulis yra tos pačios kartos. Skirtumas — steko architektūra: 1S dronas naudoja atskiras FC ir ESC plokštes, o Pavo20 turi sanglaudžiai integruotą steką, kuriame VTX, FC ir ESC — viena kompaktiška plokštė.

Burbulinio drono korpuse beveik nėra vietos tarp papildomai pritvirtinto GPS modulio ir visko, kas generuoja triukšmą.

---

## Pirmas žvilgsnis: fizinė sąranka

Prieš imantis spektro analizatoriaus, atlikau akivaizdžius patikrinimus.

![GPS modulis, sumontuotas ant DJI O4 Pro kameros korpuso viršaus — matoma plytelės antena ir ekranuotas kabelis](pavo20-gps-module.jpg)
*GPS modulis ant O4 Pro kameros viršaus. Plytelės antena (keraminis kvadratas) sumontuota ant mažos baltos platformos virš kameros korpuso. Ekranuotas kabelis eina žemyn iki FC. Tai dabartinė konfigūracija po kameros montavimo eksperimento — vis dar nepakankamas atstumas nuo BEC.*

Pradžioje pridėtas GPS modulis sėdėjo tiesiai virš integruotos FC/ESC/VTX plokštės. Antenos įžeminimo plokštuma yra PCB varis — tas pats, per kurį teka 5V BEC perjungimo srovės. Tarp GPS modulio LNA ir perjungimo reguliatoriaus nėra fizinio ekrano.

<!-- IMAGE: GPS laidų ir filtravimo bandymo nuotrauka — feritiniai karoliukai, filtravimo kondensatoriai maitinimo linijoje -->
*[TODO: Nuotrauka — GPS laidai su feritiniais karoliukais ir maitinimo linijos filtravimas]*

Pridėjau feritinių karoliukų ant GPS maitinimo linijos ir 100 µF kondensatorių prie modulio maitinimo kontaktų. Tai standartinis žemo dažnio triukšmo sprendimas. Jis nepadarė jokio išmatuojamo poveikio palydovų skaičiui.

---

## Spektro analizė — 1S dronas vs Pavo20

Laikas iš tikrųjų išmatuoti triukšmo lygį ten, kur veikia GPS. GPS L1 juosta yra **1575,42 MHz**. Konsteliacijos signalai, pasiekiantys anteną, yra nepaprastai silpni — paprastai apie −130 dBm. Bet koks vietinis trikdis 1,5–1,6 GHz diapazone juos nustelbia.

Prie kiekvieno drono steko prijungiau TinySA su trumpa vielos antena, dronams maitinamus tik iš baterijos — varikliai nestartavo, sraigtų nebuvo. Norint atskirti FC/ESC steko triukšmą nuo VTX, pradinį Pavo20 matavimą atlikau visiškai pašalinęs VTX.

![Pavo20 Pro II be VTX — TinySA trumpa vielos antena prie steko RF triukšmo matavimui](pavo20-no-vtx.jpg)
*Pavo20 be VTX. TinySA trumpa vielos antena šalia FC/ESC steko. Be VTX — bet koks triukšmas čia kilęs tik iš FC, ESC ir GPS modulio.*

![TinySA bazinio triukšmo matuojamumas, 1,2–1,8 GHz — viskas išjungta, etaloninis matavimas prieš prijungiant Pavo20](tinysa-baseline.jpg)
*Bazinis matavimas. TinySA antena pozicijoje, viskas išjungta. Plokščia triukšmo grinda apie −105 dBm visame 1,2–1,8 GHz diapazone — tai etalonas.*

<!-- IMAGE: TinySA ekrano kopija — 1S Matrix 3-in-1 skaitmeninis dronas, 1,2 GHz–1,8 GHz diapazonas, rodanti triukšmo grindą -->
*[TODO: TinySA ekrano kopija — 1S skaitmeninis dronas, 1,2–1,8 GHz diapazonas]*

![TinySA matavimas — Pavo20 Pro II su baterija, be VTX, 1,2–1,8 GHz — FC/ESC triukšmas aiškiai padidėjęs virš bazinio lygio](tinysa-pavo20-initial.jpg)
*Pavo20 su baterija (be VTX). Triukšmo grinda gerokai virš −105 dBm bazinio lygio. Aštrus smaigalys apie 1,34 GHz siekia −89 dBm — 16 dB virš bazinio. GPS juosta ties 1575 MHz jau pastebimai pakelta.*

Kontrastas ryškus. 1S drono GPS juostoje triukšmo grinda švari, matomas tik laukiamas atmosferinis fonas. Pavo20 rodo padidėjusią triukšmo grindą visame 1,2–1,8 GHz diapazone su keliais aiškiais smaigaliais 1,4–1,6 GHz regione.

---

## Perjungimo harmonikų problema

Pagrindinis triukšmo šaltinis čia yra ne tas, kurį dauguma žmonių įtaria. Dronas gulėjo ant stalo — jokie varikliai nesisuko, sraigtų nebuvo, skrydžio nebuvo. Variklio PWM čia nedalyvavo.

Tikrasis kaltininkas yra **5V BEC** (akumuliatoriaus eliminavimo grandinė) integruotoje FC/ESC plokštėje. BEC yra perjungimo reguliatoriai, ir kompaktiškame integruotame steke, kaip Pavo20, jie persijungia keliais MHz. Tai skamba nekenksmingai — keli MHz yra toli nuo 1575 MHz. Tačiau greito krašto perjungimo srovės sukuria harmonines ir intermoduliavimo produktus, spinduliuojamus per plačią spektro juostą. Praktiškai BEC triukšmas bjauriai išsilieja iki kelių GHz, o smaigaliai patenka į neprognozuojamus dažnius, priklausomai nuo konkretaus reguliatoriaus dizaino, PCB išdėstymo ir apkrovos.

Kai GPS modulis sėdi 10–15 mm tiesiai virš to BEC, ant tos pačios žemės plokštumos, ryšys yra artimojo lauko. Jis nekeliaus maitinimo linija — jis tiesiogiai spinduliuoja iš PCB takelių į GPS LNA.

Be to: **vaizdo siųstuvai** 5,8 GHz gali generuoti subharmonikas ir maišymo produktus. 5,8 GHz VTX ties 200 mW gali gaminti energiją ties 5800/4 = 1450 MHz — tiesiai GPS juostoje. VTX matavimas patvirtino, kad jį pašalinus triukšmas nepagerėjo — tai rodo, kad BEC yra pagrindinis šaltinis.

Tai patvirtinau kitoje aplinkoje — rūsyje, kur aplinkos RF triukšmas mažesnis:

![TinySA MAX HOLD matavimas — Pavo20 Pro II, 1,2–1,8 GHz, akumuliuotas ilgą laiką rūsyje — kelios harmoninių smaigaliai matomi GPS juostos srityje](tinysa-pavo20-maxhold.jpg)
*MAX HOLD skenavimas po kelių minučių akumuliacijos rūsyje. Keli smaigaliai išsklaidyti 1,2–1,6 GHz diapazone. Smaigaliai nėra pastovaus dažnio harmonikos — jie dreifuoja ir keičiasi priklausomai nuo BEC apkrovos ir temperatūros, kas būdinga perjungimo reguliatoriaus intermoduliavimo produktams.*

Lauke, GPS antena nukreipta į atvirą dangų, matomas tikrasis GPS signalo kontekstas:

![TinySA matavimas lauke — 1,2–1,8 GHz diapazonas — GPS L1 ties 1575,42 MHz vos matomas virš Pavo20 triukšmo grindos](tinysa-outside-gps.jpg)
*Matavimas lauke, atviroje erdvėje. GPS L1 signalas ties 1575,42 MHz sukuria plačią plokščiakalniui panašią elevaciją GPS juostoje — visos konsteliacijos signalai vienu metu. Suvestinis GPS signalas yra maždaug 20 dB virš bazinės triukšmo grindos. Pavo20 matavimuose matomi BEC smaigaliai yra 10–15 dB virš tos pačios bazinės grindos — ne tokie dramatiškai absoliučiu lygiu, tačiau pakankami, kad degraduotų SNR, kurį LNA turi atkurti atskirų palydovų signalams.*

Problema ne ta, kad smaigaliai nustelbia suvestinę GPS juostą — problema ta, kad jie pakelia vietinę triukšmo grindą. Atskiri palydovų signalai, kuriuos GPS modulis turi išskirti atskirai, nepereina tokio triukšmo grindos pakilimo. LNA kovoja su pakelta bazine grinda, o ne švariu dangumi.

---

## Ką išbandžiau

### 1. Feritiniai karoliukai ant GPS maitinimo

Feritiniai karoliukai ant VCC ir GND laidų į GPS modulį. Veiksmingi laidiniam triukšmui maitinimo linijoje žemesniuose dažniuose. Jokio poveikio BEC spinduliuojamam RF GPS juostoje.

**Rezultatas: Palydovų skaičius nepagerėjo.**

### 2. VTX pašalinimas

VTX visiškai pašalintas iš steko — ne tik išjungtas, fiziškai pašalintas. Jei VTX subharmonikos ties 1450 MHz būtų pagrindinis šaltinis, tai turėjo parodyti aiškų pagerėjimą.

**Rezultatas: Jokio pagerėjimo.** TinySA triukšmo profilis nepasikeitė pašalinus VTX. BEC yra dominuojantis šaltinis, ne VTX.

### 3. Ekranuotas kabelis ir dekupliavimas ant GPS modulio

Standartinė GPS laido instaliacija pakeista ekranuotu subalansuotu audio kabeliu (4 laidų su pynimo ekranu). Ekranas prijungtas prie FC žemės **tik FC gale** — modulio galo ekranas paliktas neprijungtas (plūduriuojantis). Vidiniai laidai neša maitinimą (VCC ir GND) bei duomenis (RX/TX). 1 µF ir 0,1 µF kondensatoriai prijungti lygiagrečiai tiesiai ant GPS modulio maitinimo kontaktų.

**Rezultatas: Dalinis pagerėjimas.** Palydovų skaičius kartais pasiekia 8, vietoj ankstesnio maksimalaus 5. Fiksavimas vis dar nepatikimas ir kartais visiškai nepavyksta. Geriau, bet neišspręsta.

### 4. GPS modulis ant kameros viršaus

GPS modulis perkeltas nuo steko — dabar jis sėdi ant DJI O4 Pro kameros viršaus, toliau nuo FC, ESC, BEC ir VTX. Ekranuotas kabelis driekiasi žemyn iki FC. Tikslas buvo sumažinti artimojo lauko sujungimą su BEC, pridedant fizinį atstumą tarp GPS LNA ir perjungimo reguliatoriaus.

**Rezultatas: Jokio reikšmingo pagerėjimo.** Palydovų skaičius ir fiksavimo patikimumas iš esmės nepasikeitė, palyginti su ekranuoto kabelio ir dekupliavimo rezultatu. Artimojo lauko BEC sujungimas arba vis dar pasiekia modulį per kabelį, arba BEC spinduliuojamas laukas siekia pakankamai toli, kad 2,5 colio rėmo aukštyje esantis atstumas yra nepakankamas.

---

## Pagrindinė priežasties vertinimas

Pavo20 Pro II integruoto steko dizainas teikia pirmenybę kompaktiškumui prieš RF izoliaciją. Tai sąmoningas kompromisas 2,5 colio korpusui — tiesiog nėra vietos atskirumui, kuris padarytų skirtumą.

Trukdžiai turi bent du komponentus:

1. **Spinduliuojamas RF iš 5V BEC** — integruotos FC/ESC plokštės perjungimo reguliatorius, veikiantis keliais MHz, su harmonikais ir smaigaliais, besiskleidžiančiais į GHz diapazoną. Tai pagrindinis šaltinis: buvo matomas net pašalinus VTX ir nesukant variklių.
2. **VTX subharmonikos** — 5,8 GHz siųstuvas gali gaminti smaigalius ties 5800/4 = 1450 MHz. Pasirodė, kad tai nereikšmingas veiksnys: visiškai pašalinus VTX triukšmo profilyje nebuvo jokio išmatuojamo skirtumo.

Feritiniai karoliukai veikia tik laidiniam triukšmui maitinimo linijoje — jie neturi jokio poveikio BEC spinduliuojamam RF. Fizinis atstumas tarp GPS modulio ir BEC yra vienintelis metodas, iš tikrųjų sprendžiantis 1 komponentą.

GPS modulis, naudojamas Pavo20, yra standartinis M8N/M10 variantas miniatiūrizuotame SMD pakete — virš RF LNA nėra ekrano dangtelio. Tai įprasta burbulinio GPS drono klasės dronams; prielaida, kad skrendant lauke pakanka dangaus matymo, kad būtų įveiktas pablogėjęs SNR.

Ši prielaida veikia kai kuriose skrydžio aplinkose ir visiškai neveikia kitose.

---

## Kur esu dabar

Vis dar ieškau patikimo triukšmo izoliacijos sprendimo. Ekranuotas kabelis ir dekupliavimas davė dalinį pagerėjimą; GPS modulio perkėlimas ant O4 Pro kameros viršaus (toliau nuo steko) jokio papildomo naudos nedavė. Atrodo, kad BEC artimojo lauko spinduliavimas siekia toliau, nei leidžia 2,5 colio rėmo fizinis atstumas.

Kas dar liko išbandyti:

- **Dar ilgesnis kabelis**: GPS modulio išvedimas ant prailginto kabelio iki priekinės ar galinės kojos, maksimaliai didinant atstumą nuo FC/ESC plokštės. Iki šiol išbandytas atstumas (kameros aukštis) yra nedidelis — kojų lygio atstumas artimojo lauko slopinimui gali būti visiškai kito masto.
- **Aktyvus GPS kartojimas**: išorinė aktyvi GPS antena su sava LNA, gerai išvesta už orlaivio apvalkalo ribų, prijungta plonu koaksialiniu laidu. Galutinai atsakytų, ar tai artimumu pagrįsta problema. Visiškai per sudėtinga burbuliniams dronams, bet patvirtintų pagrindinę priežastį.

---

### Tuo pačiu metu: ELRS UFL antenos modifikacija

Nesusiję su GPS, bet atlikta tuo pačiu kabelių tvarkymo metu: pridėjau UFL jungtį ELRS imtuvo antenai. Standartinė vidinė antena Pavo20 steke tinkama artimam skrydžiui, bet ne daugiau.

Su UFL modifikacija ir išorine antena ryšys laikėsi stabiliai 1 km atstumu. Ribojantis veiksnys tokiu atstumu buvo baterijos talpa 15 m/s vėjyje — ne radijo ryšys. Ši modifikacija veikė iš pirmo karto ir davė akimirksniui pastebimą rezultatą. Naudingas kontrastas GPS darbui, kuris tokio rezultato nedavė.

---

## Kiti žingsniai

Atnaujinsiu šį straipsnį eksperimentams progresuojant. Jei išsprendėte tai panašiame integruoto steko burbuliniam dronui, norėčiau išgirsti.
