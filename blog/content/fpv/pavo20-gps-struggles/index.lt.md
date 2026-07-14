---
title: "Pavo20 Pro II GPS Problemos — Triukšmas, Harmonikos ir Ieškojimas Geresnio Izoliacijos Sprendimo"
date: 2026-07-13
description: "Pavo20 Pro II vos aptinka 3 GPS palydovus ten, kur 1S skaitmeninis dronas aptinka 20+. Štai ką radau spektro analizatoriuje, ką išbandžiau ir kas lieka neišspręsta."
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
series:
  - FPV Builds
---

Pavo20 Pro II yra gabus 2,5 colio burbulinis dronas su integruotu GPS. Teoriškai tai turėtų reikšti GPS gelbėjimą mikro drono atveju — tikrą saugos tinklą. Praktiškai GPS beveik nenaudingas daugelyje skrydžio aplinkų: stebiu, kaip palydovų skaičius sustoja ties 2 ar 3, o kitas dronas tame pačiame lauke, tuo pačiu metu, prisijungia prie 20+. Tai istorija apie tai, ką radau ir kur esu dabar.

---

## Simptomas

Pirmas tos dienos skrydis. Laukas atviras, dangus giedras, pastatų nėra. Įjungiu abu dronus ir laukiu:

| Dronas | Laikas iki pirmo fiksavimo | Palydovai fiksuojant |
|--------|---------------------------|---------------------|
| 1S Matrix 3-in-1 skaitmeninis | ~90s | 20–22 |
| Pavo20 Pro II | >5min | 2–4 |

1S dronas yra mažesnis. Jo elektronika, galima sakyti, dar tankiau sugromuota. GPS modulis yra tos pačios kartos. Skirtumas — vaizdo sistema: 1S dronas naudoja DJI O3 Air Unit burbuliniame steke su kamera, o Pavo20 turi integruotą steką, kuriame VTX, FC, ESC ir GPS yra viena kompaktiška plokštė.

Burbulinio drono korpuse beveik nėra vietos tarp GPS modulio ir visko, kas generuoja triukšmą.

---

## Pirmas žvilgsnis: fizinė sąranka

Prieš imantis spektro analizatoriaus, atlikau akivaizdžius patikrinimus.

<!-- IMAGE: Pavo20 Pro II nuotrauka su sulituotu GPS moduliu ir garsiakalbiu, rodanti fizinį artumą prie VTX zonos -->
*[TODO: Nuotrauka — sulituotas GPS ir garsiakalbis ant Pavo20 Pro II steko]*

GPS antena Pavo20 ant steko viršaus, tiesiai virš ESC/VTX plokštės. Antenos įžeminimo plokštuma yra PCB varis — tas pats, per kurį teka 5V BEC perjungimo srovės ir VTX RF žemė. Tarp GPS modulio LNA ir VTX išvesties pakopo nėra fizinio ekrano.

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

Standartinė GPS laido instaliacija pakeista ekranuotu subalansuotu audio kabeliu (4 laidų su pynimo ekranu). Ekranas prijungtas prie FC žemės ir driekiamas šalia GPS modulio, suteikdamas tam tikrą vietinį ekranavimą. Vidiniai laidai neša maitinimą (VCC ir GND) bei duomenis (RX/TX). 1 µF ir 0,1 µF kondensatoriai prijungti lygiagrečiai tiesiai ant GPS modulio maitinimo kontaktų.

**Rezultatas: Dalinis pagerėjimas.** Palydovų skaičius kartais pasiekia 8, vietoj ankstesnio maksimalaus 5. Fiksavimas vis dar nepatikimas ir kartais visiškai nepavyksta. Geriau, bet neišspręsta.

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

Vis dar ieškau patikimo triukšmo izoliacijos sprendimo. Ekranuotas kabelis padėjo, bet neišsprendė problemos. Kas dar liko išbandyti:

- **Ilgesnis GPS kabelis**: modulio perkėlimas 5–8 cm toliau nuo steko plonu kabeliu. Net nedidelis fizinis atstumas dramatiškai sumažina artimojo lauko sujungimą. Kompromisas — svoris ir mechaninis sudėtingumas ant drono, kuris turėtų būti kompaktiškas.
- **Aktyvus GPS kartojimas**: išorinė aktyvi GPS antena su atskira LNA, už orlaivio apvalkalo ribų, prijungta plonu koaksialiniu laidu. Tai per sudėtinga burbuliniams dronams, bet patvirtintų, ar problema yra grynai artimumu pagrįsta.

1S Matrix dronas kiekvieną sesiją ir toliau gėdina Pavo20 palydovų skaičiumi. Kol nerasiu pataisymo, GPS gelbėjimas Pavo20 lieka kategorijoje „avarinė atsarga, kuri gali neveikti", o ne patikimas saugos elementas.

---

## Kiti žingsniai

Atnaujinsiu šį straipsnį eksperimentams progresuojant. Jei išsprendėte tai panašiame integruoto steko burbuliniam dronui, norėčiau išgirsti.
