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
  - inav
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

GPS antena Pavo20 ant steko viršaus, tiesiai virš ESC/VTX plokštės. Antenos įžeminimo plokštuma yra PCB varis — tas pats, kuriuo teka variklio perjungimo srovės ir VTX RF žemė. Tarp GPS modulio LNA ir VTX išvesties pakopo nėra fizinio ekrano.

<!-- IMAGE: GPS laidų ir filtravimo bandymo nuotrauka — feritiniai karoliukai, filtravimo kondensatoriai maitinimo linijoje -->
*[TODO: Nuotrauka — GPS laidai su feritiniais karoliukais ir maitinimo linijos filtravimas]*

Pridėjau feritinių karoliukų ant GPS maitinimo linijos ir 100 µF kondensatorių prie modulio maitinimo kontaktų. Tai standartinis žemo dažnio triukšmo sprendimas. Jis nepadarė jokio išmatuojamo poveikio palydovų skaičiui.

---

## Spektro analizė — 1S dronas vs Pavo20

Laikas iš tikrųjų išmatuoti triukšmo lygį ten, kur veikia GPS. GPS L1 juosta yra **1575,42 MHz**. Konsteliacijos signalai, pasiekiantys anteną, yra nepaprastai silpni — paprastai apie −130 dBm. Bet koks vietinis trikdis 1,5–1,6 GHz diapazone juos nustelbia.

Prie kiekvieno drono steko prijungiau TinySA su trumpa vielos antena, dronams veikiant įjungtam ir įrenginiams šoviniais (varikliai veikia per variklio bandymo jungą, be sraigtų). Norint atskirti ESC/FC triukšmą nuo VTX, pradinį Pavo20 matavimą atlikau visiškai pašalinęs VTX.

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

ESC veikia pagal PWM perjungimo dažnį — 24, 48 arba 96 kHz šiuolaikiniuose stekuose. Šių dažnių harmonikos turėtų būti garso dažniuose ir jų kartotiniuose — nieko artimo GPS L1.

Faktinis trikdžių mechanizmas yra kitoks: **variklio PWM sukuria greito krašto srovės perėjimus**, ir tie perėjimai žadina rezonanses maitinimo paskirstymo takelių, litavimo taškų ir kondensatorių parazitiniuose elementuose. Rezultatas — plačiajuostis laidinis ir spinduliuojamas triukšmas, atsirandantis netikėtais dažniais gerokai viršijantis pagrindinį perjungimo dažnį.

Be to: **vaizdo siųstuvai** 5,8 GHz gali generuoti subharmonikas ir maišymo produktus. 5,8 GHz VTX ties 200 mW gali gaminti aptinkamą energiją ties 5800/4 = 1450 MHz — tiesiai GPS juostoje.

Tai patvirtinau kitoje aplinkoje — rūsyje, kur aplinkos RF triukšmas mažesnis:

![TinySA MAX HOLD matavimas — Pavo20 Pro II, 1,2–1,8 GHz, akumuliuotas ilgą laiką rūsyje — kelios harmoninių smaigaliai matomi GPS juostos srityje](tinysa-pavo20-maxhold.jpg)
*MAX HOLD skenavimas po kelių minučių akumuliacijos rūsyje. Keli smaigaliai išsklaidyti 1,2–1,6 GHz diapazone. Smaigaliai nėra pastovaus dažnio harmonikos — jie dreifuoja ir keičiasi priklausomai nuo variklio apkrovos ir temperatūros, kas būdinga perjungimo reguliatoriaus intermoduliavimo produktams.*

Lauke, GPS antena nukreipta į atvirą dangų, matomas tikrasis GPS signalo kontekstas:

![TinySA matavimas lauke — 1,2–1,8 GHz diapazonas — GPS L1 ties 1575,42 MHz vos matomas virš Pavo20 triukšmo grindos](tinysa-outside-gps.jpg)
*Matavimas lauke, atviroje erdvėje. GPS L1 signalas ties 1575,42 MHz sukuria plačią plokščiakalniui panašią elevaciją GPS juostoje — visos konsteliacijos signalai vienu metu. Palyginkite signalo lygį su Pavo20 triukšmo smaigaliais. Triukšmo grinda GPS juostoje yra 40–50 dB aukščiau to, ką LNA bando priimti.*

GPS L1 signalas yra tikrai mikroskopiškas. Triukšmo smaigaliai, kuriuos išmatavau, yra 40–50 dB aukščiau GPS signalo lygio. GPS modulio LNA kovoja pralaimėjimo mūšį.

---

## Ką išbandžiau

### 1. Feritiniai karoliukai ant GPS maitinimo

Feritiniai karoliukai ant VCC ir GND laidų į GPS modulį. Veiksmingi laidiniam triukšmui maitinimo linijoje žemesniuose dažniuose. Jokio poveikio spinduliuojamam trikdžiui iš VTX ties 1,5 GHz — RF šiame dažnyje nekeliaus maitinimo linija.

**Rezultatas: Palydovų skaičius nepagerėjo.**

### 2. VTX galios sumažinimas

VTX nustatymas į pit režimą (0 mW) arba mažiausią galią (25 mW) GPS įgijimo fazėje. Tai sumažina 5,8 GHz pagrindinį dažnį ir jo subharmonikas.

**Rezultatas: Nedidelis pagerėjimas.** Įgijimas kartais pasiekia 6–8 palydovus, kai VTX išjungtas, tačiau tai nėra praktinis skrydžio scenarijus.

### 3. ESC PWM dažnio sumažinimas

Sumažinau ESC PWM dažnį nuo 48 kHz iki 24 kHz. Mažesnis dažnis reiškia mažiau harmonikų vienam dažnio vienetui — harmonikų tankis mažėja.

**Rezultatas: Minimalus skirtumas.** Triukšmo profilis pasikeitė, bet neišnyko iš GPS juostos.

### 4. Fizinio ekranavimo bandymai

Apvyniotas GPS modulio ir antenos plotas varinės folijos juosta, sujungta su žeme. Tai sukuria Faraday ekraną aplink modulį, tačiau antena vis tiek turi matyti dangų — o antena yra tiesiai šalia triukšmo šaltinio.

**Rezultatas: Nedidelis pagerėjimas, tačiau geometrija beveik neįmanoma ekranuoti anteną neblokuojant dangaus krypties GPS signalų.**

---

## Pagrindinė priežasties vertinimas

Pavo20 Pro II integruoto steko dizainas teikia pirmenybę kompaktiškumui prieš RF izoliaciją. Tai sąmoningas kompromisas 2,5 colio korpusui — tiesiog nėra vietos atskirumui, kuris padarytų skirtumą.

Trukdžiai turi bent du komponentus:

1. **Laidinis triukšmas** GPS modulio maitinimo linijoje iš perjungimo reguliatorių ir ESC srovės smailių
2. **Spinduliuojamas RF** iš VTX subharmonikose, krentančiose į GPS L1 juostą

Feritiniai karoliukai iš dalies šalina 1 komponentą. 2 komponentui reikia fizinio atstumo (nepasiekiamo burbuliniame drone) arba ekranuoto GPS modulio su nuosava žemės plokštuma, atskirta nuo pagrindinės steko.

GPS modulis, naudojamas Pavo20, yra standartinis M8N/M10 variantas miniatiūrizuotame SMD pakete — virš RF LNA nėra ekrano dangtelio. Tai įprasta burbulinio GPS drono klasės dronams; prielaida, kad skrendant lauke pakanka dangaus matymo, kad būtų įveiktas pablogėjęs SNR.

Ši prielaida veikia kai kuriose skrydžio aplinkose ir visiškai neveikia kitose.

---

## Kur esu dabar

Vis dar ieškau patikimo triukšmo izoliacijos sprendimo. Šiuo metu vyksta eksperimentai:

- **Ilgesnis GPS kabelis**: modulio perkėlimas 5–8 cm toliau nuo steko plonu kabeliu. Net nedidelis fizinis atstumas dramatiškai sumažina artimojo lauko sujungimą. Kompromisas — svoris ir mechaninis sudėtingumas ant drono, kuris turėtų būti kompaktiškas.
- **Aktyvus GPS kartojimas**: išorinė aktyvi GPS antena su atskira LNA, už orlaivio apvalkalo ribų, prijungta plonu koaksialiniu laidu. Tai per sudėtinga burbuliniams dronams, bet patvirtintų, ar problema yra grynai artimumu pagrįsta.
- **Migracija į INAV**: INAV navigacijos steke blogėjantis GPS ryšys tvarkomas maloniau nei Betaflight GPS gelbėjime. Jei negaliu pašalinti triukšmo, geresnis programinės įrangos stekas gali patikimai veikti ties 6–8 palydovais vietoj reikalaujamų 12+.

1S Matrix dronas kiekvieną sesiją ir toliau gėdina Pavo20 palydovų skaičiumi. Kol nerasiu pataisymo, GPS gelbėjimas Pavo20 lieka kategorijoje „avarinė atsarga, kuri gali neveikti", o ne patikimas saugos elementas.

---

## Kiti žingsniai

Atnaujinsiu šį straipsnį eksperimentams progresuojant. Jei išsprendėte tai panašiame integruoto steko burbuliniam dronui, norėčiau išgirsti.

TinySA failai ir Betaflight konfigūracijos išsaugojimai abiem dronams yra pasiekiami — suspausiu nuorodas, kai sutvarkysite katalogo struktūrą.
