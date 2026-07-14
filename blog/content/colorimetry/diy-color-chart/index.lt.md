---
title: "Namų darbo spalvų kalibravimo lentelė: nuo sublimacinio spausdintuvo iki akrilinių dažų"
date: 2025-10-26T18:00:00+02:00
description: "DIY spalvų kalibravimo lentelės kūrimas naudojant ArgyllCMS, sublimacinį spausdintuvą ir akrilinius dažus — Darktable spalvų kalibravimo darbo eigai"
thumbnail: "sublimation-charts.jpg"
author: admin
categories:
  - Spalvų mokslas
  - Fotografija
tags:
  - Spalvų mokslas
  - Darktable
  - ArgyllCMS
  - Kalibravimas
  - Pasidaryk pats
series:
  - Spalvų mokslas
---

[Ankstesniame straipsnyje](/colorimetry/reverse-engineering-cr30) minėjau norą sukurti savadarbę spalvų lentelę, kuri galėtų tarnauti kaip etalonas Darktable spalvų kalibravimo moduliui. Tas straipsnis tai pavadino „kito karto tema." Atėjo tas kitas kartas.

Idėja teoriškai paprasta: atspausdinti žinomų spalvų lopus, išmatuoti kiekvieną CR30, ir naudoti ArgyllCMS profiliui sukurti. Praktikoje prireikė dviejų visiškai skirtingų metodų, kol gavau ką nors tinkamo.

## Ko tikisi ArgyllCMS

ArgyllCMS gali sugeneruoti tikslinę lentelę — žinomų Lab reikšmių spalvų lopų tinklelį — o po to, kai išmatuoji kiekvieną lopą kolorimetru, apskaičiuoja deltą tarp to, ką atspausdinai, ir to, ką norėjai. Iš to sukuriamas ICC profilis arba Darktable spalvų kalibravimo korekcija.

Sugeneruota lentelė yra tik TIFF failas su gretutine `.cht` deskriptoriaus byla. Atspausdini, išmatuoji — viskas. Išskyrus tai, kad „tiksliai atspausdinti" pasirodė esanti sunkiausia dalis.

## Pirmas bandymas: sublimacinis spausdintuvas

Turiu seną sublimacinį spausdintuvą. Tik CMY — be juodos, be išplėsto gamuto. Spaudiniai atrodo neblogai akiai, tačiau CMY riboja tikslumą sodriai raudonoms ir tamsiai mėlynoms spalvoms.

![ArgyllCMS sugeneruotos lentelės šalia SpyderChecker 24](sublimation-charts.jpg)

Kairėje — ColorChecker24 dydžio taikinys, sugeneruotas ArgyllCMS. Dešinėje — tankesnis 96 lopų taikinys su skirtingu atsitiktiniu išdėstymu. Tikras SpyderChecker 24 fone — palyginimui.

![Dvi ArgyllCMS lentelės šalia, fone matomas sublimacinis spausdintuvas](sublimation-charts-2.jpg)

Sublimacijos spaudiniai nėra blogi. Darktable integruotam spalvų kalibravimui 24 lopų lentelė iš tikrųjų pakanka. Tačiau matavimas CR30 atskleidė problemą: CMY gautas gerokai nukrypsta sodriai raudonomis ir geltonoms spalvoms. ΔE tose lopų vietose pakankamas, kad gautą profilį padarytų nepatikimu bet kam rimtesniam.

Tinkama grubiai korekcijai, bet ne CR30 programinės įrangos tikslumo patvirtinimui.

## Antras bandymas: akriliniai dažai

Jei spausdintuvas negali tiksliai atgaminti spalvų — pieškime jas ranka. Akriliniai dažai suteikia prieigą prie tikrų pirminių spalvų — įskaitant tas, kurių CMY spausdintuvas negali atkurti — ir CR30 gali išmatuoti kiekvieną lopą tiesiogiai ant dažytų paviršių.

Substratas: maži kvadratai, iškirpti iš kieto balto plastiko lapo, dažyti atskirai, kad kiekvieną lopą galima būtų permatuoti ir perdažyti keičiant visą lentelę.

![Akrilinių dažų paletė su pirminių spalvų lustais — sausi](acrylic-palette-dry.jpg)

Pirminės spalvos: mėlyna, žalia, juoda, geltona, raudona, pilka, balta. Lustai dėkle — išdžiovinti mėginiai, iškirpti iš plastiko lapo.

![Drėgni akriliniai dažai maišomi paletės dėkle](acrylic-palette-wet.jpg)

Maišymo procesas. Viršutinėje dėklo dalyje — bazinės pirminės spalvos; apatinėje — maišiau antrines ir reguliavau šviesumą balta.

![Darbo paviršius su dažytomis lopais ir palete](acrylic-work.jpg)

Akriliniai dažai atskleidė visai kitą problemą: partijų nuoseklumas. Akrilas džiūdamas tampa šiek tiek tamsesnis nei atrodo šlapias, o ploni sluoksniai praleidžia baltą substratą, perstumiant išmatuotą spalvą link baltos. Mažiausiai du sluoksniai, idealiai trys, su CR30 matuojant po kiekvieno sluoksnio, kol spalva stabilizuosis.

Privalumas: išmatuotos spalvos iš tikrųjų yra ten, kur reikia. Ultramarine mėlyna duoda tikrą mėlyną — ne artimiausią spausdintuvo apytikslį. Taip pat su geltona, raudona ir neutraliais tonais.

## Kodėl tai svarbu Darktable

Darktable spalvų kalibravimo modulis veikia fotografuojant žinomą spalvų etaloną tomis pačiomis sąlygomis kaip ir fotografuojamas objektas, po to apskaičiuojant 3×3 matricą (su poslinkiais), kuri susieja kameros neapdorotą atsaką su etalono Lab reikšmėmis. Įprastai pirksi ColorChecker arba SpyderChecker. Šios savadarbės lentelės yra pigesnė alternatyva — jei etaloninės reikšmės, kurias paduodi Darktable, iš tikrųjų atitinka tai, kas yra ant fizinės lentelės.

Būtent tai CR30 ir suteikia: išmatuotas Lab reikšmes kiekvienam lopui D65 apšvietimo sąlygomis. Paduok jas į ArgyllCMS kartu su nuotraukos matavimais ir gausi korekciją, grindžiamą tikrais matavimais, o ne gamykliniu specifikacijų lapu.

Ar savadarbė lentelė yra *pakankamai tiksli* — dar atviras klausimas. Akriliniai lopai yra matiniai ir pakankamai vienodi, bet toli gražu nėra tokie spectriškai tolygūs ar tiksliai valdomi kaip profesionalus taikinys. CR30 ΔE ant dažytų lopų yra geras — dažniausiai iki 3–4 sodrioms spalvoms, iki 1 neutralioms — tačiau patvirtinimas prieš SpyderChecker 24 reikalauja tinkamo monitoriaus kalibratoriaus. Ir čia įstrigo.

## Kalibratoriaus problema

Dauguma vartotojų monitoriaus kalibratorių — Datacolor Spyder X, X-Rite ColorMunki Display — matuoja tik RGB (arba kelis plačius juostos). Tai suteikia pakoreguotą gamos kreivę ir baltą tašką, kas tinka ekrano kalibravimui, bet neparodo tikrosios spektrinės galios paskirstymo tavo monitoriaus pirminių spalvų. Rimtam spalvų darbui — suprasti *kodėl* monitoriaus gamas turi tokią formą, arba patvirtinti, kad ekranas iš tikrųjų gali atkurti tavo kalibravimo darbo eigoje esančias spalvas — reikia spektrinių duomenų.

CR30 gali matuoti atspindžio spektrą nuo paviršiaus. Tiesioginiai emisiniai ekranai jam neprieinami. Monitoriaus charakterizavimui reikėtų spektrofotometro, veikiančio emisiniame režime: i1Display Pro Plus arba idealiai i1Pro 3. Kainų skirtumas tarp paprastų kolorimetrų ir tikro spektrofotometro yra nemažas, ir dar neišrinkau.

Kol sprendžiu, vis dėlto užfiksavau dabartinio monitoriaus pirminių spalvų spektrinę išvestį su CR30, laikydamas jį prie ekrano — nešvariai, bet informatyviai. Mėlynas ir žalias kanalai maždaug tokie, kokių tikėtumeis iš tipinio IPS skydelio. Raudonas kanalas yra... nelabai geras. Jis pasismaigo ten, kur reikia, tačiau yra platus antrinis kupstas, kurio ten neturėtų būti — tai reiškia, kad raudonose spalvose yra netikėta indėlis iš žalio regiono. Ekrane tai akiai atrodo gerai, tačiau kritiškame spalvų darbe tokio tipo spektriniai negrynumai pasireiškia kaip sisteminga klaida, kurios jokia matricos korekcija negali visiškai ištaisyti.

Tai veda prie kitos problemos dalies: net turėdamas tobulą kalibratorių ir tobulą savadarbę lentelę, dabartinis monitorius tikriausiai nėra tinkamas įrankis nuotraukų ir vaizdo darbui. Tai atskiras pirkimo sprendimas, ir stengiuosi jo neimti tol, kol tiksliai nesuprantu, kokie iš tikrųjų yra dabartinio ekrano spektriniai apribojimai.

## Kas toliau

- Apsispręsti dėl kalibratoriaus, kuris duoda tikrus spektrinius duomenis (ne tik RGB) — dar tyrinėju
- Iki tol: fotografuoti abi lenteles kontroliuojamoje šviesoje, paleisti per ArgyllCMS, palyginti gautas korekcijas dabartiniame ekrane
- Patvirtinti akrilinės lentelės CR30 matavimus prieš SpyderChecker 24
- Išsiaiškinti, ar monitoriaus raudono kanalo negrynumas yra neįveikiama kliūtis, ar tik žinomas poslinkis, su kuriuo galima dirbti

Sublimacijos spaudiniai naudingi kaip greitas bazinis lygis. Akrilinė lentelė — tai, kuria lažinuosi dėl realaus kalibravimo darbo — kai kitame gale bus patikimas kalibratorius.
