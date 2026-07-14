---
title: "Tai niekada nebuvo prizma: mano spektroskope yra difrakcinis grotelės"
date: 2026-11-02T18:00:00+02:00
description: "Išardydamas brangakmenių spektroskopą 3D spausdinimo laikiklio projektavimui, atradau, kad visą laiką klysdamas dėl jo optikos — ir kodėl tai iš tikrųjų paaiškina mano duomenis"
thumbnail: "spectroscope-disassembled.jpg"
author: admin
categories:
  - Spalvų mokslas
  - Techninė įranga
  - Spektroskopija
tags:
  - Spektrometras
  - Difrakcinis grotelės
  - Spalvų mokslas
  - 3D spausdinimas
  - Pasidaryk pats
series:
  - Spalvų mokslas
---

Monitoriaus kalibravimo problema vis dar neišspręsta — reikia išmatuoti tikrąją monitoriaus pirminių spalvų spektrinę išvestį, o ne tik RGB. CR30 laikomas prie ekrano kažką duoda, bet jis neskirtas emisiniams šaltiniams. Reikia kišeninio brangakmenių spektroskopo, kurį naudojau visą laiką, tinkamai integruoto su OV9281 kameros moduliu.

Išardžiau jį geresniam 3D spausdintam laikikliui suprojektuoti. Ir proceso metu sužinojau, kad klydau apie jį nuo pat pradžių.

## Tai ne prizma

![Kišeninis spektroskopas su nuimtu priekiniu lęšio elementu, viduje matomas difrakcinis grotelės](spectroscope-disassembled.jpg)

Priekis atsukami ir lęšio elementas lengvai ištraukiamas. Viduje: plyšys, **difrakcinis grotelės** ir okuliaras. Ne prizma. Grotelės.

Nuo pat pirkimo šį prietaisą vadinu „juvelyro prizmos spektroskopu." Ant prietaiso parašyta "SPECTROSCOPE" ir jis parduodamas kartu su brangakmenių tikrinimo įrankiais. Tradiciniai juvelyrų spektroskopai naudoja prizmas. Šis ne. Kažkur suklysdamas prisiėmiau, niekada nepatvirtinau ir įtraukiau tą prielaidą į viską, ką apie jį rašiau.

Tai tas pats instrumentas, kurį naudojau su OV9281 kamera ir Raspberry Pi Zero visiems spektriniams matavimams. Spektroskopo korpusas, lygiavimas, kalibravimas — viskas buvo padaryta manant, kad dispersinis elementas seka Cauchy dispersiją kaip prizma. Neseka. Jis seka grotelių lygtį.

## Kodėl duomenys jau žinojo

Žvelgdamas atgal į kalibravimo kreives, tai iš tikrųjų paaiškina kažką, kas mane neramino: jos buvo įtartinai tiesinės.

Su stiklo prizma tikėtumės netiesinės bangos ilgio dispersijos — trumpesni bangos ilgiai (violetinis, mėlynas) suglausti, ilgesni (raudonas) išsklaidyti. Ryšys seka Cauchy lygtį stiklo lūžio rodikliui. Kalibruoji su polinomų tilptimi ir gauni pastebimus išlenktą pikselių pozicijos į bangos ilgį atvaizdavimą.

Mano kalibravimai vis tiek buvo beveik tiesiniai. 100-asis pikselis yra maždaug tokiu pat atstumu nuo 200-ojo, kaip 200-asis nuo 300-ojo, bangos ilgio atžvilgiu. Priskyriau tai konkrečiai prizmos geometrijai ir toliau ėjau. Tai ne prizmos geometrija. Tai todėl, kad **grotelės duoda tiesinę dispersiją** — bangos ilgis proporcingas difrakcijos kampui, ir mažuose kampuose kompaktiniame instrumente tai beveik tiesiškai atvaizduojama ant plokščio jutiklio.

Duomenys visą laiką man sakė. Aš neklausiau.

## Ką dar paaiškina grotelės

### 800nm siena

Niekada negalėjau gauti švarių duomenų virš ~800nm. Virš tos ribos signalas virsta triukšmu ir vidiniais atspindžiais — maniau, kad tai mechaninis lygiavimo laikiklyje klausimas arba kameros jautrumo kritimas.

Ne vienas, ne kitas. Tai fundamentali grotelių savybė: **difrakcijos eilės persidengią**.

Grotelių lygtis `d sin(θ) = mλ` reiškia, kad kiekvienas bangos ilgis difraguoja kiekvienoje sveikojo skaičiaus eilėje `m`. 2-osios eilės 400nm šviesos difrakcija atsiduria lygiai tame pačiame kampe kaip 1-osios eilės 800nm difrakcija. Taigi virš ~800nm jutiklis vienu metu gauna 1-osios eilės 800nm *ir* 2-osios eilės 400nm iš tos pačios grotelių pozicijos. Spektras užterštas — negalima jų atskirti be eilių rūšiavimo filtro.

Ironija: nešiojamojo spektrometro straipsnyje plačiai rašiau apie groteles, paskirstančias šviesą per kelias difrakcijos eiles, kaip trūkumą prieš prizmas. Buvau teisus. Aprašiau savo instrumento problemą. Tiesiog to nežinojau.

Prizma neturi eilių. Visa šviesa eina į vieną tolydų spektrą be persidengimo. Jei iš tikrųjų turėčiau prizmą, šios problemos nebūtų.

Sprendimas yra eilių rūšiavimo ilgabangis filtras — stiklo gabalas, blokuojantis bangos ilgius žemiau ~400nm matuojant virš 800nm, neleidžiantis 2-osios eilės UV teršti 1-osios eilės NIR. Šiuo metu praktinė viršutinė riba yra ~780nm, o tai apima visą matomą diapazoną ir pakankama spalvų mokslui bei monitoriaus charakterizavimui.

### IR filtras

Bandant kalibruoti spektrinį jautrumą prieš juodojo kūno šaltinį (kaitraliampė — CIE Illuminant A, ~2856K volframas), tikėjausi, kad išmatuotas spektras kils link IR pagal Planko kreivę. Nekyla. Nukrenta staigiai.

Kažkur optiniame kelyje yra IR pjovimo filtras — arba OV9281 modulyje, arba pačiame spektroskope. Volframo lempos Planko kreivė kyla stačiai į NIR; švarūs matavimai turėtų tai sekti. Vietoje to: skardis apie 700–720nm, tada nieko.

Tai padaro instrumentą praktiškai tik matomojo diapazono. Spalvų mokslui tinka — matomas diapazonas yra tai, kas svarbu. Tačiau tai keičia jautrumo kalibravimą: juodojo kūno etalonas veikia tik matomojoje spektro dalyje. Korekcijos kreivė turi nekontroliuojamą uodegą NIR srityje, kurią arba reikia kruopščiai tvarkyti, arba tiesiog apkarpyti.

Tarp 800nm eilių persidengimo problemos ir IR filtro, „NIR spektroskopija" su šiuo instrumentu dabartine forma neįmanoma. Tik matomas diapazonas.

## Lygiavimo kampo radimas

Grotelės ir prizmos taip pat skiriasi pozicija kameros atžvilgiu. Su prizma turi lankstumą — gali palenkti, pasukti ir rasti spektrą iš plataus kameros pozicijų spektro. Su grotelėmis pirmos eilės difraguotas spektras yra konkrečiu kampu nuo krintančios šviesos ašies, apibrėžtu grotelių lygtimi:

```
d sin(θ) = mλ
```

Fiksuotam grotelių dažniui `d`, difrakcijos kampas `θ` keičiasi su bangos ilgiu — taip spektras išsklaidomas. Tačiau matomojo spektro *centras* fiksuotu kampu, ir kamera turi būti ten pastatyta.

![Kameros modulio ir spektroskopo vaizdas iš viršaus ant pjovimo kilimėlio, žibintuvėlis kaip šaltinis](alignment-setup-top.jpg)

![Lygiavimo sąrankos šoninis kampas](alignment-setup-side.jpg)

Metodas: pritvirtinti OV9281 modulį ir spektroskopą ant pjovimo kilimėlio, šviesti Warsun žibintuvėliu į plyšį ir fiziškai sukti kamerą stebint tiesioginį vaizdą, kol spektras yra centruotas ir aštrus. Pjovimo kilimėlio kampų žymėjimai duoda tiesioginį nuskaitymą.

![Kameros tiesioginis vaizdas rodantis dispersuotą spektrą spektroskopo viduje](grating-on-screen.jpg)

Tai ką mato kamera: grotelių struktūra matoma kaip vertikalios linijos, su dispersuotu spektru kaip horizontali ryški juosta per kadro centrą. Sukti, kol ta juosta yra horizontali ir užpildo kuo daugiau kadro pločio — tada nuskaityti kampą.

Išmatuotas lygiavimas: apytiksliai **20–22°** nuo spektroskopo vamzdžio optinės ašies. Tas kampas eina į 3D modelį.

## Kas eina į laikiklio modelį

- **Kampas**: ~21° tarp spektroskopo vamzdžio ašies ir kameros montavimo ašies
- **Darbinis atstumas**: nustatytas OV9281 lęšio židinio nuotoliu ir spektroskopo išėjimo vyzdžiu — šiuo metu matuojama patvirtinti
- **Standumas**: jokio lankstumo; net vieno laipsnio kameros svyravimas pastumia spektrinę juostą nuo centro

Esamas laikiklis buvo suprojektuotas remiantis beveik koaksialios prizmos geometrijos prielaida. Jį reikia perprojektuoti nuo nulio teisingam grotelių kampui.

## Programinė įranga taip pat išaugo

Kol viską aiškinausi, taip pat perrašiau spektrometro programinę įrangą — tai, kas prasidėjo kaip greita [PySpectrometer2](https://github.com/leswright1977/PySpectrometer2) Les Wright klonas, virto žymiai didesniu daiktu. Nauja versija gyvena [github.com/foxis/PySpectrometer3](https://github.com/foxis/PySpectrometer3) (pateko į neteisingą paskyrą — turėtų būti itohio, pataisysiu).

Funkcijos, kurių nebuvo pradiniame plane: Raman poslinkio režimas, spalvų mokslo režimas su XYZ/LAB/CRI/CCT išvestimi, kišami kameros backends (Picamera2, OpenCV, RTSP, HTTP MJPEG), jautrumo korekcija prieš etaloninį apšvietimą, PDF ataskaitos, automatinis pasukimo kampas grotelių polinkiui. Tas paskutinis tiesiogiai susijęs čia — programinė įranga gali automatiškai aptikti spektro pasukimą ir jį ištaisyti, kas iš dalies kompensavo tai, kad laikiklis niekada nebuvo tiksliai teisingu kampu.

Daugiau apie programinę įrangą kitame straipsnyje.
