---
title: "Miniatūrinis brangakmenių spektroskopas yra difrakcinis grotelių prietaisas, o ne prizma"
date: 2025-11-02T18:00:00+02:00
description: "Atradus, kad kišeninis brangakmenių spektroskopas turi difrakcines groteles, randamas lygiavimo kampas su kamera 3D spausdinimo laikikliui"
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

Monitoriaus kalibravimo problema vis dar neišspręsta. Tikslas nepasikeitė: išmatuoti tikrąją monitoriaus pirminių spalvų spektrinę išvestį, o ne tik RGB. CR30 laikomas prie ekrano duoda kažką, bet jis neskirtas emisiniams šaltiniams ir skaičiai kelia abejonių. Reikia tikro spektrometro, nukreipto į ekraną.

Turiu vieną iš tų kišeninių brangakmenių tikrinimo spektroskopų — tokių, kokius juvelyrikai naudoja akmenims identifikuoti. Mažas cilindrinis vamzdis, pridedi prie akies, žiūri pro jį į šviesos šaltinį ir matai sugerties juostas. Nusipirkau seniai. Ant jo parašyta "SPECTROSCOPE." Maniau, kad jame yra prizma, nes tai tradicinis juvelyro prietaisas.

Ne. Jame yra difrakcinis grotelės.

## Išardymas

![Kišeninis spektroskopas su nuimtu priekiniu lęšio elementu](spectroscope-disassembled.jpg)

Priekis atsukami ir lęšio elementas lengvai ištraukiamas. Viduje: mažos grotelės, plyšys ir okuliaras. Grotelės yra dispersinis elementas — visai ne prizma. Laikomas lęšis yra tik priekinis objektyvas; dispersinis elementas yra giliau vamzdyje.

Tai nebūtinai blogai. Prietaisas vis tiek veikia. Tačiau tai keičia geometriją: grotelės dispersuoja fiksuotu kampu krintančios šviesos atžvilgiu, ir tas kampas priklauso nuo grotelių dažnio ir bangos ilgio. Su prizma galima sukti ir palenkti ir rasti platų darbinių kameros pozicijų spektrą. Su grotelėmis pirmos eilės difrakcija yra konkrečiu kampu — reikia pataikyti tą kampą kameros jutikliu, kitaip nefiksuosi spektro.

3D spausdinamo laikiklio modeliui reikia žinoti tą kampą tiksliai prieš modeliuojant bet ką.

## Lygiavimo kampo radimas

Metodas: pritvirtinti kameros modulį ir spektroskopą ant pjovimo kilimėlio, šviesti platjuostį šaltinį į plyšį ir fiziškai sukti kamerą stebint tiesioginį vaizdą, kol spektras yra centruotas ir aštrus. Pjovimo kilimėlio kampų žymėjimai leidžia lengvai nuskaityti kampą.

![Kameros modulio ir spektroskopo vaizdas iš viršaus ant pjovimo kilimėlio, Warsun žibintuvėlis kaip šaltinis](alignment-setup-top.jpg)

![Lygiavimo sąrankos šoninis kampas](alignment-setup-side.jpg)

Kameros modulis (OV9281 monochrominis) kairėje, spektroskopas pritvirtintas putplasčio bloke apytiksliai sulygintame su kamera, Warsun žibintuvėlis kaip platjuostis baltos šviesos šaltinis dešinėje. Putplastis yra grubus, bet efektyvus būdas laikyti kampą reguliuojant.

## Ką mato kamera

![Kameros tiesioginis vaizdas rodantis difrakcinius grotelių raštus spektroskopo viduje](grating-on-screen.jpg)

Tai kameros tiesioginis vaizdas, nukreiptas į spektroskopo išėjimą, kol žibintuvėlis apšviečia plyšį. Vertikalios linijos yra grotelių struktūra matoma per okuliarą. Ryški horizontali juosta yra dispersuotas spektras — matosi vaivorykštės spalvų srautas per kadro centrą su grotelių struktūra ant jo.

To pakanka nuskaityti kampui: sukti kamerą ant kilimėlio, kol ta spektrinė juosta yra horizontali ir užpildo kiek įmanoma daugiau kadro. Tada nuskaityti kampą tarp spektroskopo ašies ir kameros ašies iš kilimėlio žymėjimų.

Išmatuotas lygiavimo kampas yra apytiksliai **20–22°** nuo spektroskopo vamzdžio optinės ašies. Tai skaičius, kuris eina į 3D modelį — kamera turi būti sumontuota tuo kampu plyšio atžvilgiu, o ne tiesiai priešais.

## Kodėl tai svarbu 3D modeliui

Tiesioginis laikiklis — kamera koaksiali su spektroskopu — visai nefiksuotų spektro. Gautum nulinio laipsnio (nedifraguotą) spindulį tiesiai per, kas yra tiesiog baltos šviesos šaltinis, jokios spektrinės informacijos.

Laikikliu reikia:
- Fiksuoto plyšio-kameros atstumo (veikia fokusavimą ir spektrinę skiriamąją gebą)
- ~21° kampo tarp spektroskopo ašies ir kameros montavimo ašies
- Būdo laikyti kamerą standžiai tuo kampu be lankstumo

Antrinė sąlyga: naudojamas OV9281 modulis turi konkretų lęšio-jutiklio atstumą, o spektroskopo okuliaras turi savo išėjimo vyzdį. Plyšys turi būti atvaizduotas per groteles ant jutiklio santykiu artimas 1:1 maksimaliai spektrinei skiriamąjai gebai. Tai nustato darbinį atstumą, kuris taip pat eina į 3D modelį.

## Kas toliau

Suprojektuoti ir atspausdinti laikiklio 3D modelį. Kai lygiavimas bus mechaniškai užfiksuotas, kalibruoti bangos ilgio ašį naudojant žinomus spektrinius ryšius — kompaktinė liuminescencinė lemputė turi gyvsidabrio emisijos linijas ties 405, 436, 546, 578nm, kurias lengva identifikuoti. Tada: nukreipti į monitorių, išmatuoti pirmines spalvas.

Tai kelias tikrai suprasti raudono kanalo problemą neperkant €500 spektrofotometro.
