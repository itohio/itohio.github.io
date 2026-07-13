---
title: "Kaip Pagaliau Pasiduiau FPV — Dronefix.lt ir Naują Hobį"
date: 2026-07-13
description: "Kelerius metus priešinausi FPV. Tada užsiregistravau į Dronefix.lt mokymo programą, skridau simuliatoriuje, sulydžiau pirmąją konstrukciją ir visiškai pakeičiau projektų prioritetus. Spektrometras gali palaukti."
draft: true
toc: false
categories:
  - FPV
  - Asmeninis
tags:
  - fpv
  - dronefix
  - hobis
  - betaflight
  - floto
  - asmeninis
series:
  - FPV Builds
---

Kelerius metus žiūrėjau FPV turinį, prieš imdamasis veiksmų. Konstrukcijos atrodė brangios, mokymosi kreivė — staigi, o aš jau turėjau nebaigtų aparatinės įrangos projektų laukimo eilę. Spektrometras. DARP tinklo ryšio darbas. Dronų detektoriaus mokymo duomenų surinkimo kanalo, kuriam vis reikėjo daugiau duomenų.

Tada užsiregistravau į Dronefix.lt mokymo programą ir viskas pasikeitė.

---

## Dronefix.lt

Dronefix.lt vykdo struktūrizuotus FPV pilotų mokymus Lietuvoje. Ne „štai dronas, eik sudaužyk jį" — tikra programa: simuliatoriaus valandos, taisyklių ir oro erdvės teorija, praktika su realiais dronais, pirmųjų pirkimų gairės.

<!-- IMAGE: nuotrauka iš Dronefix.lt akademijos sesijos — grupė, įranga ar skrydžių zona -->
*[TODO: Nuotrauka iš Dronefix.lt mokymų]*

<!-- IMAGE: simuliatoriaus sąrankos ar ankstyvojo mokymo skrydžio nuotrauka -->
*[TODO: Nuotrauka — simuliatoriaus sesija ar ankstyvas mokymas]*

Simuliatoriaus valandos pasirodė svarbesnės, nei tikėjausi. Atėjau manydamas, kad turiu gerą erdvinį mąstymą ir elektronikos išsilavinimą — kiek sudėtinga gali būti? Atsakymas: pakankamai sudėtinga, kad laikas simuliatoriuje prieš palietant tikrą kvadrotorą išsaugo nuo daugybės brangių avarijų. Acro režimas yra nuolankumo mokykla. Skrydžio valandų nepavaduosi niekuo kitu, o simuliatoriaus valandos skaičiuojamos.

Ką programa padarė, ko nebūčiau galėjęs padaryti vienas:

- **Privertė pradėti nuo pagrindų prieš šokant į tai, ko norėjau.** Mano instinktas buvo iš karto konstruoti 5 colių freestyle kvadrotorą. Programa pradėjo nuo 2,5 colio burbulinio. Tai buvo teisingas sprendimas — burbulinio avarijos išgyvenamos.
- **Suteikė pagrindą pasirenkant įrangą.** FPV rinkoje pilna konkuruojančių standartų, nesuderinami ekosistemų ir produktų, kurie buvo geri prieš dvejus metus. Instruktoriai, skrendantys kasdien, supjaustė daug triukšmo.
- **Sujungė mane su kitais pilotais.** Bendruomenės aspektas buvo netikėtas. Kiti pilotai yra greičiausias būdas išspręsti problemas, rasti skrydžių vietas ir suprasti, kas iš tikrųjų svarbu, palyginti su tuo, kas ginčijama internete.

---

## Flotas

Po Dronefix.lt nusipirkau ne vieną kvadrotorą. Pastatiau kelis.

<!-- IMAGE: floto nuotrauka — visi dabartiniai kvadrotrai išdėstyti -->
*[TODO: Floto nuotrauka — Pavo20, 1S konstrukcija ir kiti kvadrotrai]*

<!-- IMAGE: individualių konstrukcijų nuotraukos -->
*[TODO: Individualių konstrukcijų nuotraukos]*

Dabartinis sąrašas:

**Pavo20 Pro II** — 2,5 colio GPS burbulinis. Pagrindinis GPS konfigūracijų testavimo įrankis ir [atskiro straipsnio apie GPS sunkumus](../pavo20-gps-struggles/) tema. Tai nėra gabiausias mano turimas kvadrotas, bet jis mane labiausiai išmokė apie RF trikdžius ir ESC triukšmą.

**1S Matrix 3-in-1 skaitmeninis** — konstrukcija, kuri kiekvieno seanso metu gėdina Pavo20 palydovų skaičiumi. Mažas, veikia su DJI O3 oro bloku, be problemų aptinka 20+ GPS palydovų. Faktas, kad 1S konstrukcija pralenkia specialų GPS burbulį signalo kokybėje, yra dalis to, kas mane sudomino triukšmo problema.

Sekė dar daugiau konstrukcijų. Kiekviena išmokė ko nors specifinio — variklio krypties gedimų, ESC protokolo nesuderinamumų, blackbox analizės, PID derinimo. Hobis yra tikrai edukacinis būdu, kuris jaučiasi labiau praktiškas nei dauguma programinės įrangos darbų.

---

## Ką Atidėjau

Spektrometro projektas buvo nebaigtas, kai atradau FPV. Turėjau veikiantį regimosios šviesos spektrometrą ant Raspberry Pi su TOSLINK optinio pluošto jungtimi, kalibruotą duomenų apdorojimo kanalą ir planus optinio pluošto priekinio galo atgaliniam sklaidymui eksperimentuoti.

Tas projektas vis dar laboratorijoje. Optinio pluošto darbas lėtai juda į priekį, 405 nm ir 535 nm lazerio eksperimentai vyksta, ir programinė įranga žymiai išsivystė — bet tempas sumažėjo, kai atsirado FPV. Neturiu dėl to jokio apgailestavimo. Galima būti apsėstu tik vienu dalyku vienu metu, ir šiuo metu tas dalykas yra FPV.

Planuojamas papildomas straipsnis apie spektrometrą. Trumpa versija: TOSLINK plastikinis pluoštas stipriai fluorescuoja ties 405 nm UV, kas panaikina Raman spektroskopiją per tą kanalą, o Windows neatskleidžia 10 bitų vaizdo iš fotoaparato su tinkama ekspozicijos valdymu, todėl rankinis prietaisas turi būti perprojektuotas tinkamai optinio pluošto priekiniam galui, prieš tęsiant Raman darbus.

---

## Kodėl FPV Prisijungia

Aš visų pirma nesu suinteresuotas FPV kaip filmavimo priemone ar sportu. Kas mane laiko įsitraukusį — tai sistemų darbas: RF ryšio dizainas, triukšmo analizė, GPS signalo vientisumas, variklio laiko valdymas, PID teorija. Kiekviena konstrukcija yra mažas įterptinių sistemų projektas su realaus pasaulio fizika.

Pavo20 GPS problema yra tikra RF inžinerijos problema. ELRS ryšio atsargos klausimas yra antenų teorija. Blackbox analizė yra signalų apdorojimas. Bendruomenėje pilna žmonių, kurie taiso dalykus empiriškai, kas yra greičiausia inžinerijos rūšis.

O skrydis yra tikrai malonus. Ta dalis mane nustebino labiausiai.

---

## Kas Artėja

Spektrometro straipsnis bus tikras techninis įrašas — optiniai pluoštai, spindulių skirstuvo geometrija, atgalinio sklaidymo eksperimentai, kodėl TOSLINK buvo netinkamas pasirinkimas, ką geriau daro 300 µm pluošto skirstytuvai. Prie jo grįšiu, kai bus išspręsta GPS problema arba kai pritrūks naujų dalykų laužyti.

Tuo tarpu: Pavo20 vis dar negali patikimai rasti GPS palydovų, 1S konstrukcija toliau iš jo tyčiojasi, o aš baigiu pasiteisinimų nebandyti INAV.
