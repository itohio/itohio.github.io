---
title: "Spektrometro Tęsinys — Optiniai Pluoštai, Spindulių Skirstytuvai ir TOSLINK Fluorescencijos Problema"
date: 2026-07-13
description: "Spektrometro renginyje atsirado optinio pluošto priekinis galas. 50 µm prieš 300 µm pluoštai, spindulių skirstytuvo kubeliai prieš pluošto skirstytuvus atgaliniam sklaidymui, 405 nm ir 535 nm lazeriai Raman ir fluorescencijos eksperimentams, ir kodėl Windows 10 bitų vaizdo įrašas šiam laikui nužudė Raman darbą."
draft: true
toc: true
categories:
  - Mokslo prietaisai
  - Aparatinė įranga
tags:
  - spektrometras
  - optinis-pluoštas
  - raman
  - fluorescencija
  - lazeris
  - spektroskopija
  - aparatinė-įranga
  - raspberry-pi
  - 405nm
  - 535nm
---

Originalus spektrometras buvo Raspberry Pi konstrukcija su difrakcine gardele, OV9281 kamera ir TOSLINK plastikiniais optiniais pluoštais šviesos tiekimui. Jis pakankamai gerai veikė atspindžio ir pralaidumo matavimams. Planas visada buvo išplėsti jį atgalinio sklaidymo eksperimentams — Raman ir fluorescencijos spektroskopijai — kurie reikalauja visiškai kitokios optinės geometrijos.

Tas plėtinys yra tai, apie ką šis straipsnis. Tai ankstyvojo eksperimentavimo etapo darbas. Nėra švarių rezultatų pristatyti. Turiu aprašymą, kas buvo pastatyta, kas buvo atmesta ir kur yra apribojimai.

---

## Kas Pasikeitė: Atskiras Spektrometro Stendas

Originalus rankinis prietaisas buvo vienos paskirties instrumentas. Nauja sąranka yra stendo renginia, sukurta aplink optinio pluošto kabelius — su FC jungtimis, jungiančiomis tikrą optinę įrangą.

<!-- IMAGE: stendo spektrometro stendo su optinio pluošto įvestimis nuotrauka -->
*[TODO: Nuotrauka — stendo spektrometras su FC pluošto įvestimis]*

Stendas priima standartinius FC/PC ir FC/APC jungtis, kas atveria visą komercinių optinio pluošto zondų, skirstytuvų ir sujungiklių, skirtų spektroskopijai, spektrą. Tai buvo teisingo architektūros sprendimas — kiekvieną kartą, kai pasikeičia zondo geometrija, statyti pasirinktinę jungtį yra lėčiau nei prijungti kitą pluoštą.

---

## 50 µm prieš 300 µm Pluošto Eksperimentai

Pirmasis klausimas buvo pluošto šerdies skersmuo. Spektroskopijos pluoštai paprastai patenka į dvi sritis:

| Parametras | 50 µm šerdis | 300 µm šerdis |
|-----------|-------------|--------------|
| Skaitinis apertūros | 0,22 tipiškai | 0,22–0,37 tipiškai |
| Šviesos surinkimas (esant lygiai NA) | Mažesnis | Didesnis |
| Jungimo efektyvumas į spektrometro plyšį | Geresnis (glaudesnis spindulys) | Sunkesnis (didelis išėjimo kūgis) |
| Lankstumas / tvarkymas | Trapesnis | Tvirtas |
| Pereikiga skirstytuve | Mažesnė | Didesnė |

Spektrometro plyšio geometrijai, su kuria dirbu, 300 µm praktiškai teikė geresnę signalo srautą. Didesnė šerdis surenka daugiau šviesos iš difuzinės medžiagos, kas svarbiau nei jungimo efektyvumo nuostoliai plyšyje, kai pavyzdys yra silpnas.

Išbandžiau 50 µm kabelius specialiai atgalinio sklaidymo geometrijai, kur buvo tikimasi, kad glaudus spindulys pagerina erdvinį selektyvumą. Signalas per 50 µm buvo per silpnas, kad per lazerio galias, kurias naudojau, be rizikos pažeisti mėginį, būtų gautos naudojamos Raman savybės. Apsistojau prie 300 µm.

---

## Spindulių Skirstytuvo Kubelis prieš Pluošto Skirstytuvas Atgaliniam Sklaidymui

Raman ir fluorescencijos atgalinis sklaidymas reikalauja, kad lazerio ir surinkimo keliai dalytų tą pačią ašį — apšvietite mėginį ir surenkate atgal sklaidytą šviesą einančią tuo pačiu optikos keliu. Tai padaryti su optiniais pluoštais yra du įprasti būdai.

### Spindulių Skirstytuvo Kubelio Zondas

Spindulių skirstytuvo kubelis sėdi zondo gale. Lazerio pluoštas ir surinkimo pluoštas abu jungiasi į kubelį. Kubelis siunčia 50% lazerio į mėginį ir 50% į surinkimo kelią (švaistomos), ir nukreipia 50% grįžtančio sklaidymo į surinkimo pluoštą ir 50% atgal į lazerį (taip pat švaistomos).

```
Lazerio pluoštas ──→ [SK kubelis] ──→ mėginys
                         ↕
Surinkimo pluoštas ←── sklaidos šviesa
```

Pastatiau kelias spindulių skirstytuvo zondo geometrijas. Esminis problem: 50% nuostolis įvestyje ir 50% nuostolis surinkimo metu reiškia 25% efektyvumą geriausiu atveju — 75% signalo prarandama skirstytuve. Su Raman signalais, kurie ir taip yra 6–8 eilėmis silpnesni nei žadinimo lazeris, tai yra reikšminga bauda.

Kubelis taip pat sukelia atspindžius ir klaidinamą šviesą — lazerio atgalinis atspindys nuo kubelio paviršių surinkimo pluoštą pasiekia kaip fono signalas, iš dalies persidengiantis su silpnomis Raman savybėmis.

<!-- IMAGE: spindulių skirstytuvo kubelio zondo geometrijos nuotrauka -->
*[TODO: Nuotrauka — spindulių skirstytuvo kubelio zondo surinkimas]*

### Pluošto Skirstytuvas (2-in-1 jungtis)

Alternatyva: vienas pluoštas, nešantis tiek žadinimo, tiek surinkimo kelius, atskirtas skirstytuvo sandūroje. Jungtis mėginio gale turi du pluoštus viename ferrule — vienas žadinimui, kitas surinkimui — išdėstyti arti vienas kito, kad dalintų apšvietimo tūrį.

Tai metodas, prie kurio apsistojau po spindulių skirstytuvo eksperimentų. Pagrindiniai privalumai:

- **Jokios laisvosios erdvės optikos** — jokių atspindžių, dulkių paviršių, justifikacijos
- **Geresnis jungimas** — visa šviesa yra pluošte, nuostoliai tik sujungimo nuostoliai
- **Mažesnė klaidinama šviesa** — žadinimo ir surinkimo pluoštai yra fiziškai atskirti ferrule viduje, mažinant atgalinio atspindžio problemą

Trūkumas: erdvinis atstumas tarp žadinimo ir surinkimo pluoštų ferrule viduje reiškia, kad surinkimo tūris nėra tobulai koaksialus su žadinimo spinduliu. Masinės skysto mėginiams tai nesvarbu. Paviršiaus matavimams tai įveda šiek tiek geometrinio efektyvumo nuostolių.

<!-- IMAGE: pluošto skirstytuvo / bifurkuoto pluošto zondo galo nuotrauka rodanti dviejų pluoštų ferrule -->
*[TODO: Nuotrauka — pluošto skirstytuvas bifurkuotas zondas, dviejų pluoštų ferrule galas]*

---

## Lazerio Konfigūracija: 405 nm ir 535 nm

Naudojamos dvi lazerio linijos:

| Lazeris | Bangos ilgis | Naudojimo atvejis |
|---------|------------|-------------------|
| 405 nm violetinis | UV/artimuoju UV | Raman žadinimas; UV aktyvių junginių fluorescencijos žadinimas |
| 535 nm žalias | Matomas | Raman žadinimas ilgesnio bangos ilgio; mažiau fluorescencijos fono |

Žadinimo bangos ilgio pasirinkimas veikia Raman signalo stiprumą (Raman skerspjūvis masteliuojamas kaip ~1/λ⁴, teikiant pirmenybę trumpesniems bangos ilgiams) ir fluorescencijos foną (trumpesni bangos ilgiai žadina daugiau fluorescencijos, dažnai nustelbiančios Raman savybes).

Praktiškai 405 nm teikė stipresnius Raman signalus ant neorganinių mėginių, bet buvo visiškai nustelbiamas fluorescencijos ant organinių mėginių. 535 nm davė švaresnius spektrus ant organikų, bet silpnesnius Raman signalus apskritai.

<!-- IMAGE: lazerio + filtro + pluošto jungimo sąrankos nuotrauka žadinimo gale -->
*[TODO: Nuotrauka — 405 nm lazerio jungimas į pluoštą su juostų filtru]*

---

## TOSLINK Fluorescencijos Problema

Originalus rankinis spektrometras naudojo TOSLINK plastikinį optinį pluoštą. TOSLINK yra pigus, lankstus ir lengvai nutraukiamas. Jis puikiai veikia regimojo spektro baltosios šviesos atspindžio ir pralaidumo matavimams.

Jis visiškai netinka 405 nm UV žadinimui.

TOSLINK plastikinis pluoštas (PMMA šerdis) stipriai fluorescuoja ties 405 nm apšvietimu. Fluorescencijos emisija apima platų diapazoną nuo maždaug 430 nm iki 550 nm — tiksliai ten, kur pasirodytų silpnos Raman savybės nuo daugelio dominančių junginių. Pats pluoštas tampa didžiausiu fono signalo šaltiniu.

Tai buvo atrasta, kai pluošto skirstytuvo darbas jau veikė su silicio pluoštu. Grįžimas prie TOSLINK testavimo su 405 nm lazeriu tai iš karto patvirtino: vien pluošto fonas buvo panašaus intensyvumo kaip vidutiniškai stiprus fluorescencijos mėginys.

**UV žadinimui: tik silicio pluoštas.** PMMA pluoštas yra atmestas bet kam žemiau maždaug 450 nm žadinimo.

---

## Windows 10 Bitų Vaizdo Problema

Spektrometro kamera yra OV9281 monochrominis slenkamo užrakto jutiklis. Jis gali išvesti 10 bitų — daugiau dinaminės diapazono, geresnė galimybė atskirti silpnas spektrines savybes nuo fono.

Sistemoje Windows 10 bitų vaizdo kelias iš OV9281 nėra tinkamai atskleistas. Tvarkyklė pateikia 8 bitus arba konvertuoja iš 10 bitų į 8 bitus be tinkamo valdymo. Be to, ekspozicijos laiko valdymas yra labai ribotas — ekspozicijos prieaugiai yra grubus, kas apsunkina integracijos laiko nustatymą siekiant išvengti tiek lazerio linijos sodinimo, tiek silpnų Raman savybių nepakankamai paveikslų.

Linux sistemoje V4L2 tvarkyklė atskleidžia visą 10 bitų išvestį ir tikslų ekspozicijos valdymą. Tai ne programinės įrangos problema, kurią galiu apeiti sistemoje Windows — tai tvarkyklės apribojimas.

Praktinė pasekmė: Raman spektroskopijos eksperimentai su dabartine aparatine įranga reikalauja Linux. Darbo stotis veikia Windows. Tai reiškia, kad Raman darbas yra blokuotas kol:

1. Rankinis prietaisas (Raspberry Pi pagrįstas, veikia Linux) bus atnaujintas su tinkamu optinio pluošto priekiniu galu pakeičiant TOSLINK jungimą
2. Linux mašina bus nustatyta specialiai stendo renginiui

1 variantas yra planuojamas kelias. Rankinis prietaisas turi būti atstatytas su FC pluošto įvesties prievadu vietoj TOSLINK lizdo. Tai mechaniškai nėra sudėtinga, bet reikalauja kruopštaus pluošto į plyšį jungimo sulygiavimo spektrometro korpuse.

---

## Programinės Įrangos Būklė

Nepaisant to, kad aparatinė įranga yra ankstyvajame eksperimentiniame etape, programinė įranga žymiai išsivystė. Apdorojimo kanalas tvarko:

- Bangos ilgio kalibravimą iš žinomų spektrinių linijų (neonas, gyvsidabris)
- Tamsiojo kadro atimtį ir plokščio lauko korekciją
- Slenkantis vidurkis triukšmo sumažinimui ant silpnų signalų
- Smailių aptikimą ir paprastą bibliotekos atitikimą prieš nedidelę etalonų duomenų bazę
- Fluorescencijos fono įvertinimą ir atimtį (polinominis bazinės linijos pritaikymas)

<!-- IMAGE: spektrometro programinės įrangos ekrano kopija rodanti spektrą su fono atimtimi -->
*[TODO: Programinės įrangos ekrano kopija — spektras su matoma fono atimtimi]*

Fono atimtis yra dalis, kuri buvo labiausiai peržiūrėta. Polinominis bazinės linijos pritaikymas tinkamai veikia lygiam fluorescencijos fonui, bet nepavyksta, kai fonas turi struktūrą. Kitoje iteracijoje bus naudojamas asimetrinis mažiausių kvadratų (ALS) pritaikymas, kuris geriau tvarko struktūrizuotus fonus ir yra standartinis Raman išankstinio apdorojimo metodas.

---

## Dabartinės Būklės Santrauka

| Komponentas | Būsena |
|-------------|--------|
| 300 µm silicio pluošto skirstytuvas | Veikia — teikiamas pirmenybė prieš spindulių skirstytuvo kubelį |
| 50 µm pluošto eksperimentai | Atlikti — atmesti (per mažai signalo) |
| Spindulių skirstytuvo kubelio zondai | Pastatyti — atmesti (klaidinama šviesa, efektyvumo nuostoliai) |
| 405 nm lazeris + juostų filtras | Veikia |
| 535 nm lazeris + juostų filtras | Veikia |
| Ilgaję perėjimas / notch filtras surinkimui | Veikia |
| TOSLINK pluoštas UV | Atmesta — stipriai fluorescuoja ties 405 nm |
| Windows 10 bitų vaizdo įrašas | Blokuotas — tvarkyklės apribojimas |
| Rankinio prietaiso pluošto modernizavimas | Suplanuota — nepradėta |
| Raman spektro gavimas | Blokuotas laukiant Windows pataisymo arba Linux sąrankos |
| Fluorescencijos spektroskopija | Iš dalies veikia — švarūs spektrai ant kai kurių mėginių |

Fluorescencijos spektroskopija teikia naudojamus duomenis ant kai kurių mėginių. Raman yra blokuotas dėl Windows vaizdo problemos. Pluošto skirstytuvo geometrija yra patvirtinta kaip tinkamas metodas atgaliniam sklaidymui.

Atnaujinsiu, kai rankinis prietaisas gaus optinio pluošto priekinį galą. Tai yra kitas aparatinės įrangos etapas, kuris atblokuoja Raman eksperimentus.
