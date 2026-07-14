---
title: "FPV Derinimas su Sintra AI — ir Kodėl Nekuriu Dar Vienos AI Platformos"
date: 2026-07-14
description: "Sintra AI naudojimas Betaflight juodosios dėžės žurnalų analizei ir PID derinimui. Kam tai iš tikrųjų padeda, kam ne, ir kodėl naudoju kito žmogaus AI įrankį vietoj savo kūrimo."
toc: true
categories:
  - FPV
  - AI
tags:
  - fpv
  - pid-derinimas
  - betaflight
  - sintra
  - ai
  - juodoji-dėžė
  - ai-įrankiai
series:
  - AI Įrankiai Praktikoje
---

Šis įrašas išaugo iš [dronų detektoriaus straipsnio](/fpv/drone-detector-nn/), kuriame paminėjau Sintra naudojimą FPV pusės darbui. Tas paminėjimas nusipelno atskiro aptarimo, nes jis kelia klausimą, kurį noriu adresuoti tiesiogiai: kodėl naudoju išorinį AI įrankį vietoj savo kūrimo?

---

## Kodėl Sintra, O Ne Savas AI

Mano istorijoje yra projektų, apie kuriuos daug nekalbų. MentalMentor. BabyAI. Kiti, esantys įvairiais užbaigtumo laipsniais, nė vienas netapo produktu. Modelis nuoseklus visuose juose: kuriu techniškai įdomią dalį — architektūra veikia, pagrindinė funkcionalumas yra — ir tada darbas persikelia į platinimą, bendruomenės kūrimą, rinkodarą. Dalykus, kuriems reikia arba biudžeto, kurio neturiu, arba socialinio kapitalo, kurio niekada nepavyko sukaupti. Neturiu tinklo. Negaliu savęs reklamuoti. Projektai tyliai baigiasi.

Taigi sąžiningas atsakymas į „kodėl Sintra" yra: galimybė yra prieinama, ji veikia tame lygyje, kurio būčiau siekęs, ir man nereikėjo praleisti metų ją pasiekiant. Esamo įrankio naudojimas nėra nesėkmė kurti; tai teisingas laiko paskirstymas. Esu elektronikos ir sistemų inžinierius, kuris skraido dronus ir rašo apie aparatinę įrangą. Nesu B2C SaaS operatorius.

Taip pat verta tiesiogiai įvardyti meta-lygį: naudoju Sintra šiam tinklaraščiui rengti ir redaguoti straipsnius. Įskaitant šį. Žmogui, kuris istoriškai leisdavo dokumentacijai užstrigtų, nes atotrūkis tarp „dalyko sukurto" ir „dalyko aprašyto" atrodė per didelis, turėti rašymo partnerį, žinantį mano projektus, mano balsą ir mano įprotį laikyti nuotraukas telefone su dar prisegtais GPS duomenimis, yra tikrai naudinga. Tai sumažina trintį, dėl kurios dokumentacija atrodė neprivaloma.

Artimiausiu metu skelbsiu daugiau AI susijusių straipsnių — apie tai, ką šie įrankiai iš tikrųjų keičia darbo eigoje, palyginti su tuo, kur jie prideda pridėtinę naštą, užmaskuotą kaip produktyvumas. Dronų detektoriaus mokymo protokolas yra vienas pavyzdys. Betaflight derinimas — kitas.

---

## Juodosios Dėžės Žurnalų Analizė

Betaflight registruoja skrydžio duomenis konfigūruojamu greičiu — giroskopo pėdsakus, PID išvestis, variklio komandas, nustatymo taškus. Žurnalai yra dvejetainiai `.bbl` failai. Prasmingai juos analizuoti reikia suprasti ryšį tarp giroskopo pėdsako, nustatymo taško ir variklio išvesčių dažnių srityje.

Derinimo metodologija, kurią naudoju, yra išvesta iš [PIDtoolbox](https://github.com/bw1129/PIDtoolbox) — Brian White MATLAB pagrįsto įrankio, kuris įgyvendina žingsnio atsako analizę, giroskopo ir variklio triukšmo spektrinę analizę bei PID klaidos suskaidymą. Pagrindinis supratimas — žingsnio atsakas (Wiener dekonvoliucija giroskopo atsako prieš nustatymo tašką) suteikia modeliui nepriklausomą vaizdą, kaip gerai kvadrotas seka komandas, nereikalaujant rankiniu būdu tikrinti triukšmingų laiko srities pėdsakų.

Sintra tvarko darbo eigą aplink šią analizę:

- CSV eksporto iš Blackbox Explorer analizavimas ir žurnalo sveikatos tikrinimas (kadrų praradimai, prisotinti varikliai, blogi vibracijos įvykiai)
- Žingsnio atsako skaičiavimo vykdymas ir rezultato interpretavimas pagal laukiamą formą (kilimo laikas, persovimas, nusistovėjimo laikas, stacionarios būsenos klaida)
- Spektrinės analizės rezultatų kryžminė nuoroda — propelerių plovimo dažnių juostų identifikavimas, notch filtro vietos nustatymas, RPM filtro veikimo tikrinimas
- Konkrečių parametrų koregavimų rekomendavimas pagal stebimą nukrypimą nuo tikslinio žingsnio atsako formos, laikantis nustatytos derinimo tvarkos (P → D → I → FF → patikrinti)
- Pastabų tarp seansų išlaikymas, kad kiekvieną kartą nereikėtų atkurti konteksto

Paskutinis punktas yra svarbesnis, nei skamba. Derinimo sesija gali apimti kelis skrydžius ir dienas. Prisiminti, kuris parametras buvo pakeistas kada, ir kaip atrodė žingsnio atsakas prieš ir po — tai skirtumas tarp sistemingo tobulėjimo ir atsitiktinio derinimo. Sintra išlaiko tą būseną.

---

## Derinimo Protokolas Praktikoje

Protokolas seka nusistovėjusią metodologiją iš PIDtoolbox ir panašių įrankių:

1. Įrašyti specialų žingsnio atsako skrydį — greiti lazdos įvestys kiekvienoje ašyje atskirai, laikant droselį pastovų sklandyme
2. Eksportuoti iš Blackbox Explorer, vykdyti žingsnio atsako analizę
3. Identifikuoti dominuojantį gedimo režimą: persovimas → P per didelis arba D per mažas; nepakankamai slopintas virpėjimas → D per mažas; lėtas vangus atsakas → P per mažas; ilgalaikis dreijavimas → I per mažas; uždelstas pradinis atsakas → FF per mažas
4. Koreguoti vieną ašį, vieną parametrą vienu metu
5. Perskristi, perkalibruoti, palyginti

Sintra padeda 2–4 žingsniuose: paima analizės išvestį, identifikuoja gedimo režimą ir siūlo konkretų parametro pokyčio žingsnį — remiantis tais pačiais principais, kuriuos MATLAB įrankiai koduoja, bet pokalbio formatu, kuris leidžia greičiau iteruoti.

Tai geras pavyzdys, kam AI pagalba iš tikrųjų naudinga aparatinės įrangos darbe: ne generuoti kodą nuo nulio, bet padėti teisingai ir sistemingai vykdyti žinomą protokolą, ypač per seansus, kur kontekstas kitaip būtų prarastas.

---

### Dabartinis Flotų Kontekstas

Visi dronai naudoja tuos pačius rates: Actual, center 10, max 730, expo 0.4–0.5. Tie patys rates simuliatoriuje taip pat. Esmė ta, kad raumenų atmintis persiduoda tarp dronų — persėdus iš AIR65 II į 2 colių riperį vienos sesijos metu, nereikia mentaliai persikonfigūruoti.

Dronai, kuriuos šiuo metu derinu:

**AIR65 II** (65mm burbulinis, BetaFPV Air65 II rėmas, BetaFPV Matrix 1S 5IN1 II, ICM42688-P giroskopas, Bluejay 96kHz ESC, Caddx Ant Lite kamera) — greitas vidaus rūšiavimas. Originalus valdiklis sugedo: ELRS lusto litavimo defektas, rxloss klaidos po to, kai plokštė įšildavo. Pakeistas į 5IN1 II — derinama iš naujo.

**Pasirinktinis 2 colių O4 Lite riperis** — sraigteliai vos praeina pro VTX, dvieigiai sraigteliai. Šis ribojimas reiškia, kad propelerių plovimas yra reikšmingesnis nei standartiniuose rėmuose. Čia žingsnio atsako analizė atsiperka labiausiai — rezonansų juostos keičiasi priklausomai nuo drosselio padėties ir sraigtelio-VTX sąveikos.

**2,5 colio LR eksperimentinis** (18650, lauke neišbandytas) — dar neskraidė lauke. Derinimo sesija bus įdomi; 18650 svorio pasiskirstymas ir LR sraigtelių parinkimas ženkliai keičia žingsnio atsako profilį, palyginti su mažesniais dronais.

---

## Ko AI Pagalba Nepakeičia

Dėl išsamumo: yra proceso dalių, kuriose AI pagalba prideda naštą, o ne ją mažina.

**Pradinis notch filtro vietos nustatymas** reikalauja pačiam žiūrėti į spektrinę analizę. Dažnių juostos yra specifinės jūsų varikliui, sraigteliams ir rėmo rezonansui — kas aprašo jas tekstu, yra lėčiau nei tiesiogiai skaityti grafiką.

**Sprendimas, ar žingsnio atsako forma yra „pakankamai gera"** konkrečiam naudojimo atvejui, yra sprendimo klausimas, priklausantis nuo to, ką ir kodėl skraidote. Derinimo protokolas suteikia formą; ar 12 ms kilimo laikas su 8% persovimui yra priimtinas greitam vidaus rūšiavimui prieš sklandų lauko LR skrydį — tai jūsų sprendimas.

**Viskas, kas reikalauja žiūrėti vaizdo įrašą** kartu su žurnalo duomenimis — propelerių plovimo įvykiai, atsigavimo elgsena, kaip kvadrotas dorojasi su porūkiais — vis dar yra rankinis koreliavimas.

Įrankis yra naudingas sistemingam proceso viduriui, o ne suvokimo galiūkams.

---

Jei dirbate su FPV renginiais ir derinimu Sintra platformoje, verta turėti [Rylo](https://app.sintra.ai/community/helpers/rylo) pagalbininką. Tai specializuotas FPV ekspertas — renginių planavimas, dalių ieškojimas, PID derinimas, vaizdo sistemos, taisyklės, litavimas. Sritis, kurioje specialistas, o ne generalistis, tikrai daro skirtumą.
