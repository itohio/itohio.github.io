---
title: "ESP32-C5 CSI jutimas su LHCP ir RHCP antenomis"
date: 2026-10-21
description: "Dvi UART laidu sujungtos ESP32-C5 plokštės aptinka judesį, buvimą ir kvėpavimą iš WiFi CSI. Tiesioginis signalas tarp antenų kol kas laimi."
draft: false
toc: true
categories:
  - Hardware
  - WiFi Sensing
tags:
  - esp32-c5
  - csi
  - wifi-sensing
  - presence-detection
  - breathing-detection
  - lhcp
  - rhcp
  - antennas
keywords: ["ESP32-C5 CSI", "WiFi CSI buvimo aptikimas", "WiFi kvėpavimo aptikimas", "LHCP RHCP antenų izoliacija", "ESP32 kanalo būsenos informacija"]
---

Pagaliau turėjau laisvo laiko ir išbandžiau WiFi jutimą pats, su keliais pakeitimais. Turiu dvi problemas, kurių dar neišsprendžiau: tiesioginis signalas tarp antenų per stiprus, o CSI užstringa, kai lyderis prisijungia prie maršrutizatoriaus.

Tai tik užuomina. Kodas dar bjaurus ir eksperimentinis, vėliau jį sutvarkysiu ir paskelbsiu GitHub repo. Toliau bus daugiau CSI eksperimentų, ir dalis jų labai susiję su FPV.

## Įranga

Dvi ESP32-C5 plokštės, sulituotos šalia viena kitos. Jos bendrauja UART ir turi bendrą maitinimą.

- Plokštės pačios nusprendžia, kuri yra **lyderis**, o kuri **sekėja**.
- Lyderis per WiFi talpina web programėlę.
- Abi plokštės 100 Hz dažniu siunčia neapdorotus WiFi paketus. Taip aplinka apšviečiama, o kartu kiekviena plokštė jaučia, ką siuntė kita.
- Kiekviena plokštė perduoda CSI duomenis kitai per UART. Judesį, žmogaus buvimą ir kvėpavimą randantis filtravimas veikia pačiose plokštėse.

[TODO: Andrius — kokia programinė įranga: ESP-IDF C ar TinyGo? Kokio modelio plokštės, ir ar abi plokštės tikrai vienu metu ir siunčia, ir priima?]

[TODO: Photo — proof of concept: abi C5 plokštės šalia viena kitos, matomi UART ir maitinimo laidai]

## LHCP iš vienos pusės, RHCP iš kitos

Judesys daugiausia yra atspindys. Idėja: vienoje plokštėje kairinės, kitoje dešininės apskritiminės poliarizacijos antena, kad poliarizacija iš dalies filtruotų atspindžius.

[TODO: Andrius — kuri antena siunčia, kuri priima? Ar darbinė prielaida, kad atspindys pakeičia sukimosi kryptį?]

[TODO: Photo — dvi LHCP ir RHCP antenos]

## Tiesioginio kelio problema

Tiesioginis signalas tarp antenų yra per stiprus. Kas negerai su geometrija:

- Antenos arti viena kitos, todėl atskyrimas yra artimajame lauke.
- Tarp jų esantis ekranas mažesnis už minimalų reikiamą dydį ir stovi netinkamu atstumu.
- Šalia antenų yra metalinė lentyna.
- Antenos prastai sulygiuotos.

Ekranui panaudojau dūmų nusiurbėjo filtro putas, kurios gulėjo be darbo. Išmatavau apie 600 kΩ per 5 mm, tad nusprendžiau, kad tai geriau nei nieko kaip savadarbis RF sugėriklis. Tiesioginio signalo tai neįveikė.

[TODO: Photo — separatorius su putų sugėrikliu tarp antenų]

Perkeliodamas antenas pasiekiau **26 dB** atskyrimą. Paskui klijai sukietėjo, ir atskyrimas grįžo į maždaug **4–8 dB**.

[TODO: Andrius — kaip matavai atskyrimą? Kokį dydį: RSSI, CSI amplitudę, kita?]

[TODO: Andrius — putų varža: kaip matuota? Ar žinoma, ar tokios putos tikrai sugeria 5 GHz, ar tai tik spėjimas?]

## Kaip atrodo jutimas

Lyderio web programėlė realiu laiku rodo CSI subnešlius, spektrogramas ir kvėpavimo kreivę.

[TODO: Photo — web programėlės kvėpavimo ekrano nuotrauka]

[TODO: Photo — spektrograma]

[TODO: Photo — CSI kreivės]

[TODO: Andrius — kokiomis sąlygomis gautas kvėpavimo grafikas ir kokie filtrai naudojami? Skaičiai prašomi.]

## Maršrutizatoriaus klaida

Vienu metu talpinti prieigos tašką ir srautu siųsti dvejetainius kadrus į web programėlę, kartu su realaus laiko jutimo duomenimis ir kanalo subnešliais, veikia. Čia problemų nėra.

Bet kai lyderis prisijungia prie maršrutizatoriaus, CSI po 3 minučių sustoja. Visada. Kodėl, nežinau.

[TODO: Andrius — kas tiksliai sustoja: ar nustoja ateiti paketai, ar callback, ar plokštė pakimba? Ar sekėja nukenčia?]

Priežasties kol kas nežinau, todėl jos čia nespėlioju.

## Kur dabar esame

Buvimo, judesio ir kvėpavimo aptikimas veikia plokštėse. Antenų izoliacija trukdo gauti geresnius duomenis, o klijai parodė, kokie trapūs tie 26 dB. Maršrutizatoriaus strigimas neišspręstas.

Toliau bus daugiau CSI eksperimentų, o repo atsiras sutvarkius kodą.
[TODO: Andrius — ar nori čia ką nors konkretaus apie kitą eksperimentą?]

*Lietuviška versija yra juodraštis, kurį reikia peržiūrėti.*
