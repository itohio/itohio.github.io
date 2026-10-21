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

Dvi Seeed Studio XIAO ESP32-C5 plokštės, sulituotos šalia viena kitos. Jos bendrauja UART ir turi bendrą maitinimą.

- Plokštės pačios nusprendžia, kuri yra **lyderis**, o kuri **sekėja**.
- Lyderis per WiFi talpina web programėlę.
- Plokštės dirba paeiliui. Viena 100 Hz dažniu siunčia neapdorotus WiFi paketus, apšviesdama aplinką, kita tuo metu jaučia, paskui vaidmenys apsikeičia.
- Kiekviena plokštė perduoda CSI duomenis kitai per UART. Judesį, žmogaus buvimą ir kvėpavimą randantis filtravimas veikia pačiose plokštėse.

Norėjau turėti galimybę keisti WiFi kanalus, bet tada prieigos taškas neveikė.

[TODO: Andrius — kokia programinė įranga: ESP-IDF C ar TinyGo?]

[TODO: Photo — proof of concept: abi C5 plokštės šalia viena kitos, matomi UART ir maitinimo laidai]

## LHCP iš vienos pusės, RHCP iš kitos

Judesys daugiausia yra atspindys. Idėja: vienoje plokštėje kairinės, kitoje dešininės apskritiminės poliarizacijos antena, kad poliarizacija iš dalies filtruotų atspindžius.

Darbinė prielaida: atspindys pakeičia sukimosi kryptį, todėl priešingos krypties antena turėtų labiau teikti pirmenybę atspindėtai energijai nei tiesioginiam keliui. Tai prielaida. Čia jos nepatikrinau.

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

Atskyrimo skaičiai yra RSSI.

Neturiu matavimo, ar putos apskritai ką nors sugeria WiFi dažniais. Tinkamam patikrinimui reikėtų dviejų kryptinių arba rago antenų, kurių neturiu, nors galbūt pasigaminsiu. Taip pat reikėtų bekyšulinės kameros, kurios irgi neturiu. Taigi putos yra spėjimas.

## Kaip atrodo jutimas

Lyderio web programėlė realiu laiku rodo CSI subnešlius, spektrogramas ir kvėpavimo kreivę.

[TODO: Photo — web programėlės kvėpavimo ekrano nuotrauka]

[TODO: Photo — spektrograma]

[TODO: Photo — CSI kreivės]

Kvėpavimo grafiką gavau, kai atskyrimas buvo didelis. Tačiau spektrogramose matosi triukšmas, tad kvėpavimo kreivė gali būti tiesiog triukšmas. Kol kas nežinau ir nesakau, kad tai kvėpavimas.

[TODO: Andrius — kokius filtrus naudoji judesiui, buvimui ir kvėpavimui? Skaičiai, kuriuos nori matyti kaip faktus.]

## Maršrutizatoriaus klaida

Vienu metu talpinti prieigos tašką ir srautu siųsti dvejetainius kadrus į web programėlę, kartu su realaus laiko jutimo duomenimis ir kanalo subnešliais, veikia. Čia problemų nėra.

Bet kai lyderis prisijungia prie maršrutizatoriaus, CSI po 3 minučių sustoja. Visada. Kodėl, nežinau.

Strigimo detalės:

- CSI toliau praneša galiojančius kadrus, bet visos reikšmės vienodos. Paketai baitas į baitą vienodi.
- Tai išlieka po programinio perkrovimo ir perkrovimo.
- Dingsta tik atjungus ir vėl prijungus maitinimą.

Priežasties kol kas nežinau, todėl jos čia nespėlioju.

## Kur dabar esame

Buvimo, judesio ir kvėpavimo aptikimas veikia plokštėse. Ar kvėpavimas tikras, neaišku, nes spektrogramose matosi triukšmas. Tiesioginis kelias tarp antenų per stiprus, o klijai parodė, kokie trapūs buvo tie 26 dB. Maršrutizatoriaus strigimas neišspręstas.

Tai pirmieji žingsniai link kažko įdomesnio, o repo atsiras sutvarkius kodą.

*Lietuviška versija yra juodraštis, kurį reikia peržiūrėti.*
