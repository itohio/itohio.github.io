<!-- TODO: Andrius, please review Lithuanian translation -->
---
title: "Daugiagentinės AI sistemos, kurios gerėja, o ne degraduoja"
date: 2026-09-14
description: "Be struktūros daugiagentinės AI sistemos degraduoja po antros iteracijos. Su tinkama architektūra — skill failais, savireflektyviaisiais agentais, produkcijos signalų grįžtamuoju ryšiu — jos vietoje gerėja."
draft: false
toc: true
categories:
  - Software Engineering
  - AI
tags:
  - ai
  - machine-learning
  - agentic-ai
  - multi-agent
  - software-engineering
  - llm
  - orchestration
  - prompt-engineering
keywords: ["daugiagentinės AI sistemos", "AI orkestracija", "savireflektyvi AI", "AI architektūra", "vibe coding problemos", "LLM sistemos", "gerėjančios AI sistemos"]
---

Mačiau, kaip kūrėjai tą patį AI agentą perstatė tris kartus per vieną sprintą. Kitas modelis, kitas prompt, tas pats degraduojantis rezultatas po antros iteracijos. Kodas veikia. Vieną kartą.

Tai vibe coding. Atrodo produktyviai. Demo — puiki. Po ketvirto pakeitimo užklausos niekas nebežino, kodėl kažkas veikia, o context window yra prieštaravimų šiukšlynas.

Su AI sistemomis konkrečiai šis gedimo režimas yra nematomas, kol staiga tampa matomas. Modelis nepraneša, kai jo kontekstas pasikeitė. Nepažymi, kai ankstesnis sprendimas buvo tyliai perrašytas. Tiesiog generuoja išvestį, kuri buvo techniškai koherentiška 3-iame posūkyje ir užtikrintai klaidinga 23-iame.

## Ką AI daro sau be struktūros

Vienos eilutės AI — gerai suprastas dalykas. Daugiagentinės sistemos, paleidžiamos be disciplinos, daro kažką įdomesnio: kaupia prieštaravimus. Agentas 8-ajame posūkyje perrašo apribojimą, nustatytą 2-ajame, nepastebėdamas to. Specializuotas subagentas optimizuoja savo tikslą ir generuoja išvestį, kuri kerta viršuje esančio koordinatoriaus darbą. Tikslai nukrypsta. Taisyklės pamirštamos. Sistema "veikia" ta prasme, kad generuoja išvestį, ir išvestis atrodo protingai, kol atidžiai nepažiūri.

Tikroji problema yra ta, kad tai atrodo kaip sėkmė. Ypač po pirmos iteracijos.

Be stebimos būsenos ir aiškių, ilgalaikių apribojimų, AI orchestracijos sistema yra prezervatyvas tarp produkcijos ir LLM. Tai filtras. Labai brangus.

## Architektūra, kuri leidžia gerėti

Įžvalga, prie kurios atėjau, buvo gėdingai banali: AI turi sugebėti perskaityti savo ankstesnius sprendimus.

Ne kaip pokalbio istoriją. Kaip struktūrizuotas, ilgalaikes žinias. Markdown failai su taisyklėmis. Domeno žinios skaitomoje formoje. Sprendimai su juos sukūrusia logika. Įrašas, kas veikė, kas ne, ir kodėl — palaikomas pačios sistemos.

Tai ne prompt engineering. Prompt engineering yra klausimo perrašymas. Tai, ką aprašau, yra AI darbinės atminties sukūrimas, kuri išlieka anapus context window.

Praktiškai tai atrodo kaip vienos atsakomybės skill failai: vienas failas vienai problemai, kiekvienas su taisyklėmis, apribojimais ir sukauptu mokymusi. AI juos skaito, vykdo pagal juos ir atnaujina remiantis rezultatais. Ne kiekvieną kartą. Kai kažkas lūžta arba kai išvestis nukrypsta nuo to, koks turėtų būti. Failas tampa gyvu įrašu.

Daugiagentinės hierarchijos, pastatytos ant tokio pagrindo, elgiasi kitaip nei plokščių orchestracijos grafų. Specializuotas LLM su gerai prižiūrimu domeno žinių failu gerėja savo darbe per kartotinius paleidimus. Koordinatorius virš jo gali pasikliauti nuosekliu elgesiu ir sutelkti dėmesį į koordinaciją, o ne klaidų taisymą. Tikslai ir apribojimai išlieka po 20 posūkių, nes jie nėra context window. Jie yra failuose.

## Grįžtamojo ryšio uždarymas su produkcijos signalais

Antroji dalis yra stebimumas. Ne žmonėms — AI vadybininkui.

Sentry sako, kas sulūžo. Grafana sako, kaip sistema elgiasi. Logs sako, kas iš tikrųjų nutiko. Mixpanel sako, ką daro vartotojai. Intercom arba Canny sako, dėl ko jie skundžiasi ir ko prašo.

Tai nebėra stebėjimo prietaisų skydeliai. Tai įvestys. AI vadybininkas skaito produkcijos signalus, identifikuoja šablonus ir nusprendžia, ar dabartinis elgesys yra teisingas tikslų atžvilgiu. Kai taip nėra, atnaujina atitinkamą skill failą. Kitas paleidimas yra geriau informuotas nei paskutinis.

Tai ta dalis, kuri atrodo kaip magija, kol nesupranti mechanizmo. Sistema, kuri skaito savo gedimo signalus ir tobulina savo taisykles, nėra ypatingai sudėtinga. Ji tiesiog disciplinuota.

## Kitoks AI vadybininko modelis

Didžioji dalis AI orchestracijos, kurią mačiau, yra gynybinė. AI vadybininkas sėdi tarp vartotojo užklausos ir pagrindinių LLM ir stengiasi nieko nesulaužyti. Valdo kontekstą, nukreipia, kartoja. Tai vidutinio intelekto tarpininkas.

Tai, ką aprašau, yra kitaip. AI vadybininkas kaip tikras orcherstatorius: skaito signalus iš viso produkcijos paketo, laiko apribojimus, kurie išlieka anapus bet kurio seanso, koordinuoja specializuotus LLM, kurie kiekvienas palaiko savo domeno ekspertizę, varo tobulėjimą vietoje tiesiog palaikant koherentiškumą. Hierarchinis. Savireflektyvus. Tikslai nenukenčia, nes jie saugomi, o ne spėjami iš to, kas atsitiktinai yra context window šį posūkį.

Tai ne agentinis grafų modelis. Tai ne apie agentų skaičių ar maršrutizavimo sudėtingumą. Tai apie tai, ar sistema kaupia žinias, ar jas išmeta kiekvieno pokalbio pabaigoje.

## Nuo kur tai prasidėjo

Markdown failų aplankas. Įmonės taisyklės. Projekto užrašai. Kliento kontekstas. Keli prompts. Tyrimų rezultatai.

Jokio architektūros sprendimo. Jokio orchestracijos framework. Tiesiog įprotis rašyti dalykus taip, kad AI galėtų juos perskaityti atgal. Tas įprotis, išplėstas ir formalizuotas — struktūrizuoti skill failai, savęs vertinimo ciklai, produkcijos signalų įtraukimas — yra viso to pagrindas.

Architektūra nebuvo suprojektuota. Ji atsirado darant akivaizdų dalyką nuosekliai.

Keli žingsniai iki tikrai autonominių AI sistemų. Kol kas: markdown failų aplankas ir pakankamai disciplinos neleisti AI meluoti sau pačiai. Vat taip vat.
