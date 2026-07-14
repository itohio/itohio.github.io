---
title: "Šviečiantys daiktai: pirmieji UV fluorescencijos eksperimentai"
date: 2025-11-09T18:00:00+02:00
description: "Pluošto sujungto UV fluorescencijos stendo kūrimas — TOSLINK fluorescuoja, silpnina NIR ir nepavyko su 785nm Ramanu. Trys priežastys, kodėl reikėjo pereiti"
thumbnail: "vitamin-b-cuvette.jpg"
author: admin
categories:
  - Spektroskopija
  - Techninė įranga
tags:
  - Fluorescencija
  - UV
  - Optinis pluoštas
  - TOSLINK
  - 3D spausdinimas
  - Spektroskopija
series:
  - Spalvų mokslas
---

Sureguliavus spektrometro lygiavimą ir susiformavus PySpectrometer3, akivaizdus kitas eksperimentas buvo UV fluorescencija. Mineralai, biologiniai mėginiai, brangakmeniai — daugybė įdomios chemijos pasireiškia kaip emisijos spektrai UV žadinime, kurių niekada nepamatytum baltoje šviesoje. Sąranka konceptualiai paprasta: apšviesti mėginį UV, užblokuoti žadinimą nuo spektrometro, surinkti fluorescencijos emisiją.

Praktikoje pirmoji problema atsirado dar prieš pradedant ką nors statyti.

## TOSLINK fluorescuoja

Šviesą iš šaltinių į mėginius ruošiausi tiekti per TOSLINK plastinį optinį pluoštą — jis pigus, turi standartizuotą 1mm angą, o jungtys visur randamos. Planas buvo naudoti tą patį pluoštą UV žadinimo tiekimui.

Pirma greitas testas: paimti saują TOSLINK pluošto gabaliukų, padėti po UV lempa, patikrinti, ar jie perduoda šviesą švariai.

![TOSLINK pluošto gabaliukai po UV apšvietimu, švytintys silpnai mėlynai-baltai](toslink-uv-glow.jpg)

Šviečia. Ne silpnai — aiškiai, neabejotinai. PMMA (polimetilmetakrilatas) plastikas, iš kurio pagamintas TOSLINK pluoštas, fluorescuoja UV žadinime, emituodamas mėlynai žaliame regione (~440–520nm). Lygiai tame bangos ilgių diapazone, kurį fluorescencijos eksperimentas turi matuoti aiškiai.

Tai fundamentali medžiagos savybė, ne gamintojo klausimas. Visi PMMA pluoštai taip elgsis. Pats pluoštas tampa plačiajuosčiu fono emisijos šaltiniu, ir bet koks fluorescencijos signalas iš mėginio slypi ant pluošto generuojamo triukšmo visame matomame diapazone.

TOSLINK netinka UV žadinimo nešėjui. Jis puikiai veikia matomojo diapazono perdavimo spektroskopijoje, kai šaltinio bangos ilgis yra aukščiau ~450nm, tačiau bet kam, kas apima UV žadinimą ir matomos šviesos emisijos surinkimą, jis teršia matavimą.

Yra subtilesnė tos pačios problemos versija. Net su UV pjovimo ilgabangiu filtru surinkimo angoje, jei UV lazeris švietė tiesiogiai į mėginį uždaroje dėžėje, UV sklaidosi nuo kiekvieno paviršiaus — įskaitant paties pluošto korpusą. Filtras blokuoja UV patekimą per pluošto galą, tačiau pluošto apvalkalas sugeria žadinimo fotonus per apvalkalą. PMMA reemituoja per visą matomą diapazoną. Filtras su tuo nepadeda.

Vienintelis tikras sprendimas yra silicio pluoštas, kuriam ši problema negalioja. Taip šis kelias ir baigiasi — bet dar ne dabar.

## Kas iš tikrųjų šviečia

Tai išsiaiškinus, tiesioginis eksperimentas buvo kokybinis: kas fluorescuoja, ir kokia spalva?

Trumpas atsakymas: daugiau nei tikėjaisi.

![Žaliai fluorescuojantis karoliukų vėrinys po UV — ryški žalia emisija iš plastiko](jewelry-green-uv.jpg)

Šie plastikiniai karoliukai emituoja ryškią sodžią žalią spalvą. Dažiklis beveik tikrai yra UV reaktyvus fluorescencinis polimero priedas — toks, kuris tyčia naudojamas kostiumų papuošaluose ultravioletinės šviesos efektams. Emisija tokia intensyvi, kad karoliukai atrodo kaip saviapšviečiantys.

![Mėlynai-baltai fluorescuojantis karoliukų vėrinys po UV](jewelry-blue-uv.jpg)

Kitas karoliukų vėrinys — švelnesnis mėlynai baltas spindesys. Dienine šviesa karoliukai turi pieningą permatumą. Po UV spinduliuoja plačiajuostę mėlynai baltą šviesą, būdingą tam tikrų tipų stiklo arba sintetinio opalito fluorescencijai. Vėsesnis ir difuziškesnis nei žalių karoliukų emisija.

![Raudonas karoliukų vėrinys po UV — šilta oranžinė-raudona fluorescencija](jewelry-red-uv.jpg)

Raudonai rožiniai karoliukai su šilta oranžine-raudona emisija. Emisijos spalva skiriasi nuo dieninės spalvos — baltoje šviesoje jie raudoni, po UV fluorescuoja oranžiniai-raudoni su šiek tiek skirtingu smailiu. Emisija mažiau intensyvi nei žalių karoliukų, bet aiškiai atskirta nuo žadinimo.

Nė vienas iš šių mėginių nėra moksliškai įdomus — tai dotiruoti plastikai, suprojektuoti fluorescuoti — tačiau jie naudingi tikrinant, ar sąranka veikia ir ar skirtingos emisijos spalvos atskiria.

Įdomesnis mėginys:

![Vitamino B tirpalas kvarco kiuvetėje, pritvirtintoje 3D spausdintame laikiklyje, po UV švytintis žydrai mėlynai](vitamin-b-cuvette.jpg)

Riboflavinas (vitaminas B₂), ištirpintas vandenyje, UV žadinimo metu stipriai fluorescuoja mėlynai žaliame regione — emisijos smailė apie 520nm su plačiais šlaitais. Kiuvetės spindesys yra žydras, o ne žalias, kokio galėtum tikėtis, nes fotoaparato baltos spalvos balansas traukia link violetinės-mėlynos UV lempos aplinkos. Po spektrometru tai pasirodytų kaip švari emisijos juosta.

## Kiuvetės laikiklis

Mėginių dėjimas po rankine UV lempa ir fotografavimas yra informatyvus, bet ne spektroskopija. Reikia kontroliuojamo sujungimo: UV per vieną angą, mėginys centre, fluorescencijos emisija per kitą angą, su UV žadinimu, užblokuotu nuo surinkimo pluošto.

![3D spausdintas kiuvetės laikiklis — vaizdas iš viršaus rodantis keturias kolimavimo TOSLINK angas ir centrinę kiuvetės angą](cuvette-holder-top.jpg)

Laikiklis 3D spausdintas su keturiomis angomis, simetriškai išdėstytomis aplink centrinę kiuvetės angą. Kiekviena anga tiesiogiai priima TOSLINK jungtį. Geometrija suteikia keturis nepriklausomus kanalus: du žadinimui/etalonui, du surinkimui — arba praktiškai: vienas žadinimo įvestis, viena pagrindinė surinkimo išvestis, likę du filtro įdėjimui ar antrinio etaloninio surinkimo.

Kiekviena anga turi trumpą kolimuojamąjį vamzdelį suprojektuotą laikiklye. Kolimuoti su TOSLINK yra atlaidžiai: 1mm pluošto anga priima pagrįstą kūgio kampą nereikalaudama tikslaus lygiavimo. Popieriaus lapas kaip lygiavimo gidas sufiguruoja pluoštą pakankamai arti — 1mm anga pakankama, kad maži kampiniai nukrypimai nesugriauna pralaidumo.

Tai neteisinga mažo šerdies pluoštui. Bandžiau tą patį 3D spausdinimo metodą su 50µm daugiamodu pluoštu: lygiavimo tolerancija tokiu mastu yra griežtesnė nei 3D spausdintuvas gali patikimai pagaminti. Sujungimo efektyvumas buvo baisus ir nepagerėjo rankiniu reguliavimu. TOSLINK stora šerdis čia tikrai naudinga — visas stendas tampa mechaniškai tolerantiška taip, kaip su laboratoriniu pluoštu neveiktų.

Mažas žalvarinis stulpelis centre laiko kiuvetę. Standartinės 10mm × 45mm kvarco kiuvetės telpa tiesiogiai; laikiklis išlaiko optinį kelią centre prie kiuvetės vidurio aukščio.

## Dėžė

UV fluorescencijos eksperimentams reikia tamsos dėl akivaizdžios priežasties — aplinkos šviesa užgožia silpną emisijos signalą. Tačiau dėžė egzistuoja ir dėl antros priežasties: žadinimo šaltiniai nėra LED.

Sąranka naudoja du lazerius: UV lazerį fluorescencijos žadinimui ir 785nm NIR diodą Raman eksperimentams. Abu yra tiesioginiai šaltiniai, nukreipti į mėginį, todėl reikia uždaros dėžės. Porolono išklota dėžė suteikia tamsią aplinką, saugo nuo lazerio spindulio išsiveržimo ir padaro visą sąranką atkartojama — tas pats lygiavimas kiekviename matavime.

![UV fluorescencijos matavimo dėžės vidus — porolono išklota projektinė dėžutė su UV LED, kiuvetės laikiklio mazgu ir optiniais kabeliais](uv-box-interior.jpg)

Dėžė yra standartinė ABS projektinė dėžutė su tinkintais juodo porolono įdėklais, tvirtai laikančiais kiuvetės laikiklio mazgą. Viduje: UV ir NIR lazerių laikikliai, kiuvetės laikiklis sąveikos taške ir lazerių valdiklių vielos. Optiniai kabeliai išeina per dėžės šonus per TOSLINK jungtis dėžės sienose.

Porolono išpjovos svarbios: viskas lieka vietoje uždengus dangtį. Kiuvetę galima keisti neiš­lygiuojant optinių angų — tai ir yra standartizuotos laikiklio geometrijos esmė.

TOSLINK perjungiklis įvesties pusėje leidžia perjungti tarp matavimo režimų neatidarant dėžės:

- **Perdavimas**: UV šaltinis vienoje pusėje, surinkimas priešingoje, tiesioginis perdavimas per mėginį
- **Sklaida**: 90° surinkimo geometrija, surinkimo pluoštas stačiu kampu žadinimo atžvilgiu
- **Fluorescencija su UV pjovimo filtru**: žadinimas viduje, ilgabangis filtras surinkimo angoje užblokuoja bet kokį išsklaidytą UV, renka tik Stokso poslinkio emisiją

Perjungiklis yra paprastas mechaninis pluošto jungiklis — TOSLINK jungtys ant besisukančio selektoriaus. Jokios elektronikos, jokios programinės įrangos. Grubiai, bet patikimai.

## Ką ši sąranka iš tikrųjų matuoja

Surinkimo pluošto išvestis eina į brangakmenių spektroskopą + OV9281 kamerą, aprašytus ankstesniuose straipsniuose. Su PySpectrometer3 veikiančiu matavimo režime, riboflavino tirpalo fluorescencijos spektras užtrunka apie 10 sekundžių: nustatyk mėginį, perjunk į fluorescencijos režimą, paleisk įsigijimą, gauk bangos ilgio kalibruotą emisijos spektrą.

UV fluorescencijos pusė veikia su išlygomis. TOSLINK apvalkalo sugėrimo problema, aprašyta aukščiau, reiškia, kad matavimas turi PMMA fluorescencijos foną kiekvieną kartą, kai UV lazeris šviečia dėžės viduje. Spindintys emiteriai kaip riboflavinas — valdoma. Silpni fluoroforai ar mineraliniai mėginiai, kur emisija gali būti panaši į pluošto foną — tikra problema.

### 785nm problema

785nm NIR lazeris turėjo įgalinti Raman spektroskopiją. Raman sklaida yra neelastinė — atgalinė šviesa Stokso poslinkiu pereina į ilgesnius bangos ilgius. 785nm žadinime Raman smailės tipiškiems organiniams junginiams atsiduria maždaug 800–950nm.

Du dalykai tai iš karto sunaikino.

Pirma: IR pjovimo filtras. Kaip aprašyta [ankstesniame straipsnyje](../mini-spectrometer-grating/), optiniame kelyje kažkur yra IR pjovimo filtras — staigus kritimas apie 700–720nm, tada nieko. Visas Raman spektras 785nm žadinime yra anapus to kirpimo. Spektrometras to tiesiog nemato.

Antra: TOSLINK pats neperduoda NIR. PMMA turi vidinę sugerties briauną — C-H pertoninių virpesių stiprios sugerties juostos virš ~700nm. Išmatavimu kelių kabelių ilgių radau reikšmingą, nuo ilgio priklausomą slopinimą virš 800nm. Kuo ilgesnis kabelis, tuo blogiau. Net jei IR filtro nebūtų, bet kokie 785nm žadinami Raman fotonai būtų palaipsniui praryti surinkimo pluošto prieš pasiekiant spektrometrą.

Abi problemos nurodo į tą pačią šakninę priežastį: PMMA turi netinkamą perdavimo langą bet kam virš ~700nm. Tai ne TOSLINK konstruktyvinis trūkumas — TOSLINK skirtas 660nm ryšiams ir tai daro gerai. Problema — bandymas jį stumti į NIR spektroskopiją, kur jis niekada nebuvo skirtas.

Sąranka taip pat nėra leidybos kokybės UV fluorescencijai — kiuvetės laikiklio geometrija neoptimizuota erdviniam kampui, lazerio spektras nekalibriuotas, o jautrumo korekcija daroma prieš matomos šviesos apšvietimą. Tačiau ji pakankamai funkcionali, kad galėtum identifikuoti, kurie mėginiai fluorescuoja, apibūdinti emisijos spalvą ir apytikrį smailės padėtį, ir atskirti besidengias emisijos juostas mišiniuose.

## Kas toliau

Ši pirmoji versija veikia, bet yra ankšta ir sunkiai rekonfigūruojama. TOSLINK apribojimai — fluorescencija po UV, NIR slopinimas virš 700nm — daro jį netinkamu kitiems eksperimentams. Antroji modulinė versija kuriama naudojant silicio pluoštą, kuris išsprendžia abi problemas žymiai griežtesnių lygiavimo tolerancijų kaina.

Ir kitas pluošto lygiavimo klausimo aspektas, iškilęs dirbant su problema. 200µm daugiamodo pluoštas, senas mikroskopas, 532nm lazeris, ir grubiai 3D spausdinta atgalinio sklaidos sujungtuvo konstrukcija:

![Atgalinio sklaidos Raman eksperimentas — 532nm pluoštas per pluošto daliklį, 200µm pluošto sujungimas, mikroskopo objektyvas kaip surinkimo/fokusavimo optika](raman-teaser.jpg)

Tai kitam straipsniui.
