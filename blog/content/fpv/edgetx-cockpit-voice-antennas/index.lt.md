---
title: "7 dalis: dvi antenos, dvi juostos ir dronas, kurį praradau dėl poliarizacijos"
date: 2026-08-16T15:00:00+03:00
description: "ELRS valdymo kanalo antenos yra tiesinės poliarizacijos, ne apskritiminės kaip vaizdo."
summary: "ELRS valdymo kanalo antenos yra tiesinės poliarizacijos, ne apskritiminės kaip vaizdo. Kodėl viena horizontali ir viena vertikali tikro diversiteto imtuve nugali vieną anteną."
draft: false
toc: true
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - elrs
  - antenu-diversitetas
  - poliarizacija
  - radiomaster-gx12
  - gemini
  - crsf
keywords: ["ELRS antenu poliarizacija", "tikro diversiteto imtuvas FPV", "ELRS Gemini dvi juostos", "RadioMaster GX12 ar Boxer"]
series:
  - EdgeTX Cockpit Voice
---

> **EdgeTX Cockpit Voice**, 7 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 6 dalis: telemetrijos įrašymas ir vienas skaičius, kurį turi išmatuoti pats](/fpv/edgetx-cockpit-voice-telemetry-rates/)  ·  [8 dalis: keturi dalykai, kurie čia negerai ›](/fpv/edgetx-cockpit-voice-whats-wrong/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)

Papildomi mygtukai yra tai, dėl ko šis projektas buvo malonus. Bet ne dėl jų pirkau
pultą. Tas sprendimas atsirado praradus aparatą, ir jis vertas atskiros dalies, nes
fizika ta pati, ant kurios pastatyti įspėjimai.

## Kita priežastis, kodėl pirkau šį pultą: dvi antenos, dvi juostos

Pirkau jį dėl **dviejų juostų veikimo su dviem antenomis**, ir tas sprendimas
atsirado praradus aparatą.

### Dronas, nukritęs į žolę

Su Pocket pultu man pasitaikė **poliarizacijos neatitikimas** tarp pulto ir
imtuvo antenos, ir nuotoliu dronas tiesiog nukrito iš oro į žolę.

Mechanizmą verta pasakyti tiksliai, nes FPV žmonės apie poliarizaciją įpratę
mąstyti *vaizdo* kontekste, kur konvencija yra apskritiminė. LHCP abiejuose
galuose, o LHCP su RHCP sumaišymas kainuoja apie 20 dB. Valdymo kanalas yra
kitas žvėris. **ELRS antenos yra tiesinės poliarizacijos**, dipoliai ir
monopoliai, ne spiralinės. O dvi tiesinės antenos 90° kampu viena kitos atžvilgiu
yra kryžminės poliarizacijos, o tai yra tos pačios brutalios eilės nuostolis.

Tiesinės antenos turi antrą problemą, kurią turi ir apskritiminės, bet kurią
lengviau užmiršti: dipolis spinduliuoja toru su **giliais nuliais išilgai savo
ašies**. Nukreipk antenos galą į kitą stotį, ir ten praktiškai nieko nebus. Ant
žemės to lengva išvengti. Nardymo viduryje, kai aparatas vartosi per visas
įmanomas orientacijas, išvengti negali, gali tik pasiekti, kad nulis niekada
nebūtų toje pačioje vietoje abiejose antenose vienu metu.

### Viena horizontaliai, viena vertikaliai

Todėl naujausiame aparate — **sulankstomame 4 colių**, kuris gaus savo atskirą
įrašą, kai jį paskraidysiu tiek, kad galėčiau ką nors atvirai pasakyti, naudoju
**tikro diversiteto imtuvą su dviem dviejų juostų antenomis: viena sumontuota
horizontaliai, kita vertikaliai.**

Tas statmenas derinys yra visas triukas, ir iš vienos konstrukcijos jis nupirks
du nepriklausomus dalykus:

- **Poliarizacijos aprėptis.** Kokia tuo momentu būtų pulto poliarizacija, viena
  iš dviejų priėmimo antenų yra pakankamai su ja sulygiuota. Nėra tokios
  orientacijos, kurioje abi būtų kryžminės poliarizacijos.
- **Nulių aprėptis.** Dviejų antenų nuliai nukreipti statmenomis kryptimis, tad
  jokia viena aparato orientacija negali abiejų vienu metu įstatyti į nulį.

„Tikras diversitetas“ yra ta dalis, dėl kurios tai veikia, o ne tik gerai
skamba. Tikro diversiteto imtuvas turi dvi nepriklausomas priėmimo grandines, po
vieną kiekvienai antenai, ir renkasi geresnę **kiekvienam paketui**. Tai nėra
pasyvus sumatorius ir tai nėra vienas imtuvas su jungtuku, kurį retkarčiais
perverčia.

Rezultatas ore: nardant Norvegijos krioklius, vartantis per visas aparato
orientacijas, jis tarp antenų perjungia tvarkingai, ir negaunu to ryšio nutrūkimo,
kurį geometrija sako, kad turėčiau gauti.

Pažymėtina, kad tai veikia **net kai aparate Gemini nėra.** ELRS Gemini režimas
siunčia abiem juostomis vienu metu ir reikalauja Gemini gebančio imtuvo kitame
gale. Be jo pultas vis tiek turi dvi antenas ir vis tiek tarp jų renkasi, tad
pulto diversiteto naudą gaunu ir tuose aparatuose, kurie viso Gemini negali.

### Tavo telemetrija tai jau matuoja — o manoji to nenaudoja

Štai dalis, dėl kurios rašydamas šį skyrių šiek tiek pyktelėjau ant savęs, ir ji
tiesiogiai siejasi su neegzistuojančiu ryšio kokybės įspėjimu.

Trys sensoriai, jau sėdintys mano modelyje, yra būtent diversiteto
instrumentacija:

| Sensorius | Kas tai iš tikrųjų yra |
|-----------|------------------------|
| `1RSS` | RSSI **imtuvo antenoje 1** |
| `2RSS` | RSSI **imtuvo antenoje 2** |
| `ANT`  | Kurią anteną imtuvas šiuo metu **naudoja** |

Kad būtų tikslu, kieno tai antenos: `1RSS`, `2RSS` ir `ANT` ateina iš CRSF ryšio
statistikos kadro ir aprašo **diversiteto imtuvą aparate**, o ne dvi pulto
antenas. Aukščiau aprašyta pulto pusės nauda yra atskiras mechanizmas, ir jo
neinstrumentavau, turimi atgalinio kanalo rodikliai (`TRSS`, `TQly`, `TSNR`)
matuojami pulte, bet nėra išskirti pagal anteną.

Visi trys turi `logs: 1`, tad **jie jau rašomi į CSV kas 0,3 s.** Vadinasi,
teiginys, kurį ką tik pasakiau — „tarp antenų perjungia nepriekaištingai“ — šiuo
metu yra lauko įspūdis, o ne matavimas, ir turiu duomenis jį matavimu paversti.
Sphere vaizdas [RX Blind-Spot Viewer](https://rxmap-viewer.sintra.site/rxmap/)
įrankyje sukurtas būtent tam: jis atvaizduoja blogiausią iš `1RSS`/`2RSS` pagal
azimutą ir elevaciją paties aparato atskaitos sistemoje, tad realiai veikianti
statmenų antenų pora turėtų pasirodyti kaip apvalesnė sfera su mažiau įdubimų nei
viena antena.
Suskaičiuok `ANT` perjungimus prieš `1RSS`/`2RSS` skirtumą ir gausi realų
perjungimo elgesį: kaip dažnai keičia, ar viena antena sistemiškai atlieka visą
darbą, ir ar perjungimai sutampa su orientacijos kaita juodojoje dėžėje.

Jei vieną anteną ryšys neša, o kita neduoda nieko, tai montavimo problema, ir iš
akinių ji nematoma. Savo telemetrijos rinkinyje turiu Lua skriptą antenų
diversiteto balansui; ko dar neturiu — **girdimos** versijos. Loginis jungtukas
ant `1RSS` ir `2RSS` skirtumo pasakytų apie mirusią ar blogai nuvestą anteną dar
ant stalo, prieš tai, kai ji taps pasivaikščiojimu žolėje.

Tai antras dalykas sąraše, iškart po ryšio kokybės pranešimo, ir tai ta pati
pamoka kaip ir visame šiame įraše. Informacija jau atkeliaudavo. Tik niekas jos
neklausė.

## Trumpa pastaba apie patį pultą

GX12 yra mano trečias pultas, ir vieną pastraipą būsiu neprofesionaliai entuziastingas.

Įsimylėjau jį tą pačią akimirką, kai pamačiau. Jis yra tarp RadioMaster Pocket ir
Boxer, ne toks kompaktiškas kaip Pocket, bet *gerokai* ergonomiškesnis, ir
rankose jaučiasi tikrai gerai, kaip Pocket nesijaučia. Šeši papildomi mygtukai
viršuje su atskirai adresuojama RGB yra tai, dėl ko visas šis projektas buvo
malonus, o ne varginantis.

Trumpai paskraidžiau kolegos 5 colių aparatą su Boxer, ir Boxer yra geresnis.
Geresni gimbalai, geresnė ergonomika, čia nėra ko diskutuoti. Mano pirmas
skrydis su juo baigėsi iškart, tiesiai ir vertikaliai medyje, smarkiai pralinksminęs jo
savininką. Vėliau kiek atsipirkau keliais power loop'ais per vartus,
bet medis yra ta dalis, kurią jis atsimena.

Priežastis, kodėl Boxer neturiu, yra proziška: jis netelpa. Didžioji dalis mano
skrydžių būna motociklo išvykose, o į GS Adventure bagažinę jau dabar vos
sutalpinu du dronus, akinius, baterijas ir pultą. DJI Mini 3 pakavimosi era —
kai visas komplektas dar palikdavo vietos sumuštiniams ir vandens buteliui —
jau seniai baigėsi. Ilgesnėms išvykoms teks pakuotis dar nuožmiau, o Boxer
dydžio pultas yra būtent neteisinga kryptis.

GX12 yra tas kompromisas, kuris nustojo jaustis kaip kompromisas.

Informacija jau atkeliaudavo. Tik niekas jos neklausė. Tai, daugiau ar mažiau, ir yra
visos šios serijos tezė, ir tai veda tiesiai į dalį, kurioje audituoju savo paties
darbą.


---

> **Serija:** EdgeTX Cockpit Voice, 7 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 6 dalis: telemetrijos įrašymas ir vienas skaičius, kurį turi išmatuoti pats](/fpv/edgetx-cockpit-voice-telemetry-rates/)  ·  [8 dalis: keturi dalykai, kurie čia negerai ›](/fpv/edgetx-cockpit-voice-whats-wrong/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)
