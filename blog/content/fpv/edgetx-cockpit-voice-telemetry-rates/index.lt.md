---
title: "6 dalis: telemetrijos įrašymas ir vienas skaičius, kurį turi išmatuoti pats"
date: 2026-08-16T14:00:00+03:00
description: "ELRS telemetrijos santykis, CRSF kadrų ciklinė eilė ir EdgeTX įrašymo periodas yra nuosekliai. Kodėl aritmetika nėra atsakymas ir 3D įrankis, kurį sukūriau šiems žurnalams skaityti."
draft: false
toc: true
weight: 6
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - crsf
  - elrs
  - telemetrijos-santykis
  - irasymas
  - rxmap
  - antenu-diversitetas
keywords: ["ELRS telemetrijos santykis", "CRSF kadru tipai", "EdgeTX SD logs periodas", "FPV RX akluju zonu 3D irankis"]
series:
  - EdgeTX Cockpit Voice
thumbnail: "rxmap-sphere-airframe.png"
---

Raudonas mygtukas šioje sistemoje rašo CSV į SD kortelę. Tas žurnalas pasirodo
esąs įdomiausias objektas visame projekte ir kartu tas, kurį supratau
blogiausiai.

## Telemetrijos įrašymas ir tas skaičius, kurį turi išmatuoti pats

Raudonas mygtukas valdo `LOGS` su `def: "3,1"` — **0,3 sekundės** įrašymo
periodu, rašant CSV į SD kortelę. Čia turiu nustoti teigti ir pradėti rodyti
namų darbus, nes atviras atsakymas yra tas, kad svarbiausio dalyko neišmatavau.

Įrašo tikslumo nenustato įrašymo periodas. Jį riboja trys dalykai iš eilės, ir
įrašymo periodas yra tik *paskutinis*:

1. **ELRS telemetrijos santykis**, kaip dažnai radijo kanalas apskritai skiria
   laiko tarpą atgaliniam ryšiui.
2. **CRSF kadrų ciklinė eilė**, valdiklis turi kelis skirtingus kadrų tipus, ir
   kiekviena telemetrijos galimybė nuveža vieną iš jų.
3. **EdgeTX įrašymo periodas**, kaip dažnai pultas nuskaito paskutinę gautą
   reikšmę.

Mano paties sensorių sąrašas antrą punktą padaro konkrečiu. Sugrupavus sensorius
pagal jų CRSF kadro ID:

| CRSF ID | Kadro tipas | Kokius sensorius nuveža |
|---------|-------------|-------------------------|
| `0x02` | GPS | `GPS`, `GSpd`, `Hdg`, `GAlt`, `Sats` |
| `0x08` | BATTERY_SENSOR | `RxBt`, `Curr`, `Capa`, `Bat%` |
| `0x1E` | ATTITUDE | `Ptch`, `Roll`, `Yaw` |
| `0x21` | FLIGHT_MODE | `FM` |
| `0x14` | LINK_STATISTICS | `1RSS`, `2RSS`, `RQly`, `RSNR`, `ANT`, `RFMD`, `TPWR`, `TRSS`, `TQly`, `TSNR` |

Atkreipk dėmesį, kad `Sats` ir `GAlt` atkeliauja **kartu**, tame pačiame kadre —
jie niekada negali būti nesinchronizuoti tarpusavyje. Bet `RxBt` gyvena visai
kitame kadre, todėl atsinaujina nepriklausomai ir lėčiau nei grynas telemetrijos
tarpų greitis.

```wave
{ "signal": [
  { "name": "RF paketai",        "wave": "p..............." },
  { "name": "telem. tarpas 1:4", "wave": "0.10.10.10.10" },
  { "name": "CRSF kadras",       "wave": "x.3x.4x.5x.6x",
    "data": ["GPS 0x02", "BATT 0x08", "ATT 0x1E", "FM 0x21"] },
  { "name": "RxBt naujas",       "wave": "0....1........." }
],
  "head": { "text": "Telemetrijos tarpai cikliškai keičia CRSF kadrų tipus" }
}
```

Naivi aritmetika: esant 500 Hz paketų greičiui ir 1:4 telemetrijos santykiui
gauni 125 atgalinius tarpus per sekundę, o cikliškai kaitaliojant keturis
skrydžio duomenų kadrų tipus `RxBt` atsinaujintų maždaug 31 kartą per sekundę.
Tokiu atveju 0,3 s įrašymo periodas *stipriai* per retai imtų reikšmes — įrašyčiau
vieną tašką iš dešimties ir praleisčiau kiekvieną įtampos kritimo momentą.

**Netikėk tuo skaičiumi.** Tai aritmetika iš kadrų struktūros, o ne matavimas, ir ji
ignoruoja tai, kad ELRS telemetrijos tarpai neveža daug duomenų, o CRSF GPS kadras
yra palyginti didelis, tad vienas kadras suskaidomas per kelis tarpus.

Tad išmatavau, ir matavimas jau gulėjo SD kortelėje. **Įrašymo periodas yra 0,3 s.**
Jei sensorius tikrai atkeliauja dažniau, kiekviena eilutė turi šviežią reikšmę. Jei
rečiau, CSV faile bus *iš eilės pasikartojančių identiškų reikšmių serijos*, o
vidutinis serijos ilgis yra santykis tarp tikro atvykimo intervalo ir įrašymo
periodo. Suskaičiuok serijas kiekviename stulpelyje ir turi atsakymą be jokių
prielaidų.

Trijų colių aparate, vienas dešimties minučių skrydis, gavosi taip:

| Lygis | Sensoriai | Išmatuotas intervalas | Greitis |
|---|---|---|---|
| **Valdiklio skrydžio duomenys**, per orą | `Ptch` `Roll` `Yaw` `RxBt` `Curr` `Capa` | **3,6 iki 3,9 s** | **~0,26 Hz** |
| **Pulte generuojama ryšio statistika** | `TRSS` `TSNR` `1RSS` `RQly` | 0,30 s, kiekviena eilutė | **>= 3,33 Hz** |

Kvantas yra aiškus, o ne miglotas. Iš 110 identiškų `Ptch` reikšmių serijų 76 buvo
lygiai 13 eilučių ilgio, 21 buvo 12, o ilgesnės serijos sutampa su tiksliais
kartotiniais ten, kur atnaujinimas prapuolė. Trylika eilučių po 0,3 s yra 3,9
sekundės.

**Taigi aritmetika klydo maždaug 120 kartų, ir klydo ta kryptimi, kuri svarbi.**
0,3 s įrašymo periodas ne per retai ima valdiklio duomenis. Jis ima juos maždaug
trylika kartų per dažnai. Kiekviena įtampos reikšmė įrašoma į CSV apie trylika kartų
prieš pasikeisdama.

Metodo pastaba: kiekvieną kadrą tikrink jo **didžiausios entropijos sensoriumi**,
`Curr` baterijos kadrui, `Ptch` orientacijai. `RxBt` kvantuotas iki 0,1 V, tad
išlaiko reikšmę per kelis atvykimus ir rodo lėtesnį greitį, nei jo kadras turi.

Dvi iš to sekančios išvados:

- Valdiklio sensoriams **1 iki 2 sekundžių įrašymo periodas nieko nepraranda.** Tik
  ryšio statistika pateisina 0,3 s.
- **Bet kuris įtampos slenksčio signalas turi maždaug keturias sekundes vėlinimo**
  prieš tai, kai apskritai gali pamatyti naują skaičių. Už tai trumpesni įtampos
  kritimai EdgeTX loginiams jungtukams yra nematomi. Jie egzistuoja tik juodojoje
  dėžėje.

`TRSS` yra ženklas, kodėl tie du lygiai apskritai skiriasi. Jis pasikeitė 1384 iš
1510 serijų, tai yra praktiškai kiekvienoje eilutėje. Jis generuojamas siųstuvo
modulyje ir niekada nelaukia oro kadro.

Šie skaičiai yra vienas aparatas vienoje konfigūracijoje, o telemetrijos santykio
modelio faile net nėra. Bet metodas perkeliamas, ir jis kainuoja vieną CSV, kurį jau
turi.

### Esu sukūręs įrankį, kuris skaito šiuos žurnalus

Kadangi visa raudono mygtuko esmė yra CSV failo gaminimas, turėčiau paminėti, kad
esu parašęs naršyklės įrankį, kuris būtent tą failą ir suvirškina:

**[RX Blind-Spot Viewer](https://rxmap-viewer.sintra.site/rxmap/)** — įkelk EdgeTX
SD-Logs CSV ir jis atvaizduos tavo **valdymo kanalą** trimatėje erdvėje. Viskas
veikia tik naršyklėje: niekas nėra įkeliama į serverį, paskyros nereikia, žurnalas
niekada neišeina iš tavo kompiuterio.

![RX Blind-Spot Viewer. Sphere vaizdas aparato atskaitos sistemoje, RSSI kaip empirinė antenos diagrama](rxmap-sphere-airframe.png "RX Blind-Spot Viewer. Sphere vaizdas, aparato atskaitos sistema, RSSI (blogiausias iš 1RSS/2RSS)")

Trys vaizdai:

- **Cloud**, tikros 3D skrydžio pozicijos, nuspalvintos pagal pasirinktą ryšio rodiklį
- **Sphere**, tas, kuris viršuje, ir tas, dėl kurio įrankį iš tikrųjų kūriau.
  Kiekvienas mėginys dedamas **siųstuvo kryptimi, kaip ji matoma iš aparato**, tad
  ašys yra NOSE / STBD / TAIL / PORT, o ne kompaso kryptys. **Radiusas yra signalo
  stiprumas.** Dėl to rezultatas tampa *empiriškai išmatuota antenos diagrama*
  konkrečiam tavo aparatui, o **įdubimas į vidų yra tikra imtuvo akloji zona
  tikroje orientacijoje.** Ratai žymi 0°, 30° ir 60° elevaciją. Yra atskaitos
  sistemos perjungimas — *From TX* erdviniam vaizdui, *Airframe frame* antenos
  diagramos vaizdui, ir atvaizdavimo perjungimas: *Points* neapdorotiems
  mėginiams, *Surface* glodintam kevalui, kuris pasidaro pilkas ten, kur duomenų
  nėra. Balti ir žali brūkšneliai trajektorijoje yra kurso žymekliai: baltas —
  nosis, žalias, dešinys bortas.
- **Path**, trajektorija, kurioje žymeklio dydis ir spalva atvirkščiai
  proporcingi ryšio kokybei, tad blogos vietos tampa tiesiogiai didesnės ir
  raudonesnės

Rodiklių sąrašas yra duomenų valdomas — įrankis nustato, kurie sensoriai realiai
yra tavo žurnale, ir pasiūlo juos: blogiausią iš `1RSS`/`2RSS`, `RSNR`, `RQly`,
`TRSS`, `TSNR` ir `TPWR` (traktuojamą kaip *didesnis = blogiau*, nes ELRS didina
siuntimo galią ryšiui blogėjant). Galima pasirinkti ir bet kurį neapdorotą
stulpelį. Taip pat jis automatiškai atskiria kelis skrydžius iš vieno žurnalo failo.

Tai užbaigia viso šio įrašo ratą. Pultas pasako apie ribą tuo momentu, vienu
žodžiu, kol skrendu. Peržiūros įrankis pasako *kodėl* po to, su prisegta geometrija.
Tas pats telemetrijos srautas, du tos pačios problemos galai.

Dvi jo detalės vertos atskiro paminėjimo, nes tai yra analizės pusės sprendimai
problemoms, į kurias atsitrenkiau anksčiau šiame įraše.

**Jis turi robustų žemės atskaitos tašką aukščiui**, ir tai egzistuoja būtent dėl
`GAlt` problemos iš L6 skyriaus aukščiau. `GAlt` yra metrai virš jūros lygio, o jo
*pirmieji* mėginiai yra patys blogiausi, nes fiksavimas ką tik gautas. Nustatyk
visam skrydžiui nulį pagal vieną šviežio fiksavimo mėginį, ir visas žurnalas taps
negatyvus. Todėl įrankis leidžia rinktis Auto / pagal pradžią / žemiausią / rankinį
atskaitos tašką, su pasirenkamu medianiniu filtru GPS aukščio išsišokimams, ir
traktuoja tikslius nulius `GAlt` stulpelyje kaip „nėra fiksavimo“, o ne kaip jūros
lygį. Ta pati fizika kaip aukščio įspėjimo problemoje, atakuojama iš kito galo.

Tą logiką matai veikiant ekrano nuotraukoje aukščiau, gintaro spalvos pastaba yra
įrankio pranešimas, kad žurnalas pradedamas maždaug 154 m virš savo žemiausio
taško, tad nulis buvo paimtas iš žemiausių 2 % skrydžio, o ne pasitikint pirmuoju
mėginiu. Naiviai imant atskaitą pagal pradžią, tas vienas šviežio fiksavimo
mėginys būtų padaręs visą skrydį negatyvaus aukščio.

**Jis turi srovės sensoriaus korekcijos koeficientą**, o tai yra šio įrašo
kalibracijos skyrius, padarytas veiksmingu. Jei skrydžio valdiklio srovės
sensorius blogai sumastelintas, tai kiekvienas mAh skaičius žurnale klysta
fiksuotu daugikliu, ir kiekvienas išvestinis skaičius taip pat. Nustatai korekciją
į `tikra ÷ užrašyta`, ir visas baterijos modelis persiskaičiuoja. (Betaflight'e
parametras yra `ibata_scale`, ir atkreipk dėmesį į kryptį: *mažesnis* mastelis
reiškia *didesnę* pranešamą srovę.) Papildomai jis apskaičiuoja **grįžimo namo
radiuso ratus ties įtempčiausiu skrydžio momentu**, žinant paketo talpą, naudojamą
procentą ir tavo paskelbtą saugų rezervą.

O tai yra rigoroji šio įrašo pradžioje aprašyto `rth` pranešimo versija. Pultas
duoda grubų įtampos pakaitalą pusei talpos kol esu ore, vienu žodžiu, be jokios
matematikos. Peržiūros įrankis po to pasako, ar tas žodis atėjo pakankamai anksti —
ir kurioje skrydžio dalyje nebūtų atėjęs.

Dar viena išmatuota detalė, kurią verta pažymėti: ELRS telemetrijos santykio
**modelio YAML faile nėra**. Mano `moduleData` blokas turi tik tai:

```yaml
moduleData:
   0:
      type: TYPE_CROSSFIRE
      subType: 0
      channelsStart: 0
      channelsCount: 16
      failsafeMode: NOT_SET
      mod:
         crsf:
            telemetryBaudrate: 0
```

Santykio lauko nėra, nes santykis gyvena pačiame TX modulyje ir konfigūruojamas
per ELRS Lua skriptą. Vadinasi, **dalinantis modelio YAML failu telemetrijos
santykiu nepasidalinama.** Jei nusikopijuosi mano konfigūraciją, o įrašai
atrodys kitaip nei mano, pirmiausia žiūrėk būtent čia.

Tad skyrius, kuris pradėjo kaip aritmetika, baigiasi matavimu, o tai vienintelė
kryptis, kuria toks apsikeitimas turėtų vykti. Skaičius, kurį spėjau, klydo dviem
eilėmis, ta kryptimi, dėl kurios būčiau rašęs žurnalą dešimt kartų dažniau, nei
reikia.

**Toliau:** [7 dalis, dvi antenos, dvi juostos ir dronas, kurį praradau dėl poliarizacijos](/fpv/edgetx-cockpit-voice-antennas/)
