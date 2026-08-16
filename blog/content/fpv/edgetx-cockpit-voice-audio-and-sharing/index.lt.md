---
title: "5 dalis: iš kur atkeliauja pranešimai ir kodėl negali tiesiog nusikopijuoti mano konfigūracijos"
date: 2026-08-16T13:00:00+03:00
description: "Pranešimai atkeliauja iš standartinio EdgeTX balso paketo, o ne iš to, ką būčiau įrašęs."
summary: "Pranešimai atkeliauja iš standartinio EdgeTX balso paketo, o ne iš to, ką būčiau įrašęs. Ką ištrinti iš modelio YAML prieš skelbiant ir poziciniai sensorių indeksai, dėl kurių pasidalinta konfigūracija tyliai elgiasi blogai."
draft: false
toc: true
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - wav
  - garsai
  - model-yaml
  - privatumas
  - elrs
  - telemetrija
keywords: ["EdgeTX balso paketo garsai", "EdgeTX model yml perkeliamumas", "EdgeTX telemetrijos sensoriu indeksai"]
series:
  - EdgeTX Cockpit Voice
---

> **EdgeTX Cockpit Voice**, 5 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 4 dalis: Ką pultas iš tikrųjų pasako](/fpv/edgetx-cockpit-voice-callouts/)  ·  [6 dalis: Telemetrijos įrašymas ir skaičius, kurį turi išmatuoti ›](/fpv/edgetx-cockpit-voice-telemetry-rates/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)

Pranešimai iš [4 dalies](/fpv/edgetx-cockpit-voice-callouts/) atkeliauja iš balso
paketo, kuris pateikiamas su pultu. Iš to seka du praktiniai dalykai: kur realiai
gyvena garsas ir kodėl mano konfigūracijos failo perdavimas tau yra mažiau
naudingas, nei atrodo.

## Pranešimai: rth, gpson, gpsoff, lowbat, warnng, ready

Ištarti pranešimai yra iš **balso paketo, kuris atkeliauja su pultu**. Nieko
neįrašinėjau ir nieko negeneravau. Šeši iš jų čia atlieka darbą: `rth`, `gpson`,
`gpsoff`, `lowbat`, `warnng`, `ready`.

Tai verta pasakyti atvirai, nes „pultas su manimi kalba“ skamba kaip atskiras
projektas, o jis nėra. `PLAY_TRACK` paima failo pavadinimą iš kalbai skirto garsų
katalogo SD kortelėje, angliškame pulte `/SOUNDS/en/`, ir standartinis paketas jau
turi naudingą trumpų pranešimų žodyną. Visa ši serija yra slenksčių logika,
nukreipta į failus, kurie jau buvo vietoje.

Kas yra pigiausia viso darbo dalis ir ta, kurios tikėjausi kaip sunkiausios.

Vienintelė vieta, kur tai riboja, yra tada, kai nori pranešimo, kurio pakete nėra.
Perdaryta konfigūracija 9 dalyje nori ištarto „check ok“ priešskrydžio patikrai, o
standartiniame žodyne tokio įrašo nėra. Du būdai apeiti: pasirinkti esamą paketo
įrašą, kuris pakankamai artimas pagal reikšmę, arba įdėti savo WAV į `/SOUNDS/en/`
ir pasirinkti jį. Man dar neteko dėti savo, tad formato vadovo iš spėjimų
nerašysiu.

Du garso nustatymai iš `radio.yml` verti žinojimo, nes jie keičia, kaip pranešimai
skamba:

```yaml
wavVolume: 4
beepVolume: 0
audioMuteEnable: 1
```

`beepVolume: 0` reiškia, kad pypsėjimai visiškai išjungti, o ištarti pranešimai
lieka garsūs. `audioMuteEnable: 1` tarp garsų išjungia stiprintuvą, kas sumažina
šnypštimą, bet stiprintuvui reikia akimirkos atsigauti. Jei kada pastebėsi, kad
trumpi pranešimai pameta pirmą skiemenį, tą nustatymą pirmiausia išbandyk ties `0`.
Mano skamba gerai, tad minau tai kaip dalyką, kurį verta žinoti, o ne kaip problemą,
kurią turiu.

## Dalijimasis konfigūracija: kas perkeliama ir ką ištrinti

Noriu, kad tai būtų atkartojama, tad: taip, skelbk savo YAML. Bet du įspėjimai.

### Šiuos laukus ištrink prieš skelbdamas

EdgeTX nuo 2.9 versijos pulto konfigūraciją SD kortelėje saugo YAML formatu —
`radio.yml` pultui ir po vieną failą kiekvienam modeliui `/MODELS/` kataloge.
Abiejuose manuose yra registracijos ID:

```yaml
# radio.yml
ownerRegistrationID: " 24P42P-"

# model00.yml
modelRegistrationID: " 24P42P-"
```

Prieš skelbdamas konfigūraciją, patikrink ir ištrink:

- `ownerRegistrationID` / `modelRegistrationID`
- `bluetoothName`
- Savo **ELRS binding frazę**, jos modelio YAML faile nėra, ji gyvena TX
  modulyje, bet jei dalinsi ir modulio atsarginę kopiją, ta frazė iš esmės yra
  raktas į tavo orlaivius
- Modelių pavadinimus, jei jie tave identifikuoja
- Svirčių kalibraciją (`calib:`), nekenksminga, bet niekam kitam nieko
  nereiškianti, o nukopijavus manąją tavo svirtys jausis netaisyklingai

### YAML perkeliamas mažiau, nei atrodo

Štai spąstai, ir jie tikri. Loginiai jungtukai į telemetrijos sensorius kreipiasi
**pagal poziciją**, ne pagal pavadinimą:

```yaml
def: "tele(14),40"     # sensoriaus vieta 14, kuri *mano faile* yra RxBt
```

`tele(14)` nėra „RxBt“. Tai „kas atsitiktinai atsidūrė 14-oje vietoje sensorių
atradimo metu“. Vietų tvarka priklauso nuo to, kurie kadrai atėjo pirmi, kai
atradai sensorius, o tai priklauso nuo tavo valdiklio konfigūracijos ir nuo
eiliškumo, kuriuo viską įjungei. **Tavo pulte 14-oje vietoje gali būti visai
kas kita**, ir jei taip, mano loginiai jungtukai tyliai lygins įtampos slenkstį
su tavo kursu, o visa sistema elgsis taip, kad atrodys kaip magija.

Mano vietų tvarka, kad būtų su kuo lyginti:

```text
0  1RSS   1  2RSS   2  RQly   3  RSNR   4  ANT    5  RFMD
6  TPWR   7  TRSS   8  TQly   9  TSNR  10  FM    11  Ptch
12 Roll  13  Yaw   14  RxBt  15  Curr  16  Capa  17  Bat%
18 GPS   19  GSpd  20  Hdg   21  GAlt  22  Sats
```

Tad atviras patarimas, pageidaujamumo tvarka:

1. **Perskaityk šio įrašo lenteles ir suvesk viską ranka**, naudodamas savo
   sensorių pavadinimus. Tai penkiolika minučių, ir rezultatą tikrai suprasi —
   o tai svarbu tada, kai lauke norėsi pakeisti slenkstį.
2. Jei vis tiek įmesi mano YAML: ištrink savo atrastus sensorius, atrask juos iš
   naujo, o tada **patikrink vietą po vietos**, ar loginių jungtukų puslapio
   skaičiai rodo į tuos sensorius, į kuriuos manai, kad rodo. Sąsaja rodo
   pavadinimus, tad tai lengva, tik nepraleisk.
3. `radio.yml` yra pririštas prie plokštės (manasis sako `board: gx12`) ir prie
   versijos (`semver: 2.12.2`). Nekopijuok jo į kitą pultą.

Perskaityk lenteles, naudok savo sensorių pavadinimus, ir gausi tai, ką realiai
supranti. Tai svarbu lauke, vėjyje, kai norisi pastumti slenkstį 0,1 V.


---

> **Series:** EdgeTX Cockpit Voice, 5 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 4 dalis: Ką pultas iš tikrųjų pasako](/fpv/edgetx-cockpit-voice-callouts/)  ·  [6 dalis: Telemetrijos įrašymas ir skaičius, kurį turi išmatuoti ›](/fpv/edgetx-cockpit-voice-telemetry-rates/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)
