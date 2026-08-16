---
title: "5 dalis: individualūs garso failai ir kodėl negali tiesiog nusikopijuoti mano konfigūracijos"
date: 2026-08-16T13:00:00+03:00
description: "Kur EdgeTX laiko individualius WAV pranešimus, ką ištrinti iš modelio YAML prieš skelbiant ir poziciniai sensorių indeksai, dėl kurių pasidalinta konfigūracija tyliai elgiasi blogai."
draft: false
toc: true
weight: 5
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
keywords: ["EdgeTX individualus wav garsai", "EdgeTX model yml perkeliamumas", "EdgeTX telemetrijos sensoriu indeksai"]
series:
  - EdgeTX Cockpit Voice
---

Pranešimai iš [4 dalies](/fpv/edgetx-cockpit-voice-callouts/) yra individualūs WAV
failai, ne integruoti garsai. Iš to seka du praktiniai dalykai: kur gyvena garsas
ir kodėl mano konfigūracijos failo perdavimas tau yra mažiau naudingas, nei
atrodo.

## Individualūs garso failai: rth, gpson, gpsoff, lowbat, warnng, ready

Ištarti pranešimai yra individualūs WAV failai, ne integruoti garsai. Šiandien jų
šeši: `rth`, `gpson`, `gpsoff`, `lowbat`, `warnng`, `ready`, ir septintas,
`checkok`, kai sukursiu aukščiau aprašytą pergrupuotą konfigūraciją.

Jie gyvena kalbai skirtame garsų kataloge SD kortelėje, kartu su balso paketu —
angliškam pultui tai `/SOUNDS/en/`. Failo pavadinimas be `.wav` galūnės yra tai,
ką renkiesi specialiojoje funkcijoje, ir būtent todėl visi jie sutrumpinti:
**pavadinimas ribojamas iki šešių simbolių**, todėl `warnng`, o ne `warning`.

Savuosius sugeneravau tekstą-į-kalbą įrankiu ir konvertavau į formatą, kurio
EdgeTX reikalauja. Jei tavo failai groja, bet skamba ne taip, apkirpti,
pagreitinti ar tylūs, pirmiausia tikrink formatą, nes EdgeTX groja WAV failus
tiesiogiai, be perskaičiavimo.

Vienas dalykas, kurį verta patikrinti `radio.yml` faile, jei pranešimai skamba
apkirpti pradžioje, priežasties savajame pulte galutinai nepatvirtinau:

```yaml
audioMuteEnable: 1      # stiprintuvas nutildomas tarp garsų
wavVolume: 4
beepVolume: 0
```

`audioMuteEnable: 1` tarp garsų išjungia stiprintuvą, kad būtų mažiau šnypštimo.
Kaina ta, kad stiprintuvui reikia akimirkos atsigauti, o tai gali suvalgyti pirmą
trumpo pranešimo skiemenį. Nustatymas į `0` yra testas. Minau tai kaip
kandidatą, ne kaip diagnozę.

Taip pat atkreipk dėmesį į `beepVolume: 0`, pypsėjimus nuleidau iki galo, o WAV
garsą pakėliau. Jei jau viskas su manimi kalbės, nenoriu, kad tas pats dar ir
pypsėtų.

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

**Toliau:** [6 dalis, telemetrijos įrašymas ir vienas skaičius, kurį turi išmatuoti pats](/fpv/edgetx-cockpit-voice-telemetry-rates/)
