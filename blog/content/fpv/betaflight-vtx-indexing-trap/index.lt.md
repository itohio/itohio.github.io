---
title: "Betaflight VTX spąstai: aux_channel skaičiuojamas nuo 0, o vtx_power — nuo 1"
date: 2026-07-17
description: "VTX užstrigęs pit režime, 400mW nepasiekiami, galia visada vienu lygiu per žema? Betaflight vtx komandoje aux_channel skaičiuojamas nuo 0, o vtx_power — nuo 1. Niekas apie tai neįspėja. Štai sprendimas."
toc: true
categories:
  - FPV
  - Betaflight
tags:
  - fpv
  - betaflight
  - vtx
  - vtx-power
  - smartaudio
  - cli
  - pit-mode
  - emax-th3
  - vtxtable
  - troubleshooting
keywords: ["Betaflight VTX pit režimas", "Betaflight negaliu nustatyti 400mW", "vtx_power skaičiuojamas nuo 1", "Betaflight vtx komanda indeksavimas", "SmartAudio galia vienu lygiu per žema", "Betaflight VTX galios perjungimas neveikia", "vtxtable indeksas per vieną"]
series:
  - FPV Builds
---

Betaflight CLI garsėja tuo, kad yra lakoniška, bet viduje nuosekli. AUX kanalai skaičiuojami nuo 0: AUX1 yra `0`, AUX2 yra `1`, AUX6 yra `5`. Režimai, diapazonai, reguliavimo lizdai — kelis kartus juos sukonfigūravęs, smegenys tyliai užsirašo taisyklę: *čia viskas skaičiuojama nuo nulio.* Todėl kai prijungiau VTX galios perjungimą prie ratuko, surinkau konfigūraciją taip, kaip esu surinkęs šimtą kitų CLI eilučių, tikėjausi, kad tiesiog veiks, ir judėjau toliau.

Netiesiog suveikė. Ir priežastis atsiėjo man visą vakarą, nes gedimas atrodė kaip aparatinės įrangos triktis, o ne kaip rašybos klaida.

---

## Simptomas

Konstrukcija: 2 colių analoginis „ripper" ant **SUB250F411** su **Betaflight 4.5.4** ir **Emax TH3 V02** VTX (25 / 100 / 400 mW) per SmartAudio, aparatinis UART2. Norėjau galios ant sukamojo ratuko — **AUX6** valdymo pulte — kad galėčiau pasukti išvestį aukštyn nuotoliui ir atgal žemyn artimam skrydžiui neįlįsdamas į meniu.

Vietoj to gavau:

- Kiekviena ratuko padėtis duodavo galią **vienu lygiu žemesnę** nei sukonfigūravau.
- **400 mW buvo nepasiekiami.** Jokia ratuko padėtis per visą jo eigą jų neišrinkdavo.
- Ratuko kraštuose VTX nukrisdavo į tai, kas atrodė kaip **pit režimas** — beveik nulinė išvestis, klasikinis „kodėl draugas ant stalo nemato mano vaizdo" simptomas.

Pirma atlikau įprastą aparatinės įrangos „raganų medžioklę". Perrašiau SmartAudio nustatymus. Po lupa patikrinau UART2 litavimo taškus. Perkėliau VTX ant žinomai gero SmartAudio kanalo. Tarp kiekvieno pakeitimo išjungiau ir įjungiau maitinimą. Niekas nepajudėjo. Pats VTX visą laiką buvo tvarkingas — jis darė būtent tai, ką jam liepiau. Aš tiesiog nesuvokiau, ką jam liepiau.

---

## Spąstai

Štai `vtx` komandos formatas:

```
vtx <index> <aux_channel> <vtx_band> <vtx_channel> <vtx_power> <start_range> <end_range>
```

Du iš šių laukų naudoja **priešingas indeksavimo konvencijas**, ir niekas nei dokumentacijoje, nei `help vtx` išvestyje to nepasako:

| Laukas | Konvencija | Pavyzdys |
|--------|-----------|----------|
| `aux_channel` | **skaičiuojama nuo 0** | AUX6 → `5` |
| `vtx_power` | **skaičiuojama nuo 1** | pirmas vtxtable įrašas → `1` |

Perskaityk tai dar kartą, nes tame ir slypi visas straipsnis. *Vienoje komandoje* 2-as laukas skaičiuojamas nuo nulio, o 5-as — nuo vieneto. Jei perkeli „viskas nuo nulio" taisyklę per visą eilutę — o tai ir yra natūralu — kiekviena tavo įrašyta galios reikšmė nusėda vienu lizdu per žemai.

`vtx_power` yra indeksas į tavo `vtxtable`, ir tas indeksas prasideda nuo **1**:

```mermaid
graph LR
    subgraph CLI["vtx_power reikšmė, kurią įvedi"]
        direction TB
        P0["0"]
        P1["1"]
        P2["2"]
        P3["3"]
        P4["4"]
    end
    subgraph TBL["ką iš tikrųjų adresuoja"]
        direction TB
        T0["netinkama → PIT"]
        VT0["vtxtable[0]\nPIT"]
        VT1["vtxtable[1]\n25 mW"]
        VT2["vtxtable[2]\n100 mW"]
        VT3["vtxtable[3]\n400 mW"]
    end
    P0 --> T0
    P1 --> VT0
    P2 --> VT1
    P3 --> VT2
    P4 --> VT3

    classDef bad fill:#3a1a1a,stroke:#c0392b,color:#f5d5d5;
    classDef good fill:#16281a,stroke:#27ae60,color:#d5f0d8;
    class T0,VT0 bad;
    class VT1,VT2,VT3 good;
```

Atvaizdavimas toks: `vtx_power = N` išrenka `vtxtable[N-1]`. Reikšmė `0` nėra „pirmas įrašas" — ji už diapazono ribų, o už ribų reiškia pit. Reikšmė `1` *yra* tinkama, bet standartinėje Emax tipo lentelėje `vtxtable[0]` yra **PIT** įrašas. Taigi dvi žemiausios reikšmės, kurias „natūraliai" pasiekiau, abi reiškė pit, o tikras 400 mW lizdas ties `vtxtable[3]` reikalavo `4`, kurio niekada neįvedžiau.

---

## Ką iš tikrųjų įvedžiau, ir ką gavau

Mano klaidinga konfigūracija pernešė „nuo 0" prielaidą tiesiai per visą eilutę:

```
vtx 0 5 0 0 0 1000 1250
vtx 1 5 0 0 1 1250 1500
vtx 2 5 0 0 2 1500 1750
vtx 3 5 0 0 3 1750 2000
set vtx_power = 1
```

Atkreipk dėmesį, kad `5` 2-ame lauke yra *teisingas* — AUX6, skaičiuojama nuo 0. Klaida yra priešpaskutiniame lauke — galios indekse — kiekvienoje eilutėje. Štai žala, ratuko padėtis po padėties:

| Ratuko zona | Įvedžiau `vtx_power` | Adresuoja | Tikėjausi | Iš tikrųjų gavau |
|------------|---------------------:|-----------|-----------|------------------|
| žemiausia  | `0` | netinkama | PIT (gerai) | **PIT** |
| žema       | `1` | vtxtable[0] | 25 mW | **PIT** |
| vidurinė   | `2` | vtxtable[1] | 100 mW | **25 mW** |
| aukšta     | `3` | vtxtable[2] | 400 mW | **100 mW** |
| —          | `4` | vtxtable[3] | — | **400 mW (niekada neadresuota)** |

Viskas pasislinko žemyn tiksliai per vieną eilutę. Ir mano lentelės viršus baigėsi ties 100 mW, o įrašas, kurio iš tikrųjų norėjau, `vtxtable[3]` = 400 mW, liko nepasiekiamas, nes jokia mano konfigūracijos eilutė niekada nepaduodavo `4`.

Buvo ir antras, klastingesnis nukentėjęs: `set vtx_power`.

```
set vtx_power = 1
```

Tai **atsarginė** (fallback) galia, kurią Betaflight taiko, kai AUX reikšmė neatitinka jokio sukonfigūruoto diapazono. Nustačiau `1` galvodamas „žemiausia tikra galia". Bet `set vtx_power` *taip pat* skaičiuojamas nuo 1 ir rodo į tą pačią lentelę — tad `1` = `vtxtable[0]` = **PIT**. Štai kodėl bet kuri ratuko padėtis, prasprūdusi tarp mano diapazonų, nenukrisdavo į žemą galią; ji nukrisdavo į pit. Du atskiri „nuo 1" laukai, viena klaidinga prielaida, ir efektas susideda.

---

## Sprendimas

Pastumk kiekvieną galios indeksą vienu aukštyn ir, jau būdamas čia, sutvarkyk diapazonus:

```
vtx 0 5 0 0 2 900 1333
vtx 1 5 0 0 3 1333 1666
vtx 2 5 0 0 4 1666 2100
set vtx_power = 2
```

- **`2` / `3` / `4`** dabar teisingai išrenka 25 / 100 / 400 mW. 400 mW pagaliau pasiekiami ratuko viršuje.
- **`set vtx_power = 2`** paverčia atsarginę reikšmę tikrais 25 mW vietoj pit, tad neatitikusi ratuko padėtis nukrenta į žemą galią, o ne į mirusį vaizdą.
- **Diapazonai sutraukti į tris zonas** per visą ratuko eigą be tarpų tarp jų — kiekvieno diapazono `end` yra kito diapazono `start`.
- **`900` ir `2100`** kraštuose, o ne `1000`–`2000`, sąmoningai: per CRSF kanalų galiniai taškai šiek tiek peršoka — ratuko viršus rodo apie **~2012 µs**, ne švarų 2000, o apačia nukrenta žemiau 1000. Diapazonas, kuris baigiasi tiksliai ties `2000`, palieka patį ratuko viršų neatitiktą, o tai — dėka aukščiau aprašytų atsarginės reikšmės spąstų — mane ties maksimaliu ratuku įmesdavo tiesiai į pit. Praplėtimas iki `900`–`2100` sugeria peršokimą, tad galiniai taškai lieka pririšti prie zonų, kurių noriu.

Perkrauk, ir ratukas daro tai, ką siūlo fizinis pasukimas: apačia = žema, vidurys = vidutinė, viršus = 400 mW. Jokių pit netikėtumų.

---

## Ką verta įsiminti

Betaflight CLI *yra* nuosekli — tiksliai iki tos akimirkos, kai ji tokia nustoja būti, viename lauke, be jokio įspėjimo. `aux_channel` skaičiuojamas nuo nulio, kaip ir viskas kita, ką esi įsisavinęs. `vtx_power` skaičiuojamas nuo vieneto, nes tai `vtxtable` indeksas, o ta lentelė indeksuojama nuo 1. Ta pati komanda, priešingos konvencijos, per vieną lauką viena nuo kitos.

Jei tavo VTX užstrigęs pit režime arba 400 mW tiesiog nepasirodo, kad ir kaip suksi ratuką, pirma nesigriebk lituoklio. Skaičiuok galios reikšmes nuo **1**, patikrink, ar `vtxtable[0]` yra PIT įrašas, ir atmink, kad `set vtx_power` žaidžia pagal tą pačią „nuo 1" taisyklę. Aparatinė įranga niekada nebuvo sugedusi. Sugedęs buvo indeksas.
