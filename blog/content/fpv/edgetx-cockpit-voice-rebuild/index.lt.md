---
title: "9 dalis: perdarymas, sugrupuotas skrydžio tvarka"
date: 2026-08-16T17:00:00+03:00
description: "Nuo nulio sudarytas išdėstymas: galiojimo pagalbiniai L1 iki L4, tada įrašymas, baterija ir GPS ta tvarka, kuria juos naudoju."
summary: "Nuo nulio sudarytas išdėstymas: galiojimo pagalbiniai L1 iki L4, tada įrašymas, baterija ir GPS ta tvarka, kuria juos naudoju. Plius minimalios įtampos įgarsinimas."
draft: false
toc: true
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - loginiai-jungtukai
  - prearm
  - telemetrija
  - minimali-itampa
  - gps-rescue
keywords: ["EdgeTX loginis jungtukas AND pagalbinis", "EdgeTX RxBt minimumo įgarsinimas", "EdgeTX prearm įspėjimas"]
series:
  - EdgeTX Cockpit Voice
---

> **EdgeTX Cockpit Voice**, 9 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 8 dalis: keturi dalykai, kurie čia negerai](/fpv/edgetx-cockpit-voice-whats-wrong/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)

[8 dalis](/fpv/edgetx-cockpit-voice-whats-wrong/) išvardijo keturis dalykus, kurie
negerai konfigūracijoje, su kuria skraidau šiandien. Ši dalis yra sprendimas,
suprojektuotas, bet dar neįrašytas.

## Šitą išbandysiu kaip kitą: pilnas pergrupavimas

Viskas aukščiau yra tai, su kuo realiai skraidau šiandien, netvarką įskaitant. Šį
perdarymą užrašau dalinai tam, kad tikrai imčiausi.

> **Pastaba apie numeraciją:** šis skyrius yra nuo nulio sudarytas išdėstymas, tad
> `L` numeriai žemiau **nereiškia** to, ką jie reiškia anksčiau šioje serijoje.
> Konfigūracijoje, su kuria skraidau šiandien, `L1` yra `RxBt < 4,0 V`; perdaryme
> `L1` yra galiojimo pagalbinis. Skaityk tuos du išdėstymus kaip atskirus dokumentus.

Nes atvira mano dabartinės konfigūracijos problema nėra kuris nors vienas jungtukas
— problema ta, kad ji **priaugo.** Įtampos taškus dėjau tada, kai apie juos
pagalvodavau, tada tarp jų įsprausdavau GPS ir aukštį, ir rezultatas yra vienuolika
jungtukų ta tvarka, kuria juos atsitiktinai sugalvojau. Niekas nėra blogai. Bet ir
nieko nerasi.

Tad: trys grupės, ta tvarka, kuria jas naudoju realiame gyvenime. **Įrašymas, tada
baterija, tada GPS.** Tai yra skrydžio seka, paleisk žurnalą, patikrink paketą,
palauk fiksavimo.

### Pagalbiniai pirmiausia, diapazono apačioje

Numeracijos perdarymas duoda vieną dalyką dovanų. Praeitą kartą tuos pagalbinius
buvau nubraižęs L12–L16, ir turėjau įspėti, kad jie vėluos vienu vykdymo ciklu
prieš tuos, kurie juos naudoja. Padėjus juos į **L1–L4**, ta išlyga išnyksta
visiškai: EdgeTX pereina L1 → L64 kartą per ciklą, tad pagalbinis L1 pozicijoje
visada yra šviežias, kai jį perskaito L5.

```yaml
0:                               # = L1  „telemetrija realiai yra“
   func: FUNC_VPOS
   def: "tele(14),5"             # RxBt > 0,5   (prec:1, tad 5 = 0,5 V)
   andsw: "NONE"

1:                               # = L2  baterijos įspėjimai aktyvūs IR galioja
   func: FUNC_AND
   def: "SW52,L1"                # žalias bat mygtukas + galiojimas

2:                               # = L3  GPS pranešimai aktyvūs IR galioja
   func: FUNC_AND
   def: "SW62,L1"                # mėlynas gps mygtukas + galiojimas

3:                               # = L4  priešskrydžio etapas IR galiojimas
   func: FUNC_AND
   def: "SE1,L1"                 # SE vidurys + galiojimas
```

`L1` yra pats svarbiausias. Būtent jis sustabdo visų laiptų rėkimą ant šviežios
baterijos, nes `RxBt`, sėdintis ant savo pradinės `0,0` reikšmės, neišlaiko
`RxBt > 0,5`, tad visi tolesni vartai yra neteisingi, kol neatkeliauja tikri
duomenys.

### 1 grupė — Įrašymas

Įrašymui loginio jungtuko nereikia visai; tai specialioji funkcija, valdoma
tiesiogiai raudonu mygtuku. Ji eina **pirma** specialiųjų funkcijų sąraše vien tam,
kad sąrašas būtų skaitomas skrydžio tvarka.

| Paleidėjas | Funkcija |
|---|---|
| `SW42` (raudonas `log`) | `LOGS` 0,3 s |

### 2 grupė — Baterija, L5 → L13

Laiptai mažėjančia įtampos tvarka, kuri pagaliau yra ir skaitinė tvarka.

| LS | Testas | Vartai | Garsas |
|---|---|---|---|
| **L5** | `RxBt > 4,2` | `L2` | `ready` — šviežio paketo savitikra |
| **L6** | `RxBt < 4,0` | `L2` | `Wrn1` |
| **L7** | `RxBt < 3,8` | **`L3`** | `rth` — žr. pastabą |
| **L8** | `RxBt < 3,6` | `L2` | `Sirn`, 1 s |
| **L9** | `RxBt < 3,5` | `L2` | `lowbat`, 5 s |
| **L10** | `RxBt < 2,9` | `L2` | `Alrm` |
| **L11** | `\|Δ\|≥ RxBt- 0,1` | `L2` | **įgarsina naują minimumą** |
| **L12** | `RxBt < 3,8` | `L4` | priešskrydžio klaida — `Sirn`, 2 s |
| **L13** | `RxBt > 3,8` | `L4` | priešskrydžio patikra gerai, ištartas patvirtinimas |

**`L7` vartai yra sąmoninga išimtis.** Jis yra baterijos grupėje, nes tai įtampos
slenkstis, bet pririštas prie *gps* pagalbinio, o ne prie baterijos, nes
„apsisuk“ yra tolimo skrydžio įspėjimas. Ant whoop'o viešbučio kambaryje jis būtų
triukšmas. Grupuok pagal tai, ką jungtukas matuoja; vartus dėk pagal tai, kada nori
tai išgirsti. Tai skirtingi klausimai, ir jiems visiškai normalu nesutarti.

### L11: minimalios įtampos įgarsinimas

Šis naujas, ir būtent dėl jo pradėjau perdarinėti, o ne lopyti.

EdgeTX seka einamąjį minimumą kiekvienam telemetrijos sensoriui ir pateikia jį kaip
atskirą šaltinį — `RxBt-`. Mano telemetrijos ekranai jį jau rodo. Ko niekada
nepadariau, nepriverčiau jo *kalbėti*.

```yaml
10:                              # = L11
   func: FUNC_ADIFFEGREATER      # |Δ| >= x
   def: "tele(-14),1"            # RxBt MINIMUMAS, žingsnis 0,1 V
   andsw: "L2"
```

Sujungus su `PLAY_VALUE` ant `tele(-14)`, tai reiškia: **kiekvieną kartą, kai
skrydis užfiksuoja naują žemiausią celės įtampą, pultas ją ištaria.** Ne slenkstis,
ne įspėjimas, matavimas, perskaitytas balsu tą pačią akimirką.

Dėl to gaunu įtampos kritimo duomenis iš akceleravimų dar skrisdamas, o ne po to
juodojoje dėžėje. Stiprus akceleravimas nuleidžia paketą, `RxBt-` nusileidžia su
juo, ir aš išgirstu „trys taškas keturi“. Tai skaičius, kuris man rūpi labiausiai ir
kurio ore niekada neturėjau.

Dvi detalės, dėl kurių tai veikia teisingai:

**Naudok `|Δ|`, ne `Δ`.** Minimumas tik mažėja, tad skirtumas visada negatyvus —
`Δ≥x` niekada nesuveiktų. Absoliučios reikšmės forma jį pagauna.

**Nunulink seklį kiekvienam paketui, kitaip jis nenaudingas.** Einamasis minimumas,
kuris niekada nenusinulina, tiesiog atsimena blogiausią dienos momentą. Tad `L5` —
šviežio paketo detektorius, gauna *antrą* specialiąją funkciją kartu su `ready`:

| Paleidėjas | Funkcija |
|---|---|
| `L5` | `PLAY_TRACK ready` |
| `L5` | `RESET Telemetry` |

Įjungi šviežią bateriją, pultas pasako „ready“ ir tuo pačiu metu ištrina min/max
seklius, tad `RxBt-` dabar seka *šį* paketą. Vienas jungtukas, du darbai.

0,1 V žingsnis yra paties sensoriaus kvantavimas, tad tai paskelbs kiekvieną naują
minimumą. Jei freestyle metu tai pasirodys per daug kalbanti, pakelk žingsnį iki
0,2 V, slenkstis yra garso reguliatorius.

### Kodėl egzistuoja L13: tyla nėra išlaikyta patikra

`L12` ir `L13` yra pora, ir antrasis svarbesnis, nei atrodo.

Darbo seka, kurios noriu: perjungti SE į vidurį, truputį palaukti, klausytis. Jei
niekas nesiskundžia, aktyvuoti ir skristi. Problema ta, kad **tyla reiškia du
skirtingus dalykus tuo pačiu kostiumu**:

1. baterija tvarkinga, o tai atsakymas, kurio noriu
2. telemetrija dar neatkeliavo, tad niekas neturi nuomonės

Ausimi jie neatskiriami. Perjungiu į vidurį, greitai aktyvuoju, negirdžiu nieko,
nusprendžiu, kad paketas geras, ir pakylu su baterija, kurios niekas neišmatavo.
Patikra „išlaikoma“ stipriausiai būtent tada, kai ji man nepasakė nieko.

Aviacija tai išsprendė seniai, ir taisyklę verta nusižiūrėti tiesiogiai:
**priešskrydžio patikra privalo duoti pozityvų rodmenį, o ne negatyvo nebuvimą.**
Patikra, kuri išlaikoma tylėdama, išlaikoma ir tada, kai ji sugedusi.

Tad `L13` suveikia prie `RxBt > 3,8` etape ir tai pasako balsu. Dabar perjungus SE į
vidurį nutinka lygiai vienas iš dviejų dalykų: išlaikyta patikra, kurią girdžiu,
arba įspėjimas. Tyla nebėra atsakymas. Ji reiškia „palauk ilgiau“.

### 3 grupė — GPS, L14 → L16

| LS | Testas | Vartai | Garsas |
|---|---|---|---|
| **L14** | `Sats > 10` | `L3` | **įgarsina palydovų skaičių** |
| **L15** | `Sats < 6` | `L3` | `gpsoff` — fiksavimas pablogėjo |
| **L16** | `\|Δ\|≥ GAlt 120` | `NONE` | `warnng` — aukštis |

**Įsigyti prie 10, įspėti prie 6, tarpas yra sąmoningas.** Dešimt palydovų yra
„pakankamai tvirta, kad kiltum“. Šeši yra „GPS Rescue nebėra tai, kuo pasitikėčiau“.
Nustačius abu į tą patį skaičių, jis kalbėtų kiekvieną kartą, kai skaičius
supleveniuotų per ribą; keturių palydovų neveiklumo zona reiškia, kad kiekvienas
pranešimas yra tikra būsenos kaita. Žemesnį skaičių nustatyk pagal savo
`gps_rescue_min_sats`.

`L14` pakeičia senąją Sats jungtukų porą. Skaičiaus skelbimas prie 10, o ne prie 6,
yra tas pakeitimas, kurio norėjau: žemiau dešimties nenoriu nuolatinio komentaro,
kol laukiu, noriu žinoti, kada *paruošta*.

Jei mieliau turėtum nuolatinį skaičių, kai palydovai atsiranda ir išnyksta, tai
`|Δ|≥1` ant `Sats`, pririštas už `Sats > 10` pagalbinio: dar vienas jungtukas ir
gerokai daugiau kalbėjimo.

`L16` sąmoningai lieka be vartų. Aukščio riba taikoma kiekvienam aparatui kiekvieno
skrydžio metu, tad tai vienintelis įspėjimas, kuris neturėtų turėti išjungimo
mygtuko.

### Specialiųjų funkcijų tvarka

Sąrašas pagaliau skaitomas skrydžio tvarka: žurnalas, baterija, GPS.

```text
 0  SW42  LOGS         0.3s          <- įrašymas
 1  L5    PLAY_TRACK   ready         <- baterija
 2  L5    RESET        Telemetry
 3  L6    PLAY_SOUND   Wrn1
 4  L7    PLAY_TRACK   rth
 5  L8    PLAY_SOUND   Sirn    1s
 6  L9    PLAY_TRACK   lowbat  5s
 7  L10   PLAY_SOUND   Alrm
 8  L11   PLAY_VALUE   tele(-14)     <- minimalios įtampos įgarsinimas
 9  L12   PLAY_SOUND   Sirn    2s
10  L13   PLAY_TRACK   <patikros pranešimas>
11  L14   PLAY_VALUE   tele(22)      <- GPS, palydovų skaičius
12  L15   PLAY_TRACK   gpsoff
13  L16   PLAY_TRACK   warnng
```

### Kodėl pagalbinių netiesiškumas to vertas

Vartų politika dabar gyvena **vienoje vietoje kiekvienai posistemei**, o ne
nukopijuota vienuolika kartų. Kai imsiuosi normalaus prearm jungtuko, greičiausiai
`SA`, etapavimas nuo SE nukels vienu pakeitimu: `L4` antrasis operandas pasikeis, ir
visa priešskrydžio elgsena nuseks paskui. Nė vienas slenkstis nebus paliestas.

Slenksčių logika, aktyvavimo logika ir galiojimo logika kiekviena gauna savo
sluoksnį, ir nė viena nieko nežino apie kitas.

### Ką tikrinti, o ne tikėti

Esu tikras dėl struktūros ir dėl to, kad `L<n>` yra kreipimosi forma, mano esamas
`customFn` blokas jau naudoja `swtch: "L3"`. Trys dalykai čia yra spėjimai:

- tiksli `FUNC_AND` rašyba ir jo dviejų operandų `def` formatas, nes dabartinėje
  konfigūracijoje nėra AND tipo jungtuko, iš kurio būtų galima nusikopijuoti
- kad `tele(-14)` yra pasirenkamas kaip loginio jungtuko operandas. Kaip *šaltinis*
  jis tikrai egzistuoja, mano telemetrijos ekranai jį naudoja, bet dar
  nepatvirtinau, ar pasirinkimo sąrašas siūlo min/max variantus loginio jungtuko
  viduje
- `RESET Telemetry` specialiosios funkcijos `def` formatas

Sukurk tai pulto sąsajoje, eksportuok modelį ir perskaityk, ką EdgeTX realiai
užrašė. Tada tikėk tuo, ne šiuo.

### Kur tai palieka seriją

Kiekvienas įspėjimas šiose devyniose dalyse sudėtas iš telemetrijos, kuri į pultą
jau atkeliaudavo, iš sensorių, kurie jau buvo atrasti, naudojant programinę įrangą,
kuri jau buvo įdiegta. Prie nė vieno aparato nieko nepridėta. Jokio Lua, jokios
papildomos aparatūros, nė vieno gramo kilimo masės.

Pasikeitė tik tai, kad informacija dabar keliauja į mano ausis, o ne į ekrano
kampą, į kurį nežiūriu.

Tai žemesnė riba, nei skamba, ir kartu tai didžioji dalis vertės. Tas skrydis, kai
įtampa tyliai praslydo pro negrįžimo tašką, o aš buvau užsiėmęs malonumu, nebesikartoja.
Kažkur apie pusę talpos balsas pasako „grįžk namo“, ir aš apsisuku dar turėdamas
kuro bake.

Orlaivis žinojo visą laiką. Jam tik reikėjo būdo tai pasakyti.

---

*Jei sukursi tvarkingesnę bet kurios šios dalies versiją, ypač normalų aukščio nuo
pakilimo taško įspėjimą arba nesluoksniuojamus slenksčių laiptus, labai norėčiau
tai pamatyti.*

---

> **Serija:** EdgeTX Cockpit Voice, 9 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 8 dalis: keturi dalykai, kurie čia negerai](/fpv/edgetx-cockpit-voice-whats-wrong/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)
