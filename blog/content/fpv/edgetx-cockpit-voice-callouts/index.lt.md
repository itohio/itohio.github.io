---
title: "4 dalis: ką pultas iš tikrųjų pasako"
date: 2026-08-16T14:00:00+03:00
description: "Baterijos laiptai, palydovų pranešimai ir aukščio signalas, kuris suveikia nuo pokyčio, o ne nuo absoliutaus aukščio. Su savitikra, kuri man patinka labiausiai."
summary: "Baterijos laiptai, palydovų pranešimai ir aukščio signalas, kuris suveikia nuo pokyčio, o ne nuo absoliutaus aukščio. Su savitikra, kuri man patinka labiausiai."
draft: false
toc: true
weight: 4
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - specialiosios-funkcijos
  - gps-rescue
  - betaflight
  - telemetrija
  - aukstis
  - easa
keywords: ["EdgeTX specialiosios funkcijos", "EdgeTX palydovu skaiciaus pranesimas", "EdgeTX aukscio ispejimas 120m"]
series:
  - EdgeTX Cockpit Voice
thumbnail: "special-functions-1.jpg"
---

> **EdgeTX Cockpit Voice**, 4 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 3 dalis: Trys mygtukai, trys spalvos ir AND vartai](/fpv/edgetx-cockpit-voice-buttons/)  ·  [5 dalis: Iš kur atkeliauja pranešimai ›](/fpv/edgetx-cockpit-voice-audio-and-sharing/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)

Jungtukai iš [3 dalies](/fpv/edgetx-cockpit-voice-buttons/) yra tik loginės
reikšmės, kol kažkas jų nepaverčia garsu. Tas kažkas yra EdgeTX specialiosios
funkcijos, ir čia konfigūracija nustoja būti struktūra ir tampa balsu mano ausyje.

## Specialiosios funkcijos

Čia loginė reikšmė tampa garsu.

![Specialiosios funkcijos, pirmas puslapis](special-functions-1.jpg)
![Specialiosios funkcijos, antras puslapis](special-functions-2.jpg)

```yaml
customFn:
   0:  { swtch: "L3",   func: PLAY_TRACK, def: "rth,1,1x"     }
   1:  { swtch: "L4",   func: PLAY_VALUE, def: "tele(22),1,1x" }
   2:  { swtch: "L1",   func: PLAY_SOUND, def: "Wrn1,1,1x"    }
   3:  { swtch: "L2",   func: PLAY_SOUND, def: "Sirn,1,1"     }
   4:  { swtch: "L5",   func: PLAY_TRACK, def: "gpson,1,1x"   }
   5:  { swtch: "L7",   func: PLAY_TRACK, def: "gpsoff,1,1x"  }
   6:  { swtch: "L8",   func: PLAY_TRACK, def: "lowbat,1,5"   }
   7:  { swtch: "L9",   func: PLAY_SOUND, def: "Sirn,1,2"     }
   8:  { swtch: "L6",   func: PLAY_TRACK, def: "warnng,1,1x"  }
   9:  { swtch: "SW42", func: LOGS,       def: "3,1"          }
   10: { swtch: "L10",  func: PLAY_TRACK, def: "ready,1,1x"   }
   11: { swtch: "L11",  func: PLAY_SOUND, def: "Alrm,1,1x"    }
```

Trečias `def` laukas yra pakartojimo intervalas. `1x` reiškia „paleisti vieną
kartą per suveikimą“. `1` reiškia kas sekundę, `5`, kas penkias sekundes. Tas
skirtumas svarbesnis, nei atrodo — žr. žemiau.

Bendra elgsena yra tokia:

### Baterijos laiptai — žalias mygtukas

| Vienai celei | Jungtukas | Garsas | Kartojimas | Reikšmė |
|--------------|-----------|--------|------------|---------|
| **> 4,2 V** | L10 | `ready` | vieną kartą | Šviežias paketas — sistema aktyvi ir kalba |
| **< 4,0 V** | L1 | `Wrn1` | vieną kartą | Jau skrendi, laikrodis tiksi |
| **< 3,8 V** | L3 | `rth` | vieną kartą | *Maždaug pusė talpos — apsisuk* |
| **< 3,6 V** | L2 | `Sirn` | 1 s | Grįžk namo |
| **< 3,5 V** | L8 | `lowbat` | 5 s | Leiskis, kur bebūtum |
| **< 2,9 V** | L11 | `Alrm` | vieną kartą | Paketą jau sugadinai |

`ready` pranešimas prie > 4,2 V visai nėra įspėjimas. Tai **savitikra**. Kai įjungiu bateriją ir pultas pasako „ready“,
vienu žodžiu ką tik patvirtinau, kad: telemetrija teka, RxBt sensorius gyvas,
`report_cell_voltage` tikrai nustatytas *šiame* aparate ir garso kelias veikia.
Visi keturi visos sistemos gedimo scenarijai patikrinti vienu žodžiu, dar prieš
pakylant. Jei įjungus bateriją pultas tyli, kažkas grandinėje sugedę, ir noriu
tai žinoti *dabar*, o ne 800 metrų atstumu.

Išlyga dėl LiHV: 4,35 V celei paketas 4,2 V slenkstį pralekia lengvai, tad
`ready` suveikia patikimai. Tuo tarpu LiPo, savaitę pastovėjęs lentynoje,
savaime išsikrauna maždaug iki 4,15 V ir slenksčio gali niekada nepasiekti. Tai,
tiesą sakant, teisinga elgsena, jis man pasako, kad paketas nėra pilnas.

**`rth` pranešimas prie 3,8 V yra tas, kuris tikrai išgelbėjo skrydžius.** Tai
grubus pusės talpos apytikslis vertinimas, sudarytas iš įtampos, o ne iš
kulonų, ir nesiruošiu apsimesti, kad jis tikslus. Bet jam ir nereikia būti
tiksliam. Jam reikia atvykti *tada, kai dar turiu energijos biudžetą į jį
sureaguoti*, o to kulonų skaitiklis, į kurį nežiūriu, nepasiekia. Atkreipk
dėmesį, kad jis pririštas prie **gps** mygtuko, o ne prie baterijos mygtuko: tai
tolimų misijų įspėjimas, o ant whoop'o viešbučio kambaryje jis būtų tik triukšmas.

Ir dar viena dalis apie tolimus skrydžius, nes ji nėra pasirenkama ir jokia
telemetrija jos nepakeičia: mano žmona visą laiką binokliu palaiko vizualų
kontaktą su orlaiviu. Garsiniai įspėjimai pasako apie *orlaivio* būklę. Apie
oro erdvę jie nepasako nieko.

### GPS pranešimai — mėlynas mygtukas

Palydovų skaičius yra tikrai netinkamas dalykas stebėti vizualiai, nes
akimirka, kai jis svarbiausias, yra ta pati akimirka, kai mažiausiai gali
pažiūrėti.

| Sąlyga | Jungtukas | Garsas | Reikšmė |
|--------|-----------|--------|---------|
| Sats > 6 | L4 | *įgarsina skaičių* | Artėja prie naudojamo — kiek dar? |
| Sats > 13 | L5 | `gpson` | Tvirtas fiksavimas, gelbėjimu galima tikėtis |
| Sats < 6 | L7 | `gpsoff` | **Fiksavimas pablogėjo skrydžio metu** |

`PLAY_VALUE` ant L4 yra maloniausia dalis, vietoj fiksuoto tono jis įgarsina
tikrą palydovų skaičių. Tad kol laukiu ant žemės, girdžiu „septyni“,
„devyni“, „vienuolika“, kai fiksavimas kaupiasi, ir žinau, ar laukti dar, ar
mesti, neatblokavęs pulto ekrano.

Realiai svarbus slenkstis yra **6**, nes maždaug ties tuo GPS Rescue tampa
kažkuo, kuo galima pasitikėti, ir tikslus skaičius visiškai priklauso nuo tavo
gelbėjimo konfigūracijos Betaflight'e arba INAV'e. Nustatyk jį pagal *savo*
`gps_rescue_min_sats`, ne pagal manąjį.

`gpsoff` įspėjimas prie Sats < 6 yra tas, kurio nesitikėjau, o dabar laikau
būtinu. **Akrobatika mažina palydovų skaičių.** Apversk aparatą, ir plokštelinė
antena nukreipta į žemę; stiprūs flipai bei power loop'ai skaičių numuša
reguliariai. Jei taip nutinka tolimo skrydžio metu ir aš apie tai nežinau,
skrendu su neveiksiančia gelbėjimo funkcija tikėdamas, kad turiu apsaugą. Vienas
žodis ausyje tai išsprendžia.

### Aukščio signalas — visada aktyvus

L6 turi `andsw: "NONE"`, jis aktyvus kiekvieno skrydžio metu, kiekviename
aparate. Skraidau pagal EASA A1/A3, ir 120 m nuo žemės riba nėra pageidavimas.

Bet čia turiu būti atviras dėl savo konfigūracijos, nes YAML perskaitymas man
apie ją kai ką atskleidė:

```yaml
   5:
      func: FUNC_ADIFFEGREATER    # |delta| >= x
      def: "tele(21),120"
```

`FUNC_ADIFFEGREATER` yra `|Δ| ≥ x` — **skirtumo** funkcija. Ji suveikia ne tada,
kai aukštis *viršija* 120 m. Ji suveikia, kai aukštis *pasikeitė* 120 m nuo savo
atskaitos taško.

Galėčiau apsimesti, kad taip ir suplanavau. Pasakysiu kitką: pasirodo, tai
labiau pagrįstas pasirinkimas, ir priežastis verta suprasti.

**`GAlt` CRSF telemetrijoje yra GPS aukštis, o ne aukštis virš pakilimo taško.**
Jei būčiau panaudojęs akivaizdų `a > x` su `GAlt` ir 120 m slenksčiu, jis rėktų
nuolat ir nuolatos, nes skraidau Lietuvoje, kur pati žemė yra maždaug
70–150 m virš jūros lygio. Signalas būtų teisingas dar prieš aparatui pakylant
iš rankos.

Skirtumo funkcija tai visiškai apeina: ji matuoja *pokytį*, tad atskaita yra
ten, kur pradėjau, ir 120 m pakilimo yra 120 m pakilimo nepriklausomai nuo
lauko aukščio. Tai gerokai artimiau AGL nei absoliutus GPS aukštis.

Tai nėra tobula, ir noriu netobulumus pavadinti, o ne užglaistyti:

- Jis suveikia ir nusileidus 120 m, nes tai absoliutus skirtumas. Nuskrisk nuo
  slėnio krašto, ir jis įspės.
- Suveikęs jis atnaujina atskaitą, tad vėl užsiveda ir suveikia po *sekančio*
  120 m pokyčio, o ne lieka užfiksuotas virš ribos.
- Tai įspėjimas, o ne riba. Jis pasako, kad pakilau aukštai. Laikytis
  reikalavimų vis tiek yra mano darbas.

**Būtent šią dalį labiausiai norėčiau pagerinti, ir geresnio varianto dar
neišmatavau.** Tikras atsakymas tikriausiai būtų išvesti realų aukštį nuo
pakilimo taško, kurį barometras jau duoda OSD, bet kuris nepasiekia `GAlt`
telemetrijos sensoriaus. Jei tai išsprendei EdgeTX'e gražiai, noriu išgirsti.

Tokia sistema, su kuria skraidau. Šeši ištarti pranešimai, trys palydovų būsenos ir
vienas aukščio signalas, kuris pasirodė matuojąs ne tai, ką maniau.


---

> **Series:** EdgeTX Cockpit Voice, 4 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 3 dalis: Trys mygtukai, trys spalvos ir AND vartai](/fpv/edgetx-cockpit-voice-buttons/)  ·  [5 dalis: Iš kur atkeliauja pranešimai ›](/fpv/edgetx-cockpit-voice-audio-and-sharing/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)
