---
title: "Kaip daviau savo dronui piloto kabinos balsą, 1 dalis: kodėl dronas turi su tavimi kalbėti"
date: 2026-08-16T09:00:00+03:00
description: "Įtampos skaičius OSD kamputyje yra sąsajos, o ne piloto klaida. Kodėl priverčiau savo RadioMaster GX12 kalbėti ir vienas skrydžio valdiklio nustatymas, nuo kurio viskas priklauso."
draft: false
toc: true
weight: 1
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - radiomaster-gx12
  - elrs
  - crsf
  - telemetrija
  - betaflight
  - report-cell-voltage
  - lihv
keywords: ["EdgeTX garsinis baterijos ispejimas", "report_cell_voltage Betaflight", "FPV celes itampos telemetrija", "RadioMaster GX12 nustatymai"]
series:
  - EdgeTX Cockpit Voice
thumbnail: "cover.jpg"
---

Tą skrydį žinai. Esi gerokai nuskridęs, reljefas geras, linijos plaukia, o tu
visas esi akiniuose. Kažkur OSD kamputyje įtampos skaičius jau pusantros minutės
tyliai leidžiasi, ir tu į jį nė karto nepažiūrėjai, nes buvai užsiėmęs skridimu.
Tada OSD pradeda mirksėti, ir tu suskaičiuoji: atstumas iki namų, priešpriešinis
vėjas, likęs įtampos kritimas. Ir skaičiai atsako: ne.

Tas skrydis baigiasi pasivaikščiojimu. Kartais, pasivaikščiojimu su maišeliu.

Labiausiai mane šiame gedimo scenarijuje visada trikdė tai, kad tai yra
**tik** sąsajos problema. Duomenys buvo visą laiką. Pultas žinojo. Dronas žinojo.
Vienintelė sulūžusi grandies vieta buvo ta, kad informacija buvo pateikta
mažais švytinčiais skaitmenimis periferiniame lauke žmogui, kuris tuo metu
koncentravosi į visai kitą dalyką.

## Tikras orlaivis su tavimi taip nepasielgtų

Štai kas man pasirodė absurdiška. Pasodink pilotą į „Cessną“ ir orlaivis
neleis, kad mažo kuro būklė būtų vizualinė detalė, kurią gali praleisti. Jis
pasakys. Garsiai. Ir pakartos. Įspėjimai apie neišleistą važiuomenę, apie
kritinį atakos kampą, aukščio pranešimai, įspėjimai apie reljefą. Visas
šimtmetis aviacijos žmogiškųjų faktorių inžinerijos susivedė į vieną išvadą:
**laiko atžvilgiu kritinėms būsenos kaitoms garsas nugali vaizdą, nes garsui
nereikia, kad pilotas kur nors pažiūrėtų.**

Ir vis dėlto standartinė FPV konfigūracija 250 gramų orlaiviui, kurio skrydžio
laikas keturios minutės, yra... skaičius ekrano kampe.

Tai aš tai sutvarkiau. Mano GX12 dabar su manimi kalba. Ne Lua skriptu, ne kažkuo
egzotišku, tiesiog EdgeTX loginiais jungtukais ir specialiosiomis funkcijomis,
kurios programinėje įrangoje sėdėjo visą laiką.

Tai pirmas kartas, kai tai susikonfigūravau, ir noriu pasakyti atvirai:
**kai kurias dalis galima padaryti kur kas mažiau nerangiai.** Parodysiu
konkrečiai, kur mano variantas nerangus ir kodėl, nes tai naudingiau nei
apsimesti, kad viską padariau teisingai iš pirmo karto. Bet esmė veikia, ir
vienas konkretus įspėjimas, pranešimas „grįžk namo“ maždaug prie pusės
talpos, man tikrai išgelbėjo skrydžius tolimose misijose. Jis duoda signalą
pradėti planuoti kelią atgal, kol dar turiu energijos biudžetą tai padaryti, o
ne atrasti problemą tada, kai biudžetas jau išleistas.

![RadioMaster GX12](cover.jpg)

## Nulinis žingsnis: tegu visi dronai kalba ta pačia kalba

Tai vienintelis pakeitimas, dėl kurio visa sistema tampa įmanoma, ir jis
atliekamas skrydžio valdiklyje, ne pulte.

Pagal nutylėjimą CRSF baterijos kadras praneša **paketo įtampą**. Kaip visai
flotilei bendras slenkstis tai yra nenaudinga, nes mano flotilė apima nuo 1S
iki 4S. Slenkstis „3,5 V“ nieko nereiškia, kai vienas aparatas skrenda su vienu
18650 elementu, o kitas, su 4S LiHV paketu. Man reikėtų atskiro slenksčių
rinkinio kiekvienam modeliui, palaikomo ranka, amžinai.

Todėl visiems aparatams nustačiau pranešti **vidutinę celės įtampą**.
Betaflight'e tai vienas parametras:

```text
set report_cell_voltage = ON
save
```

Tas pats yra ir Betaflight konfigūratoriuje, skirtuke *Power & Battery* —
„Report cell voltage instead of pack voltage in telemetry“. Skrydžio valdiklis
padalija paketo įtampą iš savo nustatyto celių skaičiaus dar prieš tai, kai
reikšmė pasiekia telemetrijos kadrą.

Dabar `3,5 V` reiškia tą patį fizinį dalyką ir ant 1S whoop'o, ir ant 2S
riperių, ir ant 4S trijų colių. Vienas slenksčių laiptų rinkinys visai flotilei.

> **Dėl INAV atitikmens:** visi čia aktualūs aparatai pas mane skrenda su
> Betaflight, tad patikrinta tik Betaflight'e. Jei naudoji INAV — pasitikrink
> parametro pavadinimą, o nedaryk prielaidos, kad jis identiškas. Aš to
> neišmatavau.

### Kodėl nedalinti EdgeTX pusėje?

Galima. EdgeTX leidžia telemetrijos sensoriui nustatyti savą **Ratio** koeficientą,
tad galėtum leisti valdikliui pranešti paketo įtampą ir dalinti iš celių
skaičiaus jau pulte.

Aš to sąmoningai nedariau, ir tai matosi konfigūracijoje. RxBt sensoriui
netaikoma jokia korekcija:

```yaml
telemetrySensors:
   14:
      id1:
         id: 8              # CRSF kadras 0x08, BATTERY_SENSOR
      id2:
         instance: 0
      label: "RxBt"
      unit: 1               # voltai
      prec: 1               # vienas skaičius po kablelio
      cfg:
         custom:
            ratio: 0        # be mastelio
            offset: 0       # be poslinkio
```

Dvi priežastys daryti tai orlaivyje:

1. **Pulte koeficientas yra vienam modeliui, o celių skaičius, vienam
   paketui.** Dalyba iš keturių pulto pusėje tampa neteisinga tą pačią
   sekundę, kai tą patį aparatą paskraidinu su 3S paketu.
2. **LiHV sugriauna fiksuotą spėjimą.** Mano trijų colių skrenda su 4S LiHV —
   4,35 V celei pilnai įkrautas, tai yra 17,4 V pakete. Pultas, kuriam pasakyta
   „laikyk, kad 4S“, susitvarko, bet pultas, kuris *bando nustatyti* celių
   skaičių iš jau padalintos reikšmės, ne. Skrydžio valdiklis savo celių skaičių
   jau žino iš tikros nustatymo logikos. Tegu skaičiuoja tas, kuris žino.

Kompromisas atviras: darant tai valdiklio pusėje, kiekvienam naujam aparatui
reikia tos CLI eilutės, ir jei pamirši, įspėjimai suveiks absurdišku momentu. Tad
tai priklauso naujo aparato paruošimo sąrašui, kartu su tais dalykais, kurių irgi
nesimato.

Vienas slenksčių rinkinys dabar reiškia tą patį fizinį dalyką kiekvienam mano
aparatui. Toliau: trys mygtukai, kurie nusprendžia, kuriems įspėjimams leidžiama
kalbėti, ir AND vartai, neleidžiantys jiems vienas kitam po kojų kliudyti.

**Toliau:** [3 dalis, trys mygtukai, trys spalvos ir AND vartai](/fpv/edgetx-cockpit-voice-buttons/)
