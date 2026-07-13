---
title: "Dažnos FPV buildo klaidos"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "troubleshooting", "motors", "esc", "build"]
---

Rinkinys „pirmų 10 naujo buildo minučių“ nesėkmių ir kaip jas diagnozuoti. Beveik kiekvieną iš jų patyriau asmeniškai — kai kurias ne po vieną kartą.

---

## Neteisinga vieno motoro sukimosi kryptis

**Simptomas:** dronas pakyla, bet dreifuoja arba sukasi apie yaw ašį; viena šaka iškart pasvyra žemyn davus throttle.

**Priežastis:** vienas motoras sukasi ne ta kryptimi. Motorai būna CW ir CCW variantų; kai kurie turi apkeistus bullet jungtis. Jei apkeiti bet kuriuos du iš trijų motoro laidų, motoras pasisuka atgal.

**Sprendimas:**  
Su DSHOT sukimosi kryptį apsuki be perlaidinimo (kryptis saugoma ESC'e, o ne Betaflight `set` kintamajame):

- **Betaflight Motors tab** (propai NUIMTI): pažymėk *Reverse* varnelę probleminiam motorui. Betaflight išsiunčia DShot komandą, kuri įrašo naują kryptį į tą ESC.
- Arba **BLHeli_32 / AM32 konfigūratoriuje**: nustatyk Motor Direction (Normal / Reversed) atitinkamam ESC.

Arba fiziškai apkeisk bet kuriuos du iš trijų fazės laidų problemiškame motore.

---

## Visi motorai sukasi ne ta kryptimi

**Simptomas:** dronas kiekvieną kartą iškart apsiverčia ta pačia kryptimi davus throttle. Gali arm ir iškart nukristi.

**Priežastis:** visi keturi motorai apsukti — arba ESC buvo įrašytas su apsuktais numatytaisiais, arba visi motorai prilaidinti atbulai.

**Diagnozė:** Betaflight Motors tab'e (propai NUIMTI) suk kiekvieną motorą ir palygink sukimąsi su [Betaflight motorų išdėstymo schema](https://betaflight.com/docs/wiki/guides/current/Motor-Spin-Directions) tavo rėmui.

**Sprendimas:** **Motors tab'e** (propai NUIMTI) pažymėk *Reverse* visiems keturiems motorams arba nustatyk juos Reversed BLHeli_32 / AM32 konfigūratoriuje. Jei tik kai kurie neteisingi, apsuk tik juos.

---

## Propai ant ne tų motorų (dronas apsiverčia armindamas)

**Simptomas:** dronas iškart apsiverčia į vieną pusę tą akimirką, kai duodi throttle, nesvarbu koks derinimas.

**Priežastis:** CW propas ant CCW motoro arba propai priskirti ne toms šakoms.

**Sprendimas:** patikrink Betaflight motorų išdėstymo schemą. Kiekviena motoro pozicija pažymėta 1–4 su laukiama sukimosi kryptimi. CW propai (priekinė briauna sulinkusi pagal laikrodžio rodyklę, žiūrint iš viršaus) eina ant CCW besisukančių motorų ir atvirkščiai.

Visada pirma suk motorus su NUIMTAIS propais ir vizualiai patikrink kryptį prieš uždėdamas propus.

---

## ESC desync

**Simptomas:** motoras trūkčioja arba trumpam sustoja po apkrova; staigus throttle atsako praradimas; girdimas spragtelėjimas/trūkčiojimas, po kurio seka apsivertimas.

**Priežastis:** ESC praranda sinchronizaciją su motoro back-EMF signalu. Dažni sukėlėjai: agresyvus greitėjimas, susidėvėję guoliai, per didelis RPM ESC'o PWM dažniui, blogas filtravimas.

**Sprendimo variantai:**
1. Sumažink RPM filtravimo poreikį — įjunk RPM filtrą (bidirectional DSHOT)
2. Padidink ESC PWM dažnį: 48 kHz arba 96 kHz (žr. [ESC kHz](../../motors-esc/esc-khz/))
3. Sumažink motoro timing'ą, jei naudoji advance timing
4. Patikrink motoro guolius — susidėvėję guoliai sukelia netolygų back-EMF
5. Sumažink D-term, jei osciliacija verčia motorus greitai keisti kryptį

---

## FC pasileidžia su propais ir iškart arm'inasi

**Simptomas:** motorai suksti iškart įjungus maitinimą.

**Priežastis:** arm jungiklis buvo paliktas ARM padėtyje boot'o metu arba `motor_stop` išjungtas ir throttle kalibracija nesutvarkyta.

**Sprendimas:** visada įjunk maitinimą su arm jungikliu DISARM padėtyje. Įjunk prearm arba arm switch mode Modes tab'e. Niekada nenaudok throttle-stick arm skrydyje be atskiro arm jungiklio.

---

## OSD nerodo

**Simptomas:** juodas vaizdas arba vaizdas be OSD sluoksnio.

**Priežastis (analog):** OSD lustas neinicializuotas, ne tas UART arba kamera prijungta ne prie FC vaizdo įėjimo (prijungta tiesiai prie VTX).

**Priežastis (digital):** MSP OSD neįjungtas, priskirtas ne tas serial portas.

```
# For digital systems — set MSP OSD
set osd_displayport_device = MSP
set displayport_msp_serial = 1   # match the UART connected to air unit
save
```

---

## Vaizdo triukšmas / linijos analoginiame sraute

**Simptomas:** horizontalios linijos, riedantis triukšmo raštas arba visiškas baltas vaizdas.

**Dažnos priežastys:**
- VTX įsijungia anksčiau, nei kamera turi stabilų maitinimą — pridėk mažą LC filtrą ant VTX maitinimo
- ESC perjungimo triukšmas patenka į vaizdo įžeminimą — atskirk maitinimo linijas, pridėk kondensatorių ant baterijos laidų
- VTX ir kamera dalinasi triukšminga 5V linija — kamerai naudok atskirą LC filtruotą reguliatorių

---

## Neleidžia arm (Prearm / Throttle High)

**Simptomas:** dronas neleidžia arm; OSD rodo RXLOSS, THROTTLE arba ANGLE arm vėliavą.

**Sprendimo sąrašas:**
1. Throttle stick'as minimume
2. Arm jungiklis DISARM padėtyje prieš įjungiant maitinimą
3. RC ryšys prijungtas (patikrink RXLOSS vėliavą)
4. Angle mode (jei įjungtas) — horizontas turi būti maždaug lygus
5. GPS fix reikalingas, jei nustatytas `GPS_FIX` arm reikalavimas

Paleisk `status` CLI'e, kad pamatytum visas aktyvias arm vėliavas — jis tiksliai pasako, kodėl dronas su tavimi nekalba.
