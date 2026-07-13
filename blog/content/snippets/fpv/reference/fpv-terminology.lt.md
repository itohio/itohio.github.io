---
title: "FPV terminų žinynas"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "reference", "glossary", "terminology", "beginner"]
---

Vieno puslapio žinynas su santrumpomis ir terminais, kurie užpildo kiekvieną FPV forumo postą ir Discord serverį. Pradžioje šitą lentelę laikiau atidarytą antrame monitoriuje — tų santrumpų tiek, kad galva sukasi. Sudėliota pagal sritis, kad susiję terminai liktų kartu.

---

## Aparatūros santrumpos

| Santrumpa | Pilnas pavadinimas | Ką daro |
|---------|-----------|--------------|
| **FC** | Flight Controller | Smegenys — sukasi Betaflight/INAV, skaito gyro, siunčia komandas motorams |
| **ESC** | Electronic Speed Controller | Verčia FC motorų komandas į trifazę AC brushless motorams |
| **4-in-1 ESC** | Four-in-one ESC | Visi keturi ESC kanalai vienoje plokštėje; sėdi po FC arba virš jo stack'e |
| **VTX** | Video Transmitter | Belaidžiu būdu siunčia FPV vaizdą į goggles; paprastai 5,8 GHz analog arba 2,4/5,8 GHz digital |
| **VRX** | Video Receiver | Goggles pusės imtuvas analoginiam vaizdui; digital sistemose jį pakeičia display'us |
| **OSD** | On-Screen Display | Uždeda telemetriją (bateriją, RSSI, skrydžio režimą) ant FPV vaizdo |
| **RX / Receiver** | RC Receiver | Priima RC ryšį iš pulto ir perduoda stick'ų komandas FC |
| **TX / Transmitter** | RC Transmitter | Tavo pultas; siunčia valdymo ryšį (ELRS, CRSF ir t. t.) |
| **GPS** | Global Positioning System | Duoda poziciją, aukštį, greitį; būtinas GPS Rescue ir navigacijos režimams |
| **IMU** | Inertial Measurement Unit | Gyro + akselerometras kartu ant FC; matuoja sukimosi greitį ir linijinį pagreitį |
| **MPU / ICM** | Motion Processing Unit | Konkretūs gyro čipų prekiniai ženklai: MPU-6000, ICM-42688-P — IMU čipas |
| **BEC** | Battery Eliminator Circuit | Įtampos reguliatorius stack'e; duoda 5 V arba 9 V linijas FC, VTX, kamerai |
| **PDB** | Power Distribution Board | Paskirsto baterijos įtampą visiems keturiems ESC ir BEC; dažnai integruotas į 4-in-1 ESC |
| **XT60 / XT30** | Xt jungtys (60 A / 30 A) | Standartinės LiPo baterijų jungtys; XT60 5" ir didesniems, XT30 micro/whoop'ams |
| **Cap** | Capacitor | Low-ESR elektrolitinis kondensatorius, prilituotas prie baterijos padų; slopina įtampos šuolius, kurie „nudroppina“ FC |
| **AIO** | All-in-one | FC + ESC + (kartais) VTX vienoje plokštėje; dažnas whoop'uose ir micro'uose |
| **Stack** | FC stack | FC plokštė ir ESC plokštė, suveržtos M2/M3 standoff'ais |

---

## Motorai ir propai

| Terminas | Reikšmė |
|------|---------|
| **KV** | Motoro greičio konstanta: RPM voltui be apkrovos. 2400 KV ant 4S → ~40 000 RPM be apkrovos prie 16,8 V. Mažesnis KV = daugiau sukimo momento, didesni propai. |
| **Stator size** | Motoro statoriaus skersmuo × aukštis mm. „2306“ = 23 mm skersmuo, 6 mm statoriaus aukštis. Didesnis statorius = daugiau galios. |
| **Pole count** | Magnetų polių skaičius; veikia RPM vs elektrinį dažnį. 12N14P = 12 statoriaus dantų, 14 magnetų polių. |
| **eRPM** | Elektriniai RPM = mechaniniai RPM × (poliai / 2). Naudoja RPM filter ir DSHOT telemetrija. |
| **Prop notation** | pvz., **5148**: pirmi du skaitmenys = skersmuo colio dešimtosiomis (5,1"), kiti du = pitch dešimtosiomis (4,8"). |
| **Pitch** | Kiek propas pasistumia per apsisukimą idealiame ore. Didesnis pitch = daugiau greičio, daugiau apkrovos. |
| **Tri-blade / Bi-blade** | Mentelių skaičius. Tri-blade = daugiau traukos vienam RPM, daugiau triukšmo. Bi-blade = sklandžiau, efektyviau greityje. |
| **CW / CCW** | Sukimasis pagal / prieš laikrodžio rodyklę. Standartinis Betaflight motorų išdėstymas: M1 CCW, M2 CW, M3 CW, M4 CCW (Betaflight props-in). |
| **T-mount** | Propo tvirtinimas centriniu varžtu. Naudojamas 5" ir didesniems. |
| **Press-fit** | Propas užspaudžiamas tiesiai ant motoro veleno — dažnas whoop'uose. Patikrink užspaudimo tvirtumą prieš kiekvieną skrydį. |
| **AUW** | All-up weight (bendras skrydžio svoris). Visa skrendanti masė su baterija. TWR = trauka / AUW. |
| **TWR** | Traukos ir svorio santykis (visa full-throttle trauka ÷ AUW). ~4:1 patogiam freestyle; 6:1+ racing'ui / maksimaliam vikrumui. |

---

## Baterija ir maitinimas

| Terminas | Reikšmė |
|------|---------|
| **LiPo** | Lithium Polymer. Standartinė FPV baterijų chemija. Didelis išsikrovimo greitis, didelis energijos tankis, gaisro rizika, jei pažeista. |
| **Li-Ion** | Lithium Ion (18650/21700 celės). Mažesnis C-rating nei LiPo, didesnė talpa, geriau long-range dronams. |
| **Cell count** | Nuosekliai sujungtų celių skaičius. „4S“ = 4 celės × 4,2 V = 16,8 V pilnai įkrovus; 14,8 V nominalas. |
| **mAh** | Miliampervalandės. Talpa. 1300 mAh = gali duoti 1,3 A vieną valandą arba 13 A 6 minutes. |
| **C-rating** | Maksimalaus nuolatinio išsikrovimo daugiklis. 75C ant 1300 mAh = 97,5 A maks. nuolat. C-rating'us vertink skeptiškai — gamintojai juos pučia. |
| **IR / Internal Resistance** | Įtampos kritimas celės viduje po apkrova. Mažas IR = sveika celė. Matuok baterijų testeriu ramybėje ir po apkrova. |
| **Sag** | Įtampos kritimas patraukus full throttle. Stiprus sag → FC brownout perkrovimas. |
| **Storage voltage** | 3,8 V vienai celei. Laikyk LiPo čia, jei neskrisi ilgiau nei dieną. Laikymas pilnai įkrautų ar tuščių greitina degradaciją. |
| **Balance lead** | JST-XH jungtis su vienu laidu vienai celei. Naudojama kiekvienai celei balansuoti krovimo metu. Niekada neskrisk su nesubalansuotu paku. |
| **UBEC** | Universal BEC. Impulsinis reguliatorius; efektyvesnis nei linijiniai BEC prie didesnių įtampos kritimų. |

---

## RC ryšys ir radijas

| Terminas | Reikšmė |
|------|---------|
| **ELRS** | ExpressLRS. Atviro kodo long-range RC ryšys. 2,4 GHz arba 900 MHz. Itin maža latencija (3–6 ms prie 500 Hz), didelis nuotolis, nemokamas. |
| **CRSF** | Crossfire Serial protokolas iš TBS. Full-duplex 400 kbps serial tarp TX modulio ir FC; taip pat frame formatas, kurį naudoja ELRS. |
| **FrSky** | Radijų gamintojas; D8/D16/ACCESS protokolai. Legacy, bet vis dar labai dažnas (Taranis/Jumper). |
| **SBUS** | Skaitmeninis serial protokolas (invertuotas UART, 100 kbps) RX → FC. Vienas laidas, 16 kanalų. |
| **PWM** | Pulse Width Modulation. Legacy vienas-laidas-vienam-kanalui RC signalas. 1000–2000 µs. Retai naudojamas moderniuose FC stack'uose. |
| **RSSI** | Received Signal Strength Indicator. Signalo stiprumas dBm arba kaip 0–100% (0 = prarasta, −90 dBm tipiškas minimumas). |
| **LQ** | Link Quality. Gautų paketų procentas iš paskutinių 100 (ELRS). Naudingesnis nei vien RSSI; LQ <70% = įspėjimas, <50% = skrisk namo. |
| **SNR** | Signal-to-Noise Ratio. Kiek signalas virš triukšmo grindų dB. Neigiamas SNR ELRS'e vis dar tinkamas naudoti. |
| **Packet rate** | ELRS atnaujinimo dažnis: 50 Hz, 150 Hz, 250 Hz, 500 Hz. Didesnis = mažesnė latencija, mažesnis nuotolis. |
| **Telemetry** | Downlink duomenys iš drono į radiją: baterijos įtampa, GPS pozicija, RSSI, LQ, skrydžio režimas. |
| **Failsafe** | Elgesys, kai prarandamas RC ryšys. Betaflight Stage 2 procedūros: Drop (motorai iškart išsijungia), Land (kontroliuojamas nusileidimas), GPS Rescue (grįžimas namo — reikia GPS). Atskirai, *imtuvo* „Hold“ režimas toliau siunčia paskutines stick'ų reikšmes (pavojinga — FC niekada nemato praradimo). |
| **BVLOS** | Beyond Visual Line of Sight. Beveik visose reguliavimo sistemose reikalauja specialaus leidimo. |

---

## Betaflight / Firmware

| Terminas | Reikšmė |
|------|---------|
| **Betaflight** | Populiariausias FPV skrydžio kontrolerio firmware. Orientuotas į acro (rate režimą). `github.com/betaflight/betaflight` |
| **INAV** | iNav. Į navigaciją orientuotas Betaflight fork'as. Prideda GPS waypoint'us, fiksuoto sparno palaikymą, RTH, cruise režimus. |
| **Acro mode** | Grynas rate valdymas — FC tik koreguoja gyro dreifą, stick = sukimosi greitis. Numatytasis FPV skrydžio režimas. |
| **Horizon / Angle** | Savaiminio išsilyginimo režimai. FC naudoja akselerometrą, kad išlaikytų lygią padėtį. Naudinga tik pradedantiesiems. |
| **Arming** | Būsena, kai motorai įjungti. Daugumai setup'ų reikia: geras RC ryšys, jokių arming flag'ų, arm jungiklis. |
| **PID** | Proportional-Integral-Derivative. Valdymo kilpa Betaflight branduolyje. Žr. [PID Basics](../../tuning/pid-basics/). |
| **Rates** | Kaip stick'o pakrypimas verčiamas į sukimosi greitį (°/s). Keturi stiliai: Betaflight, Actual, KISS, Quickrates. Žr. [Rate Modes](../../tuning/rate-modes/). |
| **Blackbox** | Skrydžio duomenų registratorius, įmontuotas Betaflight. Log'ina gyro, setpoint, motorus, PID iki 4 kHz. Žr. [Blackbox Logging](../../tuning/blackbox-logging/). |
| **CLI** | Command Line Interface. Tekstinis terminalas Betaflight Configurator'e tiesioginei parametrų prieigai. `diff all` atsarginei kopijai. |
| **RPM filter** | Dinaminiai notch filtrai, prikabinti prie motorų elektrinių RPM per DSHOT telemetriją. Reikia bidirectional DSHOT. |
| **Dynamic notch** | Savaime prisitaikantis notch filtras, sekantis dominuojančius triukšmo dažnius gyro signale. |
| **TPA** | Throttle PID Attenuation. Sumažina P (ir pasirinktinai D) prie aukšto throttle, kad kompensuotų greitesnį motorų atsaką prie aukštų RPM. |
| **iterm_relax** | Slopina I-term integraciją per greitus stick'ų judesius (flip'us, roll'us). Neleidžia I-term windup ir atšokimo. |
| **anti_gravity** | Laikinai pakelia I-term, kai throttle keičiasi staigiai. Neleidžia aukščiui kristi staigiai numetus throttle. |
| **d_min / d_max** | D-term, kuris kinta tarp mažesnės reikšmės ramybėje ir didesnės per greitus manevrus. Sumažina motorų kaitimą hover'e. |
| **FF / Feedforward** | Prideda motoro komandą proporcingai stick'o judėjimo greičiui. Sumažina sekimo vėlavimą. |
| **DSHOT** | Skaitmeninis motorų protokolas (DSHOT150/300/600/1200). Pakeičia analoginį PWM binariniu frame'u. Būtinas RPM filter'iui. |
| **Bidirectional DSHOT** | DSHOT su telemetrija atgal iš ESC į FC. Duoda eRPM kiekvienam motorui → įgalina RPM filter. |
| **Turtle mode** | Flip-over-after-crash. Apsuka motorus, kad apverstų droną teisinga puse, be reikalo eiti prie jo. |
| **Air Mode** | Palaiko PID aktyvų prie nulinio throttle. Neleidžia yaw trūkčioti numetus throttle. Būtinas akrobatikai. |
| **GPS Rescue** | Avarinis grįžimas namo naudojant GPS. Reikia GPS lock ir nustatyto home taško. |
| **Master multiplier** | Proporcingai skaliuoja visus PID gain'us. Naudinga keičiant motorų/propų derinius. |
| **diff all** | CLI komanda, kuri išveda tik ne-numatytuosius nustatymus. Naudok atsarginėms kopijoms. Mažesnė nei `dump`, bet vis tiek pilna. |

---

## Vaizdo sistemos

| Terminas | Reikšmė |
|------|---------|
| **Analog** | Tradicinis FPV: 5,8 GHz FM vaizdas, NTSC/PAL. Maža latencija (~1 ms), maža raiška (~600 TVL), pigu. Goggles: Fatshark HDO2, Skyzone. |
| **Digital HD** | DJI O3/O4, Walksnail Avatar, HDZero. 1080p vaizdas, didesnė latencija (20–35 ms), geresnė vaizdo kokybė. |
| **Latency (glass-to-glass)** | Bendras vėlavimas nuo kameros lęšio iki goggles ekrano. Analog: 3–7 ms. Digital: 20–40 ms. Kritiška proximity skrydžiui. |
| **TVL** | TV Lines. Analoginės kameros raiška. 1200 TVL maždaug atitinka SD vaizdą. |
| **WDR** | Wide Dynamic Range. Kameros funkcija, suspaudžianti šviesius ir tamsius plotus. Naudinga mišrioje patalpų/lauko šviesoje. |
| **DVR** | Digital Video Recorder. Įrašo FPV vaizdą goggles'uose krašų peržiūrai ir turiniui. |
| **OSD** | FC uždėti elementai ant vaizdo. Betaflight OSD elementai: baterija, RSSI, LQ, režimas, greitis, pozicija. |
| **VTX power** | Perdavimo galia mW. 25 mW/200 mW tipiška. Virš 25 mW daugelyje šalių/aplinkų yra nelegalu be leidimo. |
| **vtxtable** | Betaflight CLI lentelė, susiejanti VTX galios lygius su mW reikšmėmis. Reikia smart audio / Tramp valdymui. |

---

## Aerodinamika (greita nuoroda)

| Terminas | Reikšmė |
|------|---------|
| **Downwash** | Pagreitinto oro stulpas, stumiamas rotoriaus žemyn. Nusidriekia kelis propo skersmenis žemiau drono. |
| **Propwash** | Turbulencija, patiriama, kai dronas leidžiasi į savo paties downwash'ą. Sukelia tą būdingą virpėjimą išeinant iš nardymo. Žr. [Propwash](../../aerodynamics/propwash/). |
| **Ground effect** | Padidintas keliamosios jėgos efektyvumas skrendant maždaug iki 1 propo skersmens virš žemės. Downwash negali pilnai išsivystyti; sklinda radialiai. |
| **Tip vortex** | Slėgio nuotėkis ties mentelės galu. Sumažina efektyvų disko plotą. Ducted fans jį panaikina — žr. [Ducted Fans](../../aerodynamics/ducted-fans/). |
| **AoA** | Angle of Attack (atakos kampas). Kampas tarp propo mentelės stygos linijos ir įtekančio oro. Susijęs su trauka ir stall. |
| **Blade pitch** | Propo mentelės kampas sukimosi plokštumos atžvilgiu. Didesnis pitch = daugiau pasistūmimo per apsisukimą, daugiau apkrovos. |
| **Thrust curve** | Ryšys tarp motoro throttle komandos ir realios traukos. Netiesinis — trauka maždaug proporcinga RPM². |

---

## Prop notacijos dekoderis

```
 5 1 4 8
 │ │ │ │
 │ │ └─┴── Pitch × 10 in inches → 48 = 4.8"
 └─┴────── Diameter × 10 in inches → 51 = 5.1"
```

Pavyzdžiai: `5148` = 5,1" skersmuo, 4,8" pitch | `4045` = 4,0" skersmuo, 4,5" pitch | `3020` = 3,0" skersmuo, 2,0" pitch

Kai kurie gamintojai prideda trečią segmentą: `5148-3` = 5,1" × 4,8", tri-blade.

---

## Motoro žymėjimo dekoderis

Motorų žymėjimai laisvai seka `DDHHkv-NNNN`:

```
 2 3 0 6   2 4 5 0 K V
 │ │ │ │   └───────┘
 └─┴─┴─┴── Stator: 23mm diameter × 06mm height
           KV rating: 2450 RPM/V
```

Statoriaus dydis lemia maksimalią galią: didesnis statorius → daugiau sukimo momento → sunkesnis propas, aukštesnė įtampa.

| Statinio tipas | Tipiškas motoras | Tipiškas propas | Baterija |
|-----------|--------------|-------------|---------|
| 5" freestyle | 2306–2407, 1700–2450 KV | 5145–5148 tri | 4S–6S |
| 5" racing | 2204–2306, 2400–2600 KV | 5040–5140 bi | 4S–6S |
| 3.5" / Cinelog | 1404–1507, 3600–4000 KV | 3540–3545 | 3S–4S |
| Tinywhoop 75mm | 0802–0803, 19000–25000 KV | 40mm ducted | 1S |
| Pavo20 / Meteor | 1102–1103, 8700–11000 KV | 2" ducted | 1S |
