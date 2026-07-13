---
title: "DSHOT ir RPM filter"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "dshot", "rpm-filter", "bidirectional", "noise", "filtering"]
---

DSHOT yra skaitmeninis ESC protokolas. Naudojamas bidirectional režime, jis grąžina motoro RPM atgal į skrydžio kontrolerį, įgalindamas RPM filter — didžiausią atskirą motoro triukšmo slopinimo pagerinimą moderniuose Betaflight tune'uose. Na, jei ir įsijungsi tik vieną dalyką iš viso šio sąrašo, tegul tai būna šis. Apie byte lygio vaizdą, kaip freimai ir eRPM telemetrija koduojami laide, žr. [DSHOT on the Wire](../dshot-protocol/).

---

## DSHOT variantai

| Protokolas | Greitis     | Pastabos                                |
|------------|-------------|-----------------------------------------|
| DSHOT150   | 150 kbps    | Legacy; lėtas; nerekomenduojamas        |
| DSHOT300   | 300 kbps    | Default; suderinamas su dauguma ESC     |
| DSHOT600   | 600 kbps    | Greitesni atnaujinimai; reikalingas kai kurioms 8K kilpoms |
| DSHOT1200  | 1200 kbps   | Retai naudojamas; mažai ESC jį palaiko  |

**DSHOT300** yra default ir veikia ant kiekvieno modernaus ESC. Naudok DSHOT600 tik jei suki 8K/8K arba ESC jį aiškiai palaiko.

---

## Bidirectional DSHOT

Bidirectional DSHOT prideda grįžtamąjį signalą iš ESC į FC — kiekvienas ESC siunčia savo motoro RPM atgal tuo pačiu metu, kai gauna throttle komandas.

Įjunk per CLI:
```
set dshot_bidir = ON
set motor_pwm_protocol = DSHOT300
save
```

Įjungęs, eik į **Motors** tab'ą Configurator'yje — kiekvienas motoras turėtų rodyti gyvą RPM rodmenį, kai jį pasuki ranka (prop'ai nuimti). Jei kuris motoras rodo 0 RPM, patikrink, ar ESC firmware palaiko bidirectional DSHOT (BLHeli_32 ≥ 32.7 arba AM32 su bidir palaikymu).

---

## RPM filter

RPM filter naudoja realaus laiko motoro RPM telemetriją, kad statytų notch filtrus tiksliai ant motoro triukšmo harmonikų. Tai pašalina poreikį plataus diapazono lowpass filtrams, kurie prideda fazės vėlinimą.

Įjunk:
```
set rpm_filter_harmonics = 3
set rpm_filter_q = 500
save
```

Filtras automatiškai seka pagrindinį motoro dažnį ir jo harmonikas (paprastai 2× ir 3×). Kai RPM keičiasi skrydžio metu, notch pozicijos juda kartu su juo.

**Poveikis tune'ui:**
- Ženkliai sumažina motoro triukšmą gyro signale
- Leidžia aukštesnius P/D gain'us be osciliacijų
- Įgalina mažesnę statinio notch filtro naštą (dynamic notch galima sumažinti)
- Padaro tune'ą nuoseklesnį per visą throttle diapazoną

---

## RPM filter ir Betaflight versija

- **BF 4.1+**: RPM filter prieinamas ir stabilus
- **BF 4.2+**: Pagerintas sekimas; multi-harmonic palaikymas
- **BF 4.3+**: Veikia kartu su Dynamic Notch v2 (platesnis, protingesnis notch)

Nesuk RPM filter ant BF versijų žemiau 4.1.

---

## Dynamic Notch Filter (kompanionas)

Net su RPM filter, dynamic notch filter tvarko ne-motoro triukšmą (rėmo rezonansą, propwash, guolių triukšmą). Palik jį įjungtą:

```
set dyn_notch_count = 4
set dyn_notch_q = 250
set dyn_notch_min_hz = 100
set dyn_notch_max_hz = 600
save
```

Kai RPM filter tvarko motoro harmonikas, dynamic notch galima nustatyti konservatyviai (mažiau notch'ų, platesnis Q), kad išvengtum per didelio fazės vėlinimo.

---

## Lowpass filtrai

Su įjungtu RPM filter gali sumažinti lowpass filtrų agresyvumą:

```
# Gyro lowpass — more permissive with RPM filter
set gyro_lowpass_hz = 0           # disable static lowpass (RPM filter handles it)
set gyro_lowpass2_hz = 0

# D-term lowpass — keep this for D-noise
set dterm_lowpass_hz = 100
set dterm_lowpass2_hz = 200
```

Betaflight cinematic ir freestyle preset'ai nustato juos automatiškai, kai juos pritaikai.

---

## Troubleshooting

| Simptomas                          | Priežastis                                  |
|------------------------------------|---------------------------------------------|
| RPM rodomas kaip 0 Motors tab'e    | ESC nepalaiko bidir DSHOT; bidir neįjungtas; ne tas protokolas |
| Osciliacijos blogesnės po įjungimo | RPM filter Q per aukštas; per daug harmonikų |
| ESC keistai pypsi po įjungimo      | Kai kuriems ESC reikia reflash, kad įjungtų bidir režimą |

Prieš darydamas prielaidą, kad tai aparatūros gedimas, patikrink ESC firmware changelog dėl bidirectional DSHOT palaikymo — dažniausiai kaltas ne lituoklis, o praleista eilutė changelog'e.
