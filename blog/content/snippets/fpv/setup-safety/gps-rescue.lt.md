---
title: "GPS Rescue konfigūracija"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "gps", "rescue", "failsafe", "return-to-home", "long-range"]
---

GPS Rescue — tai Betaflight return-to-home failsafe. Kai suveikia (ryšio praradimas, žema baterija ar rankinis įjungimas), dronas pakyla iki sukonfigūruoto aukščio ir parskrenda ten, kur buvo arm'intas. Skamba kaip magija — bet magija veikia tik tada, kai viską sukonfigūravai teisingai.

---

## Būtinos sąlygos

- GPS modulis prijungtas ir priskirtas UART'ui (įjunk **GPS** Ports tab'e tam UART'ui)
- 6+ palydovų fiksacija prieš arm (nustatomas minimumas)
- GPS Rescue mode pridėtas Modes tab'e (ant jungiklio arba pririštas prie failsafe)

---

## Bazinė konfigūracija (CLI)

```
set gps_provider = UBLOX       # or NMEA for generic modules
set gps_baudrate = 115200      # match GPS module baud rate

set gps_rescue_max_angle = 30       # max tilt angle during rescue (degrees)
set gps_rescue_initial_climb = 15   # climb above arm point before heading home (m)
set gps_rescue_return_alt = 30      # cruise altitude for the return flight (m)
set gps_rescue_ground_speed = 750   # return speed (cm/s ≈ 27 km/h)
set gps_rescue_descent_dist = 20    # meters from home where descent begins
set gps_rescue_descend_rate = 150   # descent speed approaching home (cm/s)
set gps_rescue_disarm_threshold = 15  # impact detection for auto-disarm on landing

save
```

---

## Integracija su failsafe

GPS Rescue integruojasi su Betaflight failsafe sistema. Failsafe tab'e:

1. Nustatyk **Failsafe Stage 2** į `GPS RESCUE`
2. Nustatyk protingą Stage 1 guard laiką (2–3 sekundės ryšio praradimo prieš Stage 2 suveikimą)
3. Įjunk `GPS_RESCUE` mode ant AUX jungiklio rankiniam įjungimui

```
set failsafe_procedure = GPS-RESCUE
set failsafe_delay = 20       # 20 × 0.1s = 2 seconds
set failsafe_recovery_delay = 20
```

---

## Arm patikrinimai

Kai GPS Rescue nustatytas kaip failsafe, Betaflight reikalauja GPS fix prieš arm (nustatoma):

```
set gps_rescue_min_sats = 6    # minimum satellites required
set gps_rescue_allow_arming_without_fix = OFF  # don't arm without fix
```

Bandomiesiems skrydžiams žinomose saugiose vietose gali laikinai leisti arm be fix — bet niekada neskrisk su GPS Rescue kaip failsafe be realaus GPS fix.

---

## Derinimo patarimai

- **Initial climb** turi apeiti vietines kliūtis. Atviruose laukuose 15 m užtenka. Prie medžių ar statinių padidink iki 30–50 m.
- **Return altitude** nustato aukštį, naudojamą parskridimo metu. Turi būti virš aukščiausios kliūties parskridimo kelyje.
- **Return speed** (numatytoji 300 cm/s = 3 m/s) yra atsargi. Long-range buildams padidink iki 500–700 cm/s.
- **Descent distance** — tai kaip toli nuo namų dronas pradeda galutinį leidimąsi. Padidink vėjuotomis sąlygomis.

---

## Testavimas

Prieš pasikliaudamas GPS Rescue, visada išbandyk jį ramią dieną virš atviros vietos („turėtų veikti" — tai ne testas):

1. Arm ir pakilk; užfiksuok GPS home tašką
2. Nuskrisk 100 m
3. Įjunk GPS Rescue jungikliu (o ne nukirsdamas ryšį)
4. Patikrink, ar dronas pakyla, pasisuka namų link ir leidžiasi

Būk pasiruošęs atgauti valdymą, jei elgesys neteisingas.

---

## Pastabos

- GPS Rescue reikalauja magnetometro (kompaso) patikimam kursui. Kai kurie buildai naudoja GPS modulyje įmontuotą kompasą (įjunk Betaflight GPS tab'e).
- Be kompaso dronas kursui vertinti naudoja akselerometrą + GPS greitį — mažiau tikslu, bet veikia.
- GPS tikslumas kinta priklausomai nuo palydovų skaičiaus ir atmosferos sąlygų. HDOP < 2.0 yra priimtina rescue'ui; < 1.5 yra ideali.
