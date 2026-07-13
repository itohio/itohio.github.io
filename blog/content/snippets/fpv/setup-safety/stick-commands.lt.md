---
title: "Betaflight stick komandos"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "stick-commands", "configuration", "cli"]
---

Stick komandos leidžia įjungti Betaflight funkcijas tiesiai iš siųstuvo stick'ų, be kompiuterio. Naudinga lauke — kai nešiojamąjį palikai namie (arba tiesiog tingi jį tempti į lauką).

Visoms komandoms dronas turi būti **disarm**. Throttle visada minimalioje padėtyje, nebent nurodyta kitaip.

---

## Stick'ų padėtys

```
         PITCH UP
            ↑
YAW LEFT ←     → YAW RIGHT
            ↓
         PITCH DOWN

ROLL maps to: ← LEFT  |  RIGHT →
```

---

## Dažniausios komandos

| Veiksmas                      | Throttle | Yaw   | Pitch | Roll  |
|-------------------------------|----------|-------|-------|-------|
| Įeiti į CLI / Config mode     | LOW      | LOW   | HIGH  | HIGH  |
| Išsaugoti konfigūraciją (CLI mode) | LOW | LOW   | HIGH  | LOW   |
| Įeiti į akselerometro kalibraciją | LOW  | LOW   | HIGH  | CENTER|
| Įjungti/išjungti Blackbox     | LOW      | LOW   | LOW   | HIGH  |
| Įeiti į OSD meniu             | CENTER   | LOW   | CENTER| CENTER (laikyti) |

> Tikslios stick'ų padėtys skiriasi priklausomai nuo Betaflight versijos ir ar naudoji mode 1/2. Lentelė aukščiau skirta **Mode 2** (throttle kairysis stick'as).

---

## Įėjimas į CLI per stick'us (lauko konfigūracija)

Laikyk: **Throttle LOW · Yaw LOW · Pitch HIGH · Roll HIGH** ~5 sekundes.

FC pereina į konfigūracijos būseną. Iš čia gali:
- Pasiekti OSD meniu (jei sukonfigūruota)
- Įjungti kalibraciją

Kad įeitum į tikrąjį CLI, vis tiek reikia USB + Betaflight Konfigūratoriaus arba Bluetooth/WiFi adapterio. Stick'ais valdomas „config mode“ daugiausia skirtas OSD meniu prieigai.

---

## OSD meniu prieiga (dažniausias lauko naudojimas)

Su **CONFIGURATOR MSP** mode įjungtu ant jungiklio arba per stick komandą:

Laikyk **Throttle CENTER · Yaw LEFT · Pitch CENTER** ~3 sekundes.

Naviguok:
- **Pitch UP/DOWN** — judinti žymeklį
- **Roll RIGHT** — pasirinkti / įeiti
- **Roll LEFT** — atgal
- **Yaw LEFT** — išeiti iš meniu

Tai leidžia keisti PID profilius, rate profilius, VTX galią ir kitus nustatymus be telefono ar nešiojamojo.

---

## Pastabos

- Stick komandos patikimai suveikia tik laikant stick'us tiksliai kraštinėse padėtyse (>95% nuokrypis). Sukalibruok savo siųstuvo endpoint'us.
- Kai kurios komandos naujesnėse Betaflight versijose apsaugotos dvigubo įvedimo seka, kad išvengtų netyčinių suveikimų.
- Arm paprastai yra ant **Yaw RIGHT** (arba atskiro arm jungiklio). Niekada nemaišyk arm su config stick padėtimis.

---

## Nuoroda

Pilna stick komandų lentelė: [Betaflight Wiki — Stick Commands](https://betaflight.com/docs/wiki/guides/current/stick-commands)
