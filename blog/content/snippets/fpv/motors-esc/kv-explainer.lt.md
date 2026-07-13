---
title: "Motoro KV paaiškinta"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "motor", "kv", "voltage", "rpm", "efficiency", "selection"]
---

KV yra labiausiai nesuprasta motoro specifikacija FPV pasaulyje. Tai **ne** motoro kokybės ar galios matas — tai santykis, kuris nulemia, kiek RPM gauni vienam voltui. (Ir ne, didesnis skaičius nereiškia „geresnis motoras“, kad ir kaip parduotuvės aprašymas norėtų, jog taip pagalvotum.)

---

## Apibrėžimas

**KV = RPM voltui (be apkrovos)**

Motoras, kurio reitingas 2400 KV, besisukantis ant 16.8 V (4S pilnai įkrautas), suksis:

```
RPM = KV × Voltage
RPM = 2400 × 16.8 ≈ 40,320 RPM (no load)
```

Su apkrova (su prop'u) RPM bus mažesnis — paprastai 65–85% no-load RPM, priklausomai nuo prop dydžio ir throttle.

---

## Ką KV veikia

| Aukštesnis KV           | Žemesnis KV                 |
|-------------------------|-----------------------------|
| Aukštesnis RPM          | Žemesnis RPM                |
| Labiau tinka mažiems prop'ams | Labiau tinka dideliems prop'ams |
| Mažiau torque           | Daugiau torque              |
| Kaista labiau su apkrova | Kaista mažiau su apkrova    |
| Reikia žemesnės įtampos (2S, 3S) | Reikia aukštesnės įtampos (4S, 6S) |

Stator apvijos nulemia KV: mažiau, storesnės apvijos = aukštesnis KV; daugiau, plonesnės apvijos = žemesnis KV.

---

## Įtampa ir KV kartu

KV yra beprasmis nežinant darbinės įtampos. Tą patį RPM galima pasiekti skirtingomis KV × Voltage kombinacijomis:

```
2400 KV × 14.8 V (4S nominal) = 35,520 RPM
1750 KV × 22.2 V (6S nominal) = 38,850 RPM
```

Abu setup'ai duoda panašų RPM — bet 6S motoras turi daugiau torque, kad efektyviai suktų didesnį prop'ą.

---

## Praktinis KV gidas pagal rėmo dydį

| Rėmo dydis | Dažna įtampa   | Tipinis KV diapazonas | Prop dydis     |
|------------|---------------|-----------------|----------------|
| 1" whoop   | 1S (3.7 V)    | 15,000–20,000 KV| 31 mm          |
| 2.5" micro | 2S–3S         | 5,000–8,000 KV  | 2.5"           |
| 3" toothpick | 3S–4S       | 3,000–5,000 KV  | 3"             |
| 5" freestyle | 4S–6S       | 1,700–2,500 KV  | 5"             |
| 7" long range| 4S–6S       | 1,300–1,800 KV  | 7"             |
| 10" cinematic| 4S–6S       | 700–1,000 KV    | 9"–10"         |

---

## KV skaičiuoklė

**Tikslinis RPM iš KV ir įtampos:**
```
RPM = KV × Voltage
```

**KV, reikalingas tiksliniam RPM prie žinomos įtampos:**
```
KV = Target RPM ÷ Voltage
```

**Apkrauto RPM įvertis (apytikslis):**
```
Loaded RPM ≈ KV × Voltage × 0.75
```

Pavyzdys: nori ~25,000 apkrauto RPM ant 4S (14.8 V nominal):
```
KV ≈ 25,000 ÷ (14.8 × 0.75) ≈ 25,000 ÷ 11.1 ≈ 2,250 KV
```
→ 2300–2450 KV motoras ant 4S yra protingas taikinys.

---

## KV ir efektyvumas

Žemesnio KV motorai, veikiantys prie aukštesnės įtampos, paprastai efektyvesni tokiai pačiai išeinamai galiai. Motoro varinės apvijos turi tą pačią varžą nepriklausomai nuo KV, bet aukštesnės įtampos ir žemesnės srovės darbas sumažina I²R nuostolius (karštį apvijose).

Būtent todėl 6S build'ai ant 5" rėmo pasiekia geresnius skrydžio laikus, nepaisant panašaus fizinio dydžio — žemesnio KV motoras ant 6S kaista mažiau ir mažiau energijos švaisto kaip karštį.

---

## Pastabos

- KV yra no-load specifikacija. Realaus pasaulio RPM priklauso nuo prop pitch, prop skersmens, throttle ir oro tankio.
- Stator matmenys (pvz., 2306 = 23 mm skersmuo × 6 mm aukštis) nulemia torque talpą ir šilumos sklaidą, nepriklausomai nuo KV.
- Du motorai su tuo pačiu stator ir tuo pačiu KV iš skirtingų gamintojų gali elgtis labai skirtingai — apvijų kokybė, magnetų stiprumas ir guolių tikslumas visi turi reikšmės. Taigi popierius popieriumi, o pajusi tik tada, kai užsuksi.
