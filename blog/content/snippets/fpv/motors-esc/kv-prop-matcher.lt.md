---
title: "KV ir prop derinimas — tip speed, apkrova ir parinkimas"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "motor", "kv", "props", "tip-speed", "efficiency", "thrust", "matching"]
---

Motoro KV derinimas su prop dydžiu yra pats svarbiausias veiksnys build'o efektyvumui ir motoro ilgaamžiškumui. Tikslas: laikyti tip speed efektyviame diapazone ir laikyti motoro temperatūrą protingą. Skamba kaip nuobodi matematika — bet būtent ji lemia, ar variklis keps, ar skris visą pack'ą.

---

## Prop tip speed

Propelerio tip speed — tai greitis, kuriuo mentės galas juda per orą. Kai tip speed artėja prie garso greičio (~343 m/s), efektyvumas smarkiai krenta, o triukšmas dramatiškai auga.

**Praktinis efektyvus diapazonas: 100–150 m/s tip speed prie hover/cruise throttle.**

### Skaičiavimas

```
Tip Speed (m/s) = (π × Prop Diameter [m] × RPM) / 60
```

Pavyzdys — 5" prop (0.127 m skersmuo) ant 20,000 RPM:
```
Tip Speed = (π × 0.127 × 20,000) / 60
           = (3.1416 × 0.127 × 20,000) / 60
           ≈ 7,980 / 60
           ≈ 133 m/s
```
→ 133 m/s yra efektyviame diapazone.

---

## Greita nuorodų lentelė

| Prop skersmuo | Maks. efektyvus RPM (150 m/s) | Tipinis KV ant 4S (14.8V) |
|---------------|----------------------------|--------------------------|
| 3" (76 mm)    | ~37,600 RPM                | ~3,500–4,500 KV          |
| 4" (102 mm)   | ~28,100 RPM                | ~2,600–3,200 KV          |
| 5" (127 mm)   | ~22,500 RPM                | ~2,000–2,500 KV          |
| 5" (127 mm)   | ~22,500 RPM                | ~1,500–1,800 KV ant 6S   |
| 7" (178 mm)   | ~16,100 RPM                | ~1,300–1,600 KV ant 4S–6S |
| 10" (254 mm)  | ~11,300 RPM                | ~700–900 KV ant 6S       |

---

## Motoro apkrovos indeksas (thrust-to-weight)

Naudingas sanity-check prie hover: kiekvienas motoras neša ketvirtadalį viso svorio.

**Hover thrust vienam motorui:**
```
Hover thrust per motor = AUW / 4        (for a quad)
```

~500–700 g 5" kvadrui tai yra ~125–175 g vienam motorui, o tai ant sveiko build'o paprastai atsiduria maždaug ties 40–50% throttle.

**Thrust-to-weight santykis (TWR)** lygina *bendrą full-throttle* thrust su AUW:
```
TWR = (4 × max thrust per motor) / AUW
```

TWR 4:1 yra tipiškas freestyle, 3:1 tinka cinematic, o racing nori 6:1+. 4:1 kvadras pakelia savo paties svorį naudodamas tik ketvirtadalį prieinamo thrust — likusi dalis yra atsarga punch-out'ams.

---

## Prop pitch ir motoro parinkimas

Pitch — tai teorinis atstumas, kurį prop pasistumia per apsisukimą. Aukštesnis pitch = agresyvesnis kandimas = daugiau greičio, bet daugiau drag = motoras dirba sunkiau.

| Naudojimo atvejis | Pitch rekomendacija          |
|------------------|------------------------------|
| Efektyvumas / long range | Žemas pitch (3.8"–4.3") |
| Freestyle        | Vidutinis (4.8"–5.1")        |
| Racing / top speed | Aukštesnis pitch (5.1"–6.0") |

**Aukštesnis pitch reikalauja daugiau torque → žemesnio KV motoro ant aukštesnės įtampos** tokiam pačiam efektyvumui.

---

## Derinimo darbo eiga

1. **Pasirink rėmo dydį** → nustato prop skersmens diapazoną
2. **Pasirink baterijos įtampą** → nustato įtampos įvestį į motorą
3. **Pasirink naudojimo atvejį** → nustato tikslinį RPM diapazoną ir pitch prioritetą
4. **Apskaičiuok reikalingą KV:**
   ```
   KV = Target Cruise RPM ÷ (Voltage × 0.75)
   ```
5. **Patikrink tip speed** prie įvertinto maks. RPM:
   ```
   Max RPM = KV × Max Voltage
   Tip Speed = (π × Diameter_m × MaxRPM) / 60
   ```
   Turėtų likti žemiau ~170 m/s; idealiai žemiau 150 m/s.
6. **Patikrink stator dydį** — didesnis stator (pvz. 2306 vs 2204) tvarko daugiau karščio prie duoto KV, tad sunkesnės prop kombinacijos reikalauja didesnio stator.

---

## Išspręstas pavyzdys — 5" Freestyle 4S

- Rėmas: 5" (0.127 m), baterija: 4S (16.8 V pilnas įkrovimas)
- Tikslinis full-throttle *apkrautas* RPM ≈ 24,000–28,000 RPM

Dirbk atgal nuo apkrauto maks. RPM iki KV (apkrautas ≈ 75% no-load `KV × Voltage`):
```
KV = Max loaded RPM ÷ (Voltage × 0.75)
KV = 26,000 ÷ (16.8 × 0.75) = 26,000 ÷ 12.6 ≈ 2,060 KV
```
→ Tip speed prie to apkrauto maks.: `π × 0.127 × 26,000 / 60 ≈ 173 m/s`. Tai virš 150 m/s efektyvaus plafono — normalu freestyle, kuris iškeičia dalį efektyvumo į punch (racing suka dar aukščiau).

Praktikoje **2000–2450 KV** motorai yra nusistovėjusi sweet spot 5" ant 4S, atitinkanti aukščiau esančią greitą nuorodų lentelę. (Žymėjimas kaip „2306“ yra *stator dydis* — 23 mm × 6 mm — ne KV vertė.)

---

## Prop parinkimo nykščio taisyklės

- **Skersmuo** — nulemtas rėmo peties ilgio ir motoro tvirtinimo tarpo. Neviršyk rėmo ribų.
- **Pitch** — aukštesnis pitch greičiui ir reaktyvumui; žemesnis pitch efektyvumui ir hover laikui.
- **Menčių skaičius** — 3 mentės: efektyvumo ir valdymo balansas. 4/5 mentės: daugiau thrust tame pačiame skersmenyje, daugiau triukšmo ir mažiau efektyvumo.
- **Tip speed** — visada patikrink apskaičiuotą tip speed prie max throttle. Jei jis viršija 180 m/s, palieki efektyvumą ant stalo ir generuoji nereikalingą triukšmą.

---

## Pastabos

- Šie skaičiavimai duoda teorinį RPM. Realus apkrautas RPM su prop'u paprastai yra 70–80% no-load KV × voltage.
- Motorų thrust lentelės iš gamintojų matuojamos prie konkrečių įtampų ant konkrečių prop'ų — visada persitikrink savo tiksliai kombinacijai.
- Maži skirtumai tarp prop gamintojų (HQ, Gemfan, DAL) veikia realų RPM, thrust ir efektyvumą net ant nominaliai identiškų prop'ų. Nepaisant to — skaičiuoklė nuves tave iki teisingo kvartalo, o galutinį adresą vis tiek surasi jausdamas skrydį.
