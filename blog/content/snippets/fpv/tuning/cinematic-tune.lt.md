---
title: "Cinematic tune iš Betaflight preset'ų"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "cinematic", "tune", "presets", "pid", "filters"]
---

Betaflight ateina su Presets biblioteka, kurioje yra paruošti cinematic tune'ai. Jie mažina agresyvumą, suminkština filtrus ir sumažina propwash artefaktus, kad kadrai būtų sklandūs ir „plaukiantys“. Na, o gera žinia ta, kad nereikia nieko išrasti iš naujo — kažkas jau padirbėjo už mus.

---

## Preset'o įkėlimas

Betaflight Configurator → **Presets** skirtukas → ieškok `cinematic`.

Geri atspirties taškai:
- **Cinematic Freestyle** — saikingas filtravimas, žemas D-term, sklandus step response
- **Cinematic HD** — sunkesnis filtravimas giroskopo triukšmui slopinti, tinka didesniems dronams su gimbalais
- **Slow Flyer / Whoop Cinematic** — suderintas mažiems 2" ar toothpick rėmams

Spustelk preset'ą → **Preview**, kad pamatytum, kokias CLI komandas jis pritaikys → **Apply and Save**.

---

## Ką keičia cinematic tune

| Parametras       | Freestyle tipiškai | Cinematic tikslas  |
|------------------|--------------------|--------------------|
| P-term           | Vidutinis–aukštas  | Žemesnis (mažiau snappy)|
| D-term           | Vidutinis          | Žemesnis           |
| I-term           | Vidutinis          | Panašus / kiek aukštesnis (laiko poziciją) |
| Feedforward      | Vidutinis–aukštas  | Žemas (mažina stick atsako aštrumą) |
| RPM filter       | ON                 | ON (būtinas)       |
| Dynamic notch    | ON                 | ON, platesnės juostos |
| TPA breakpoint   | ~1250              | ~1150 (ankstesnis suminkštėjimas) |
| Rates            | 633–733            | 333–433            |

---

## Rankiniai koregavimai po preset'o

Preset'ai — tai atspirties taškas. Pritaikius:

1. **Patikrink RPM filter**, ar įjungtas ir ar veikia bidirectional DSHOT (`dshot_bidir = ON`, patikrink `rpmfilter` CLI).
2. **Sumažink RC smoothing**, jei kadrai atrodo, lyg „ieškotų“ ties centru:
   ```
   set rc_smoothing_auto_factor = 50
   ```
3. **Sumažink feedforward**, jei dronas jaučiasi trūkčiojantis per lėtus panoraminius kadrus:
   ```
   set feedforward_transition = 100
   set feedforward_averaging = 4_POINT
   ```
4. **Suderink rates į 333 ar 433** — cinematic skraidymui retai reikia daugiau nei 400 °/s maks. roll.
5. **Įjunk I-term relax**, jei matai atšokimą po rollų:
   ```
   set iterm_relax = RP
   set iterm_relax_type = SETPOINT
   ```

---

## Blackbox patikra

Po pirmo cinematic skrydžio ištrauk blackbox logą ir ieškok:
- Švarios giroskopo trajektorijos be aukšto dažnio šuolių
- Mažo variklio išvesties svyravimo (jokių osciliacijų)
- Sklandaus setpoint vs. giroskopo sekimo per lėtus judesius

Propwash per greitus krypties keitimus yra normalu ir jį sunkiau pašalinti vien filtravimu — throttle valdymas ir skraidymo technika cinematic darbui svarbesni (kitaip tariant, dalį darbo vis tiek teks atlikti nykščiais, ne CLI).

---

## Pastabos

- RPM filter yra privalomas moderniems cinematic tune'ams — be jo reikės daug sunkesnio lowpass filtravimo, kuris prideda vėlinimo ir fazės delsos.
- Preset tune'ai priklauso nuo versijos. Prieš pritaikydamas visada patikrink, kuriai Betaflight versijai preset'as skirtas.
