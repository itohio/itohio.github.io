---
title: "Rate profilio parinkimas ir Persistent Stats"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "rates", "osd", "persistent-stats", "switch"]
---

Rate profilių perjungimas vidury sesijos per siųstuvo jungiklį ir asmeninių rekordų statistikos saugojimas į OSD — abi naudingos funkcijos, bet jos sąveikauja netikėtu būdu. Hmm, „netikėtu“ — švelniai pasakius; aš kurį laiką rimtai galvojau, kad man sugedo FC.

---

## Persistent Stats

Betaflight gali sekti skrydžio statistiką per maitinimo ciklus ir rodyti ją OSD:

- Maks. aukštis, maks. greitis, maks. G-jėga, maks. srovė, maks. throttle
- Bendras armed laikas, bendras skrydžių skaičius

**Kuo jos naudingos:** persistent stats leidžia matyti asmeninius rekordus OSD skrendant — pravartu peržiūrint įrašus ir žinant, ar skrydis buvo greičiausias/sunkiausias. Jie išlieka po disarm ir po išjungimo, tad gali peržiūrėti juos tarp sesijų.

### Persistent Stats įjungimas

Betaflight Configurator → **OSD** skirtukas → nusileisk iki Stats sekcijos, įjunk norimą statistiką. Ji pasirodo po-skrydžio statistikos ekrane po disarm.

Per CLI:
```
set stats = ON
set stats_total_flights = ON
set stats_total_time = ON
save
```

---

## Rate profilio parinkimas per jungiklį

Betaflight palaiko iki 6 rate profilių. Gali perjunginėti tarp jų skrydžio metu naudodamas siųstuvo kanalą, priskirtą mode sąlygai.

Configurator → **Modes** skirtuke pridėk `RATES PROFILE 1`, `RATES PROFILE 2`, `RATES PROFILE 3` sąlygas pasirinktuose AUX kanalų diapazonuose. Kiekvienas profilis saugo savo RC Rate / Super Rate / Expo reikšmes.

Tipinis setup'as: 3 pozicijų jungiklis → po vieną rate profilį kiekvienai pozicijai (pvz. 533 → 633 → 733).

---

## Konfliktas: kodėl stats nesaugomi po disarm

**Kai rate profilio perjungimas įjungtas, persistent stats nebus išsaugoti po disarm.**

Štai kodėl: rate profilio perjungimas modifikuoja skrydžio kontrolerio konfigūraciją (tai config keitimas, o ne tiesiog runtime reikšmė). Kad tas keitimas išliktų po perkrovimo, Betaflight turėtų įrašyti config į flash. Betaflight **nedaro** automatinio config išsaugojimo po disarm — sąmoningai, kad išvengtų flash dėvėjimosi ir atsitiktinių perrašymų vidury sesijos.

Kadangi tas pats saugojimo mechanizmas dalinamas tarp „kuris rate profilis aktyvus“ ir „dabartiniai statistikos skaitikliai“, ir kadangi joks auto-save nepasileidžia po disarm, tavo statistikos skaitikliai atsistato per kitą maitinimo ciklą.

---

## Apėjimo būdai

### 1 variantas — Išsaugoti config rankiniu būdu po nusileidimo

Paleisk `save` komandą iš Betaflight Configurator ar per OSD stick komandą iškart po nusileidimo, prieš išimant bateriją. Tai įrašo dabartinį config (įskaitant, kuris rate profilis aktyvus, ir naujausius statistikos skaitiklius) į flash. Taip, tai reiškia — atsimink tą `save`; aš pusę savo rekordų praradau būtent todėl, kad neatsiminiau.

```
# In CLI after each session
save
```

### 2 variantas — Išjungti rate profilio perjungimą

Jei tau nereikia rate profilio keitimų skrendant, pašalink `RATES PROFILE` mode sąlygas iš Modes skirtuko. Su išjungtu profilių perjungimu aktyvus profilis niekada nekeičia config runtime metu, ir statistika normaliai saugoma po disarm.

### 3 variantas — Naudoti vieną profilį ir derinti rates pagal stilių

Užuot perjunginėjęs profilius skrydžio metu, laikyk vieną profilį nustatytą į rates, kuriais skraidai dažniausiai. Jei nori išbandyti kitokius rates, keisk profilį ant stalo tarp sesijų ir išsaugok rankiniu būdu.

---

## Santrauka

| Funkcijų kombinacija            | Ar stats išlieka po disarm? |
|---------------------------------|--------------------------|
| Persistent stats, be profilio perjungimo | ✅ Taip             |
| Persistent stats + profilio perjungimas  | ❌ Ne (saugok rankiniu būdu)|
| Profilio perjungimas, stats išjungti      | N/A                  |
