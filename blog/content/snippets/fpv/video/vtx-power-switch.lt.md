---
title: "VTX galios valdymas per siųstuvą"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "betaflight", "vtx", "power", "switch", "aux", "cli"]
---

Priskirk siųstuvo svirtelę, ratuką ar jungiklį, kad Betaflight'e perjunginėtum VTX galios lygius. Naudinga pit mode paleidimo metu, mažai galiai skrendant arti ir pilnai galiai long range.

---

## Kaip veikia

Betaflight skaito AUX kanalą ir susieja PWM ruožus su VTX galios lentelės indeksais. VTX turi būti prijungtas prie FC per SmartAudio, IRC Tramp ar MSP VTX — tiesioginis valdymas be FC passthrough neveiks.

---

## VTX galios lentelė

Apibrėžk leidžiamus galios lygius VTX lentelėje. Betaflight Configurator → **Video Transmitter** tab'as → suredaguok galios lygius, kad atitiktų tai, ką tavo VTX palaiko.

Per CLI:
```
vtxtable powervalues 25 100 200 400 600
vtxtable powerlabels 25 100 200 400 600
```

---

## Mode Condition nustatymas

Galia iš jungiklio konfigūruojama Configurator → **VTX** tab'e (išsaugoma kaip `vtx` CLI eilutės); pit mode naudoja `VTX PIT MODE` mode **Modes** tab'e.

3 pozicijų jungikliui, perjunginėjančiam 25 / 200 / 600 mW:
```
# AUX2 range mapping example (1000-2000 µs typical)
# Position LOW  (1000-1300) → pit / 25 mW
# Position MID  (1300-1700) → 200 mW
# Position HIGH (1700-2000) → 600 mW
```

---

## Radiomaster Pocket — svirtelės / ratuko pavyzdys

Radiomaster Pocket turi scrollinamą ratuką (S1), kuris išveda sklandų 1000–2000 µs ruožą AUX kanale — idealu VTX galiai.

Pirma patvirtink VTX galios lentelę CLI (tai realios komandos):

```
vtxtable powervalues 25 100 200 400 600
vtxtable powerlabels 25 100 200 400 600
save
```

Priskirk ratuką **AUX3** siųstuvo mikseryje, tada pridėk juostas Configurator → **VTX** tab'e arba `vtx <slot> <aux> <band> <channel> <power> <start> <end>` eilutėmis (AUX numeruojami nuo 0, tad AUX3 = 2; band/channel/power 0 = nekeisti; ruožai 25 µs žingsniu; power — `vtxtable` lygis nuo 1). Padalink 1000–2000 µs ratuko eigą į nepersidengiančias juostas:

| Ratuko juosta | µs ruožas | Priskyrimas |
|-----------|----------|--------|
| LOW | 1000–1325 | `VTX PIT MODE` (Modes tab'as) |
| MID | 1350–1650 | `vtx 0 2 0 0 3 1350 1650` (3 lygis = 200 mW) |
| HIGH | 1675–2000 | `vtx 1 2 0 0 5 1675 2000` (5 lygis = 600 mW) |

Už visų juostų lieka paskutinis pasirinkimas; tikslių vidurio taškų nereikia, tik nepersidengiančių ruožų.

---

## Pit mode per arm / disarm

Laikyk ratuką/svirtelę minimume (pit mode), kol esi pite. VTX nukris iki minimalios galios (arba RF off, jei tavo VTX tai palaiko), kol nepastumsi svirtelės aukštyn. Taip nešaudai kitiems pilotams į goggle'us pilna galia, kol pats dar prie stalo.

---

## Patikra

Po nustatymo patikrink OSD arba Betaflight Configurator → Video Transmitter tab'e, ar galios lygis keičiasi, kai judini ratuką. Jei turi, patikrink signalo matuokliu ar dažnių skeneriu.

---

## Pastabos

- **SmartAudio v2.1+** būtinas patikimam galios perjungimui skrydžio metu. Ankstesnės versijos gali reikalauti save/reboot ciklo.
- Kai kurie VTX reikalauja reboot'o, kad pritaikytų galios pakeitimus — patikrink prieš pasikliaudamas tuo lauke.
- IRC Tramp taip pat pilnai palaikomas; pasirink jį kaip VTX periferiją Configurator → **Ports** tab'e (VTX (Tramp)) ant to UART, prie kurio prijungtas VTX, o ne per kokį `set` kintamąjį.
