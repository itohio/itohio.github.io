---
title: "LiPo baterijų pagrindai — C rating'as, celių skaičius ir priežiūra"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "battery", "lipo", "c-rating", "cell-count", "storage", "safety"]
---

LiPo baterijos — jautriausias priežiūrai komponentas visame FPV builde. Netinkamas naudojimas jas greitai užmuša arba sukelia gaisrą. Turiu porą išpūstų pakų lentynoje kaip priminimą, kad taisyklės čia ne šiaip sau.

---

## Celės įtampa

| Būsena          | Įtampa vienai celei |
|----------------|-----------------|
| Pilnai įkrauta  | 4.20 V           |
| Nominali        | 3.70 V           |
| Sandėliavimo    | 3.80–3.85 V      |
| Žemas cutoff    | 3.50 V           |
| Mirusi / pažeista | < 3.30 V         |

**Niekada neišnaudok žemiau 3.5 V vienai celei** esant apkrovai. Nustatyk FC low-battery įspėjimą suveikti ties ~3.5–3.6 V vienai celei (įtampa esant apkrovai, o ne ramybėje).

---

## Celių skaičius

| Konfigūracija | Nominali | Maks. | Tipiškas naudojimas       |
|--------|---------|-------|--------------------------|
| 1S     | 3.7 V   | 4.2 V | Maži whoop'ai            |
| 2S     | 7.4 V   | 8.4 V | 2.5" micro dronai        |
| 3S     | 11.1 V  | 12.6 V| 3" micro dronai          |
| 4S     | 14.8 V  | 16.8 V| 5" standartas            |
| 6S     | 22.2 V  | 25.2 V| 5" high performance, 7"+ |

---

## C rating'as

C rating'as — tai iškrovos srovės daugiklis, susietas su talpa.

```
Max continuous current (A) = Capacity (Ah) × C rating
```

Pavyzdys — 1500 mAh, 100C baterija:
```
Max current = 1.5 Ah × 100 = 150 A
```

**C rating'as — tai marketingas** ant daugumos pigesnių pakų. Traktuok jį kaip grubią gairę. Realaus pasaulio taisyklė:

- 5" freestyle quad'as: taikyk į **1500–2200 mAh, 80C+** ant 4S
- 5" efektyvumui: 2200–3000 mAh, 50–80C ant 4S ar 6S
- Racing: 650–1300 mAh, 100C+ ant 4S–6S

Jei tavo pakas smarkiai prasėda (voltage sag) throttle punch metu (įtampa nukrenta 1+ V vienai celei), C rating'o neužtenka tavo buildo srovės poreikiui.

---

## Sandėliavimo įkrova

LiPo baterijas visada laikyk ties **3.80–3.85 V vienai celei**. Niekada nelaikyk pilnai įkrautų ar pilnai išsekintų.

- Pilnai įkrautų laikymas greitina talpos praradimą ir sukelia pūtimąsi (puffing).
- Pilnai išsekintų laikymas rizikuoja pažeisti celes dėl tolesnės savaiminės iškrovos žemiau minimalios įtampos.

Dauguma įkroviklių turi **Storage** įkrovos režimą — naudok jį, jei neskrisi per artimiausias 24–48 valandas.

---

## Pūtimasis (puffing)

Išsipūtęs pakas turi viduje susikaupusių dujų — tai celės pažeidimo požymis, dažniausiai dėl per gilios iškrovos, perkrovos ar didelės srovės streso.

- Nedidelis pūtimasis: dar naudotina, bet stebėk atidžiai. Netrukus išimk iš naudojimo.
- Ryškus pūtimasis: nedelsiant išmesk iš naudojimo. Nekrauk.
- Utilizuok elektronikos perdirbimo punkte — pirma pilnai iškrauk sūraus vandens kibire.

---

## Saugus naudojimas

- Krauk ant **LiPo-safe maišelio ar nedegaus paviršiaus** — niekada be priežiūros.
- Niekada nekrauk pažeisto, išsipūtusio ar po krašo esančio pako, jo neapžiūrėjęs.
- Krauk **1C** kaip numatytą greitį (t. y. 1500 mAh pakas → 1.5 A įkrovos greitis). Didesni greičiai mažina ciklų kiekį.
- Niekada nepalik kraunamos LiPo be priežiūros.

---

## Baterijos ilgaamžiškumas

Iš kokybiško pako tikėkis 150–300 ciklų, jei elgiesi gerai. End-of-life požymiai:
- Ryškus talpos praradimas esant apkrovai
- Padidėjusi vidinė varža (išmatuok su savo įkrovikliu)
- Nuolatinis pūtimasis po skrydžių
- Voltage sag viršija 0.5 V/celei esant apkrovai ties vidutiniu throttle
