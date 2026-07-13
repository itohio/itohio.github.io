---
title: "Neuroninio Tinklo Apmokymas Dronų Aptikimui iš Garso"
date: 2026-07-13
description: "10 klasių akustinis klasifikatorius Shahed-136 dronams ir vartotojų kvadrotoriams, apmokytas beveik nuo nulio, nes duomenų tiesiog nėra. Kas veikė, kas nepavyko ir kodėl INT8 kvantizacija dar nėra atsakymas."
draft: true
toc: true
categories:
  - Mašininis mokymas
  - Aparatinė įranga
tags:
  - mašininis-mokymas
  - garsas
  - neuroninis-tinklas
  - pytorch
  - shahed
  - dronų-aptikimas
  - onnx
  - kvantizacija
  - drone-detector
---

*Gyva demonstracija: [drone-detector.sintra.site](https://drone-detector.sintra.site)*

<!-- IMAGE: drone-detector.sintra.site sąsajos ekrano kopija rodo gyvąjį išvadą -->
*[TODO: Gyvosios aptikimo svetainės ekrano kopija naršyklėje]*

---

## Problema

Shahed-136 veikia su dvitakčiu stumiamojo sraigto varikliu. Pagrindinis dažnis yra apie 83 Hz su harmoninėmis, besitęsiančiomis iki maždaug 4 kHz. Jis skamba nieko bendra neturinčiai su vartojimo kvadratoriu, fiksuotasparnio RC lėktuvu ar eismu. Akustinis parašas yra ryškus, jei žinai, ko klausai.

Inžinerinė problema: negalite atsisiųsti pažymėto Shahed-136 garso duomenų rinkinio. Ši etiketė neegzistuoja AudioSet 527 klasėse. Nėra DADS įrašo, jokios Freesound kategorijos, jokio akademinio etalono. Įrašote patys arba sintezuojate iš turimos medžiagos — o turima medžiaga yra tai, kas buvo viešai paskelbta iš konflikto zonų, o tai yra negausa, nesuderinama pagal mikrofono atstumą ir dažnai stipriai užteršta fono triukšmu.

Nustatyta užduotis: 10 klasių vienos etiketės klasifikatorius, veikiantis realiuoju laiku slenkančiame 10 sekundžių lange su 1 sekundės postūmiu. Klasės yra: **dronas, spiečius, kvadrotas, sraigtasparnis, reaktyvinis, lėktuvas, motociklas, vejos pjoviklis, traktorius, laukimas**. Modelis turi veikti naršyklėje ant procesoriaus ir vidutinės klasės mobiliojoje aparatinėje įrangoje.

---

## Duomenų rinkinio surinkimas

Šis skyrius svarbiausias visiems, kas bando pakartoti. Duomenų rinkinys gaunamas iš penkių skirtingų šaltinių, kiekvienas su skirtingomis kokybės charakteristikomis ir gedimų būdais.

| Šaltinis | Klasės | Pastabos |
|----------|--------|---------|
| Vartotojo mikrofono įrašai | dronas (Shahed proxy), spiečius | Vienintelė galimybė — viešo duomenų rinkinio nėra |
| HuggingFace DADS | kvadrotas | Vartojimo UAV; etiketė=1 tik |
| ESC-50 | sraigtasparnis, lėktuvas, vejos pjoviklis, laukimas | Švari, kuruota, 50 klasių, 2000 įrašų |
| AudioSet per yt-dlp | sraigtasparnis, reaktyvinis, lėktuvas, motociklas, traktorius, laukimas | MID pagrįstas segmentų atsisiuntimas; ~30% nesėkmių rodiklis |
| DREGON / SPCup19 (Inria) | kvadrotas (savojo triukšmo) | Laivo įrašai; kelių kanalų → mono |
| Zenodo 15190811 | kvadrotas (lauke) | 14 realių dronų modelių; pasirinkti 3–4 dydžiui valdyti |

Drono įrašus padariau pats lauke. Gauti pakankamai įvairovės — skirtingų atstumų, skirtingų kampų, skirtingų fono sąlygų — prireikė kelių seansų. Akustinė tų įrašų įvairovė lemia, kaip gerai modelis apibendrina realiam diegimui.

### AudioSet atsisiuntimo nesėkmės

Atsisiuntimas iš AudioSet per yt-dlp naudojant MID kodus konkrečioms klasėms turi maždaug 30% nesėkmės rodiklį: vaizdo įrašai buvo ištrinti, padaryti privatūs arba apriboti regionu nuo etikečių surinkimo. Iš atsisiųstų įrašų dar viena dalis neišlaiko juostos energijos kokybės patikrinimo — garso segmentas turi teisingą etiketę, bet daugiausia yra tyla arba ekstremalus iškraipymas. Šiuos atmetiau tyliai ir registravau atmetimo rodiklį; kai kuriose klasių grupėse jis siekė 15%.

### Nulinės reikšmės įrašų taisymas

Trumpi šaltinio įrašai, paplėtoti iki 10 sekundžių, įveda struktūrinį artefaktą: modelis mato tą patį garso šabloną kartojantį, kas nepanašu į bet ką diegimo aplinkoje. Sukūriau paprastą RMS pagrįstą turinio ilgio detektorių, kuris nustato faktinį garso turinio langą įraše, atmeta trumpesnius nei 3 sekundžių įrašus po tylos apkarpymo ir taiko atsitiktinę padėties nustatymą vietoj paplėtimo.

### Spiečiaus sintezė

Spiečiaus klasė yra 100% sintetinė. Nėra viešo "kelių Shahed orlaivių" garso. Visi 300 spiečiaus įrašų buvo sugeneruoti iš dronų įrašų baseino:

```mermaid
flowchart TD
    DR["dronų įrašai\n(vartotojo įrašai)"]
    PS["aukščio poslinkis ±2 pustoniai\nstiprinimas −4..0 dB\nlaiko dreifas 0..0.5s"]
    MX["maišyti 2–4 egzempliorius"]
    BG["laukimo fonas\nSNR 8–18 dB\np=0.6"]
    SW["spiečiaus įrašas\n10 s @ 16 kHz"]

    DR --> PS --> MX --> BG --> SW
```

Kiekvienas egzempliorius gauna nepriklausomą aukščio poslinkį (±2 pustoniai), stiprinimo atsitiktinumą (−4 iki 0 dB) ir laiko dreifą (iki 0,5 sekundžių). Su 60% tikimybe primaišomas laukimo fonas ties 8–18 dB SNR.

### Augmentacijos rinkinys

Taikoma tik mokymo metu, niekada validavimo metu:

- Asimetrinis stiprinimas: −20 iki +6 dB
- Aukščio poslinkis (kiekvienam įrašui, atsitiktinis ±3 pustonių ribose)
- Priedinis triukšmas ties 3–25 dB SNR
- Aukštų dažnių ir žemų dažnių filtravimas (atsitiktiniai ribiniai dažniai)
- Laukimo fono mišinys

### Klasės balansas

Apribotas iki 1000 įrašų vienai klasei mokymo dalyje. `WeightedRandomSampler` kompensuoja likusį disbalansą. Validavimas niekada nebuvo ribojamas.

Galutinis duomenų rinkinys: ~13 200 įrašų iš viso. Mokymo/validavimo skirstymas yra sluoksniuotas.

---

## Architektūra

```mermaid
graph LR
    W["bangos forma\n10s @ 32kHz"]
    MEL["AugmentMelSTFT\n128 mel × 1000 kadrų"]
    BB["MobileNetV3 stuburas\nmn20_as\n~16.1M parametrų\nAudioSet-apmokyta"]
    POOL["AdaptiveAvgPool2d(1)"]
    HEAD["DroneHead\n256 paslėptų, dropout 0.4\n→ 10 logitų"]
    OUT["sigmoid → vienai klasei tikimybė"]

    W --> MEL --> BB --> POOL --> HEAD --> OUT
```

### Kodėl MobileNetV3 / EfficientAT

Stuburas yra `mn20_as` iš EfficientAT šeimos — MobileNetV3-Large, išmasteluotas iki 20 pločio dauginamojo, iš anksto apmokytas ant visų 527 AudioSet klasių. Ties ~16,1 mln. parametrų ir mAP 0,478 ant AudioSet, tai yra auksinė vidurį: `mn10` nepakankamai apmokytas šiai užduočiai, `mn40` per sunkus mobiliajam išvedimui.

```python
class DroneHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden=256, dropout=0.4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(self.pool(x).flatten(1))
```

---

## Mokymo protokolas

```mermaid
sequenceDiagram
    participant P1 as 1 fazė (tik galva)
    participant P2 as 2 fazė (pilnas reguliavimas)
    participant CKPT as Geriausias patikros taškas

    Note over P1: Stuburas užšaldytas<br />lr = 1e-4<br />Mokyti kol val F1 ≥ 0.50

    P1->>CKPT: išsaugoti kai pagerėja val F1
    P1->>P2: pereiti kai pasiekiama riba

    Note over P2: Visi parametrai atšildyti<br />stuburo lr = 5e-6<br />galvos lr = 1e-4<br />35 epochos

    P2->>CKPT: išsaugoti pagal kalibruotą makro F1
    P2->>P1: atkurti geriausią, atgal į P1 kol kantrybė=8 išnaudota

    loop Ciklai 1–5
        P1-->>P2: kartoti
        P2-->>P1: kartoti
    end
```

Kodėl pakaitomis: stuburas stabilizuojasi, kai užšaldytas. Kai jį atšildote, galvos gradiento signalas pernelyg stipriai traukia stuburą ties dideliu mokymosi greičiu — štai kodėl stuburo mokymosi greitis 2 fazėje yra 20× mažesnis nei galvos (5e-6 prieš 1e-4).

### Nuostolių funkcija

`BCEWithLogitsLoss` su kiekvienos klasės `pos_weight`:

```python
pos_weight = (n_neg / n_pos).clamp(min=NUM_CLASSES, max=30.0)
```

---

## Rezultatai ir kiekvienos klasės analizė

<!-- IMAGE: mokymo kreivė — nuostoliai ir makro F1 per pakaitomuosius P1↔P2 ciklus, pažymėti fazių perėjimai -->
*[TODO: Mokymo kreivė — nuostoliai ir makro F1 per ciklus]*

<!-- IMAGE: 10×10 painiavos matricos šilumos žemėlapis -->
*[TODO: 10×10 painiavos matrica]*

### Kiekvienos klasės F1

| Klasė | F1 | Pastaba |
|-------|----|---------|
| motociklas | 0.997 | Lengviausia — ryškus aukšto RPM harmoninis profilis |
| traktorius | 0.981 | Žemo dažnio pagrindinis, nuoseklus parašas |
| reaktyvinis | 0.974 | Aukštų dažnių plačiajuostis |
| lėktuvas | 0.961 | Nuoseklus sraigto parašas |
| sraigtasparnis | 0.944 | Pagrindinis rotorius ties 15–25 Hz |
| laukimas | 0.921 | Fonas; painiojamas su vejapjove |
| kvadrotas | 0.906 | Rotoriaus triukšmas sutampa su laukimu mažame droslyje |
| dronas | 0.913 | Mažas duomenų rinkinys; painiojamas su spiečiumi |
| spiečius | 0.854 | Sunkiausia — sintetinis; dalinasi parašu su dronu |
| vejapjovė | 0.887 | Painiojama su laukimu; abu plačiajuosčiai, žemo dažnio |

**Makro F1 (kalibruotas): 0.940**

### Kiekvienos klasės ribos kalibravimas

Vienoda 0,5 riba yra neteisinga. Kalibruotos ribos ties drono ≥ 90% tikslumo tikslu:

| Klasė | Riba | Tikslumas | Atpaukimas |
|-------|------|----------|------------|
| dronas | 0.863 | 0.901 | 0.912 |
| spiečius | 0.973 | 0.902 | 0.771 |
| vejapjovė | 0.413 | 0.800 | 0.978 |
| traktorius | 0.934 | 0.993 | 0.974 |

---

## Kvantizacija: kas nutiko ir kodėl nepavyko

**INT8 modelis nepasiekė diegimo lygio.**

```mermaid
graph TD
    FP32["ONNX FP32\n48.5 MB\nmakro F1 = 0.940\nlat = 16.7ms  p95 = 23.6ms"]
    INT8["ONNX INT8 W8A8\n13.0 MB  3.7× mažesnis\nmakro F1 = 0.790\nlat = 7.8ms  p95 = 9.8ms"]
    GUARD["F1 apsauga = 0.85\nINT8 kalibruotas = 0.828\nNepavyko → aktyvus = FP32"]

    FP32 -->|"quantize_static\nQDQ formatas\nMinMax kalibravimas"| INT8
    INT8 --> GUARD
```

Entropija agresyviai apkerta aktyvacijos diapazoną. Spiečius turi retų, bet didelių aktyvacijų — entropija jas traktuoja kaip anormalijas. Spiečiaus atpaukimas nukrito nuo 0,77 iki 0,52.

MinMax kalibravimas išlaiko visą stebimą diapazoną. Geriau, bet makro F1 tik 0,828 po kiekvienos klasės ribų rekalibravimai ant INT8 modelio.

Pagrindinė priežastis: INT8 keičia kiekvienos klasės tikimybių skales netolygiai. Traktoriaus riba persikėlė nuo 0,934 (FP32) iki 0,010 (INT8). Vejapjovės riba persikėlė nuo 0,413 iki 0,157. Tai ne apvalinimo klaidos — INT8 logitų pasiskirstymas yra struktūriškai skirtingas nuo FP32 tam tikroms klasėms.

---

## Gyvosios išvados architektūra

```mermaid
flowchart LR
    MIC["mikrofonas\n32kHz"]
    BUF["žiedinis buferis\n15s"]
    WIN["10s langas\niškviesti kas 1s"]
    MEL["mel spektrograma\n128×1000"]
    ONNX["ONNX Runtime\nFP32 išvada\n16.7ms vid."]
    MED["einamoji mediana\nper paskutinius 5 langus"]
    THR["kiekvienos klasės riba\ntaikyti kalibruotas reikšmes"]
    OUT["aptikimo išvestis"]

    MIC --> BUF --> WIN --> MEL --> ONNX --> MED --> THR --> OUT
```

---

## Atviri klausimai

**QAT**: Aiškiausias kelias į INT8 modelį, atitinkantį F1 apsaugą. Dar nebandyta.

**TFLite eksportas**: Blokuojamas dėl platformos suderinamumo — ONNX → TFLite konvertavimo kelias per tf2onnx buvo nepatikimas šiam modelio grafiko topologijai.

**Srautinis išvedimas įterptinėje aparatinėje įrangoje**: 10 sekundžių langas + 1 sekundės postūmio architektūra buvo sukurta programoms, toleruojančioms delsą. Tinkama srautinė architektūra sumažintų delsą iki mažiau nei 2 sekundžių, tačiau reikalauja architektūros pakeitimų.

**Daugiau dronų įrašų**: Drono klasė yra silpniausia duomenų rinkinio vieta. Daugiau įrašų iš skirtingų orlaivio konfigūracijų, atstumų ir fonų sumažintų drono ↔ spiečiaus painiavą.

Modelis nėra baigtas. Tačiau ties makro F1 0,940 ant FP32 procesoriaus ties 16,7 ms delsa, jis yra pakankamai toli, kad verta diegti ir gauti realaus pasaulio atsiliepimų.
