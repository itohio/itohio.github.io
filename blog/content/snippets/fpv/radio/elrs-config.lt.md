---
title: "ELRS konfigūracija — bind phrase, FCC vs LBT"
date: 2026-07-13
draft: false
category: "fpv"
tags: ["fpv", "elrs", "expressLRS", "radio", "link", "bind", "fcc", "lbt"]
---

ExpressLRS (ELRS) — atviro kodo RC linko protokolas, žinomas dėl itin mažo vėlinimo ir didelio range. Du kritiniai nustatymai turi sutapti tarp siųstuvo modulio (TX) ir imtuvo (RX): **regulatory domain** ir **bind phrase**.

---

## Regulatory domain: FCC vs LBT (ELRS/CE)

ELRS firmware yra sukompiliuojamas konkrečiam radijo reguliaciniam domenui, kuris nulemia, kaip jis šokinėja per dažnius ir koks perdavimo elgesys leidžiamas.

| Domenas | Kur naudojamas                    | Elgesys                                          |
|--------|----------------------------------|--------------------------------------------------|
| **FCC**| JAV, Kanada, didžioji Amerikų dalis| Pilnas dažnių hopping, be listen-before-talk    |
| **LBT**| ES, JK, Australija, Lietuva 🇱🇹 | Listen-Before-Talk — patikrina kanalą prieš TX   |
| **CE** | Daugelyje kontekstų — LBT sinonimas | Tas pats kaip LBT; firmware'e ELRS naudoja „LBT“ |

**LBT (Listen Before Talk)** prieš kiekvieną perdavimą prideda trumpą kanalo patikrą, kad atitiktų ES spektro reguliavimą. Tai prideda šiek tiek vėlinimo, palyginti su FCC režimu.

### Kodėl jie turi sutapti

TX ir RX turi būti flashinti su **tuo pačiu regulatory domain**. FCC TX nesusibaindins su LBT RX ir atvirkščiai. Jie veikia skirtingomis kanalų sekomis ir laikinėmis prielaidomis.

Nesutampantys domenai = **nėra linko**, net iš arti.

Jei atsiveži modulį iš JAV ir naudoji jį ES, perflashink ir TX, ir RX į LBT firmware.

---

## Kaip patikrinti savo domeną

ELRS Configurator (flashinimo ar atnaujinimo metu):
- Ieškok `Regulatory Domain` dropdown'o — jis rodys `FCC_915`, `EU_868`, `ISM_2400`, `EU_CE_2400` ir pan.
- 2,4 GHz turi du build'us: `ISM_2400` (be LBT) ir `EU_CE_2400` (su LBT, ES/CE naudojimui).

Ir 900 MHz, ir 2,4 GHz atveju domenas svarbus: TX ir RX turi veikti tame pačiame domene, kad susijungtų.

---

## Bind phrase

Bind phrase — tai vartotojo pasirinkta frazė, kurią TX ir RX hashina į tą patį UID. Ji pakeičia tradicinį baindinimą mygtuko paspaudimu.

- TX modulis ir RX **turi naudoti tą pačią bind phrase**.
- Dažniausiai ji nustatoma flashinant, bet ją galima pakeisti ir vėliau neperflashinus (pvz., per WiFi web UI / runtime options).
- Ji neprivaloma: be frazės sukompiliuotas firmware naudoja tradicinį bind (bind režimas RX + Bind TX Lua skripte).

**Nustatymas:**
1. Atidaryk [ELRS Configurator](https://github.com/ExpressLRS/ExpressLRS-Configurator)
2. Pasirink savo aparatūros target
3. Įvesk savo pasirinktą bind phrase laukelyje `Binding Phrase`
4. Pasirink regulatory domain
5. Flashink ir TX modulį, ir visus imtuvus su **identiškais nustatymais**

```
# Example bind phrase (use something unique to you)
MY_UNIQUE_PHRASE_2024
```

Frazė yra hashinama — jai nereikia būti slaptai, tik nuosekliai vienodai.

---

## Packet rate ir telemetry

Susibaindinęs, TX modulio LUA skripte (paleidžiamame iš radijo tools meniu) nustatyk pageidaujamą packet rate ir telemetry ratio.

| Packet rate | Vėlinimas | Range kompromisas         |
|-------------|----------|---------------------------|
| 50 Hz       | ~20 ms   | Geriausias range          |
| 150 Hz      | ~6,7 ms  | Geras balansas            |
| 250 Hz      | ~4 ms    | Mažas vėlinimas, trumpesnis range|
| 500 Hz      | ~2 ms    | Mažiausias vėlinimas      |
| F1000 / D500| ~1 ms    | Race režimas, labai trumpas range|

Kinematografijai ir bendram skraidymui: **150 Hz** — tvirtas numatytasis. Racing'ui: **250 Hz ar daugiau**.

Telemetry ratio (`1:n`) valdo, kaip dažnai RX siunčia duomenis atgal. Didesnis ratio = mažiau TX laiko telemetrijai = geresnis link budget RX → TX kryptimi.

---

## Dažniausios ELRS problemos

| Simptomas                 | Tikėtina priežastis                           |
|---------------------------|-----------------------------------------------|
| Nesusibaindina po flashinimo | Skirtinga bind phrase arba regulatory domain |
| Protarpiais krenta linkas | Antenos orientacija; TX ir RX domenai skiriasi |
| Telemetrija neveikia      | Telemetry ratio nustatytas į `Off`; blogas serial portas |
| RSSI rodo, bet nėra valdymo | UART nesukonfigūruotas Betaflight'e; CRSF nenustatytas kaip receiver tipas |

Betaflight CLI:
```
set serialrx_provider = CRSF
set serialrx_inverted = OFF
save
```
Ports tab'e priskirk prie ELRS RX prijungtą UART kaip `Serial RX`.
