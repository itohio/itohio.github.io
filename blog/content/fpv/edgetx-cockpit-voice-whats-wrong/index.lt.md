---
title: "8 dalis: keturi dalykai, kurie čia negerai"
date: 2026-08-16T10:00:00+03:00
description: "Sluoksniuojami slenksčiai, nulinis atkirtimo laikas prieš įtampos kritimą, signalas, suveikiantis dar prieš pirmą telemetrijos kadrą, ir ryšio kokybės įspėjimas, kurio niekada neprijungiau."
summary: "Sluoksniuojami slenksčiai, nulinis atkirtimo laikas prieš įtampos kritimą, signalas, suveikiantis dar prieš pirmą telemetrijos kadrą, ir ryšio kokybės įspėjimas, kurio niekada neprijungiau."
draft: false
toc: true
weight: 8
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - edgetx
  - loginiai-jungtukai
  - itampos-kritimas
  - rysio-kokybe
  - juodoji-deze
keywords: ["EdgeTX loginiu jungtuku atkirtimo laikas", "itampos kritimas netikras signalas FPV", "RQly rysio kokybes ispejimas"]
series:
  - EdgeTX Cockpit Voice
---

> **EdgeTX Cockpit Voice**, 8 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 7 dalis: Dvi antenos, dvi juostos](/fpv/edgetx-cockpit-voice-antennas/)  ·  [9 dalis: Perdarymas, sugrupuotas skrydžio tvarka ›](/fpv/edgetx-cockpit-voice-rebuild/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)

Viskas iki šiol yra tai, su kuo realiai skraidau. Ši dalis yra atviras
atsiskaitymas. Pradžioje sakiau, kad kai kurias dalis galima padaryti mažiau
nerangiai, ir štai konkrečiai, kur manoji nerangi.

## Kas čia nerangu

Sakiau, kad būsiu konkretus. Perskaitęs savo konfigūraciją šviežiomis akimis,
štai kas joje ne taip.

### 1. Slenksčiai sluoksniuojasi, ir sirenos kraunasi vienas ant kito

`a < x` yra **lygio**, o ne krašto tikrinimas. Nukritęs žemiau 3,5 V, tuo pačiu
esi ir žemiau 3,6 V, ir žemiau 4,0 V, ir žemiau 3,8 V. Visi tie loginiai
jungtukai yra teisingi vienu metu.

Prie 3,4 V celei mano pultas vykdo:

| Jungtukas | Būsena | Garsas | Kartojimas |
|-----------|--------|--------|------------|
| L1 (< 4,0) | teisinga | `Wrn1` | vieną kartą — jau suveikė, tyli |
| L3 (< 3,8) | teisinga | `rth` | vieną kartą — jau suveikė, tyli |
| L2 (< 3,6) | teisinga | `Sirn` | **kas 1 s, be galo** |
| L8 (< 3,5) | teisinga | `lowbat` | **kas 5 s, be galo** |

Taigi žemiau 3,5 V girdžiu sireną kas sekundę *ir* ištartą „low battery“ kas
penkias sekundes, vienas ant kito. Tuo momentu tai turbūt net gerai, dėmesį
tikrai atkreipia, bet tai nėra *informatyvu*. Sirena, kuri niekada nenustoja,
neperduoda daugiau informacijos nei sirena, kuri suveikia vieną kartą.

Sprendimas, padaryti kiekvieną laiptelį išskirtinį, sujungiant kiekvieną
slenkstį su žemiau esančio slenksčio negacija: „žemiau 3,6 **ir ne** žemiau
3,5“. EdgeTX tai gali antru loginių jungtukų sluoksniu. Aš dar neperdariau.

### 2. Nulinis atkirtimo laikas, visur

Kiekvienas loginis jungtukas turi `delay: 0, duration: 0`. Filtravimo nėra
jokio, o tai reiškia, kad **bet koks trumpalaikis įtampos kritimas sukelia
nuolatinį įspėjimą.**

Man tai nėra teorija. Mano trijų colių 4S aparate juodosios dėžės įrašas iš
staigaus akceleravimo parodė paketo įtampos kritimą iki **3,065 V celei** prie
83 A srovės, momentinis įvykis, visiškai atsistatęs po sekundės dalies. Tai yra
165 mV atsargos iki mano 2,9 V „paketą sugadinai“ signalo suveikimo, ir tai
pakete, kuriam viskas buvo gerai.

Įtampos kritimas nėra įkrovos lygis. Įtampos slenkstis be laiko kriterijaus
skirtumo pamatyti negali. 3,5 V ir 2,9 V laipteliai yra pažeidžiami, nes būtent
tokias reikšmes praeini po apkrova gerokai anksčiau, nei jas pasiekiam ramybėje.

EdgeTX įrankius turi: **Duration** reikalauja, kad sąlyga išsilaikytų N
sekundžių prieš jungtukui tampant teisingu, o **Delay** atideda perėjimą.
Uždėjus sekundės ar dviejų trukmę žemiesiems laipteliams, netikri signalai dėl
įtampos kritimo išnyktų beveik visiškai.

Neskelbsiu konkrečių skaičių, nes jų dar neišvedžiau iš savo įrašų, o rinktis
juos iš nuojautos būtų būtent tas spėjimas, prieš kurį visas šis įrašas ir
argumentuoja. Teisingas būdas, pasižiūrėti į savo juodosios dėžės įtampos
kritimų trukmių pasiskirstymą ir pasirinkti trukmę, ilgesnę už ilgiausią savo
akceleravimą. Tai matavimas, ir įrašus jam turiu.

### 3. Jis ant manęs rėkia dar prieš pasisveikinimą

Tai labiausiai kasdien trikdantis defektas ir tas, kurio dar neišsprendžiau.

**Kai įjungiu bateriją, pultas paskelbia įspėjimus ir „low battery“ garsą _prieš_
tai, kai pasako „ready“.** Kiekvieną kartą. Skamba taip, tarsi aparatui būtų bėda
tą pačią sekundę, kai jis pabunda.

Priežastis yra `a<x` savybė, kuri atrodo akivaizdi po fakto ir yra nematoma tada,
kai sistemą kuri: **lygio palyginimas negali atskirti „kritiškai žemai“ nuo
„dar nėra duomenų“.**

Užsimezgus ryšiui pultas jau turi jungtį, bet CRSF baterijos kadras dar
neatkeliavo, tad `RxBt` sensorius vis dar sėdi ant savo pradinės reikšmės
**0,0 V**. O `0,0` yra mažiau nei 4,0, ir mažiau nei 3,6, ir mažiau nei 3,5, ir —
gražiausia dalis, mažiau nei **2,9**. Tad visi laiptai suveikia vienu metu,
įskaitant žemiausią laiptelį: pultas džiugiai informuoja, kad sugadinau paketą —
ant šviežios baterijos, dar prieš tai, kai atėjo pirmas tikras įtampos rodmuo.

Tada atkeliauja baterijos kadras, `RxBt` šokteli į tikrą reikšmę, visi jungtukai
tampa neteisingi, `L10` pamato `> 4,2 V` ir pasako „ready“ — ir viskas gerai. Bet
pirmas dalykas, kurį išgirstu, yra signalas.

Tai dar nemaloniai persidengia su kadrų greičiu iš ankstesnio skyriaus. Ta mirusi
zona nėra milisekundės, ji tęsiasi tol, kol atkeliauja pirmas baterijos kadras, o
tie kadrai nėra dažni.

Sprendimas, kurio dar nepritaikiau, yra trivialus: **sujungti kiekvieną žemos
įtampos jungtuką su galiojimo sąlyga**, kažkuo panašiu į `RxBt > 0,5`, kad
„nėra telemetrijos“ būtų skaitoma kaip „nėra nuomonės“, o ne kaip „katastrofa“.
Taip pat veiktų `Duration`, pakankamai ilgas, kad pergyventų paleidimo tarpą.

Sprendimas, kurį jau *pradėjau*, yra įdomesnis, ir jis paaiškina jungtuką, kuris
kitaip atrodytų kaip šiukšlė. **L9 yra `RxBt < 3,8 V IR SE-`**, pririštas prie
3 pozicijų jungtuko SE vidurinės pozicijos, o ne prie žalio `bat` mygtuko. Tai
sąmoninga: **aktyvavimą (arm) uždėjau ant SE**, o **įspėjimus ant SE vidurio**, kad
visa įspėjimų sistema būtų aktyvi *prearm* momentu, o ne kol aparatas stovi ant
žemės nieko neveikdamas. Prearm yra teisinga vieta priešskrydžio įtampos
patikrinimui, tai momentas, kai jau ruošiesi įsipareigoti.

**Prearm dar nesukonfigūravau.** Puikiai žinau, kada tai padarysiu: pirmą kartą,
kai pakelsiu droną, pultas atsitrenks man į krūtinę ir arm jungtukas persivers.
Esu gana tikras, kad tai bus pakankamai įsimintina pamoka, kad tą patį vakarą
darbas būtų padarytas, su sąlyga, kad dar turėsiu visus pirštus rašyti.

Kas yra blogas planas. Bet tai sąžiningas mano tikrojo plano aprašymas.

### 4. Jokio ryšio kokybės įspėjimo

Tai didžiausia reali spraga. Mano modelyje yra:

```yaml
rssiSource: none
rfAlarms:
   warning: 65
   critical: 35
```

Radijo signalo slenksčiai sukonfigūruoti, bet `rssiSource` yra `none`, tad
niekas nėra prijungta, kad juos paleistų. Tuo tarpu `RQly`, `RSNR`, `ANT` ir abu
`1RSS`/`2RSS` sensoriai sėdi sensorių sąraše, pilnai užpildyti, su `logs: 1`
kiekvienam iš jų, ir visiškai nenaudojami nė vieno loginio jungtuko.

Turint galvoje, kad visas šis projektas egzistuoja tam, kad nebeprašvilptum ribos,
į kurią nežiūrėjau, tai, kad nepritaikiau jo **ryšio kokybei**, ribai, kuri
skrydžius realiai baigia, krūme, toli nuo mašinos, yra praleidimas, kurį
pastebėjau tai rašydamas. `RQly < 70 → PLAY_TRACK "link"` yra vienas loginis
jungtukas ir viena specialioji funkcija, ir tai sekantis punktas sąraše.

Ir yra dar blogiau, dėl to, kas būtent tie sensoriai yra. Žr. žemiau.

Ironija manęs neaplenkia: būtent tuos pačius sensorius skaito mano
[RX Blind-Spot Viewer](https://rxmap-viewer.sintra.site/rxmap/), kad sukurtų 3D
antenos diagramą. Aš mielai praleisiu vakarą analizuodamas ryšio kokybę trimis
dimensijomis po skrydžio, o vieno jungtuko, dėl kurio pultas skrydžio metu pasakytų
„link“, dar neprijungiau.

Keturi defektai, visi mano, visus galima sutvarkyti per vakarą. Jų užrašymas ir yra
ta dalis, dėl kurios jie tampa sutvarkomi.


---

> **Series:** EdgeTX Cockpit Voice, 8 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 7 dalis: Dvi antenos, dvi juostos](/fpv/edgetx-cockpit-voice-antennas/)  ·  [9 dalis: Perdarymas, sugrupuotas skrydžio tvarka ›](/fpv/edgetx-cockpit-voice-rebuild/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)
