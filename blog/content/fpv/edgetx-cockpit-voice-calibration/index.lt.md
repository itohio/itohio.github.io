---
title: "2 dalis: kalibracija, ant kurios stovi kiekvienas baterijos įspėjimas"
date: 2026-08-16T10:00:00+03:00
description: "Blogai sukalibruotas įtampos rodmuo neatrodo sugedęs, jis atrodo tikėtinas. Kodėl report_cell_voltage padaro, kad vbat_scale klaida propaguotųsi du kartus."
summary: "Blogai sukalibruotas įtampos rodmuo neatrodo sugedęs, jis atrodo tikėtinas. Kodėl report_cell_voltage padaro, kad vbat_scale klaida propaguotųsi du kartus."
draft: false
toc: true
categories:
  - FPV
  - EdgeTX
tags:
  - fpv
  - betaflight
  - vbat-scale
  - kalibracija
  - baterija
  - lihv
  - edgetx
  - telemetrija
keywords: ["Betaflight vbat_scale kalibracija", "baterijos įtampos kalibracija dronui", "report_cell_voltage celių skaičius"]
series:
  - EdgeTX Cockpit Voice
---

> **EdgeTX Cockpit Voice**, 2 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 1 dalis: kodėl dronas turi su tavimi kalbėti](/fpv/edgetx-cockpit-voice-why/)  ·  [3 dalis: trys mygtukai, trys spalvos ir AND vartai ›](/fpv/edgetx-cockpit-voice-buttons/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)

Kiekvienas šios serijos baterijos įspėjimas yra palyginimas su skaičiumi. Tad prieš
visa kita tas skaičius turi būti tikras. Šis skyrius apie nustatymą, kuris tai
nusprendžia, ir būtent jį akivaizdžiausiai esu padaręs neteisingai.

## Visi laiptai stovi ant kalibracijos, kurios tu tikriausiai nepadarei

Šį skyrių turiu įterpti iškart po ankstesniojo, nes visa, kas seka, nuo jo
priklauso, ir nenoriu, kad kas nors tai statytų ant blogo pamato.

**Tavo baterijos įspėjimai yra būtent tokie geri, kokia yra tavo įtampos
kalibracija.**

Užrašyta tai skamba akivaizdžiai. Praktikoje neakivaizdu, nes blogai
sukalibruotas įtampos rodmuo neatrodo sugedęs. Jis atrodo kaip visiškai
tikėtinas skaičius, kuris tiesiog klysta per 200 mV, ir kiekvienas aukščiau esančių
laiptų slenkstis tą klaidą tyliai paveldi.

Turiu du aparatus, kurie šiuo metu yra blogai sukalibruoti, vadinasi, **jų
įspėjimai suveikia per vėlai.** Ne „šiek tiek netiksliai“, o per vėlai, ta
kryptimi, kuri kainuoja paketą. Aš tai žinau ir dar nesutvarkiau, būtent tokiems
prisipažinimams šis tinklaraštis ir egzistuoja.

Reguliavimo parametras yra `vbat_scale` Betaflight'e. Jis pataiso ADC daliklio
santykį pagal realius tavo plokštės rezistorius, kurie tarp plokščių skiriasi, o
nustatytas jis yra į bendrą numatytąją reikšmę, kuri tinka niekam konkrečiai.

### 3S → 4S spąstai

Konkretus būdas, kuriuo tai mane pagavo, vertas aprašymo, nes tai natūralus
veiksmas ir jokio įspėjimo nėra.

Turėjau aparatus, sukonfigūruotus ir skraidančius su **3S**, o tada perkėliau
juos į **4S** testams. Niekas tame perėjime nepasako, kad tavo kalibracija dabar
kainuoja daugiau. Bet kainuoja, nes klaidos susideda.

`report_cell_voltage = ON` reiškia, kad valdiklis dalija paketo įtampą iš savo
**nustatyto** celių skaičiaus. Ir tas nustatymas pats yra išvestas iš išmatuotos
paketo įtampos įjungimo metu, valdiklis dalija tai, ką perskaito, iš
maksimalios celės įtampos konstantos ir apvalina. Tad įtampos klaida
propaguojasi **du kartus**:

1. Tiesiogiai — į pranešamą vienos celės reikšmę.
2. Galimai dar kartą, nustumdama nustatytą celių skaičių į neteisingą sveikąjį
   skaičių.

Antrasis kelias yra bjaurusis, nes jis suklysta *tyliai ir tikėtinai*. Jei
blogai sumastelintas 4S paketas perskaitomas pakankamai žemai, kad valdiklis
nuspręstų, jog žiūri į 3S, tai jis dalija iš trijų, o ne iš keturių, ir pultui
atiduoda vienos celės reikšmę, kuri patogiai sėdi normaliame diapazone, būdama
visiškai fiktyvi. Tada kiekvienas mano laiptų slenkstis matuotų dydį, kurio
nėra, o `ready` savitikra puikiai suveiktų, nes neteisingas skaičius virš 4,2 V
vis tiek yra skaičius virš 4,2 V.

Savitikra, kurią aprašysiu 4 dalyje, patikrina, ar veikia
signalo kelias. **Ji nepatikrina, ar skaičius yra tikras.** Tai skirtingi
teiginiai, ir noriu būti aiškus, kurį iš jų turiu.

### Regresija naujame konfigūratoriuje

Štai praktinis nepatogumas, ir būtent dėl jo tai gaus atskirą įrašą, o ne
pastraipą.

Anksčiau kalibruodavau taip: pakeldavau motorus iki nedidelės apkrovos —
maždaug 2 A iš paketo, ir tada perjungdavau į kalibracijos skirtuką **motorams
vis dar veikiant**, kad kalibruotų realiame darbo taške, o ne tuščiąja eiga. Tai
svarbu: nori, kad rodmuo būtų patikimas ten, kur jį realiai naudoji, po
apkrova, ne tik ramybėje ant stalo.

Dabartiniame Betaflight konfigūratoriuje taip nebegalima. **Išėjus iš skirtuko
motorai išsijungia.** Tos darbo sekos tiesiog nebėra.

Teisingos pakeičiančios procedūros dar neišsiaiškinau, tad jos čia neišradinėsiu.
Tai bus kitas įrašas: tinkama įtampos kalibracija su dabartiniu
konfigūratoriumi, kas pasikeitė, ir kaip gauti patikimą rodmenį po apkrova be
senojo triuko.

### Viena atvira pastaba apie skaičių, kurį pateiksiu 8 dalyje

3,065 V celei įtampos kritimo reikšmė, kurią pateiksiu 8 dalyje, iš 83 A
akceleravimo mano trijų colių aparate, turi tą pačią priklausomybę. Tai yra tai,
ką skrydžio valdiklis *užrašė*, ir jos tikslumas stovi ant to, kad to aparato
įtampos kalibracija yra tvarkinga. To konkretaus aparato `vbat_scale` prieš
etaloninį matuoklį nepatikrinau nepriklausomai. Traktuok tai kaip stiprų
problemos formos rodiklį, o ne kaip metrologiškai švarų matavimą.

Jei sukursi šioje serijoje aprašytą įspėjimų sistemą ir praleisi kalibraciją,
sukūrei kažką, kas ramiu balsu užtikrintai pasakys tau neteisingą dalyką. Tai,
ko gero, blogiau nei skaičius ekrano kampe.

Du mano aparatai vis dar sako tiesą per vėlai. Žinau, kurie du, ir dar
nesutvarkiau, o tai yra tas dalykas, kuris labiau tinka laboratoriniam sąsiuviniui
nei vadovui.


---

> **Serija:** EdgeTX Cockpit Voice, 2 dalis iš 9. Kaip priverčiau RadioMaster GX12 įgarsinti savo telemetriją, kad žema baterija būtų tai, ką išgirstu, o ne tai, ko pamiršau pažiūrėti.
>
> [‹ 1 dalis: kodėl dronas turi su tavimi kalbėti](/fpv/edgetx-cockpit-voice-why/)  ·  [3 dalis: trys mygtukai, trys spalvos ir AND vartai ›](/fpv/edgetx-cockpit-voice-buttons/)  ·  [Pradėti nuo 1 dalies](/fpv/edgetx-cockpit-voice-why/)
