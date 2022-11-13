---
title: "Pirmasis įrašas Lietuvių kalba"
date: 2022-05-13T18:26:47+02:00
subtitle: "This is a test post to check how the multilanguage feature works"
authod: admin
thumbnail: https://preview.redd.it/0x68eoaoez071.jpg?width=640&crop=smart&auto=webp&s=3365f78f7da8accfbb5667076b02c87f69204469
---

Šiuo įrašu noriu pasidžiaugti statinio turinio svetainės priežiūra. O konkrečiai - [Hugo](https://gohugo.io).

Šiaip esu gana tingus žmogus su daugybe interesų. Ir išmokau tokios strategijos - jeigu nepavyksta kažko padaryti kuo greičiau, 
tai tikriausiai neverta to daryti visai.
Išskirtinais atvejais jeigu matau realią naudą iš potencialaus projekto(ar tai asmeninė satisfakcija, arba komercinė nauda, arba nauda visuomenei) - tuomet galiu jo imtis rimtai.
Priešingu atveju norėdamas patenkinti smalsumą susikūriu smėlio dėžę ir išsibandau idėjas kol neatsiranda kas nors įdomesnio. Geriausiai šitą strategiją apibūtina štai toks komiksas:

![Apie projektus](https://preview.redd.it/0x68eoaoez071.jpg?width=640&crop=smart&auto=webp&s=3365f78f7da8accfbb5667076b02c87f69204469)

Taigi, Pirma šio blogo versija buvo Django projektas su visokiais templeitais ir panašiai. Jį prižiūrėti buvo gana įdomu, tačiau visgi reikalaudavo daug mentalinių pastangų.
Nekalbant jau apie žingsnių kiekį norint atnaujinti turinį net ir su visom automatizacijom.

Tuomet atėjo laikas prie CMS, šiuo atveju dokerizacija bei Mezzanine. Tuo atveju priežiūra būdavo daug paprastesnė - tereikėdavo prisijungti prie admin panelės, įkelti naują įrašą ir viskas.
Tačiau kai atėjo laikas eksperimentuoti su įvairiais protokolais ir norėjau perpanaudoti savo serverį tiems tikslams supratau, jog atstatinėti tokį daiktą nėra labai paprastą. Juolabiau vis labiau tolau nuo Python ir vis arčiau Go.

Dabar gi, po kokių dviejų-trijų metų tylos ir kaltės jausmo, jog absoliučiai niekuo niekur nesidalinu, ir po to kai iš pelenų prikėliau Mezzanine pagrindu sukurtą blogą(ir kuomet suvokiau kiek reikia šokti su bugnu aplink tiekėją, kad gaučiau SSL sertifikatą:D), suvokiau, jog net ir jungtis prie admin panelės nenoriu. O noriu, pasirodo, dirbti tiesiogiai su Github.

Tiksliau planas atrodė labai paprastas:
- palikti puslapio github repo
- sukurti deployment sripts
- sukurti Github Pages su automatiniu deploymentu
- nukreipti DNS įrašus į Github Pages

Tokiu būdu gaunu du svarbius dalykus - turinį valdyti ir atnaujint labai paprasta. Deplojint - labai paprasta (`git commit` ir `git push`). Eksperimentuot su Go backend'e - dar paprasčiau(Docker konteinerių rotacija automatizuota) - nors šiuo metu ir nereikia to backendo :)

Rezultate:
- sumažinau sau nemažai trinties ką nors rašyti
- SSL dabar rūpinasi Github (big win)
- Smarkiai sumažėjo apkrova nuo mano serverio
- Smarkiai padidėjo galimybės eksperimentuoti su Go - pagaliau galėsiu eksperimentuoti su įvairiais web apps'ais prieš migruodamas juos į AWS arba Heroku

