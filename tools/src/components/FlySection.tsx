"use client";

import { useState, useEffect, useCallback } from "react";
import { useLang, type LS, type Lang } from "@/i18n";

// ── types ─────────────────────────────────────────────────────────────────────
interface CheckItem {
  id: string;
  text: LS;
  hint?: LS;
  link?: string;
  linkLabel?: LS;
  warn?: boolean;
}
interface ChecklistDef {
  id: string;
  name: LS;
  badge: string;
  color: string;
  description: LS;
  cert: LS;
  defaultMinutes: number;
  preItems: CheckItem[];
  postItems: CheckItem[];
}
interface FlightState {
  phase: "pre" | "flying" | "post" | "done";
  checkedPre: string[];
  checkedPost: string[];
  flightStart?: number;
  plannedEnd?: number;
  location?: string;
}

const L = (en: string, lt: string): LS => ({ en, lt });

// ── common checklist items ────────────────────────────────────────────────────
const COMMON_PRE: CheckItem[] = [
  { id:"reg",    text:L("Operator registration number marked on drone", "UAS naudotojo registracijos numeris pažymėtas ant drono") },
  { id:"cert",   text:L("Pilot certificate accessible (phone/card)", "Piloto pažymėjimas pasiekiamas (telefone / kortelė)") },
  { id:"airsp",  text:L("Airspace checked — no restricted zones active at this location", "Oro erdvė patikrinta — šioje vietoje nėra aktyvių ribojamų zonų"),
    hint:L("Use official drone app for your country. Check CTR, UAS zones, prohibited/conditional areas.", "Naudokite oficialią savo šalies dronų programėlę. Patikrinkite CTR, UAS geografines zonas, draudžiamas / sąlygines zonas.") },
  { id:"notam",  text:L("NOTAMs checked for today", "Šiandienos NOTAM pranešimai patikrinti"),
    hint:L("Military exercises, VIP movements, air shows can appear with short notice. Check AIS portal.", "Karinės pratybos, VIP judėjimas, aviacijos šou gali atsirasti su trumpu įspėjimu. Tikrinkite AIS portalą.") },
  { id:"wx",     text:L("Weather assessed: wind within drone limit, visibility OK, no precipitation", "Oras įvertintas: vėjas neviršija drono ribų, matomumas geras, nėra kritulių"),
    hint:L("Check gusts, not just average wind. Assess the full planned flight window, not just now.", "Tikrinkite gūsius, ne tik vidutinį vėją. Įvertinkite visą planuojamo skrydžio laiką, ne tik dabartį.") },
  { id:"bat",    text:L("Battery: fully charged, no swelling, resting voltage OK", "Akumuliatorius: pilnai įkrautas, neišsipūtęs, ramybės įtampa tinkama"),
    hint:L("LiPo full: 4.20 V/cell (or 4.35 V/cell LiHV). Reject any swollen or physically damaged pack.", "Pilnas LiPo: 4,20 V/celei (LiHV — 4,35 V/celei). Nenaudokite išsipūtusių ar fiziškai pažeistų pakuočių.") },
  { id:"props",  text:L("Props: no chips or cracks, securely tightened, correct rotation direction", "Sraigtai: be įskilimų ir nuoskalų, tvirtai priveržti, teisinga sukimosi kryptis"),
    hint:L("Run a finger along each blade. Even small nicks cause vibration and IMU noise.", "Perbraukite pirštu kiekvieną mentę. Net mažos nuoskalos sukelia vibraciją ir IMU triukšmą.") },
  { id:"frame",  text:L("Frame: no cracks, all screws tight, arms locked or folded correctly", "Rėmas: be įtrūkimų, visi varžtai priveržti, rankos užfiksuotos arba teisingai išlankstytos") },
  { id:"motors", text:L("Motors: spin freely by hand, no grinding or roughness", "Varikliai: ranka sukasi laisvai, be trynimosi ar šiurkštumo") },
  { id:"gps",    text:L("GPS: 3D fix acquired, ≥10 satellites recommended", "GPS: gauta 3D fiksacija, rekomenduojama ≥10 palydovų"),
    hint:L("Wait for solid lock before arming. Don't rush — GPS-assisted modes need stable lock.", "Prieš aktyvuodami palaukite stabilios fiksacijos. Neskubėkite — GPS režimams reikia stabilaus signalo.") },
  { id:"comp",   text:L("Compass: calibrated for this location (especially first flight here)", "Kompasas: kalibruotas šioje vietoje (ypač pirmam skrydžiui čia)"),
    hint:L("Recalibrate whenever moving to a new area significantly, or near metal structures.", "Perkalibruokite persikėlę į gerokai kitą vietovę arba prie metalinių konstrukcijų.") },
  { id:"fsc",    text:L("Failsafe/RTH: altitude set above tallest obstacle in any direction", "Failsafe / RTH: aukštis nustatytas virš aukščiausios kliūties bet kuria kryptimi"),
    hint:L("RTH must clear everything between the drone and home point, including terrain behind you.", "RTH turi praskristi virš visko tarp drono ir namų taško, įskaitant reljefą už jūsų.") },
  { id:"rc",     text:L("RC link: full signal, stick inputs respond correctly in all axes", "RC ryšys: pilnas signalas, valdymo svirtys teisingai reaguoja visomis ašimis") },
  { id:"cam",    text:L("Camera/gimbal: secure, horizon level, SD card present and writable", "Kamera / stabilizatorius: pritvirtinti, horizontas lygus, SD kortelė įdėta ir įrašoma") },
  { id:"elz",    text:L("Emergency landing zones identified in all directions", "Avarinio tūpimo vietos numatytos visomis kryptimis") },
  { id:"clear",  text:L("Take-off area clear of people, animals, and loose objects", "Kilimo vieta be žmonių, gyvūnų ir palaidų daiktų") },
  { id:"vlos",   text:L("VLOS confirmed: can clearly see the drone at planned operating distance", "VLOS patvirtintas: dronas aiškiai matomas planuojamu atstumu") },
];

const COMMON_POST: CheckItem[] = [
  { id:"land",    text:L("Drone safely landed, not rolling", "Dronas saugiai nutūpė, nerieda") },
  { id:"disarm",  text:L("Motors disarmed, props fully stopped", "Varikliai išjungti, sraigtai visiškai sustojo") },
  { id:"batdis",  text:L("Battery disconnected", "Akumuliatorius atjungtas") },
  { id:"battemp", text:L("Battery not excessively hot — wait before charging", "Akumuliatorius neperkaitęs — prieš įkraunant palaukite"),
    hint:L("Above 40°C after flight is normal. Don't charge until below 35°C.", "Virš 40 °C po skrydžio — normalu. Nekraukite, kol neatvės žemiau 35 °C.") },
  { id:"batstor", text:L("Battery: set storage voltage (3.7–3.85 V/cell) if done flying for today", "Akumuliatorius: jei šiandien baigėte skraidyti, nustatykite saugojimo įtampą (3,7–3,85 V/celei)"),
    hint:L("Don't leave LiPo at 100% — storage charge protects cell health long-term.", "Nepalikite LiPo įkrauto 100 % — saugojimo įkrova ilgam saugo celių būklę.") },
  { id:"inspect", text:L("Props and frame inspected for new damage", "Sraigtai ir rėmas patikrinti dėl naujų pažeidimų") },
  { id:"media",   text:L("Memory card secured / footage backed up", "Atminties kortelė išsaugota / medžiaga nukopijuota") },
  { id:"log",     text:L("Flight time, location, and incidents logged", "Skrydžio laikas, vieta ir incidentai užregistruoti") },
  { id:"pack",    text:L("All equipment packed and accounted for", "Visa įranga supakuota ir suskaičiuota") },
];

// ── checklist definitions ─────────────────────────────────────────────────────
const CHECKLISTS: ChecklistDef[] = [
  {
    id:"a1", name:L("Open A1", "Atviroji A1"), badge:"A1", color:"blue",
    description:L("C0/C1 CE-marked drone · near (not over assemblies of) people · A1/A3 cert",
                  "C0/C1 klasės CE dronas · šalia (bet ne virš sambūrių) žmonių · A1/A3 pažymėjimas"),
    cert:L("A1/A3 certificate", "A1/A3 pažymėjimas"), defaultMinutes:10,
    preItems:[
      ...COMMON_PRE,
      { id:"a1-cls",   text:L("Drone is C0 (<250 g) or C1 (<900 g) CE class-marked", "Dronas pažymėtas C0 (<250 g) arba C1 (<900 g) CE klasės žyma"),
        hint:L("A1 requires CE class marking C0 or C1. Privately built drones and drones without CE class marks operate in A3 only, not A1.",
               "A1 reikalauja C0 arba C1 klasės žymos. Savadarbiai dronai ir dronai be klasės žymos gali skraidyti tik A3, ne A1.") },
      { id:"a1-rid",   text:L("[C1 only] Remote ID active and updated", "[Tik C1] Nuotolinis identifikavimas aktyvus ir atnaujintas"),
        hint:L("EU 2019/947 UAS.OPEN.020(5)(d): C1 drones in A1 must have Remote ID active. C0 and private builds: not required.",
               "ES 2019/947 UAS.OPEN.020(5)(d): C1 dronai A1 pakategorėje privalo turėti aktyvų nuotolinį identifikavimą. C0 ir savadarbiams — neprivaloma.") },
      { id:"a1-noass", text:L("Not flying over any assembly of people", "Neskraidoma virš žmonių sambūrių"), warn:true,
        hint:L("Assemblies (events, markets, crowds) are prohibited in ALL Open subcategories including A1. This is absolute.",
               "Sambūriai (renginiai, turgūs, minios) draudžiami VISOSE atvirosios kategorijos pakategorėse, įskaitant A1. Išimčių nėra.") },
    ],
    postItems:COMMON_POST,
  },
  {
    id:"a2", name:L("Open A2", "Atviroji A2"), badge:"A2", color:"green",
    description:L("C2 CE-marked drone · 30 m low-speed / 50 m normal from people · A2 CoC required",
                  "C2 klasės CE dronas · 30 m mažo greičio / 50 m įprastu režimu nuo žmonių · reikalingas A2 CoC"),
    cert:L("A2 CoC + A1/A3 certificate", "A2 CoC + A1/A3 pažymėjimas"), defaultMinutes:10,
    preItems:[
      ...COMMON_PRE,
      { id:"a2-coc",   text:L("A2 Certificate of Competency (CoC) present", "A2 kompetencijos pažymėjimas (CoC) su savimi") },
      { id:"a2-c2",    text:L("Drone is C2 CE class-marked (<4 kg) — only C2 may fly A2", "Dronas pažymėtas C2 CE klasės žyma (<4 kg) — A2 gali skraidyti tik C2"),
        hint:L("No CE class marking = privately built = A3 only. You cannot fly A2 with an unmarked or privately built drone.",
               "Be klasės žymos = savadarbis = tik A3. Su nepažymėtu ar savadarbiu dronu A2 skraidyti negalima.") },
      { id:"a2-rid",   text:L("Remote ID active and updated", "Nuotolinis identifikavimas aktyvus ir atnaujintas"),
        hint:L("C2 drones must have Remote ID (EU 2019/945 Part 3). Must be active during flight.", "C2 dronai privalo turėti nuotolinį identifikavimą (ES 2019/945, 3 dalis). Turi būti aktyvus skrydžio metu.") },
      { id:"a2-lsm",   text:L("Low-speed mode configured if approaching within 30–50 m of people", "Mažo greičio režimas sukonfigūruotas, jei artėjama 30–50 m prie žmonių"),
        hint:L("Low-speed mode (≤3 m/s) enables the 30 m minimum. Without it: 50 m minimum applies.", "Mažo greičio režimas (≤3 m/s) leidžia 30 m minimumą. Be jo galioja 50 m minimumas.") },
      { id:"a2-noass", text:L("Not flying over any assembly of people", "Neskraidoma virš žmonių sambūrių"), warn:true },
    ],
    postItems:COMMON_POST,
  },
  {
    id:"a3", name:L("Open A3", "Atviroji A3"), badge:"A3", color:"orange",
    description:L("Any class OR private build · 150 m from all built-up areas · A1/A3 cert",
                  "Bet kuri klasė ARBA savadarbis · 150 m nuo užstatytų teritorijų · A1/A3 pažymėjimas"),
    cert:L("A1/A3 certificate", "A1/A3 pažymėjimas"), defaultMinutes:12,
    preItems:[
      ...COMMON_PRE,
      { id:"a3-150",  text:L("Confirmed 150 m+ from ALL residential, commercial, recreational, and industrial areas", "Patvirtinta: 150 m ir daugiau nuo VISŲ gyvenamųjų, komercinių, poilsio ir pramoninių teritorijų"), warn:true,
        hint:L("150 m from the nearest building/settlement. Parks with people, campsites, and farms with workers count.", "150 m nuo artimiausio pastato / gyvenvietės. Parkai su žmonėmis, kempingai ir ūkiai su darbuotojais taip pat skaičiuojasi.") },
      { id:"a3-nop",  text:L("No uninvolved people in the planned flight area", "Planuojamoje skrydžio zonoje nėra nesusijusių asmenų") },
      { id:"a3-pv",   text:L("Private build or CE-class drone — both allowed in A3", "Savadarbis arba CE klasės dronas — A3 leidžiami abu") },
    ],
    postItems:COMMON_POST,
  },
  {
    id:"specific-bvlos", name:L("Specific: BVLOS", "Specialioji: BVLOS"), badge:"BVLOS", color:"purple",
    description:L("Beyond Visual Line of Sight with private/any build · Specific category required (EU 2019/947 Art. 5)",
                  "Skrydis už tiesioginio matomumo ribų · reikalinga specialioji kategorija (ES 2019/947, 5 str.)"),
    cert:L("Operational authorisation from NAA (or STS-02 declaration)", "TKA veiklos leidimas (arba STS-02 deklaracija)"), defaultMinutes:20,
    preItems:[
      ...COMMON_PRE.filter(i => !["vlos"].includes(i.id)),
      { id:"bv-auth",  text:L("Operational authorisation (or STS-02 declaration) from your NAA present and valid", "Veiklos leidimas (arba STS-02 deklaracija) iš nacionalinės institucijos su savimi ir galiojantis"), warn:true,
        hint:L("BVLOS is NOT permitted in Open category (EU 2019/947 Art. 4(d)). You MUST have Specific category authorisation. For STS-02: drone ≤3 kg, ≤30 m AGL, observers positioned, remote pilot certificate.",
               "BVLOS atvirojoje kategorijoje NELEIDŽIAMAS (ES 2019/947, 4 str. d p.). PRIVALOMAS specialiosios kategorijos leidimas. STS-02: dronas ≤3 kg, ≤30 m virš žemės, išdėstyti stebėtojai, nuotolinio piloto pažymėjimas.") },
      { id:"bv-man",   text:L("Operations manual available and specific BVLOS procedures reviewed", "Veiklos vadovas pasiekiamas, BVLOS procedūros peržiūrėtos") },
      { id:"bv-obs",   text:L("BVLOS observers briefed, positioned, and in communication", "Oro erdvės stebėtojai instruktuoti, išdėstyti ir palaiko ryšį") },
      { id:"bv-cond",  text:L("All authorisation conditions confirmed met for this flight", "Patvirtinta, kad šiam skrydžiui įvykdytos visos leidimo sąlygos") },
      { id:"bv-tele",  text:L("Telemetry and video link confirmed reliable at planned BVLOS range", "Telemetrija ir vaizdo ryšys patikimi planuojamu BVLOS atstumu") },
      { id:"bv-rtl",   text:L("RTH / contingency procedure confirmed for link loss beyond VLOS range", "RTH / nenumatytų atvejų procedūra patvirtinta ryšio praradimui už VLOS ribų"),
        hint:L("Failsafe must be appropriate for BVLOS. RTH must navigate back safely without requiring visual guidance.", "Failsafe turi tikti BVLOS. RTH turi saugiai grįžti be vizualaus valdymo.") },
      { id:"bv-area",  text:L("Flight corridor confirmed clear of manned aircraft and people", "Skrydžio koridorius patvirtintas be pilotuojamų orlaivių ir žmonių") },
    ],
    postItems:[
      ...COMMON_POST,
      { id:"bv-log",   text:L("BVLOS operational log completed per authorisation requirements", "BVLOS veiklos žurnalas užpildytas pagal leidimo reikalavimus") },
    ],
  },
  {
    id:"specific-crowd", name:L("Specific: Over People", "Specialioji: virš žmonių"), badge:"CROWD", color:"purple",
    description:L("Events / concerts / clubs with drone over or near crowds · Specific category required",
                  "Renginiai / koncertai / klubai su dronu virš ar šalia minios · reikalinga specialioji kategorija"),
    cert:L("Operational authorisation from NAA · EU 2019/947 Art. 5", "TKA veiklos leidimas · ES 2019/947, 5 str."), defaultMinutes:15,
    preItems:[
      ...COMMON_PRE,
      { id:"cr-auth",  text:L("Operational authorisation from NAA present and valid — flying over assemblies requires Specific category", "Veiklos leidimas su savimi ir galiojantis — skrydžiams virš sambūrių reikalinga specialioji kategorija"), warn:true,
        hint:L("Flying over assemblies of people is prohibited in ALL Open subcategories (EU 2019/947 Art. 4(c)). There are NO exceptions. Specific category authorisation is mandatory.",
               "Skrydžiai virš žmonių sambūrių draudžiami VISOSE atvirosios kategorijos pakategorėse (ES 2019/947, 4 str. c p.). Išimčių NĖRA. Specialiosios kategorijos leidimas privalomas.") },
      { id:"cr-man",   text:L("Operations manual available, crowd operation procedures reviewed", "Veiklos vadovas pasiekiamas, skrydžių virš minios procedūros peržiūrėtos") },
      { id:"cr-org",   text:L("Event organiser coordination confirmed — they know and consent to the operation", "Suderinta su renginio organizatoriumi — jis žino ir sutinka") },
      { id:"cr-safe",  text:L("Ground safety measures in place: crowd barriers, exclusion zones, marshals", "Antžeminės saugos priemonės: užtvarai, draudžiamos zonos, tvarkdariai") },
      { id:"cr-em",    text:L("Emergency procedures briefed to all crew including crowd evacuation plan", "Avarinės procedūros išaiškintos visai komandai, įskaitant minios evakuacijos planą") },
      { id:"cr-ins",   text:L("Adequate insurance confirmed for crowd operations — verify policy covers this scenario", "Pakankamas draudimas skrydžiams virš minios — patikrinkite, ar polisas dengia šį scenarijų"),
        hint:L("Standard recreational or basic commercial policies often exclude operations over crowds. Verify explicitly.", "Standartiniai mėgėjiški ar baziniai komerciniai polisai dažnai neapima skrydžių virš minios. Pasitikrinkite aiškiai.") },
      { id:"cr-parac", text:L("Drone equipped with parachute/safety system per authorisation requirements (if required)", "Dronas su parašiutu / saugos sistema pagal leidimo reikalavimus (jei reikalaujama)") },
      { id:"cr-auth2", text:L("All authorisation conditions confirmed met", "Patvirtinta, kad įvykdytos visos leidimo sąlygos") },
    ],
    postItems:[
      ...COMMON_POST,
      { id:"cr-log",   text:L("Operational log completed per authorisation requirements", "Veiklos žurnalas užpildytas pagal leidimo reikalavimus") },
    ],
  },
  {
    id:"fpv-indoor", name:L("FPV Indoor", "FPV patalpose"), badge:"FPV", color:"blue",
    description:L("Indoor drone flying / freestyle / racing — aviation law generally does not apply indoors",
                  "Skraidymas patalpose / freestyle / lenktynės — aviacijos teisė patalpoms paprastai netaikoma"),
    cert:L("No EASA cert required indoors · check venue insurance and local safety rules", "Patalpose EASA pažymėjimo nereikia · pasitikrinkite vietos draudimą ir saugos taisykles"), defaultMinutes:8,
    preItems:[
      { id:"in-venue",  text:L("Venue confirmed: indoor space, not in controlled airspace", "Vieta patvirtinta: uždara patalpa, ne kontroliuojama oro erdvė"),
        hint:L("EU drone regulations (EU 2019/947) apply to airspace — outdoor operations. Indoor flight is generally outside their scope. However, check your specific country's national law as some may regulate indoor public events.",
               "ES dronų reglamentai (ES 2019/947) taikomi oro erdvei — skrydžiams lauke. Skrydžiai patalpose paprastai į jų taikymo sritį nepatenka. Vis dėlto pasitikrinkite nacionalinę teisę — kai kurios šalys reguliuoja viešus renginius patalpose.") },
      { id:"in-clear",  text:L("Flying area free of people in the flight path — or spectators behind adequate barriers", "Skrydžio trasoje nėra žmonių — arba žiūrovai už tinkamų užtvarų") },
      { id:"in-net",    text:L("Safety net or protective enclosure in place if spectators are present", "Jei yra žiūrovų — įrengtas apsauginis tinklas ar aptvaras") },
      { id:"in-ins",    text:L("Event/venue liability insurance confirmed for indoor drone operation", "Renginio / vietos civilinės atsakomybės draudimas apima dronų skrydžius patalpose") },
      { id:"in-props",  text:L("Props: no cracks, secure, correct rotation", "Sraigtai: be įtrūkimų, tvirti, teisinga sukimosi kryptis") },
      { id:"in-bat",    text:L("Battery charged, not swollen", "Akumuliatorius įkrautas, neišsipūtęs") },
      { id:"in-motors", text:L("Motors spin freely, no roughness", "Varikliai sukasi laisvai, be šiurkštumo") },
      { id:"in-bind",   text:L("RC bind confirmed, failsafe tested (motors cut on link loss for indoor — NOT RTH)", "RC susiejimas patvirtintas, failsafe išbandytas (patalpose praradus ryšį — variklių išjungimas, NE RTH)") },
      { id:"in-arm",    text:L("Arming sequence known — do NOT arm pointing at people", "Aktyvavimo seka žinoma — NEAKTYVUOKITE nukreipę į žmones") },
      { id:"in-buzz",   text:L("Battery buzzer armed/set: land before battery cutoff causes uncontrolled descent", "Akumuliatoriaus įspėjimas nustatytas: nutūpkite, kol išsikrovimas nesukėlė nevaldomo kritimo") },
    ],
    postItems:[
      { id:"in-disarm", text:L("Drone disarmed, props stopped", "Dronas išjungtas, sraigtai sustojo") },
      { id:"in-bat2",   text:L("Battery disconnected, check temperature", "Akumuliatorius atjungtas, patikrinta temperatūra") },
      { id:"in-ins2",   text:L("Props and frame inspected for damage", "Sraigtai ir rėmas patikrinti dėl pažeidimų") },
      { id:"in-media",  text:L("Memory card secured", "Atminties kortelė išsaugota") },
    ],
  },
  {
    id:"specific", name:L("Specific (General)", "Specialioji (bendra)"), badge:"STS", color:"purple",
    description:L("General Specific category — for operations not fitting Open category rules",
                  "Bendra specialioji kategorija — veiklai, kuri netelpa į atvirosios kategorijos taisykles"),
    cert:L("Operational authorisation or STS declaration from NAA", "TKA veiklos leidimas arba STS deklaracija"), defaultMinutes:20,
    preItems:[
      ...COMMON_PRE,
      { id:"sp-auth",  text:L("Operational authorisation document present and valid", "Veiklos leidimo dokumentas su savimi ir galiojantis"), warn:true,
        hint:L("Specific category is triggered when ANY Open category requirement is not met (EU 2019/947 Art. 5). BVLOS, over crowds, >120 m AGL, >25 kg all require Specific or Certified.",
               "Specialioji kategorija taikoma, kai neįvykdomas BET KURIS atvirosios kategorijos reikalavimas (ES 2019/947, 5 str.). BVLOS, virš minios, >120 m virš žemės, >25 kg — visiems reikia specialiosios arba sertifikuotosios kategorijos.") },
      { id:"sp-man",   text:L("Operations manual available and current", "Veiklos vadovas pasiekiamas ir aktualus") },
      { id:"sp-cond",  text:L("All authorisation conditions reviewed and confirmed met for THIS flight", "Visos leidimo sąlygos peržiūrėtos ir patvirtintos ŠIAM skrydžiui") },
      { id:"sp-risk",  text:L("Risk mitigations in place per authorisation", "Rizikos mažinimo priemonės įgyvendintos pagal leidimą") },
      { id:"sp-crew",  text:L("All crew briefed on the operation and emergency procedures", "Visa komanda instruktuota apie veiklą ir avarines procedūras") },
      { id:"sp-area",  text:L("Operational area controlled or secured per authorisation requirements", "Veiklos zona kontroliuojama ar apsaugota pagal leidimo reikalavimus") },
    ],
    postItems:[
      ...COMMON_POST,
      { id:"sp-log",   text:L("Detailed operational log completed per authorisation requirements", "Detalus veiklos žurnalas užpildytas pagal leidimo reikalavimus") },
    ],
  },
  {
    id:"poland", name:L("Poland / PANSA", "Lenkija / PANSA"), badge:"PL", color:"red",
    description:L("Open category in Poland · PansaUTM flight registration required in most zones",
                  "Atviroji kategorija Lenkijoje · daugumoje zonų privaloma skrydžio registracija PansaUTM"),
    cert:L("A1/A3 or A2 CoC + PansaUTM registration", "A1/A3 arba A2 CoC + PansaUTM registracija"), defaultMinutes:10,
    preItems:[
      ...COMMON_PRE,
      { id:"pl-reg",   text:L("EU operator registration accepted in Poland (home country reg is valid in EU)", "ES naudotojo registracija galioja Lenkijoje (gimtosios šalies registracija galioja visoje ES)"),
        hint:L("Poland accepts EU UAS operator registration. No separate Polish registration needed for EU operators.", "Lenkija pripažįsta ES UAS naudotojo registraciją. ES naudotojams atskira lenkiška registracija nereikalinga.") },
      { id:"pl-zone",  text:L("DRA zone type confirmed at this location — DRA-P (prohibited), DRA-R (restricted) or DRA-free", "Patvirtintas DRA zonos tipas šioje vietoje — DRA-P (draudžiama), DRA-R (ribojama) arba be DRA"),
        hint:L("Poland uses DRA (Drone Restricted Area) zones. Even Open category operations may need coordination in DRA-R zones near airports and cities.", "Lenkija naudoja DRA (Drone Restricted Area) zonas. Net atvirosios kategorijos skrydžiams DRA-R zonose prie oro uostų ir miestų gali reikėti suderinimo."),
        link:"https://pansa.pl", linkLabel:L("Check PANSA zones", "Tikrinti PANSA zonas") },
      { id:"pl-utm",   text:L("PansaUTM: flight plan created, submitted, and approved", "PansaUTM: skrydžio planas sukurtas, pateiktas ir patvirtintas"), warn:true,
        link:"https://pansa.pl", linkLabel:L("PansaUTM", "PansaUTM"),
        hint:L("Required for many zones. Create the plan in PansaUTM (via pansa.pl), submit it, and receive approval before flying. Keep the plan ID.", "Privaloma daugelyje zonų. Sukurkite planą PansaUTM (per pansa.pl), pateikite ir gaukite patvirtinimą prieš skrydį. Išsaugokite plano ID.") },
      { id:"pl-win",   text:L("PansaUTM: flight time window currently active (not too early / not expired)", "PansaUTM: skrydžio laiko langas šiuo metu aktyvus (ne per anksti / nepasibaigęs)") },
      { id:"pl-id",    text:L("PansaUTM plan ID / reference noted for inspection purposes", "PansaUTM plano ID / nuoroda užsirašyta patikrinimui") },
    ],
    postItems:[
      ...COMMON_POST,
      { id:"pl-close", text:L("PansaUTM: flight marked as completed / closed in the system", "PansaUTM: skrydis sistemoje pažymėtas kaip užbaigtas / uždarytas"), warn:true,
        link:"https://pansa.pl", linkLabel:L("Close in PansaUTM", "Uždaryti PansaUTM"),
        hint:L("Failing to close a PansaUTM flight can block future approvals. Always close it immediately after landing.", "Neuždarytas PansaUTM skrydis gali blokuoti būsimus patvirtinimus. Visada uždarykite iškart po nutūpimo.") },
    ],
  },
];

// ── reference data ─────────────────────────────────────────────────────────────
// Source: EU 2019/945 Annex Parts 1-4 + EU 2019/947 Annex Part A
const CLASS_TABLE: { cls: LS; mtom: string; remoteId: LS; subcats: string; note: LS }[] = [
  { cls:L("C0","C0"), mtom:"< 250 g", remoteId:L("NOT required","NEprivalomas"), subcats:"A1, A3", note:L("No cert needed if no camera. Private build <250g: same.", "Be kameros pažymėjimo nereikia. Savadarbis <250 g: tas pats.") },
  { cls:L("C1","C1"), mtom:"< 900 g", remoteId:L("Required (product req.)","Privalomas (gaminio reikalavimas)"), subcats:"A1, A3", note:L("A1/A3 cert. RID must be active in A1.", "A1/A3 pažymėjimas. A1 pakategorėje RID turi būti aktyvus.") },
  { cls:L("C2","C2"), mtom:"< 4 kg",  remoteId:L("Required (product req.)","Privalomas (gaminio reikalavimas)"), subcats:"A2",     note:L("A2 CoC required. Low-speed mode (≤3 m/s).", "Reikalingas A2 CoC. Mažo greičio režimas (≤3 m/s).") },
  { cls:L("C3","C3"), mtom:"< 25 kg", remoteId:L("Required (product req.)","Privalomas (gaminio reikalavimas)"), subcats:"A3",     note:L("A1/A3 cert. 150 m from built-up areas.", "A1/A3 pažymėjimas. 150 m nuo užstatytų teritorijų.") },
  { cls:L("C4","C4"), mtom:"< 25 kg", remoteId:L("Only if NAA requests","Tik jei reikalauja TKA"),    subcats:"A3",     note:L("No autonomous modes allowed. A1/A3 cert.", "Autonominiai režimai neleidžiami. A1/A3 pažymėjimas.") },
  { cls:L("Private build","Savadarbis"), mtom:L("< 25 kg for Open","< 25 kg atvirojoje").en, remoteId:L("NOT required","NEprivalomas"), subcats:L("A3 only","tik A3").en, note:L("No CE class req. applies. A3 only. No Remote ID.", "CE klasės reikalavimai netaikomi. Tik A3. Be nuotolinio identifikavimo.") },
];

const CAP_MATRIX: { op: LS; none: LS; a1a3: LS; a2: LS }[] = [
  { op:L("C0 (<250g, no camera)","C0 (<250 g, be kameros)"),     none:L("A1/A3 — no cert needed","A1/A3 — pažymėjimo nereikia"), a1a3:L("✓ A1/A3","✓ A1/A3"), a2:L("✓ A1/A3","✓ A1/A3") },
  { op:L("C0 with camera","C0 su kamera"),                        none:L("⚠ register first","⚠ pirma registruokitės"),       a1a3:L("✓ A1/A3","✓ A1/A3"), a2:L("✓ A1/A3","✓ A1/A3") },
  { op:L("C1 (<900g, CE-marked)","C1 (<900 g, CE žyma)"),         none:L("✗","✗"),                      a1a3:L("✓ A1/A3","✓ A1/A3"), a2:L("✓ A1/A3","✓ A1/A3") },
  { op:L("C2 (<4kg) — A2 sub.","C2 (<4 kg) — A2 pakat."),         none:L("✗","✗"),                      a1a3:L("✗ A3 only","✗ tik A3"), a2:L("✓ A2","✓ A2") },
  { op:L("C3/C4 (<25kg)","C3/C4 (<25 kg)"),                       none:L("✗","✗"),                      a1a3:L("✓ A3 only","✓ tik A3"), a2:L("✓ A3 only","✓ tik A3") },
  { op:L("Private build (any size)","Savadarbis (bet kokio dydžio)"), none:L("✗","✗"),                  a1a3:L("✓ A3 only","✓ tik A3"), a2:L("✓ A3 only","✓ tik A3") },
  { op:L("Night flying","Skrydis naktį"),                         none:L("✗","✗"),                      a1a3:L("✓ + green flashing light","✓ + žalia mirksinti šviesa"), a2:L("✓ + green flashing light","✓ + žalia mirksinti šviesa") },
  { op:L("FPV (goggles)","FPV (akiniai)"),                        none:L("✗","✗"),                      a1a3:L("✓ + trained VLOS observer","✓ + apmokytas VLOS stebėtojas"), a2:L("✓ + trained VLOS observer","✓ + apmokytas VLOS stebėtojas") },
  { op:L("BVLOS","BVLOS"),                                        none:L("✗","✗"),                      a1a3:L("✗","✗"), a2:L("✗ — Specific cat. only","✗ — tik specialioji kat.") },
  { op:L("Over assemblies of people","Virš žmonių sambūrių"),     none:L("✗","✗"),                      a1a3:L("✗","✗"), a2:L("✗ — Specific cat. only","✗ — tik specialioji kat.") },
];

const SPECIFIC_ALLOWS: { icon: string; item: LS; note: LS }[] = [
  { icon:"✓", item:L("BVLOS (Beyond Visual Line of Sight)","BVLOS (už tiesioginio matomumo ribų)"), note:L("EU 2019/947 Art. 5 — triggered by non-compliance with VLOS rule in Art. 4(d)","ES 2019/947, 5 str. — taikoma, kai nesilaikoma VLOS taisyklės (4 str. d p.)") },
  { icon:"✓", item:L("Flying over assemblies of people","Skrydžiai virš žmonių sambūrių"), note:L("With adequate risk mitigation and authorisation","Su tinkamu rizikos mažinimu ir leidimu") },
  { icon:"✓", item:L("Operations above 120 m AGL","Skrydžiai aukščiau 120 m virš žemės"), note:L("If justified by risk assessment and authorisation","Jei pagrįsta rizikos vertinimu ir leidimu") },
  { icon:"✓", item:L("Drones ≥ 25 kg (up to ~150 kg)","Dronai ≥ 25 kg (iki ~150 kg)"), note:L("Above 25 kg is outside Open — needs Specific or Certified","Virš 25 kg — ne atviroji kategorija; reikia specialiosios arba sertifikuotosios") },
  { icon:"✓", item:L("Operations in congested areas not possible in A3","Skrydžiai tankiai apgyvendintose vietovėse, negalimi A3"), note:L("Urban surveys, infrastructure inspection","Miesto tyrimai, infrastruktūros apžiūra") },
  { icon:"✓", item:L("Night operations beyond Open category limits","Nakties skrydžiai už atvirosios kategorijos ribų"), note:L("E.g. without mandatory light","Pvz., be privalomos šviesos") },
  { icon:"✓", item:L("Standard Scenario declarations (STS-01 / STS-02)","Standartinių scenarijų deklaracijos (STS-01 / STS-02)"), note:L("STS-01: VLOS populated area. STS-02: BVLOS with observers. Both ≤3 kg, ≤30 m AGL. Self-declaration, no individual NAA authorisation.","STS-01: VLOS apgyvendintoje vietovėje. STS-02: BVLOS su stebėtojais. Abu ≤3 kg, ≤30 m virš žemės. Deklaracija, be individualaus TKA leidimo.") },
];
const SPECIFIC_NOT_ALLOWS: { icon: string; item: LS; note: LS }[] = [
  { icon:"✗", item:L("Flying without operational authorisation or STS declaration","Skrydis be veiklos leidimo ar STS deklaracijos"), note:L("Every Specific operation needs either a full authorisation (Art. 12) or STS declaration (Art. 5(5))","Kiekvienai specialiosios kategorijos veiklai reikia leidimo (12 str.) arba STS deklaracijos (5 str. 5 d.)") },
  { icon:"✗", item:L("Transport of people / dangerous goods over crowds","Žmonių / pavojingų krovinių vežimas virš minios"), note:L("That triggers Certified category (EU 2019/947 Art. 6)","Tam taikoma sertifikuotoji kategorija (ES 2019/947, 6 str.)") },
  { icon:"✗", item:L("Operations where UAS requires type certification (Certified cat.)","Veikla, kuriai UAS reikia tipo sertifikato (sertifikuotoji kat.)"), note:L("e.g. aircraft-category UAS operated commercially at scale","pvz., orlaivio kategorijos UAS komerciniu mastu") },
  { icon:"✗", item:L("Ignoring national additional rules","Nacionalinių papildomų taisyklių ignoravimas"), note:L("Member states may add national restrictions on top of EU rules","Valstybės narės gali nustatyti papildomus apribojimus prie ES taisyklių") },
];

// Country resource links
const COUNTRY_LINKS: {country:LS, flag:string, authority:string, dronePortal:string, dronePortalUrl:string, notamUrl:string, notes:LS}[] = [
  { country:L("Lithuania","Lietuva"), flag:"🇱🇹", authority:"LTSA + TKA (Transporto kompetencijų agentūra)", dronePortal:"utm.ans.lt", dronePortalUrl:"https://utm.ans.lt", notamUrl:"https://www.ans.lt", notes:L("UAS geographical zones: Oro navigacija map utm.ans.lt. Operator registration and pilot certificates: sertifikatai.tka.lt. NOTAM/AIS: ans.lt.","UAS geografinės zonos: Oro navigacijos žemėlapis utm.ans.lt. Naudotojo registracija ir pilotų pažymėjimai: sertifikatai.tka.lt. NOTAM/AIS: ans.lt.") },
  { country:L("Poland","Lenkija"), flag:"🇵🇱", authority:"PANSA (Polska Agencja Żeglugi Powietrznej)", dronePortal:"pansa.pl", dronePortalUrl:"https://pansa.pl", notamUrl:"https://aim.pansa.pl/notamdrones/", notes:L("PansaUTM required in many zones. DRA zones (P=prohibited, R=restricted). DroneRadar app.","PansaUTM privaloma daugelyje zonų. DRA zonos (P = draudžiama, R = ribojama). DroneRadar programėlė.") },
  { country:L("Latvia","Latvija"), flag:"🇱🇻", authority:"CAA Latvia (Civilās aviācijas aģentūra)", dronePortal:"caa.lv/en/uas", dronePortalUrl:"https://www.caa.lv/en/uas", notamUrl:"https://www.caa.lv/en/uas", notes:L("Registration via CAA Latvia. Latvian UAS zones on drone map.","Registracija per CAA Latvia. Latvijos UAS zonos dronų žemėlapyje.") },
  { country:L("Estonia","Estija"), flag:"🇪🇪", authority:"ECAA (Lennuamet)", dronePortal:"ecaa.ee/en/transport/aviation/unmanned-aircraft", dronePortalUrl:"https://ecaa.ee/en/transport/aviation/unmanned-aircraft", notamUrl:"https://notam.ecaa.ee", notes:L("Register on ECAA portal. Drone map via ecaa.ee. Tallinn CTR requires coordination.","Registracija ECAA portale. Dronų žemėlapis per ecaa.ee. Talino CTR reikia suderinimo.") },
  { country:L("Sweden","Švedija"), flag:"🇸🇪", authority:"Transportstyrelsen + LFV", dronePortal:"aro.lfv.se", dronePortalUrl:"https://aro.lfv.se", notamUrl:"https://aro.lfv.se/bin/aro/public/frameset.jsp?id=notam", notes:L("'Open' category ops: check Swedish national zones. CTR around Stockholm/Gothenburg/Malmö. LFV handles airspace.","Atvirosios kategorijos skrydžiai: tikrinkite Švedijos nacionalines zonas. CTR aplink Stokholmą / Geteborgą / Malmę. Oro erdvę tvarko LFV.") },
  { country:L("Norway","Norvegija"), flag:"🇳🇴", authority:"Luftfartstilsynet (CAA Norway)", dronePortal:"luftfartstilsynet.no/en/drones", dronePortalUrl:"https://luftfartstilsynet.no/en/drones/", notamUrl:"https://avinor.no/en/airport/oslo-airport/information-for-pilots/notam", notes:L("Norway is EEA — EU rules apply. Register at CAA Norway. Many national parks have extra restrictions. Fjord/mountain areas: orographic turbulence risk.","Norvegija — EEE, galioja ES taisyklės. Registracija CAA Norway. Daug nacionalinių parkų su papildomais apribojimais. Fjordai / kalnai: orografinės turbulencijos rizika.") },
  { country:L("Germany","Vokietija"), flag:"🇩🇪", authority:"LBA (Luftfahrt-Bundesamt)", dronePortal:"dipul.de", dronePortalUrl:"https://www.dipul.de", notamUrl:"https://secureskytech.com/germany-drone-flying/", notes:L("DIPUL is the official German digital platform for UAS (UAS.bund.de / dipul.de). Many populated areas and nature reserves have national additional restrictions. Pilot certificate (EU A1/A3) required even for C0 above 250g.","DIPUL — oficiali Vokietijos UAS platforma (dipul.de). Daug apgyvendintų vietovių ir draustinių su papildomais nacionaliniais apribojimais.") },
  { country:L("Austria","Austrija"), flag:"🇦🇹", authority:"Austro Control", dronePortal:"dronespace.at", dronePortalUrl:"https://www.dronespace.at", notamUrl:"https://www.austrocontrol.at/en/austrocontrol/services/notam", notes:L("Dronespace.at is Austro Control's official drone portal. Many national parks and Alpine areas have additional restrictions. Vienna CTR is large and complex.","Dronespace.at — oficialus Austro Control dronų portalas. Daug nacionalinių parkų ir Alpių vietovių su papildomais apribojimais. Vienos CTR didelis ir sudėtingas.") },
  { country:L("Switzerland","Šveicarija"), flag:"🇨🇭", authority:"FOCA (BAZL / Bundesamt für Zivilluftfahrt)", dronePortal:"bazl.admin.ch/en/aircraft/drones", dronePortalUrl:"https://www.bazl.admin.ch/en/aircraft/drones.html", notamUrl:"https://www.skybriefing.com/o/notam", notes:L("Switzerland is not EU — has its own rules but largely aligned. Drone map at map.geo.admin.ch. FOCA registration required. Many national parks restricted. Mountain flying: altitude, density altitude, turbulence risks.","Šveicarija — ne ES, taisyklės savos, bet daugiausia suderintos. Dronų žemėlapis map.geo.admin.ch. Privaloma FOCA registracija. Daug ribojamų nacionalinių parkų. Kalnuose: aukščio, tankio aukščio ir turbulencijos rizika.") },
  { country:L("Portugal","Portugalija"), flag:"🇵🇹", authority:"ANAC Portugal", dronePortal:"anac.pt/vPT/Generico/uav", dronePortalUrl:"https://www.anac.pt/vPT/Generico/uav/Paginas/UAS.aspx", notamUrl:"https://www.nav.pt/en-us/aeronautical-information/notam", notes:L("ANAC is the Portuguese CAA. Algarve coastal areas popular — check local restrictions. Historical sites often have additional no-fly zones.","ANAC — Portugalijos aviacijos institucija. Algarvės pakrantė populiari — tikrinkite vietinius apribojimus. Istorinėse vietose dažnai papildomos neskraidymo zonos.") },
];

// ── UI strings ────────────────────────────────────────────────────────────────
const T = {
  fly:            L("Fly","Skrydis"),
  flySub:         L("Pre/post-flight checklists with timer. Open A1/A2/A3, BVLOS, over people, FPV indoor, Poland/PANSA. Timer persists across browser close.",
                    "Kontroliniai sąrašai prieš ir po skrydžio su laikmačiu. Atviroji A1/A2/A3, BVLOS, virš žmonių, FPV patalpose, Lenkija/PANSA. Laikmatis išlieka uždarius naršyklę."),
  inProgress:     L("✈ Flight in progress:","✈ Vyksta skrydis:"),
  timeLeft:       L("left","liko"),
  timeExpired:    L("time expired","laikas baigėsi"),
  openCat:        L("Open Category","Atviroji kategorija"),
  specificCat:    L("Specific Category & Special Scenarios","Specialioji kategorija ir ypatingi scenarijai"),
  reference:      L("Reference","Žinynas"),
  quickRef:       L("Quick Reference","Trumpas žinynas"),
  quickRefBlurb:  L("C0–C4 + private build class table, license matrix, A1/A2/A3 distance rules, Specific category explainer, country portals for LT/PL/LV/EE/SE/NO/DE/AT/CH/PT.",
                    "C0–C4 ir savadarbių klasių lentelė, pažymėjimų matrica, A1/A2/A3 atstumų taisyklės, specialiosios kategorijos paaiškinimas, šalių portalai LT/PL/LV/EE/SE/NO/DE/AT/CH/PT."),
  classTable:     L("Class table","Klasių lentelė"),
  specificShort:  L("Specific cat.","Specialioji kat."),
  countryLinks:   L("Country links","Šalių nuorodos"),
  open:           L("Open →","Atidaryti →"),
  openChecklist:  L("Open checklist →","Atidaryti sąrašą →"),
  pre:            L("pre","prieš"),
  post:           L("post","po"),
  notStarted:     L("Not started","Nepradėta"),
  flying:         L("✈ Flying","✈ Skrenda"),
  postFlight:     L("Post-flight","Po skrydžio"),
  done:           L("✓ Done","✓ Baigta"),
  back:           L("← Back","← Atgal"),
  cert:           L("Cert:","Pažymėjimas:"),
  stepPre:        L("Pre-flight","Prieš skrydį"),
  stepFlying:     L("Flying","Skrydis"),
  stepPost:       L("Post-flight","Po skrydžio"),
  stepDone:       L("Done","Baigta"),
  locationLabel:  L("Location / notes","Vieta / pastabos"),
  locationPh:     L("e.g. Klaipėda beach, north end","pvz., Klaipėdos paplūdimys, šiaurinis galas"),
  cleared:        L("✓ CLEARED FOR TAKEOFF","✓ LEIDŽIAMA KILTI"),
  plannedTime:    L("Planned flight time","Planuojama skrydžio trukmė"),
  startTimer:     L("Start flight timer →","Paleisti skrydžio laikmatį →"),
  remainingItems: L("items remaining before takeoff","punktai iki kilimo"),
  remaining:      L("REMAINING","LIKO"),
  elapsed:        L("Elapsed:","Praėjo:"),
  expiredLand:    L("⚠ Planned time expired — land now","⚠ Planuotas laikas baigėsi — tūpkite dabar"),
  landBtn:        L("Land — go to post-flight checklist →","Nutūpiau — į sąrašą po skrydžio →"),
  markComplete:   L("Mark flight complete ✓","Pažymėti skrydį baigtu ✓"),
  flightComplete: L("Flight complete","Skrydis baigtas"),
  flightTime:     L("Flight time:","Skrydžio trukmė:"),
  location:       L("Location:","Vieta:"),
  newFlight:      L("New flight","Naujas skrydis"),
  reset:          L("↺ Reset & start over","↺ Atstatyti ir pradėti iš naujo"),
  refSource:      L("Source: EU 2019/945 + EU 2019/947 (fetched from EUR-Lex)","Šaltinis: ES 2019/945 + ES 2019/947 (EUR-Lex)"),
  secClass:       L("Drone class requirements (C0–C4 + private build)","Dronų klasių reikalavimai (C0–C4 + savadarbiai)"),
  thClass:        L("Class","Klasė"),
  thMtom:         L("MTOM","MTOM"),
  thRid:          L("Remote ID","Nuotolinis ID"),
  thSubcats:      L("Subcategories","Pakategorės"),
  thNotes:        L("Notes","Pastabos"),
  ridNote1:       L("Remote ID is a ","Nuotolinis identifikavimas yra "),
  ridNoteBold:    L("product requirement","gaminio reikalavimas"),
  ridNote2:       L(" for CE-class-marked drones (EU 2019/945). Privately built drones are not C-class and do not require Remote ID. C1 Remote ID must be \"active and updated\" when flying in A1 (EU 2019/947 UAS.OPEN.020(5)(d)).",
                    " CE klasės dronams (ES 2019/945). Savadarbiai dronai nėra C klasės ir nuotolinio identifikavimo neprivalo turėti. C1 nuotolinis ID turi būti „aktyvus ir atnaujintas“ skrendant A1 (ES 2019/947 UAS.OPEN.020(5)(d))."),
  secMatrix:      L("License × capability matrix","Pažymėjimų ir galimybių matrica"),
  thOp:           L("Operation","Veikla"),
  thNoCert:       L("No cert","Be pažymėjimo"),
  thA1A3:         L("A1/A3 cert","A1/A3 pažymėjimas"),
  thA2:           L("A2 CoC","A2 CoC"),
  secSpecific:    L("Specific category — what it allows and doesn't","Specialioji kategorija — kas leidžiama ir kas ne"),
  triggeredBy:    L("Triggered by:","Taikoma, kai:"),
  triggeredText:  L("Any violation of Open category requirements (EU 2019/947 Art. 5(1)). The key Open rules that push you to Specific: VLOS (→ BVLOS), no-assemblies (→ over crowds), 120 m AGL, <25 kg MTOM.",
                    "pažeidžiamas bet kuris atvirosios kategorijos reikalavimas (ES 2019/947, 5 str. 1 d.). Pagrindinės taisyklės, vedančios į specialiąją: VLOS (→ BVLOS), be sambūrių (→ virš minios), 120 m virš žemės, <25 kg MTOM."),
  stsHead:        L("Standard Scenarios (STS) — no full NAA authorisation needed:","Standartiniai scenarijai (STS) — pilno TKA leidimo nereikia:"),
  sts01:          L("STS-01: VLOS over controlled ground area in populated environment (drone ≤3 kg, ≤30 m AGL).","STS-01: VLOS virš kontroliuojamos žemės teritorijos apgyvendintoje aplinkoje (dronas ≤3 kg, ≤30 m virš žemės)."),
  sts02:          L("STS-02: BVLOS with airspace observers in sparsely populated area (drone ≤3 kg, ≤30 m AGL).","STS-02: BVLOS su oro erdvės stebėtojais retai apgyvendintoje vietovėje (dronas ≤3 kg, ≤30 m virš žemės)."),
  stsBoth:        L("Both require: self-declaration to NAA, remote pilot certificate, ops manual.","Abiem reikia: deklaracijos TKA, nuotolinio piloto pažymėjimo, veiklos vadovo."),
  allows:         L("Specific category allows:","Specialioji kategorija leidžia:"),
  notAllows:      L("Specific category does NOT allow:","Specialioji kategorija NELEIDŽIA:"),
  secDist:        L("Distance rules at a glance","Atstumų taisyklės trumpai"),
  secCountries:   L("Country drone portals & apps","Šalių dronų portalai ir programėlės"),
  notams:         L("NOTAMs","NOTAM"),
  panEU:          L("Pan-European apps","Europinės programėlės"),
  verifyOfficial: L("Always verify with the official national authority source — apps may lag official changes.","Visada pasitikrinkite oficialiame nacionalinės institucijos šaltinyje — programėlės gali atsilikti nuo oficialių pakeitimų."),
};

const DIST_CARDS: { cat: string; drone: LS; color: string; rules: LS[] }[] = [
  { cat:"A1", drone:L("C0/C1","C0/C1"), color:"blue", rules:[L("Not over assemblies of people","Ne virš žmonių sambūrių"),L("C0: may overfly isolated individuals","C0: gali praskristi virš pavienių asmenų"),L("C1: minimise time over individuals","C1: kuo trumpiau virš asmenų"),L("120 m max AGL","Ne aukščiau 120 m virš žemės")] },
  { cat:"A2", drone:L("C2 only","tik C2"), color:"green", rules:[L("50 m min from uninvolved persons","Ne mažiau 50 m nuo nesusijusių asmenų"),L("30 m with low-speed mode (≤3 m/s)","30 m su mažo greičio režimu (≤3 m/s)"),L("Not over any people","Ne virš žmonių"),L("120 m max AGL","Ne aukščiau 120 m virš žemės")] },
  { cat:"A3", drone:L("Any / private","Bet kuris / savadarbis"), color:"orange", rules:[L("150 m from residential/commercial/","150 m nuo gyvenamųjų / komercinių /"),L("recreational/industrial areas","poilsio / pramoninių teritorijų"),L("No uninvolved persons in area","Zonoje nėra nesusijusių asmenų"),L("120 m max AGL","Ne aukščiau 120 m virš žemės")] },
];

// ── localStorage ──────────────────────────────────────────────────────────────
const fsKey = (id: string) => `fly_state_${id}`;
function loadState(id: string): FlightState {
  try { const r = localStorage.getItem(fsKey(id)); if (r) return JSON.parse(r); } catch {}
  return { phase:"pre", checkedPre:[], checkedPost:[] };
}
function saveState(id: string, s: FlightState) {
  try { localStorage.setItem(fsKey(id), JSON.stringify(s)); } catch {}
}

function useNow(active: boolean) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}
function fmtMs(ms: number) {
  if (ms < 0) ms = 0;
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s/60)).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;
}

// ── ChecklistView ─────────────────────────────────────────────────────────────
function ChecklistView({ def, onBack }: { def: ChecklistDef; onBack: () => void }) {
  const { t } = useLang();
  const [state, setState] = useState<FlightState>(() => loadState(def.id));
  const [planMin, setPlanMin] = useState(def.defaultMinutes);
  const [location, setLocation] = useState(state.location ?? "");
  const [hints, setHints] = useState<Set<string>>(new Set());
  const now = useNow(state.phase === "flying");

  const save = useCallback((s: FlightState) => { setState(s); saveState(def.id, s); }, [def.id]);
  const togglePre = (id: string) => save({ ...state, checkedPre: state.checkedPre.includes(id) ? state.checkedPre.filter(x=>x!==id) : [...state.checkedPre, id] });
  const togglePost = (id: string) => save({ ...state, checkedPost: state.checkedPost.includes(id) ? state.checkedPost.filter(x=>x!==id) : [...state.checkedPost, id] });
  const toggleHint = (id: string) => setHints(p => { const n=new Set(p); n.has(id)?n.delete(id):n.add(id); return n; });

  const allPreDone = def.preItems.every(it => state.checkedPre.includes(it.id));
  const allPostDone = def.postItems.every(it => state.checkedPost.includes(it.id));
  const preCount = def.preItems.filter(it => state.checkedPre.includes(it.id)).length;
  const postCount = def.postItems.filter(it => state.checkedPost.includes(it.id)).length;

  const elapsed = state.flightStart ? now - state.flightStart : 0;
  const remaining = state.plannedEnd ? state.plannedEnd - now : 0;
  const timerPct = state.plannedEnd && state.flightStart ? Math.max(0,Math.min(100,(remaining/(state.plannedEnd-state.flightStart))*100)) : 100;
  const timerColor = timerPct > 30 ? "green" : timerPct > 10 ? "orange" : "red";

  const phases: Array<"pre"|"flying"|"post"|"done"> = ["pre","flying","post","done"];
  const phaseIdx = phases.indexOf(state.phase);

  function ItemList({ items, checked, onToggle }: { items: CheckItem[], checked: string[], onToggle: (id:string)=>void }) {
    return (
      <ul className="fly-items">
        {items.map(item => {
          const isChecked = checked.includes(item.id);
          const expanded = hints.has(item.id);
          return (
            <li key={item.id} className={`fly-item ${isChecked?"checked":""} ${item.warn?"warn":""}`}>
              <button className="fly-check" onClick={() => onToggle(item.id)}>
                <span className="fly-checkbox">{isChecked?"✓":""}</span>
                <span className="fly-item-text">{t(item.text)}</span>
              </button>
              {(item.hint||item.link) && <button className="fly-hint-btn" onClick={()=>toggleHint(item.id)}>ⓘ</button>}
              {expanded && (
                <div className="fly-hint-body">
                  {item.hint && <p>{t(item.hint)}</p>}
                  {item.link && <a href={item.link} target="_blank" rel="noopener">{item.linkLabel ? t(item.linkLabel) : item.link}</a>}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    );
  }

  return (
    <div className="fly-view">
      <div className="fly-hdr">
        <button className="fc-back" onClick={onBack}>{t(T.back)}</button>
        <span className={`fly-badge fly-badge--${def.color}`}>{def.badge}</span>
        <h1>{t(def.name)}</h1>
        <p className="fly-desc">{t(def.description)}</p>
        <p className="fly-cert">{t(T.cert)} <strong>{t(def.cert)}</strong></p>
      </div>
      <div className="fly-stepper">
        {[T.stepPre,T.stepFlying,T.stepPost,T.stepDone].map((label, i) => (
          <div key={i} className={`fly-step ${i<phaseIdx?"done":i===phaseIdx?"active":"future"}`}>
            <span>{i<phaseIdx?"✓":i+1}</span><small>{t(label)}</small>
          </div>
        ))}
      </div>

      {state.phase === "pre" && (
        <div className="fly-phase">
          <div className="fly-prog-row">
            <span>{preCount}/{def.preItems.length}</span>
            <div className="fly-prog-bar"><div className="fly-prog-fill" style={{width:`${def.preItems.length?Math.round(preCount/def.preItems.length*100):0}%`}}/></div>
            <span>{def.preItems.length?Math.round(preCount/def.preItems.length*100):0}%</span>
          </div>
          <div className="fly-loc-row">
            <label>{t(T.locationLabel)}</label>
            <input type="text" value={location} placeholder={t(T.locationPh)} onChange={e=>setLocation(e.target.value)}/>
          </div>
          <ItemList items={def.preItems} checked={state.checkedPre} onToggle={togglePre}/>
          {allPreDone ? (
            <div className="fly-cleared">
              <div className="fly-cleared-banner">{t(T.cleared)}</div>
              <div className="fly-plan-row">
                <label>{t(T.plannedTime)}</label>
                <div className="fly-plan-btns">
                  {[5,8,10,12,15,20,30].map(m => (
                    <button key={m} className={planMin===m?"on":""} onClick={()=>setPlanMin(m)}>{m} min</button>
                  ))}
                </div>
              </div>
              <button className="fly-start-btn" onClick={()=>{
                const start=Date.now();
                save({...state, phase:"flying", flightStart:start, plannedEnd:start+planMin*60_000, location});
              }}>{t(T.startTimer)}</button>
            </div>
          ) : (
            <p className="fly-progress-note">{def.preItems.length-preCount} {t(T.remainingItems)}</p>
          )}
        </div>
      )}

      {state.phase === "flying" && (
        <div className="fly-phase fly-phase--flying">
          <div className="fly-timer-display" data-color={timerColor}>
            <div className="fly-timer-label">{t(T.remaining)}</div>
            <div className="fly-timer-big">{fmtMs(remaining)}</div>
            <div className="fly-timer-elapsed">{t(T.elapsed)} {fmtMs(elapsed)}</div>
            <div className="fly-timer-bar-wrap">
              <div className="fly-timer-bar-fill" style={{width:`${timerPct}%`,background:timerColor==="green"?"#3fb950":timerColor==="orange"?"#f0a020":"#f85149"}}/>
            </div>
          </div>
          {state.location && <div className="fly-loc-display">📍 {state.location}</div>}
          {remaining <= 0 && <div className="fly-timer-expired">{t(T.expiredLand)}</div>}
          <button className="fly-end-btn" onClick={()=>save({...state,phase:"post"})}>
            {t(T.landBtn)}
          </button>
        </div>
      )}

      {state.phase === "post" && (
        <div className="fly-phase">
          <div className="fly-prog-row">
            <span>{postCount}/{def.postItems.length}</span>
            <div className="fly-prog-bar"><div className="fly-prog-fill" style={{width:`${def.postItems.length?Math.round(postCount/def.postItems.length*100):0}%`}}/></div>
          </div>
          <ItemList items={def.postItems} checked={state.checkedPost} onToggle={togglePost}/>
          {allPostDone && (
            <button className="fly-start-btn" onClick={()=>{
              save({...state,phase:"done"});
            }}>{t(T.markComplete)}</button>
          )}
        </div>
      )}

      {state.phase === "done" && (
        <div className="fly-phase fly-done">
          <div className="fly-done-icon">✓</div>
          <h2>{t(T.flightComplete)}</h2>
          {state.flightStart && <p>{t(T.flightTime)} {fmtMs(state.flightStart?(state.plannedEnd?Math.min(now,state.plannedEnd)-state.flightStart:elapsed):0)}</p>}
          {state.location && <p>{t(T.location)} {state.location}</p>}
          <button className="fly-start-btn" onClick={()=>save({phase:"pre",checkedPre:[],checkedPost:[]})}>{t(T.newFlight)}</button>
        </div>
      )}

      {state.phase !== "done" && state.phase !== "pre" && (
        <button className="fly-reset-link" onClick={()=>save({phase:"pre",checkedPre:[],checkedPost:[]})}>{t(T.reset)}</button>
      )}
    </div>
  );
}

// ── ReferenceView ─────────────────────────────────────────────────────────────
function ReferenceView({ onBack }: { onBack: () => void }) {
  const { t, lang } = useLang();
  const [open, setOpen] = useState<string|null>("class");
  const tog = (s: string) => setOpen(p => p===s?null:s);
  const mtomLabel = (r: typeof CLASS_TABLE[number]) => r.cls.en === "Private build" ? (lang === "lt" ? "< 25 kg atvirojoje" : "< 25 kg for Open") : r.mtom;
  const subcatLabel = (r: typeof CLASS_TABLE[number]) => r.cls.en === "Private build" ? (lang === "lt" ? "tik A3" : "A3 only") : r.subcats;

  return (
    <div className="fly-view">
      <div className="fly-hdr">
        <button className="fc-back" onClick={onBack}>{t(T.back)}</button>
        <h1>{t(T.quickRef)}</h1>
        <p className="fly-desc">{t(T.refSource)}</p>
      </div>

      {/* Class table */}
      <div className="fly-ref-section">
        <button className="fly-ref-toggle" onClick={()=>tog("class")}>{open==="class"?"▼":"▶"} {t(T.secClass)}</button>
        {open==="class" && (
          <div className="fly-ref-scroll">
            <table className="fly-ref-table">
              <thead><tr><th>{t(T.thClass)}</th><th>{t(T.thMtom)}</th><th>{t(T.thRid)}</th><th>{t(T.thSubcats)}</th><th>{t(T.thNotes)}</th></tr></thead>
              <tbody>
                {CLASS_TABLE.map(r=>(
                  <tr key={r.cls.en}>
                    <td><strong>{t(r.cls)}</strong></td>
                    <td>{mtomLabel(r)}</td>
                    <td className={r.remoteId.en.includes("NOT")?"fly-yes":""}>{t(r.remoteId)}</td>
                    <td>{subcatLabel(r)}</td>
                    <td className="fly-note">{t(r.note)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="fly-ref-note">{t(T.ridNote1)}<strong>{t(T.ridNoteBold)}</strong>{t(T.ridNote2)}</p>
          </div>
        )}
      </div>

      {/* License matrix */}
      <div className="fly-ref-section">
        <button className="fly-ref-toggle" onClick={()=>tog("matrix")}>{open==="matrix"?"▼":"▶"} {t(T.secMatrix)}</button>
        {open==="matrix" && (
          <div className="fly-ref-scroll">
            <table className="fly-ref-table">
              <thead><tr><th>{t(T.thOp)}</th><th>{t(T.thNoCert)}</th><th>{t(T.thA1A3)}</th><th>{t(T.thA2)}</th></tr></thead>
              <tbody>
                {CAP_MATRIX.map(r=>{
                  const cls = (v: LS) => v.en.startsWith("✗") ? "fly-no" : (v.en.startsWith("✓") || v.en.includes("A1")) ? "fly-yes" : "";
                  return (
                    <tr key={r.op.en}>
                      <td>{t(r.op)}</td>
                      <td className={cls(r.none)}>{t(r.none)}</td>
                      <td className={cls(r.a1a3)}>{t(r.a1a3)}</td>
                      <td className={cls(r.a2)}>{t(r.a2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Specific category explainer */}
      <div className="fly-ref-section">
        <button className="fly-ref-toggle" onClick={()=>tog("specific")}>{open==="specific"?"▼":"▶"} {t(T.secSpecific)}</button>
        {open==="specific" && (
          <div className="fly-specific-body">
            <p className="fly-specific-intro">
              <strong>{t(T.triggeredBy)}</strong> {t(T.triggeredText)}
            </p>
            <p className="fly-specific-intro">
              <strong>{t(T.stsHead)}</strong><br/>
              {t(T.sts01)}<br/>
              {t(T.sts02)}<br/>
              {t(T.stsBoth)}
            </p>
            <h3>{t(T.allows)}</h3>
            <ul className="fly-specific-list">
              {SPECIFIC_ALLOWS.map((r,i) => (
                <li key={i}><span className="fly-yes">{r.icon}</span> <strong>{t(r.item)}</strong> <span className="fly-note-inline">{t(r.note)}</span></li>
              ))}
            </ul>
            <h3>{t(T.notAllows)}</h3>
            <ul className="fly-specific-list">
              {SPECIFIC_NOT_ALLOWS.map((r,i) => (
                <li key={i}><span className="fly-no">{r.icon}</span> <strong>{t(r.item)}</strong> <span className="fly-note-inline">{t(r.note)}</span></li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Distance quick ref */}
      <div className="fly-ref-section">
        <button className="fly-ref-toggle" onClick={()=>tog("dist")}>{open==="dist"?"▼":"▶"} {t(T.secDist)}</button>
        {open==="dist" && (
          <div className="fly-dist-grid">
            {DIST_CARDS.map(c=>(
              <div key={c.cat} className={`fly-dist-card fly-dist-${c.color}`}>
                <div className="fly-dist-head"><span className={`fly-badge fly-badge--${c.color}`}>{c.cat}</span><span>{t(c.drone)}</span></div>
                <ul>{c.rules.map((r,i)=><li key={i}>{t(r)}</li>)}</ul>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Country resources */}
      <div className="fly-ref-section">
        <button className="fly-ref-toggle" onClick={()=>tog("countries")}>{open==="countries"?"▼":"▶"} {t(T.secCountries)}</button>
        {open==="countries" && (
          <div className="fly-country-grid">
            {COUNTRY_LINKS.map(c=>(
              <div key={c.country.en} className="fly-country-card">
                <div className="fly-country-head">
                  <span className="fly-country-flag">{c.flag}</span>
                  <div>
                    <strong>{t(c.country)}</strong>
                    <span className="fly-country-auth">{c.authority}</span>
                  </div>
                </div>
                <div className="fly-country-links">
                  <a href={c.dronePortalUrl} target="_blank" rel="noopener">🗺 {c.dronePortal}</a>
                  {c.notamUrl && c.notamUrl !== c.dronePortalUrl && (
                    <a href={c.notamUrl} target="_blank" rel="noopener">📋 {t(T.notams)}</a>
                  )}
                </div>
                <p className="fly-country-note">{t(c.notes)}</p>
              </div>
            ))}
            <div className="fly-country-card fly-country-card--wide">
              <strong>{t(T.panEU)}</strong>
              <div className="fly-country-links">
                <a href="https://www.openaip.net" target="_blank" rel="noopener">OpenAIP</a>
                <a href="https://droneradar.eu" target="_blank" rel="noopener">DroneRadar EU</a>
                <a href="https://app.airmap.com" target="_blank" rel="noopener">AirMap</a>
              </div>
              <p className="fly-country-note">{t(T.verifyOfficial)}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── FlySection (hub) ──────────────────────────────────────────────────────────
interface Props { route?: string; onBack: () => void; navigate: (r: string) => void; }

export default function FlySection({ route="", onBack, navigate }: Props) {
  const { t } = useLang();
  const sub = route.replace(/^fly\/?/,"");
  if (sub==="reference") return <ReferenceView onBack={()=>navigate("fly")}/>;
  const def = CHECKLISTS.find(c=>c.id===sub);
  if (def) return <ChecklistView def={def} onBack={()=>navigate("fly")}/>;

  const states = CHECKLISTS.map(c=>({ id:c.id, state:loadState(c.id) }));
  const active = states.filter(s=>s.state.phase==="flying");

  const Card = ({ c }: { c: ChecklistDef }) => {
    const s=states.find(x=>x.id===c.id)!.state;
    const pre=c.preItems.filter(it=>s.checkedPre.includes(it.id)).length;
    const hasProgress=s.phase!=="pre"||pre>0;
    const phaseLabel=s.phase==="pre"?(pre>0?`${pre}/${c.preItems.length} ${t(T.pre)}`:t(T.notStarted)):s.phase==="flying"?t(T.flying):s.phase==="post"?t(T.postFlight):t(T.done);
    return (
      <li className={`hub-card hub-card--${c.color}`}>
        <button className="hub-card-btn" onClick={()=>navigate(`fly/${c.id}`)}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
            <span className={`fly-badge fly-badge--${c.color}`}>{c.badge}</span>
            {hasProgress && <span className="fly-state-chip" data-phase={s.phase}>{phaseLabel}</span>}
          </div>
          <h2>{t(c.name)}</h2><p>{t(c.description)}</p>
          <div className="hub-tags"><span>{c.preItems.length} {t(T.pre)}</span><span>{c.postItems.length} {t(T.post)}</span></div>
          <span className="hub-go">{t(T.openChecklist)}</span>
        </button>
      </li>
    );
  };

  return (
    <div className="hub">
      <header className="hub-head">
        <p className="hub-eyebrow"><button className="hub-link-btn" onClick={onBack}>{t({en:"ITOHI Tools",lt:"ITOHI įrankiai"})}</button></p>
        <h1>{t(T.fly)}</h1>
        <p className="hub-sub">{t(T.flySub)}</p>
      </header>

      {active.length > 0 && (
        <div className="fly-active-banner">
          {t(T.inProgress)}{" "}
          {active.map(f => {
            const d=CHECKLISTS.find(c=>c.id===f.id)!;
            const rem=f.state.plannedEnd?f.state.plannedEnd-Date.now():0;
            return <button key={f.id} className="fly-active-link" onClick={()=>navigate(`fly/${f.id}`)}>{t(d.name)} — {rem>0?fmtMs(rem)+" "+t(T.timeLeft):t(T.timeExpired)} →</button>;
          })}
        </div>
      )}

      <h2 className="hub-section">{t(T.openCat)}</h2>
      <ul className="hub-grid hub-grid--3">
        {CHECKLISTS.filter(c=>["a1","a2","a3"].includes(c.id)).map(c=><Card key={c.id} c={c}/>)}
      </ul>

      <h2 className="hub-section">{t(T.specificCat)}</h2>
      <ul className="hub-grid hub-grid--3">
        {CHECKLISTS.filter(c=>["specific-bvlos","specific-crowd","fpv-indoor","specific","poland"].includes(c.id)).map(c=><Card key={c.id} c={c}/>)}
      </ul>

      <h2 className="hub-section">{t(T.reference)}</h2>
      <ul className="hub-grid">
        <li className="hub-card hub-card--blue">
          <button className="hub-card-btn" onClick={()=>navigate("fly/reference")}>
            <h2>{t(T.quickRef)}</h2>
            <p>{t(T.quickRefBlurb)}</p>
            <div className="hub-tags"><span>{t(T.classTable)}</span><span>{t(T.specificShort)}</span><span>{t(T.countryLinks)}</span></div>
            <span className="hub-go">{t(T.open)}</span>
          </button>
        </li>
      </ul>
    </div>
  );
}

export type { Lang };
