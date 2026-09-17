"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { Question, QuestionL10n } from "@/types/exam";

export type Lang = "en" | "lt";
/** A localized string pair. */
export type LS = { en: string; lt: string };

const LANG_KEY = "itohi_lang";

function detectLang(): Lang {
  try {
    const saved = localStorage.getItem(LANG_KEY);
    if (saved === "en" || saved === "lt") return saved;
    if ((navigator.language || "").toLowerCase().startsWith("lt")) return "lt";
  } catch {}
  return "en";
}

const LangCtx = createContext<{ lang: Lang; setLang: (l: Lang) => void }>({ lang: "en", setLang: () => {} });

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>("en");
  useEffect(() => { setLangState(detectLang()); }, []);
  useEffect(() => { try { document.documentElement.lang = lang; } catch {} }, [lang]);
  const setLang = (l: Lang) => { setLangState(l); try { localStorage.setItem(LANG_KEY, l); } catch {} };
  return <LangCtx.Provider value={{ lang, setLang }}>{children}</LangCtx.Provider>;
}

export function useLang() {
  const { lang, setLang } = useContext(LangCtx);
  const t = (s: LS | string): string => (typeof s === "string" ? s : s[lang]);
  return { lang, setLang, t };
}

export function LangToggle() {
  const { lang, setLang } = useLang();
  return (
    <div className="lang-toggle" role="group" aria-label="Language">
      <button className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>EN</button>
      <button className={lang === "lt" ? "on" : ""} onClick={() => setLang("lt")}>LT</button>
    </div>
  );
}

/** Apply a translation map to a question bank. Unmapped questions stay in the source language. */
export function localizeQuestions(
  qs: Question[],
  lang: Lang,
  map?: Record<string, QuestionL10n>,
  categories?: Record<string, string>,
): Question[] {
  if (lang === "en") return qs;
  return qs.map(q => {
    const m = map?.[q.id];
    const category = q.category && categories?.[q.category] ? categories[q.category] : q.category;
    if (!m) return category === q.category ? q : { ...q, category };
    return { ...q, category, q: m.q, options: m.options, explanation: m.explanation ?? q.explanation };
  });
}

/* ── UI strings ──────────────────────────────────────────────────────────── */
export const S = {
  // common
  itohiTools:      { en: "ITOHI Tools", lt: "ITOHI įrankiai" },
  exams:           { en: "Exams", lt: "Egzaminai" },
  allTools:        { en: "← All tools", lt: "← Visi įrankiai" },
  back:            { en: "← Back", lt: "← Atgal" },
  open:            { en: "Open →", lt: "Atidaryti →" },
  start:           { en: "Start →", lt: "Pradėti →" },
  pass:            { en: "pass", lt: "išlaikymo riba" },
  min:             { en: "min", lt: "min" },
  q:               { en: "Q", lt: "kl." },
  // home
  homeEyebrow:     { en: "tools", lt: "įrankiai" },
  homeSub:         { en: "FPV tools and exam study from ITOHI. Everything runs in your browser — no account, no upload, data stays on your device.",
                     lt: "FPV įrankiai ir egzaminų mokymasis iš ITOHI. Viskas veikia jūsų naršyklėje — be paskyros, be įkėlimo, duomenys lieka jūsų įrenginyje." },
  toolFly:         { en: "Fly", lt: "Skrydis" },
  toolFlyBlurb:    { en: "Pre/post-flight checklists for Open A1/A2/A3, BVLOS, over people, FPV indoor, Poland/PANSA. Clearance gate + timer that survives browser close.",
                     lt: "Kontroliniai sąrašai prieš ir po skrydžio: atviroji A1/A2/A3, BVLOS, virš žmonių, FPV patalpose, Lenkija/PANSA. Leidimo vartai ir laikmatis, išliekantis uždarius naršyklę." },
  toolRx:          { en: "RX Blind-Spot Viewer", lt: "RX aklųjų zonų žiūryklė" },
  toolRxBlurb:     { en: "Load an EdgeTX telemetry CSV and visualise your control link in 3D. Antenna-pattern sphere, per-flight splitting, RTH risk.",
                     lt: "Įkelkite EdgeTX telemetrijos CSV ir pamatykite valdymo ryšį 3D. Antenos diagramos sfera, skaidymas pagal skrydžius, RTH rizika." },
  toolExams:       { en: "Exam Study", lt: "Egzaminų mokymasis" },
  toolExamsBlurb:  { en: "EASA A1/A3 (336 Q), A2 CoC (60 Q), and radio operator exams in Lithuanian and English — {n} questions total. SM-2 flashcards + timed exams.",
                     lt: "EASA A1/A3 (336 kl.), A2 CoC (60 kl.) ir radijo mėgėjų egzaminai lietuvių bei anglų kalbomis — iš viso {n} klausimų. SM-2 kortelės ir egzaminai su laikmačiu." },
  tagChecklists:   { en: "Checklists", lt: "Kontroliniai sąrašai" },
  tagTimer:        { en: "Timer", lt: "Laikmatis" },
  tagCountry:      { en: "Country links", lt: "Šalių nuorodos" },
  tagTelemetry:    { en: "telemetry", lt: "telemetrija" },
  // exams hub
  examStudy:       { en: "Exam Study", lt: "Egzaminų mokymasis" },
  examStudySub:    { en: "Drone and radio operator exam preparation. SM-2 spaced repetition flashcards + timed mock exams. Progress saved in browser.",
                     lt: "Pasiruošimas dronų ir radijo mėgėjų egzaminams. SM-2 intervalinio kartojimo kortelės ir bandomieji egzaminai su laikmačiu. Pažanga saugoma naršyklėje." },
  easaDroneExams:  { en: "EASA Drone Exams", lt: "EASA dronų egzaminai" },
  radioExams:      { en: "Radio Operator Exams", lt: "Radijo mėgėjų egzaminai" },
  a1a3CardBlurb:   { en: "EU Open category remote pilot competency. 336 questions across 9 official Lithuanian CAA categories. Reasoning questions tagged ⚡.",
                     lt: "ES atvirosios kategorijos nuotolinio piloto kompetencija. 336 klausimai 9 oficialiose TKA (CAA) kategorijose. Samprotavimo klausimai pažymėti ⚡." },
  a2CardBlurb:     { en: "A2 Certificate of Competency — fly C2 drones within 30–50 m of people. Advanced meteorology, human factors, UAS technical knowledge.",
                     lt: "A2 kompetencijos pažymėjimas — C2 klasės dronai 30–50 m nuo žmonių. Išplėstinė meteorologija, žmogiškieji veiksniai, UAS techninės žinios." },
  categories9:     { en: "9 categories", lt: "9 kategorijos" },
  reasoning:       { en: "⚡ reasoning", lt: "⚡ samprotavimas" },
  a2specific:      { en: "A2 specific", lt: "A2 specifika" },
  flashcardsExam:  { en: "Flashcards & exam →", lt: "Kortelės ir egzaminas →" },
  harecBlurb:      { en: "{n} English questions with explanations — HAREC syllabus: propagation, electronics, procedures, antennas, safety.",
                     lt: "{n} klausimų anglų kalba su paaiškinimais — HAREC programa: sklidimas, elektronika, procedūros, antenos, sauga." },
  explanations:    { en: "Explanations", lt: "Su paaiškinimais" },
  // A1/A3 hub
  a1a3Title:       { en: "EASA A1/A3 — Open Category", lt: "EASA A1/A3 — atviroji kategorija" },
  a1a3Sub1:        { en: "Study for the EASA", lt: "Mokykitės EASA" },
  a1a3CertName:    { en: "A1/A3 remote pilot competency certificate", lt: "A1/A3 nuotolinio piloto kompetencijos pažymėjimui" },
  a1a3Sub2:        { en: "{n} questions across 9 categories matching the Lithuanian CAA exam syllabus. Spaced repetition (SM-2) in flashcards. Exam questions proportionally distributed across categories.",
                     lt: "{n} klausimų 9 kategorijose pagal Lietuvos TKA egzamino programą. Intervalinis kartojimas (SM-2) kortelėse. Egzamino klausimai proporcingai paskirstyti pagal kategorijas." },
  fcSR:            { en: "Flashcards + Spaced Repetition", lt: "Kortelės + intervalinis kartojimas" },
  a1a3FcBlurb:     { en: "All {n} questions with SM-2 scheduling. Hard cards come back sooner; easy ones space out. Filter by category, reasoning questions tagged ⚡.",
                     lt: "Visi {n} klausimai su SM-2 planavimu. Sunkios kortelės grįžta greičiau, lengvos — rečiau. Filtras pagal kategoriją, samprotavimo klausimai pažymėti ⚡." },
  startFlashcards: { en: "Start flashcards →", lt: "Pradėti korteles →" },
  examPractice:    { en: "Exam Practice", lt: "Egzamino pratybos" },
  a1a3ExBlurb:     { en: "Timed mock exam — 40 questions, 60 min, 75% pass. Proportional distribution across all 9 categories, ⚡ reasoning questions included. Session saved — resume anytime.",
                     lt: "Bandomasis egzaminas su laikmačiu — 40 klausimų, 60 min, 75 % išlaikymo riba. Proporcingas paskirstymas visose 9 kategorijose, įskaitant ⚡ samprotavimo klausimus. Sesija išsaugoma — tęskite bet kada." },
  qProportional:   { en: "40 Q proportional", lt: "40 kl. proporcingai" },
  startExam:       { en: "Start exam →", lt: "Pradėti egzaminą →" },
  catBreakdown:    { en: "Category breakdown", lt: "Klausimai pagal kategorijas" },
  catEN:           { en: "Category (EN)", lt: "Kategorija (EN)" },
  catLT:           { en: "Kategorija (LT)", lt: "Kategorija (LT)" },
  total:           { en: "Total", lt: "Iš viso" },
  realExam:        { en: "Real exam:", lt: "Tikras egzaminas:" },
  realExamLT:      { en: "Lithuania —", lt: "Lietuva —" },
  // A2 hub
  a2Title:         { en: "EASA A2 Certificate of Competency", lt: "EASA A2 kompetencijos pažymėjimas" },
  a2Sub:           { en: "Study for the", lt: "Mokykitės" },
  a2CocName:       { en: "A2 CoC", lt: "A2 CoC pažymėjimui" },
  a2Sub2:          { en: "— fly C2 drones within 30–50 m of uninvolved persons. Prerequisite: A1/A3 certificate. {n} questions across {c} categories.",
                     lt: "— skraidykite C2 klasės dronais 30–50 m nuo nesusijusių asmenų. Būtina sąlyga: A1/A3 pažymėjimas. {n} klausimų {c} kategorijose." },
  flashcards:      { en: "Flashcards", lt: "Kortelės" },
  a2FcBlurb:       { en: "All {n} A2-specific questions: advanced meteorology, human performance, UAS technical knowledge, A2 operational rules. SM-2 scheduling.",
                     lt: "Visi {n} A2 klausimai: išplėstinė meteorologija, žmogaus galimybės, UAS techninės žinios, A2 veiklos taisyklės. SM-2 planavimas." },
  a2ExBlurb:       { en: "40 questions, 30 min, 75% pass. Post-exam review with explanations and category breakdown. Session saved — resume anytime.",
                     lt: "40 klausimų, 30 min, 75 % išlaikymo riba. Peržiūra po egzamino su paaiškinimais ir kategorijų suvestine. Sesija išsaugoma — tęskite bet kada." },
  a2RealExam:      { en: "Contact your national CAA for A2 CoC test location and booking.", lt: "Dėl A2 CoC egzamino vietos ir registracijos kreipkitės į nacionalinę aviacijos instituciją (Lietuvoje — TKA)." },
  // radio hub
  radioHubTitle:   { en: "Radijo Mėgėjų Egzaminai", lt: "Radijo mėgėjų egzaminai" },
  // exam configs
  a1a3Name:        { en: "EASA A1/A3 Remote Pilot Competency", lt: "EASA A1/A3 nuotolinio piloto kompetencija" },
  a1a3Desc:        { en: "EU Open category online theory test. 40 questions, 60 minutes, 75% pass (30/40 correct).",
                     lt: "ES atvirosios kategorijos teorijos testas internetu. 40 klausimų, 60 minučių, 75 % išlaikymo riba (30/40 teisingų)." },
  a2Name:          { en: "EASA A2 Certificate of Competency", lt: "EASA A2 kompetencijos pažymėjimas" },
  a2Desc:          { en: "A2 CoC theory test. 40 questions, 30 minutes, 75% pass. Prerequisite: A1/A3 certificate.",
                     lt: "A2 CoC teorijos testas. 40 klausimų, 30 minučių, 75 % išlaikymo riba. Būtina sąlyga: A1/A3 pažymėjimas." },
  // exam session
  sessionInProgress:{ en: "Session in progress", lt: "Sesija vykdoma" },
  answered:        { en: "answered", lt: "atsakyta" },
  left:            { en: "left", lt: "liko" },
  review:          { en: "review", lt: "peržiūra" },
  timeExpired:     { en: "time expired", lt: "laikas baigėsi" },
  continue:        { en: "Continue →", lt: "Tęsti →" },
  discard:         { en: "Discard", lt: "Atmesti" },
  questionsInPool: { en: "Questions in pool", lt: "Klausimų banke" },
  passMark:        { en: "Pass mark", lt: "Išlaikymo riba" },
  time:            { en: "Time", lt: "Laikas" },
  attempts:        { en: "attempt(s)", lt: "bandymai" },
  last:            { en: "Last", lt: "Paskutinis" },
  best:            { en: "Best", lt: "Geriausias" },
  qThisSession:    { en: "Questions this session", lt: "Klausimų šioje sesijoje" },
  all:             { en: "All", lt: "Visi" },
  catDist:         { en: "Category distribution ({n} questions, proportional):", lt: "Paskirstymas pagal kategorijas ({n} klausimų, proporcingai):" },
  estTime:         { en: "Estimated time: {n} min", lt: "Numatoma trukmė: {n} min" },
  passUpper:       { en: "PASS", lt: "IŠLAIKYTA" },
  failUpper:       { en: "FAIL", lt: "NEIŠLAIKYTA" },
  correct:         { en: "correct", lt: "teisingų" },
  passLabel:       { en: "Pass", lt: "Riba" },
  prev:            { en: "← Prev", lt: "← Ankstesnis" },
  next:            { en: "Next →", lt: "Kitas →" },
  finish:          { en: "Finish →", lt: "Baigti →" },
  notAnswered:     { en: "Not answered", lt: "Neatsakyta" },
  newExam:         { en: "← New exam", lt: "← Naujas egzaminas" },
  end:             { en: "End", lt: "Baigti" },
  flag:            { en: "⚑ Flag", lt: "⚑ Pažymėti" },
  flagged:         { en: "⚑ Flagged", lt: "⚑ Pažymėtas" },
  reasoningTag:    { en: "⚡ Reasoning", lt: "⚡ Samprotavimas" },
  today:           { en: "today", lt: "šiandien" },
  yesterday:       { en: "yesterday", lt: "vakar" },
  dAgo:            { en: "{n}d ago", lt: "prieš {n} d." },
  wAgo:            { en: "{n}w ago", lt: "prieš {n} sav." },
  moAgo:           { en: "{n}mo ago", lt: "prieš {n} mėn." },
  // flashcards
  fcTitleSuffix:   { en: "— Flashcards", lt: "— kortelės" },
  category:        { en: "Category", lt: "Kategorija" },
  shuffled:        { en: "⇄ Shuffled", lt: "⇄ Sumaišyta" },
  inOrder:         { en: "⇄ In order", lt: "⇄ Iš eilės" },
  restartSession:  { en: "↺ Restart session", lt: "↺ Iš naujo" },
  due:             { en: "Due", lt: "Kartoti" },
  newCards:        { en: "New", lt: "Nauji" },
  learned:         { en: "Learned", lt: "Išmokta" },
  again:           { en: "Again", lt: "Dar kartą" },
  thisSession:     { en: "This session: {n} reviewed", lt: "Šioje sesijoje: {n} peržiūrėta" },
  sessionComplete: { en: "Session complete!", lt: "Sesija baigta!" },
  cardsReviewed:   { en: "{n} cards reviewed.", lt: "Peržiūrėta kortelių: {n}." },
  wereHard:        { en: "{n} were hard — they'll come back sooner.", lt: "{n} buvo sunkios — jos grįš greičiau." },
  allCorrect:      { en: "All answered correctly.", lt: "Į visas atsakyta teisingai." },
  comeBack:        { en: "Cards scheduled for future sessions will appear when due. Come back tomorrow!", lt: "Kortelės, suplanuotos vėliau, pasirodys, kai ateis jų laikas. Sugrįžkite rytoj!" },
  startNewSession: { en: "Start new session", lt: "Pradėti naują sesiją" },
  backToModule:    { en: "← Back to module", lt: "← Atgal į modulį" },
  clickToReveal:   { en: "Click or Space to reveal", lt: "Spauskite arba tarpas — atskleisti" },
  correctAnswer:   { en: "Correct answer", lt: "Teisingas atsakymas" },
  hideExplanation: { en: "Hide explanation", lt: "Slėpti paaiškinimą" },
  showExplanation: { en: "Show explanation", lt: "Rodyti paaiškinimą" },
  spaceToFlip:     { en: "Space to flip", lt: "Tarpas — apversti" },
  seeSoon:         { en: "See soon", lt: "Netrukus" },
  hard:            { en: "Hard", lt: "Sunku" },
  tomorrow:        { en: "Tomorrow", lt: "Rytoj" },
  good:            { en: "Good", lt: "Gerai" },
  easy:            { en: "Easy", lt: "Lengva" },
  longInterval:    { en: "Long interval", lt: "Ilgas intervalas" },
  keysHint:        { en: "1 = Again · 2 = Hard · 3 = Good · 4 = Easy", lt: "1 = Dar kartą · 2 = Sunku · 3 = Gerai · 4 = Lengva" },
  noCards:         { en: "No cards match this filter.", lt: "Šiam filtrui kortelių nėra." },
} as const satisfies Record<string, LS>;

export function fmt(s: string, vars: Record<string, string | number>) {
  return s.replace(/\{(\w+)\}/g, (_, k) => String(vars[k] ?? ""));
}
