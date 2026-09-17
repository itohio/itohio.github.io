"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import { a1a3Questions, a1a3Categories } from "@/data/a1a3/index";
import { a2Questions } from "@/data/a2";
import { radioQuestions } from "@/data/radio";
import { radioLtuBQuestions } from "@/data/radio-ltu-b";
import { radioLtuAQuestions } from "@/data/radio-ltu-a";
import { a1a3Lt, a1a3CategoriesLt } from "@/data/lt/a1a3";
import { a2Lt, a2CategoriesLt } from "@/data/lt/a2";
import type { ExamConfig } from "@/types/exam";
import { LangProvider, LangToggle, useLang, localizeQuestions, S, fmt, ltCount, pct } from "@/i18n";

const RxViewerClient         = dynamic(() => import("@/components/RxViewerClient"),         { ssr: false });
const FlashcardSessionClient = dynamic(() => import("@/components/FlashcardSessionClient"), { ssr: false });
const ExamSessionClient      = dynamic(() => import("@/components/ExamSessionClient"),      { ssr: false });
const FlySection             = dynamic(() => import("@/components/FlySection"),             { ssr: false });

// ── exam configs ──────────────────────────────────────────────────────────────
function useExamConfigs(): Record<string, ExamConfig> {
  const { t } = useLang();
  return useMemo(() => ({
    a1a3: {
      id: "a1a3", shortName: "A1/A3",
      name: t(S.a1a3Name), description: t(S.a1a3Desc),
      questionCount: 40, timeMinutes: 60, passPercent: 75, color: "#58a6ff",
    },
    a2: {
      id: "a2", shortName: "A2 CoC",
      name: t(S.a2Name), description: t(S.a2Desc),
      questionCount: 40, timeMinutes: 30, passPercent: 75, color: "#3fb950",
    },
    radio: {
      id: "radio", shortName: "HAREC",
      name: "Amateur Radio — HAREC Examination",
      description: "HAREC foundation exam. 26 questions, 30 minutes, 70% pass. Recognised across IARU Region 1 / CEPT.",
      questionCount: 26, timeMinutes: 30, passPercent: 70, color: "#f0a020",
    },
    "radio-b": {
      id: "radio-b", shortName: "Klasė B",
      name: "Radijo mėgėjų B klasės kvalifikacinis egzaminas",
      description: "Oficialūs RRT egzamino klausimai. 30 klausimų, 45 min, 70 % išlaikymo riba. Šaltinis: hamradio.lt / va1da5/ham-radio-exam-prep-ltu.",
      questionCount: 30, timeMinutes: 45, passPercent: 70, color: "#f0a020",
    },
    "radio-a": {
      id: "radio-a", shortName: "Klasė A",
      name: "Radijo mėgėjų A klasės kvalifikacinis egzaminas",
      description: "Oficialūs RRT egzamino klausimai. 40 klausimų, 60 min, 70 % išlaikymo riba. Šaltinis: hamradio.lt / va1da5/ham-radio-exam-prep-ltu.",
      questionCount: 40, timeMinutes: 60, passPercent: 70, color: "#f0a020",
    },
  }), [t]);
}

// ── hash router ───────────────────────────────────────────────────────────────
function getHash() {
  if (typeof window === "undefined") return "";
  // Support both /#/path and legacy ?tool=rxmap
  const hash = window.location.hash.replace(/^#\/?/, "").replace(/\/$/, "");
  if (hash) return hash;
  const tool = new URLSearchParams(window.location.search).get("tool");
  if (tool === "rxmap") return "rxmap";
  return "";
}

function navigate(path: string) {
  window.location.hash = path ? `/${path}` : "/";
}

function Eyebrow({ trail }: { trail?: Array<{ label: string; to: string }> }) {
  const { t } = useLang();
  return (
    <p className="hub-eyebrow">
      <button className="hub-link-btn" onClick={() => navigate("")}>{t(S.itohiTools)}</button>
      {trail?.map(x => <span key={x.to}> · <button className="hub-link-btn" onClick={() => navigate(x.to)}>{x.label}</button></span>)}
    </p>
  );
}

// ── hub components ────────────────────────────────────────────────────────────
function qCount(n: number, lang: "en" | "lt") { return lang === "lt" ? ltCount(n, "klausimas", "klausimai", "klausimų") : `${n} question${n === 1 ? "" : "s"}`; }

function A1A3Hub() {
  const { t, lang } = useLang();
  const total = a1a3Questions.length;
  const catName = (c: { key: string; lt: string }) => (lang === "lt" ? c.lt : c.key);
  return (
    <div className="hub">
      <header className="hub-head">
        <Eyebrow trail={[{ label: t(S.exams), to: "exams" }]} />
        <h1>{t(S.a1a3Title)}</h1>
        <p className="hub-sub">
          {t(S.a1a3Sub1)} <strong>{t(S.a1a3CertName)}</strong>.{" "}
          {fmt(t(S.a1a3Sub2), { n: qCount(total, lang) })}
        </p>
      </header>
      <ul className="hub-grid">
        <li className="hub-card hub-card--blue">
          <button className="hub-card-btn" onClick={() => navigate("a1a3/flashcards")}>
            <h2>{t(S.fcSR)}</h2>
            <p>{fmt(t(S.a1a3FcBlurb), { n: total })}</p>
            <div className="hub-tags">{a1a3Categories.map(c => <span key={c.key}>{catName(c)} ({c.count})</span>)}</div>
            <span className="hub-go">{t(S.startFlashcards)}</span>
          </button>
        </li>
        <li className="hub-card hub-card--green">
          <button className="hub-card-btn" onClick={() => navigate("a1a3/exam")}>
            <h2>{t(S.examPractice)}</h2>
            <p>{t(S.a1a3ExBlurb)}</p>
            <div className="hub-tags"><span>{t(S.qProportional)}</span><span>60 {t(S.min)}</span><span>{pct(75, lang)} {t(S.pass)}</span><span>{t(S.reasoning)}</span></div>
            <span className="hub-go">{t(S.startExam)}</span>
          </button>
        </li>
      </ul>
      <div className="hub-cat-table">
        <h3>{t(S.catBreakdown)}</h3>
        <table>
          <thead><tr><th>{t(S.catEN)}</th><th>{t(S.catLT)}</th><th>{t(S.q)}</th></tr></thead>
          <tbody>
            {a1a3Categories.map(c => (
              <tr key={c.key}><td>{c.key}</td><td className="lt">{c.lt}</td><td className="num">{c.count}</td></tr>
            ))}
            <tr className="total"><td colSpan={2}>{t(S.total)}</td><td className="num">{total}</td></tr>
          </tbody>
        </table>
      </div>
      <div className="hub-note">
        <strong>{t(S.realExam)}</strong> {t(S.realExamLT)} <a href="https://sertifikatai.tka.lt/lt/login" target="_blank" rel="noopener">TKA (sertifikatai.tka.lt)</a> · <a href="https://utm.ans.lt" target="_blank" rel="noopener">utm.ans.lt</a>. 40 {t(S.q)} · 60 {t(S.min)} · {pct(75, lang)} {t(S.pass)}.
      </div>
      <p className="hub-back"><button className="hub-link-btn" onClick={() => navigate("")}>{t(S.allTools)}</button></p>
    </div>
  );
}

function A2Hub() {
  const { t, lang } = useLang();
  const total = a2Questions.length;
  const cats = Array.from(new Set(a2Questions.map(q => q.category ?? "General"))).sort();
  return (
    <div className="hub">
      <header className="hub-head">
        <Eyebrow trail={[{ label: t(S.exams), to: "exams" }]} />
        <h1>{t(S.a2Title)}</h1>
        <p className="hub-sub">
          {t(S.a2Sub)} <strong>{t(S.a2CocName)}</strong> {fmt(t(S.a2Sub2), { n: qCount(total, lang), c: cats.length })}
        </p>
      </header>
      <ul className="hub-grid">
        <li className="hub-card hub-card--blue">
          <button className="hub-card-btn" onClick={() => navigate("a2/flashcards")}>
            <h2>{t(S.flashcards)}</h2>
            <p>{fmt(t(S.a2FcBlurb), { n: total })}</p>
            <div className="hub-tags">{cats.map(c => <span key={c}>{lang === "lt" ? (a2CategoriesLt[c] ?? c) : c}</span>)}</div>
            <span className="hub-go">{t(S.startFlashcards)}</span>
          </button>
        </li>
        <li className="hub-card hub-card--green">
          <button className="hub-card-btn" onClick={() => navigate("a2/exam")}>
            <h2>{t(S.examPractice)}</h2>
            <p>{t(S.a2ExBlurb)}</p>
            <div className="hub-tags"><span>40 {t(S.q)}</span><span>30 {t(S.min)}</span><span>{pct(75, lang)} {t(S.pass)}</span></div>
            <span className="hub-go">{t(S.startExam)}</span>
          </button>
        </li>
      </ul>
      <div className="hub-note">
        <strong>{t(S.realExam)}</strong> {t(S.a2RealExam)}
      </div>
      <p className="hub-back"><button className="hub-link-btn" onClick={() => navigate("")}>{t(S.allTools)}</button></p>
    </div>
  );
}

function RadioHub() {
  const { t } = useLang();
  return (
    <div className="hub">
      <header className="hub-head">
        <Eyebrow trail={[{ label: t(S.exams), to: "exams" }]} />
        <h1>{t(S.radioHubTitle)}</h1>
        <p className="hub-sub">
          Oficialūs RRT egzamino klausimai (Klasė B ir A) iš{" "}
          <a href="https://hamradio.lt" target="_blank" rel="noopener">hamradio.lt</a>, ir HAREC konceptualūs klausimai anglų kalba.
          Visi su SM-2 pasikartojimo planavimu ir egzamino pratybomis.
        </p>
      </header>
      <ul className="hub-grid">
        <li className="hub-card hub-card--orange">
          <button className="hub-card-btn" onClick={() => navigate("radio/b")}>
            <h2>Klasė B — Oficialūs klausimai</h2>
            <p>{radioLtuBQuestions.length} oficialių RRT egzamino klausimų lietuvių kalba. Tiksliai tie klausimai, kurie naudojami realiame egzamine.</p>
            <div className="hub-tags"><span>{radioLtuBQuestions.length} kl.</span><span>Lietuvių kalba</span><span>RRT oficialūs</span></div>
            <span className="hub-exam-sub">30 kl. · 45 min · 70 %</span>
            <span className="hub-go">Pradėti →</span>
          </button>
        </li>
        <li className="hub-card hub-card--orange">
          <button className="hub-card-btn" onClick={() => navigate("radio/a")}>
            <h2>Klasė A — Oficialūs klausimai</h2>
            <p>{radioLtuAQuestions.length} oficialių RRT egzamino klausimų lietuvių kalba. A klasės licencija suteikia daugiau teisių nei B.</p>
            <div className="hub-tags"><span>{radioLtuAQuestions.length} kl.</span><span>Lietuvių kalba</span><span>RRT oficialūs</span></div>
            <span className="hub-exam-sub">40 kl. · 60 min · 70 %</span>
            <span className="hub-go">Pradėti →</span>
          </button>
        </li>
        <li className="hub-card hub-card--blue">
          <button className="hub-card-btn" onClick={() => navigate("radio/harec")}>
            <h2>HAREC — Conceptual (EN)</h2>
            <p>{radioQuestions.length} English questions with explanations covering the HAREC syllabus conceptually — propagation, electronics, operating procedures, safety. SM-2 scheduling.</p>
            <div className="hub-tags"><span>{radioQuestions.length} Q</span><span>English</span><span>With explanations</span></div>
            <span className="hub-exam-sub">26 Q · 30 min · 70% pass</span>
            <span className="hub-go">Start →</span>
          </button>
        </li>
      </ul>
      <div className="hub-note">
        <strong>Egzamino informacija:</strong>{" "}
        <a href="https://www.rrt.lt/radijo-spektras/radijo-megejai/radijo-megeju-egzaminai/" target="_blank" rel="noopener">RRT egzaminai</a>
        {" · "}<a href="https://www.lrmd.lt" target="_blank" rel="noopener">LRMD</a>
        {" · "}Klausimai: <a href="https://github.com/va1da5/ham-radio-exam-prep-ltu" target="_blank" rel="noopener">va1da5/ham-radio-exam-prep-ltu</a> (MIT)
      </div>
      <p className="hub-back"><button className="hub-link-btn" onClick={() => navigate("")}>{t(S.allTools)}</button></p>
    </div>
  );
}

function RadioLtuHub({ level }: { level: "a" | "b" }) {
  const { t } = useLang();
  const configs = useExamConfigs();
  const isB = level === "b";
  const questions = isB ? radioLtuBQuestions : radioLtuAQuestions;
  const title = isB ? "Klasė B" : "Klasė A";
  const config = configs[isB ? "radio-b" : "radio-a"];
  return (
    <div className="hub">
      <header className="hub-head">
        <Eyebrow trail={[{ label: t(S.exams), to: "exams" }, { label: "Radijo egzaminai", to: "radio" }]} />
        <h1>Radijo mėgėjų {title} kvalifikacinis egzaminas</h1>
        <p className="hub-sub">
          {questions.length} oficialių RRT egzamino klausimų lietuvių kalba.
          {isB ? " B klasės licencija — pradinis lygis." : " A klasės licencija — aukštesnis lygis, daugiau teisių."}
          {" "}Šaltinis: <a href="https://hamradio.lt" target="_blank" rel="noopener">hamradio.lt</a>.
        </p>
      </header>
      <ul className="hub-grid">
        <li className="hub-card hub-card--blue">
          <button className="hub-card-btn" onClick={() => navigate(`radio/${level}/flashcards`)}>
            <h2>Kortelės (SM-2)</h2>
            <p>Visi {questions.length} klausimai su SM-2 planavimu. Sunkesni klausimai kartojasi dažniau. Filtravimas, maišymas.</p>
            <div className="hub-tags"><span>{questions.length} kl.</span><span>SM-2</span><span>Lietuvių k.</span></div>
            <span className="hub-go">Pradėti →</span>
          </button>
        </li>
        <li className="hub-card hub-card--green">
          <button className="hub-card-btn" onClick={() => navigate(`radio/${level}/exam`)}>
            <h2>Egzamino pratybos</h2>
            <p>{config.questionCount} klausimų, {config.timeMinutes} min, {config.passPercent} % reikalinga. Sesija išsaugoma — galima tęsti vėliau.</p>
            <div className="hub-tags"><span>{config.questionCount} kl.</span><span>{config.timeMinutes} min</span><span>{config.passPercent} %</span></div>
            <span className="hub-go">Pradėti egzaminą →</span>
          </button>
        </li>
      </ul>
      <div className="hub-note">
        <strong>Pastaba:</strong> Tai oficialūs RRT egzamino klausimai iš{" "}
        <a href="https://github.com/va1da5/ham-radio-exam-prep-ltu" target="_blank" rel="noopener">va1da5/ham-radio-exam-prep-ltu</a> (MIT).
        Klausimai be paaiškinimų — kaip ir realiame egzamine.
      </div>
      <p className="hub-back"><button className="hub-link-btn" onClick={() => navigate("radio")}>← Radijo egzaminai</button></p>
    </div>
  );
}

// ── exams hub ─────────────────────────────────────────────────────────────────
function ExamsHub() {
  const { t, lang } = useLang();
  return (
    <div className="hub">
      <header className="hub-head">
        <Eyebrow />
        <h1>{t(S.examStudy)}</h1>
        <p className="hub-sub">{t(S.examStudySub)}</p>
      </header>

      <h2 className="hub-section">{t(S.easaDroneExams)}</h2>
      <ul className="hub-grid">
        <li className="hub-card hub-card--blue">
          <button className="hub-card-btn" onClick={() => navigate("a1a3")}>
            <h2>EASA A1/A3</h2>
            <p>{t(S.a1a3CardBlurb)}</p>
            <div className="hub-tags"><span>336 {t(S.q)}</span><span>{t(S.categories9)}</span><span>{t(S.reasoning)}</span></div>
            <span className="hub-exam-sub">40 {t(S.q)} · 60 {t(S.min)} · {pct(75, lang)} {t(S.pass)}</span>
            <span className="hub-go">{t(S.flashcardsExam)}</span>
          </button>
        </li>
        <li className="hub-card hub-card--green">
          <button className="hub-card-btn" onClick={() => navigate("a2")}>
            <h2>EASA A2 CoC</h2>
            <p>{t(S.a2CardBlurb)}</p>
            <div className="hub-tags"><span>60 {t(S.q)}</span><span>{t(S.a2specific)}</span></div>
            <span className="hub-exam-sub">40 {t(S.q)} · 30 {t(S.min)} · {pct(75, lang)} {t(S.pass)}</span>
            <span className="hub-go">{t(S.flashcardsExam)}</span>
          </button>
        </li>
      </ul>

      <h2 className="hub-section">{t(S.radioExams)}</h2>
      <ul className="hub-grid hub-grid--3">
        <li className="hub-card hub-card--orange">
          <button className="hub-card-btn" onClick={() => navigate("radio/b")}>
            <h2>Klasė B</h2>
            <p>{radioLtuBQuestions.length} oficialių RRT egzamino klausimų lietuvių kalba. Pradinis lygis.</p>
            <div className="hub-tags"><span>{radioLtuBQuestions.length} kl.</span><span>LT</span><span>RRT oficialūs</span></div>
            <span className="hub-exam-sub">30 kl. · 45 min · 70 %</span>
            <span className="hub-go">Pradėti →</span>
          </button>
        </li>
        <li className="hub-card hub-card--orange">
          <button className="hub-card-btn" onClick={() => navigate("radio/a")}>
            <h2>Klasė A</h2>
            <p>{radioLtuAQuestions.length} oficialių RRT egzamino klausimų lietuvių kalba. Aukštesnis lygis, daugiau teisių.</p>
            <div className="hub-tags"><span>{radioLtuAQuestions.length} kl.</span><span>LT</span><span>RRT oficialūs</span></div>
            <span className="hub-exam-sub">40 kl. · 60 min · 70 %</span>
            <span className="hub-go">Pradėti →</span>
          </button>
        </li>
        <li className="hub-card hub-card--blue">
          <button className="hub-card-btn" onClick={() => navigate("radio/harec")}>
            <h2>HAREC (English)</h2>
            <p>{fmt(t(S.harecBlurb), { n: radioQuestions.length })}</p>
            <div className="hub-tags"><span>{radioQuestions.length} Q</span><span>EN</span><span>{t(S.explanations)}</span></div>
            <span className="hub-exam-sub">26 Q · 30 min · 70% pass</span>
            <span className="hub-go">{t(S.start)}</span>
          </button>
        </li>
      </ul>
      <p className="hub-back"><button className="hub-link-btn" onClick={() => navigate("")}>{t(S.allTools)}</button></p>
    </div>
  );
}

// ── landing page ──────────────────────────────────────────────────────────────
function Home() {
  const { t, lang } = useLang();
  const totalQ = radioLtuBQuestions.length + radioLtuAQuestions.length + radioQuestions.length + a1a3Questions.length + a2Questions.length;
  const tools = [
    { slug: "fly",   name: t(S.toolFly),   blurb: t(S.toolFlyBlurb), tags: [t(S.tagChecklists), t(S.tagTimer), "BVLOS", t(S.tagCountry)], color: "green" },
    { slug: "rxmap", name: t(S.toolRx),    blurb: t(S.toolRxBlurb),  tags: ["FPV", "EdgeTX", "3D", t(S.tagTelemetry)], color: "blue" },
    { slug: "exams", name: t(S.toolExams), blurb: fmt(t(S.toolExamsBlurb), { n: qCount(totalQ, lang) }), tags: ["A1/A3", "A2", "Klasė B/A", "HAREC"], color: "orange" },
  ];
  return (
    <div className="hub">
      <header className="hub-head">
        <p className="hub-eyebrow"><a href="https://itohi.com">itohi.com</a> · {t(S.homeEyebrow)}</p>
        <h1>{t(S.itohiTools)}</h1>
        <p className="hub-sub">{t(S.homeSub)}</p>
      </header>

      <ul className="hub-grid">
        {tools.map(x => (
          <li key={x.slug} className={`hub-card hub-card--${x.color}`}>
            <button className="hub-card-btn" onClick={() => navigate(x.slug)}>
              <h2>{x.name}</h2>
              <p>{x.blurb}</p>
              <div className="hub-tags">{x.tags.map(tag => <span key={tag}>{tag}</span>)}</div>
              <span className="hub-go">{t(S.open)}</span>
            </button>
          </li>
        ))}
      </ul>

      <footer className="hub-foot">
        <p>
          <a href="https://itohi.com">itohi.com</a>
          {" · "}<a href="https://itohi.com/fpv">FPV</a>
          {" · "}<a href="https://github.com/itohio">GitHub</a>
        </p>
      </footer>
    </div>
  );
}

// ── SPA root ──────────────────────────────────────────────────────────────────
function Router() {
  const { lang } = useLang();
  const configs = useExamConfigs();
  const [route, setRoute] = useState("");

  const handleHash = useCallback(() => { setRoute(getHash()); }, []);

  useEffect(() => {
    handleHash();
    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, [handleHash]);

  // Flashcard / exam back = go to parent hub
  const back = useCallback(() => {
    const parts = route.split("/");
    if (parts.length > 1) navigate(parts[0]);
    else navigate("");
  }, [route]);

  const a1a3 = useMemo(() => localizeQuestions(a1a3Questions, lang, a1a3Lt, a1a3CategoriesLt), [lang]);
  const a2   = useMemo(() => localizeQuestions(a2Questions,   lang, a2Lt,   a2CategoriesLt),   [lang]);

  // ── render ──
  if (!route || route === "/") return <Home />;
  if (route === "rxmap") return <RxViewerClient />;

  if (route === "a1a3")             return <A1A3Hub />;
  if (route === "a1a3/flashcards")  return <FlashcardSessionClient questions={a1a3} title="EASA A1/A3" storagePrefix="a1a3" onBack={back} />;
  if (route === "a1a3/exam")        return <ExamSessionClient questions={a1a3} config={configs.a1a3} onBack={back} />;

  if (route === "a2")               return <A2Hub />;
  if (route === "a2/flashcards")    return <FlashcardSessionClient questions={a2} title="EASA A2 CoC" storagePrefix="a2" onBack={back} />;
  if (route === "a2/exam")          return <ExamSessionClient questions={a2} config={configs.a2} onBack={back} />;

  if (route === "radio")              return <RadioHub />;
  // Lithuanian official exam routes
  if (route === "radio/b")            return <RadioLtuHub level="b" />;
  if (route === "radio/b/flashcards") return <FlashcardSessionClient questions={radioLtuBQuestions} title="Radijo mėgėjų Klasė B" storagePrefix="radio-b" onBack={() => navigate("radio/b")} />;
  if (route === "radio/b/exam")       return <ExamSessionClient questions={radioLtuBQuestions} config={configs["radio-b"]} onBack={() => navigate("radio/b")} />;
  if (route === "radio/a")            return <RadioLtuHub level="a" />;
  if (route === "radio/a/flashcards") return <FlashcardSessionClient questions={radioLtuAQuestions} title="Radijo mėgėjų Klasė A" storagePrefix="radio-a" onBack={() => navigate("radio/a")} />;
  if (route === "radio/a/exam")       return <ExamSessionClient questions={radioLtuAQuestions} config={configs["radio-a"]} onBack={() => navigate("radio/a")} />;
  // HAREC English
  if (route === "radio/harec")            return <FlashcardSessionClient questions={radioQuestions} title="Amateur Radio HAREC" storagePrefix="radio-harec" onBack={() => navigate("radio")} />;
  if (route === "radio/harec/flashcards") return <FlashcardSessionClient questions={radioQuestions} title="Amateur Radio HAREC" storagePrefix="radio-harec" onBack={() => navigate("radio")} />;
  if (route === "radio/harec/exam")       return <ExamSessionClient questions={radioQuestions} config={configs.radio} onBack={() => navigate("radio")} />;

  if (route === "exams") return <ExamsHub />;

  if (route === "fly" || route.startsWith("fly/")) {
    return <FlySection route={route} onBack={() => navigate("")} navigate={navigate} />;
  }

  // unknown hash → home
  return <Home />;
}

export default function App() {
  return (
    <LangProvider>
      <LangToggle />
      <Router />
    </LangProvider>
  );
}
