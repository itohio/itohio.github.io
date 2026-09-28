"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import type { Question, ExamConfig } from "@/types/exam";
import { useLang, S, fmt, pct, type Lang } from "@/i18n";

// ── helpers ──────────────────────────────────────────────────────────────────
function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function pickProportional(questions: Question[], total: number): Question[] {
  const bycat: Record<string, Question[]> = {};
  questions.forEach(q => {
    const c = q.category ?? "General";
    if (!bycat[c]) bycat[c] = [];
    bycat[c].push(q);
  });
  const entries = Object.entries(bycat);
  const totalQ = questions.length;
  const picked: Question[] = [];
  let remaining = total;
  const allocs = entries.map(([cat, qs]) => {
    const exact = (qs.length / totalQ) * total;
    return { cat, qs, alloc: Math.floor(exact), frac: exact - Math.floor(exact) };
  });
  allocs.forEach(e => {
    const n = Math.min(e.alloc, e.qs.length);
    picked.push(...shuffle(e.qs).slice(0, n));
    remaining -= n;
  });
  if (remaining > 0) {
    const sorted = allocs.filter(e => e.alloc < e.qs.length).sort((a, b) => b.frac - a.frac);
    for (const e of sorted) {
      if (remaining === 0) break;
      const extra = shuffle(e.qs).find(q => !picked.find(p => p.id === q.id));
      if (extra) { picked.push(extra); remaining--; }
    }
  }
  return shuffle(picked).slice(0, total);
}

function fmtDate(ts: number, lang: Lang) {
  const L = (k: keyof typeof S) => S[k][lang];
  const d = new Date(ts);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffD = Math.floor(diffMs / 86_400_000);
  if (diffD === 0) return L("today");
  if (diffD === 1) return L("yesterday");
  if (diffD < 7) return fmt(L("dAgo"), { n: diffD });
  if (diffD < 30) return fmt(L("wAgo"), { n: Math.floor(diffD / 7) });
  return fmt(L("moAgo"), { n: Math.floor(diffD / 30) });
}

// ── persistence types ─────────────────────────────────────────────────────────
interface Answer { chosen: number | null; flagged: boolean; }

interface SavedSession {
  poolIds: string[];
  answers: Answer[];
  current: number;
  endTime: number;   // absolute ms — time keeps running while tab is closed
  phase: "running" | "review";
  reviewIdx: number;
  qCount: number;
  savedAt: number;
}

export interface ExamResult {
  date: number;
  score: number;
  correct: number;
  total: number;
  passed: boolean;
  durationSec: number;
}

function sessionKey(id: string) { return `exam_session_${id}`; }
function historyKey(id: string) { return `exam_history_${id}`; }

function loadSession(id: string): SavedSession | null {
  try {
    const raw = localStorage.getItem(sessionKey(id));
    if (!raw) return null;
    const s: SavedSession = JSON.parse(raw);
    // Discard sessions older than 7 days
    if (Date.now() - s.savedAt > 7 * 86_400_000) { localStorage.removeItem(sessionKey(id)); return null; }
    return s;
  } catch { return null; }
}

function saveSession(id: string, s: SavedSession) {
  try { localStorage.setItem(sessionKey(id), JSON.stringify({ ...s, savedAt: Date.now() })); } catch {}
}

function clearSession(id: string) {
  try { localStorage.removeItem(sessionKey(id)); } catch {}
}

function loadHistory(id: string): ExamResult[] {
  try { return JSON.parse(localStorage.getItem(historyKey(id)) ?? "[]"); } catch { return []; }
}

function appendHistory(id: string, result: ExamResult) {
  try {
    const prev = loadHistory(id);
    const next = [result, ...prev].slice(0, 20);
    localStorage.setItem(historyKey(id), JSON.stringify(next));
  } catch {}
}

// ── component ─────────────────────────────────────────────────────────────────
interface Props { questions: Question[]; config: ExamConfig; onBack?: () => void; }
type Phase = "setup" | "running" | "review";

export default function ExamSession({ questions, config, onBack }: Props) {
  const { t, lang } = useLang();
  const [phase, setPhase] = useState<Phase>("setup");
  const [qCount, setQCount] = useState(config.questionCount);
  const [pool, setPool] = useState<Question[]>([]);
  const [answers, setAnswers] = useState<Answer[]>([]);
  const [current, setCurrent] = useState(0);
  const [endTime, setEndTime] = useState(0);       // absolute ms
  const [secondsLeft, setSecondsLeft] = useState(0);
  const [reviewIdx, setReviewIdx] = useState(0);
  const [savedSession, setSavedSession] = useState<SavedSession | null>(null);
  const [history, setHistory] = useState<ExamResult[]>([]);
  const [startTime, setStartTime] = useState(0);   // ms when exam began

  // Load saved session + history on mount
  useEffect(() => {
    setSavedSession(loadSession(config.id));
    setHistory(loadHistory(config.id));
  }, [config.id]);

  // Auto-save whenever running state changes
  useEffect(() => {
    if (phase !== "running" && phase !== "review") return;
    if (!pool.length) return;
    saveSession(config.id, {
      poolIds: pool.map(q => q.id),
      answers,
      current,
      endTime,
      phase,
      reviewIdx,
      qCount,
      savedAt: Date.now(),
    });
  }, [phase, answers, current, reviewIdx, endTime, pool, qCount, config.id]);

  // Timer
  useEffect(() => {
    if (phase !== "running") return;
    const tick = () => {
      const left = Math.max(0, Math.round((endTime - Date.now()) / 1000));
      setSecondsLeft(left);
      if (left === 0) finish();
    };
    tick();
    const t = setInterval(tick, 500);
    return () => clearInterval(t);
  }, [phase, endTime]); // eslint-disable-line

  const finish = useCallback((fromTimer = false) => {
    setReviewIdx(0);
    setPhase("review");
    // Save result to history
    setPool(p => {
      setAnswers(a => {
        const correct = a.filter((ans, i) => ans.chosen === p[i]?.answer).length;
        const score = p.length ? Math.round((correct / p.length) * 100) : 0;
        const durationSec = Math.round((Date.now() - startTime) / 1000);
        const result: ExamResult = { date: Date.now(), score, correct, total: p.length, passed: score >= config.passPercent, durationSec };
        appendHistory(config.id, result);
        setHistory(loadHistory(config.id));
        return a;
      });
      return p;
    });
  }, [config.id, config.passPercent, startTime]);

  const startFresh = useCallback((n: number) => {
    const p = pickProportional(questions, Math.min(n, questions.length));
    const now = Date.now();
    const dur = Math.round((n / config.questionCount) * config.timeMinutes * 60);
    setPool(p);
    setAnswers(p.map(() => ({ chosen: null, flagged: false })));
    setCurrent(0);
    setEndTime(now + dur * 1000);
    setSecondsLeft(dur);
    setStartTime(now);
    setReviewIdx(0);
    setPhase("running");
    clearSession(config.id);
    setSavedSession(null);
  }, [questions, config]);

  const resumeSession = useCallback((s: SavedSession) => {
    // Hydrate question pool from IDs
    const qMap = new Map(questions.map(q => [q.id, q]));
    const p = s.poolIds.map(id => qMap.get(id)).filter(Boolean) as Question[];
    if (p.length < s.poolIds.length * 0.9) {
      // Too many IDs missing (question bank changed) — discard
      clearSession(config.id);
      setSavedSession(null);
      return;
    }
    setPool(p);
    setAnswers(s.answers);
    setCurrent(s.current);
    setEndTime(s.endTime);
    setSecondsLeft(Math.max(0, Math.round((s.endTime - Date.now()) / 1000)));
    setStartTime(Date.now() - (s.qCount / config.questionCount * config.timeMinutes * 60 * 1000 - (s.endTime - Date.now())));
    setQCount(s.qCount);
    setReviewIdx(s.reviewIdx);
    // If time ran out while away, go to review
    if (s.endTime <= Date.now() && s.phase === "running") {
      setPhase("review");
    } else {
      setPhase(s.phase);
    }
    setSavedSession(null);
  }, [questions, config]);

  const discardAndSetup = useCallback(() => {
    clearSession(config.id);
    setSavedSession(null);
  }, [config.id]);

  const resetToSetup = useCallback(() => {
    clearSession(config.id);
    setPhase("setup");
    setSavedSession(null);
  }, [config.id]);

  const setAnswer = (chosen: number) =>
    setAnswers(prev => prev.map((a, i) => i === current ? { ...a, chosen } : a));
  const toggleFlag = () =>
    setAnswers(prev => prev.map((a, i) => i === current ? { ...a, flagged: !a.flagged } : a));

  const mm = String(Math.floor(secondsLeft / 60)).padStart(2, "0");
  const ss = String(secondsLeft % 60).padStart(2, "0");
  const timerPct = endTime && startTime
    ? Math.max(0, Math.min(100, Math.round(((endTime - Date.now()) / (endTime - startTime)) * 100)))
    : 100;
  const timerWarn = secondsLeft > 0 && secondsLeft < 120;

  const correct = answers.filter((a, i) => a.chosen === pool[i]?.answer).length;
  const score = pool.length ? Math.round((correct / pool.length) * 100) : 0;
  const passed = score >= config.passPercent;
  const answered = answers.filter(a => a.chosen !== null).length;
  const flagged = answers.filter(a => a.flagged).length;

  const catStats = useMemo(() => {
    if (phase !== "review") return {};
    const stats: Record<string, { total: number; right: number }> = {};
    pool.forEach((q, i) => {
      const cat = q.category ?? "General";
      if (!stats[cat]) stats[cat] = { total: 0, right: 0 };
      stats[cat].total++;
      if (answers[i]?.chosen === q.answer) stats[cat].right++;
    });
    return stats;
  }, [phase, pool, answers]);

  const catCounts = useMemo(() => {
    const bycat: Record<string, number> = {};
    questions.forEach(q => { const c = q.category ?? "General"; bycat[c] = (bycat[c] ?? 0) + 1; });
    return Object.entries(bycat)
      .sort((a, b) => b[1] - a[1])
      .map(([cat, count]) => ({ cat, count, exam: Math.max(1, Math.round((count / questions.length) * qCount)) }));
  }, [questions, qCount]);

  const bestScore = history.length ? Math.max(...history.map(h => h.score)) : null;
  const lastResult = history[0] ?? null;

  /* ── SETUP ─────────────────────────────────────────────────────────────── */
  if (phase === "setup") {
    const maxQ = questions.length;
    const opts = [20, 30, 40, maxQ].filter((n, i, a) => n <= maxQ && a.indexOf(n) === i).sort((a, b) => a - b);
    return (
      <div className="ex-setup">
        <div className="hdr-row">
          <button className="fc-back-btn" onClick={onBack}>{t(S.back)}</button>
          <h1>{config.name}</h1>
        </div>
        <p className="ex-sub">{config.description}</p>

        {/* Resume banner */}
        {savedSession && (
          <div className="ex-resume-banner">
            <div className="ex-resume-info">
              <strong>{t(S.sessionInProgress)}</strong>
              <span>
                {savedSession.answers.filter(a => a.chosen !== null).length}/{savedSession.poolIds.length} {t(S.answered)}
                {savedSession.phase === "running" && savedSession.endTime > Date.now() && (
                  <> · {Math.round((savedSession.endTime - Date.now()) / 60000)} {t(S.min)} {t(S.left)}</>
                )}
                {savedSession.phase === "review" && <> · {t(S.review)}</>}
                {savedSession.endTime <= Date.now() && savedSession.phase === "running" && <> · {t(S.timeExpired)}</>}
              </span>
            </div>
            <div className="ex-resume-actions">
              <button className="ex-resume-btn" onClick={() => resumeSession(savedSession)}>
                {t(S.continue)}
              </button>
              <button className="ex-discard-btn" onClick={discardAndSetup}>
                {t(S.discard)}
              </button>
            </div>
          </div>
        )}

        <div className="ex-info-grid">
          <div><span>{t(S.questionsInPool)}</span><strong>{maxQ}</strong></div>
          <div><span>{t(S.passMark)}</span><strong>{pct(config.passPercent, lang)}</strong></div>
          <div><span>{t(S.time)}</span><strong>{config.timeMinutes} {t(S.min)}</strong></div>
        </div>

        {/* History summary */}
        {history.length > 0 && (
          <div className="ex-history-bar">
            <span>{history.length} {lang === "en" ? (history.length !== 1 ? "attempts" : "attempt") : (history.length === 1 ? "bandymas" : (history.length % 10 === 0 || (history.length % 100 >= 11 && history.length % 100 <= 19)) ? "bandymų" : "bandymai")}</span>
            {lastResult && (
              <span className={lastResult.passed ? "ok" : "bad"}>
                {t(S.last)}: {pct(lastResult.score, lang)} ({lastResult.correct}/{lastResult.total}) — {fmtDate(lastResult.date, lang)}
              </span>
            )}
            {bestScore !== null && <span className="ok">{t(S.best)}: {pct(bestScore, lang)}</span>}
          </div>
        )}

        <div className="ex-setup-row">
          <label>{t(S.qThisSession)}</label>
          <div className="ex-q-options">
            {opts.map(n => (
              <button key={n} className={qCount === n ? "on" : ""} onClick={() => setQCount(n)}>
                {n === maxQ ? `${t(S.all)} (${n})` : n}
              </button>
            ))}
          </div>
        </div>

        <div className="ex-dist-preview">
          <p className="ex-dist-label">{fmt(t(S.catDist), { n: qCount })}</p>
          <div className="ex-dist-grid">
            {catCounts.map(({ cat, count, exam }) => (
              <div key={cat} className="ex-dist-row">
                <span className="ex-dist-cat">{cat}</span>
                <span className="ex-dist-count">{exam} / {count}</span>
              </div>
            ))}
          </div>
        </div>

        <p className="ex-time-note">{fmt(t(S.estTime), { n: Math.round((qCount / config.questionCount) * config.timeMinutes) })}</p>
        <button className="ex-start" onClick={() => startFresh(qCount)}>{t(S.startExam)}</button>
      </div>
    );
  }

  /* ── REVIEW ────────────────────────────────────────────────────────────── */
  if (phase === "review") {
    const rq = pool[reviewIdx];
    const ra = answers[reviewIdx];
    return (
      <div className="ex-review">
        <div className="ex-result-header">
          <h1 className={passed ? "pass" : "fail"}>{passed ? t(S.passUpper) : t(S.failUpper)} — {pct(score, lang)}</h1>
          <p>{correct} / {pool.length} {t(S.correct)} · {t(S.passLabel)} {pct(config.passPercent, lang)} · {config.name}</p>
        </div>

        <div className="ex-cat-breakdown">
          {Object.entries(catStats).sort().map(([cat, s]) => (
            <div key={cat} className="ex-cat-row">
              <span>{cat}</span>
              <span className={s.right / s.total >= config.passPercent / 100 ? "ok" : "bad"}>
                {s.right}/{s.total}
              </span>
            </div>
          ))}
        </div>

        <div className="ex-rev-nav">
          <button onClick={() => setReviewIdx(i => Math.max(0, i - 1))} disabled={reviewIdx === 0}>{t(S.prev)}</button>
          <span className="ex-rev-counter">{t(S.qPrefix)}{reviewIdx + 1} / {pool.length}</span>
          <button onClick={() => setReviewIdx(i => Math.min(pool.length - 1, i + 1))} disabled={reviewIdx === pool.length - 1}>{t(S.next)}</button>
        </div>

        <div className={`ex-rev-card ${ra?.chosen === rq.answer ? "correct" : "wrong"}`}>
          {rq.category && <span className="fc-cat">{rq.category}</span>}
          {rq.reasoning && <span className="fc-cat fc-reasoning-tag">{t(S.reasoningTag)}</span>}
          <p className="ex-rev-q">{rq.q}</p>
          <ul className="ex-rev-opts">
            {rq.options.map((opt, i) => (
              <li key={i} className={i === rq.answer ? "correct-opt" : i === ra?.chosen ? "wrong-opt" : ""}>
                {i === rq.answer && "✓ "}{i === ra?.chosen && i !== rq.answer && "✗ "}{opt}
              </li>
            ))}
          </ul>
          {ra?.chosen === null && <p className="ex-skipped">{t(S.notAnswered)}</p>}
          {rq.explanation && <p className="ex-rev-explain">{rq.explanation}</p>}
        </div>

        <div className="ex-rev-actions">
          <button onClick={resetToSetup}>{t(S.newExam)}</button>
          <div className="ex-rev-dots">
            {pool.map((q, i) => (
              <button key={i}
                className={`ex-dot ${answers[i]?.chosen === q.answer ? "ok" : answers[i]?.chosen === null ? "skip" : "err"} ${i === reviewIdx ? "active" : ""}`}
                onClick={() => setReviewIdx(i)} title={`Q${i + 1}`} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* ── RUNNING ───────────────────────────────────────────────────────────── */
  const q = pool[current];
  const a = answers[current];
  return (
    <div className="ex-wrap">
      <div className="ex-topbar">
        <span className="ex-counter">{current + 1} / {pool.length}</span>
        <div className="ex-timer-bar">
          <div className={`ex-timer-fill ${timerWarn ? "warn" : ""}`} style={{ width: `${timerPct}%` }} />
        </div>
        <span className={`ex-time ${timerWarn ? "warn" : ""}`}>{mm}:{ss}</span>
        <span className="ex-answered">{answered}/{pool.length}</span>
        {flagged > 0 && <span className="ex-flagged">⚑ {flagged}</span>}
        <button className="ex-end-btn" onClick={() => finish()}>{t(S.end)}</button>
      </div>

      <div className="ex-body">
        <div className="ex-q-head">
          {q.category && <span className="fc-cat">{q.category}</span>}
          {q.reasoning && <span className="fc-cat fc-reasoning-tag">{t(S.reasoningTag)}</span>}
          {a.flagged && <span className="fc-cat" style={{ background: "rgba(240,160,32,.2)", color: "#f0a020" }}>{t(S.flagged)}</span>}
        </div>
        <p className="ex-question">{q.q}</p>
        <ul className="ex-options">
          {q.options.map((opt, i) => (
            <li key={i}>
              <button className={`ex-opt ${a.chosen === i ? "selected" : ""}`} onClick={() => setAnswer(i)}>
                <span className="ex-opt-letter">{String.fromCharCode(65 + i)}</span>{opt}
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="ex-footer">
        <button onClick={() => setCurrent(c => Math.max(0, c - 1))} disabled={current === 0}>{t(S.prev)}</button>
        <button className={a.flagged ? "on" : ""} onClick={toggleFlag}>{a.flagged ? t(S.flagged) : t(S.flag)}</button>
        {current < pool.length - 1
          ? <button onClick={() => setCurrent(c => c + 1)}>{t(S.next)}</button>
          : <button className="ex-start" onClick={() => finish()}>{t(S.finish)}</button>}
      </div>

      <div className="ex-q-strip">
        {pool.map((_, i) => (
          <button key={i}
            className={`ex-strip-btn ${i === current ? "active" : ""} ${answers[i].chosen !== null ? "done" : ""} ${answers[i].flagged ? "flagged" : ""}`}
            onClick={() => setCurrent(i)}>{i + 1}</button>
        ))}
      </div>
    </div>
  );
}
