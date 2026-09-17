"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import type { Question } from "@/types/exam";
import { useLang, S, fmt } from "@/i18n";

// ── preference persistence ────────────────────────────────────────────────
function loadPrefs(prefix: string): { cat: string; doShuffle: boolean } {
  try { return JSON.parse(localStorage.getItem(`fc_prefs_${prefix}`) ?? "{}"); } catch { return { cat: "All", doShuffle: false }; }
}
function savePrefs(prefix: string, cat: string, doShuffle: boolean) {
  try { localStorage.setItem(`fc_prefs_${prefix}`, JSON.stringify({ cat, doShuffle })); } catch {}
}

// ── SM-2 Spaced Repetition ────────────────────────────────────────────────
interface SRCard {
  interval: number;    // days
  reps: number;
  ef: number;          // ease factor, starts 2.5
  due: number;         // timestamp ms
}

const SR_KEY = (prefix: string, id: string) => `sr_${prefix}_${id}`;
const DAY = 86_400_000;

function loadSR(prefix: string, id: string): SRCard | null {
  try {
    const raw = localStorage.getItem(SR_KEY(prefix, id));
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

function saveSR(prefix: string, id: string, card: SRCard) {
  try { localStorage.setItem(SR_KEY(prefix, id), JSON.stringify(card)); } catch {}
}

// quality: 0=Again, 1=Hard, 2=Good, 3=Easy  →  SM-2 q: 0, 3, 4, 5
const QUALITY_MAP = [0, 3, 4, 5];

function nextSR(prev: SRCard | null, quality: 0 | 1 | 2 | 3): SRCard {
  const q = QUALITY_MAP[quality];
  let { interval = 0, reps = 0, ef = 2.5 } = prev ?? {};
  if (q >= 3) {
    if (reps === 0) interval = 1;
    else if (reps === 1) interval = 6;
    else interval = Math.round(interval * ef);
    reps++;
  } else {
    interval = 1;
    reps = 0;
  }
  ef = Math.max(1.3, ef + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02));
  return { interval, reps, ef, due: Date.now() + interval * DAY };
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

interface Props { questions: Question[]; title: string; storagePrefix?: string; onBack?: () => void; }

type QueueItem = { q: Question; insertIdx: number; };

export default function FlashcardSession({ questions, title, storagePrefix = "fc", onBack }: Props) {
  const { t } = useLang();
  const categories = useMemo(
    () => ["All", ...Array.from(new Set(questions.map(q => q.category ?? "General"))).sort()],
    [questions]
  );
  const [cat, setCat] = useState(() => loadPrefs(storagePrefix).cat ?? "All");
  const [doShuffle, setDoShuffle] = useState(() => loadPrefs(storagePrefix).doShuffle ?? false);

  // SM-2 queue state
  const [queue, setQueue] = useState<Question[]>([]);
  const [queueIdx, setQueueIdx] = useState(0);
  const [againItems, setAgainItems] = useState<Question[]>([]); // cards to re-show
  const [sessionDone, setSessionDone] = useState(0);
  const [sessionAgain, setSessionAgain] = useState(0);

  // Card display state
  const [flipped, setFlipped] = useState(false);
  const [showExplain, setShowExplain] = useState(false);

  // SR data (live, reloaded on cat/shuffle change)
  const [srData, setSrData] = useState<Record<string, SRCard | null>>({});

  const prefix = `${storagePrefix}_${cat.replace(/\s+/g, "_")}`;

  const filtered = useMemo(() => {
    let qs = questions.filter(q => cat === "All" || q.category === cat);
    return doShuffle ? shuffle(qs) : qs;
  }, [questions, cat, doShuffle]);

  // Persist preferences
  useEffect(() => { savePrefs(storagePrefix, cat, doShuffle); }, [storagePrefix, cat, doShuffle]);

  // Build queue from SR state
  const buildQueue = useCallback(() => {
    const now = Date.now();
    const sr: Record<string, SRCard | null> = {};
    filtered.forEach(q => { sr[q.id] = loadSR(storagePrefix, q.id); });
    setSrData(sr);

    const due = filtered.filter(q => sr[q.id] && sr[q.id]!.due <= now);
    const newCards = filtered.filter(q => !sr[q.id]);
    const future = filtered.filter(q => sr[q.id] && sr[q.id]!.due > now);

    // Due first (sorted by overdue-ness), then new, then future
    const ordered = [
      ...due.sort((a, b) => (sr[a.id]?.due ?? 0) - (sr[b.id]?.due ?? 0)),
      ...newCards,
      ...future.sort((a, b) => (sr[a.id]?.due ?? 0) - (sr[b.id]?.due ?? 0)),
    ];
    setQueue(ordered);
    setQueueIdx(0);
    setAgainItems([]);
    setSessionDone(0);
    setSessionAgain(0);
    setFlipped(false);
    setShowExplain(false);
  }, [filtered, storagePrefix]);

  useEffect(() => { buildQueue(); }, [buildQueue]);

  const card = queue[queueIdx] ?? againItems[0] ?? null;
  const isFromAgain = queueIdx >= queue.length && againItems.length > 0;

  const totalInSession = useMemo(() => {
    const now = Date.now();
    return filtered.filter(q => {
      const s = srData[q.id];
      return !s || s.due <= now;
    }).length;
  }, [filtered, srData]);

  const dueCount = useMemo(() => {
    const now = Date.now();
    return filtered.filter(q => srData[q.id] && srData[q.id]!.due <= now).length;
  }, [filtered, srData]);

  const newCount = filtered.filter(q => !srData[q.id]).length;

  const rate = useCallback((quality: 0 | 1 | 2 | 3) => {
    if (!card) return;
    const prev = loadSR(storagePrefix, card.id);
    const next = nextSR(prev, quality);
    saveSR(storagePrefix, card.id, next);
    setSrData(d => ({ ...d, [card.id]: next }));

    if (quality === 0) {
      setAgainItems(a => [...a.filter(x => x.id !== card.id), card]);
      setSessionAgain(n => n + 1);
    } else {
      setSessionDone(n => n + 1);
      if (isFromAgain) {
        setAgainItems(a => a.filter(x => x.id !== card.id));
      } else {
        setQueueIdx(i => i + 1);
      }
    }
    setFlipped(false);
    setShowExplain(false);
  }, [card, storagePrefix, isFromAgain]);

  const flip = useCallback(() => setFlipped(f => !f), []);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (["INPUT","SELECT","TEXTAREA"].includes((e.target as HTMLElement).tagName)) return;
      if ((e.key === " " || e.key === "Enter") && !flipped) { e.preventDefault(); flip(); }
      if (flipped) {
        if (e.key === "1") rate(0);
        if (e.key === "2") rate(1);
        if (e.key === "3") rate(2);
        if (e.key === "4") rate(3);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [flip, rate, flipped]);

  const allDone = queueIdx >= queue.length && againItems.length === 0;

  return (
    <div className="fc-wrap">
      <div className="fc-header">
        <button className="fc-back-btn" onClick={onBack}>{t(S.back)}</button>
        <h1>{title} {t(S.fcTitleSuffix)}</h1>
        <div className="fc-controls">
          <label>
            {t(S.category)}
            <select value={cat} onChange={e => { setCat(e.target.value); }}>
              {categories.map(c => <option key={c} value={c}>{c === "All" ? t(S.all) : c}</option>)}
            </select>
          </label>
          <button className={doShuffle ? "on" : ""} onClick={() => setDoShuffle(s => !s)}>
            {doShuffle ? t(S.shuffled) : t(S.inOrder)}
          </button>
          <button onClick={buildQueue}>{t(S.restartSession)}</button>
        </div>
      </div>

      <div className="fc-sr-stats">
        <span className="sr-due">{t(S.due)}: {dueCount}</span>
        <span className="sr-new">{t(S.newCards)}: {newCount}</span>
        <span className="sr-learned">{t(S.learned)}: {filtered.filter(q => srData[q.id] && srData[q.id]!.reps > 0).length}</span>
        {againItems.length > 0 && <span className="sr-again">{t(S.again)}: {againItems.length}</span>}
        {sessionDone > 0 && <span className="sr-session">{fmt(t(S.thisSession), { n: sessionDone })}</span>}
      </div>

      {allDone ? (
        <div className="fc-done">
          <h2>{t(S.sessionComplete)}</h2>
          <p>{fmt(t(S.cardsReviewed), { n: sessionDone })} {sessionAgain > 0 ? fmt(t(S.wereHard), { n: sessionAgain }) : t(S.allCorrect)}</p>
          <p className="fc-done-sub">{t(S.comeBack)}</p>
          <button onClick={buildQueue}>{t(S.startNewSession)}</button>
          <button className="fc-back-btn" style={{marginLeft:"12px"}} onClick={onBack}>{t(S.backToModule)}</button>
        </div>
      ) : card ? (
        <>
          <div className="fc-prog-row">
            <span className="fc-prog-label">
              {isFromAgain ? `${t(S.again)} ${againItems.indexOf(card) + 1}/${againItems.length}` : `${queueIdx + 1}/${queue.length}`}
            </span>
            <div className="fc-prog-bar">
              <div className="fc-prog-fill" style={{ width: isFromAgain ? "100%" : `${((queueIdx) / queue.length) * 100}%` }} />
            </div>
          </div>

          <div className="fc-scene" onClick={!flipped ? flip : undefined}>
            <div className={`fc-card ${flipped ? "flipped" : ""}`}>
              <div className="fc-face fc-front">
                {card.category && <span className="fc-cat">{card.category}</span>}
                {card.reasoning && <span className="fc-cat fc-reasoning-tag">{t(S.reasoningTag)}</span>}
                {isFromAgain && <span className="fc-cat" style={{background:"rgba(248,81,73,.2)",color:"#f85149"}}>↺ {t(S.again)}</span>}
                <p className="fc-question">{card.q}</p>
                <span className="fc-hint">{t(S.clickToReveal)}</span>
              </div>
              <div className="fc-face fc-back">
                {card.category && <span className="fc-cat">{card.category}</span>}
                {card.reasoning && <span className="fc-cat fc-reasoning-tag">{t(S.reasoningTag)}</span>}
                <p className="fc-answer-label">{t(S.correctAnswer)}</p>
                <p className="fc-answer">{card.options[card.answer]}</p>
                <div className="fc-all-opts">
                  {card.options.map((opt, i) => (
                    <div key={i} className={`fc-opt-row ${i === card.answer ? "correct" : "wrong"}`}>
                      <span>{String.fromCharCode(65+i)}.</span> {opt}
                    </div>
                  ))}
                </div>
                {card.explanation && (
                  <>
                    <button className="fc-explain-btn" onClick={e => { e.stopPropagation(); setShowExplain(s => !s); }}>
                      {showExplain ? t(S.hideExplanation) : t(S.showExplanation)}
                    </button>
                    {showExplain && <p className="fc-explain">{card.explanation}</p>}
                  </>
                )}
              </div>
            </div>
          </div>

          {!flipped ? (
            <p className="fc-keys">{t(S.spaceToFlip)}</p>
          ) : (
            <>
              <div className="fc-rate-row">
                <button className="fc-rate fc-rate-again" onClick={() => rate(0)}>
                  <span>↺</span> {t(S.again)}
                  <small>{t(S.seeSoon)}</small>
                </button>
                <button className="fc-rate fc-rate-hard" onClick={() => rate(1)}>
                  <span>~</span> {t(S.hard)}
                  <small>{t(S.tomorrow)}</small>
                </button>
                <button className="fc-rate fc-rate-good" onClick={() => rate(2)}>
                  <span>✓</span> {t(S.good)}
                  <small>{srData[card.id] ? `~${Math.round((srData[card.id]!.interval || 1) * 2.5)}d` : "3d"}</small>
                </button>
                <button className="fc-rate fc-rate-easy" onClick={() => rate(3)}>
                  <span>★</span> {t(S.easy)}
                  <small>{t(S.longInterval)}</small>
                </button>
              </div>
              <p className="fc-keys">{t(S.keysHint)}</p>
            </>
          )}
        </>
      ) : (
        <div className="fc-empty">{t(S.noCards)}</div>
      )}
    </div>
  );
}
