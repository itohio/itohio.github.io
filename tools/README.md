# ITOHI Tools — source

Source of the browser tools published at https://itohio.sintra.site
(EASA A1/A3 and A2 CoC exam prep in EN/LT, Lithuanian RRT radio exams,
pre/post-flight checklists, RX Blind-Spot Viewer).

Next.js 14 App Router, static export (`output: 'export'`). No backend.

## Layout

- `src/app/page.tsx` — hash router and all hub pages
- `src/i18n.tsx` — EN/LT language context, toggle, UI strings, `localizeQuestions`
- `src/components/` — exam session (timed, proportional), flashcards (SM-2), Fly checklists + Quick Reference
- `src/data/a1a3/*.ts`, `src/data/a2.ts` — English question banks (source of truth for answer index)
- `src/data/lt/*.ts` — Lithuanian text keyed by question id; answer index is shared with EN
- `src/data/radio*.ts` — HAREC (EN) and RRT Klasė A/B (LT, from va1da5/ham-radio-exam-prep-ltu, MIT)
- `public/legacy/` — frozen build snapshot of the RX Blind-Spot Viewer. Its React source was lost;
  the viewer is embedded from this snapshot via `src/components/RxViewerClient.tsx`.

## Build

```
npm install
npm run dev     # http://localhost:3000
npm run build   # static export to out/
```

Maintenance rule: this directory is the source of truth. Edit here, commit, then republish.
