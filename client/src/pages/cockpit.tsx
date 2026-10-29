// Cockpit layout primitives (2026-10-29, human-asked on a 1920 monitor: "can we
// get it to fill the page … with how the UI looks on the bot instead of the
// strip"). Forecast, Brain and the ticker dossier share this shell: a page that
// fills the content width (CSS .vt-ck; index.css releases the app shell's 920px
// column for it), a KPI strip of hero tiles and a responsive panel grid in the
// AI Engine's visual language — its eyebrows (.vt-bot-eb), mono (.vt-bot-m),
// hero numbers (.vt-bot-hero) and tone classes. Layout only: every number and
// label is the page's own; nothing here reads data. Phone and tablet stack the
// same panels in one or two columns (index.css, the .vt-ck block).
import type { ReactNode } from "react";

export type Tone = "g" | "r" | "o" | "b" | "p" | "t3";
const TONE: Record<Tone, string> = { g: "vt-bot-g", r: "vt-bot-r", o: "vt-bot-o", b: "vt-bot-b", p: "vt-bot-p", t3: "vt-bot-t3" };
export const toneClass = (t?: Tone): string => (t ? TONE[t] : "");

/** The strip across the top: `n` tiles (4–6) laid out like the AI Engine's. */
export function KpiStrip({ n, label, children }: { n: number; label: string; children: ReactNode }) {
  return <div className="vt-ck-kpi" data-n={n} role="group" aria-label={label}>{children}</div>;
}

/** One tile: eyebrow, hero value (optional smaller suffix), one mono line.
 *  With `onClick` it is a real button (the Brain page's status filters). */
export function KpiTile({ label, value, sub, subTone, line, tone, title, onClick, pressed, status }: {
  label: string; value: string; sub?: string; subTone?: Tone; line: string; tone?: Tone; title?: string;
  onClick?: () => void; pressed?: boolean; status?: string;
}) {
  const body = (
    <>
      <span className="vt-bot-eb">{label}</span>
      <span className={`vt-bot-hero ${toneClass(tone)}`}>{value}{sub ? <span className={`vt-ck-sub2 ${toneClass(subTone)}`}>{sub}</span> : null}</span>
      <span className="vt-ck-line">{line}</span>
    </>
  );
  return onClick
    ? <button type="button" className="vt-ck-tile" title={title} onClick={onClick} aria-pressed={pressed} data-status={status}>{body}</button>
    : <div className="vt-ck-tile" title={title} data-status={status}>{body}</div>;
}

/** A grid panel in the bot's panel language: title row (an h2, so the page's
 *  heading order survives the grid), optional right-aligned meta, body. */
export function Panel({ id, title, meta, span, className, children, status }: {
  id: string; title: ReactNode; meta?: ReactNode; span?: 2 | "full"; className?: string; children: ReactNode; status?: string;
}) {
  const cls = ["vt-ck-panel", span === 2 ? "vt-ck-span2" : span === "full" ? "vt-ck-full" : "", className ?? ""].filter(Boolean).join(" ");
  return (
    <section className={cls} aria-labelledby={id} data-status={status}>
      <div className="vt-ck-ph"><h2 className="vt-ck-ti" id={id}>{title}</h2>{meta ? <span className="vt-ck-meta">{meta}</span> : null}</div>
      <div className="vt-ck-pb">{children}</div>
    </section>
  );
}

/** A full-width row inside the grid for a section heading and its controls. */
export function Bar({ id, title, meta, children }: { id: string; title: ReactNode; meta?: ReactNode; children?: ReactNode }) {
  return (
    <div className="vt-ck-bar">
      <h2 id={id}>{title}</h2>
      {meta ? <span className="vt-ck-meta">{meta}</span> : null}
      {children}
    </div>
  );
}
