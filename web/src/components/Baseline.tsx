// How a signed-in person's reading compares with their own normal (v3's baseline).
import type { Signal } from "../api";
import { SIGNAL_NAMES } from "../levels";

const NEAR = 0.05;
const ORDER: Signal[] = ["text", "voice", "face", "wrist"];

// Each signal learns its own normal, so progress is shown per signal.
export function BaselineProgress({ counts, needed }: { counts: Partial<Record<Signal, number>>; needed: number }) {
  const used = ORDER.filter((s) => counts[s] !== undefined);
  return (
    <div className="baseline">
      <span className="b-name">Learning your normal</span>
      {used.length === 0 && <p className="muted">Your first check-in starts it.</p>}
      {used.map((s) => {
        const n = Math.min(counts[s] ?? 0, needed);
        return (
          <div key={s} className="baseline-row">
            <div className="b-head">
              <span>{SIGNAL_NAMES[s]}</span>
              <span className="muted">{n >= needed ? "Learned" : `${n} of ${needed}`}</span>
            </div>
            <div className="steps" role="progressbar" aria-valuemin={0} aria-valuemax={needed} aria-valuenow={n} aria-label={`${SIGNAL_NAMES[s]}: check-ins toward your normal`}>
              {Array.from({ length: needed }, (_, i) => <span key={i} className={i < n ? "done" : ""} />)}
            </div>
          </div>
        );
      })}
      <p className="note">
        After {needed} check-ins with a signal, SafeSpace knows your usual level for it, and results are compared with
        your normal. Until then, results are close to a simple average of your signals.
      </p>
    </div>
  );
}

export function ChangeFromNormal({ change }: { change: number }) {
  const points = Math.round(change * 100);
  const [label, tone] =
    Math.abs(change) < NEAR ? ["About your usual", "low"] : change > 0 ? ["Higher than your usual", change >= 0.2 ? "high" : "medium"] : ["Lower than your usual", "low"];
  return (
    <div className={`compare ${tone}`}>
      <span className="b-name">{label}</span>
      <span className="compare-value">{points > 0 ? "+" : ""}{points} <small>points</small></span>
    </div>
  );
}
