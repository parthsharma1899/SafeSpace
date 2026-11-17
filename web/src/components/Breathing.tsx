import { useEffect, useState } from "react";

const PHASES = [
  { label: "Breathe in", seconds: 4, grow: true },
  { label: "Hold", seconds: 4, grow: true },
  { label: "Breathe out", seconds: 6, grow: false },
];
const ROUNDS = 4;

// A guided breathing exercise: four rounds of in 4 s, hold 4 s, out 6 s.
export function Breathing({ onClose }: { onClose: () => void }) {
  const [round, setRound] = useState(1);
  const [phase, setPhase] = useState(0);
  const [done, setDone] = useState(false);
  const [started, setStarted] = useState(false);

  // Mount small, then grow on the next frame so the first breath animates too.
  useEffect(() => {
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setStarted(true)));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    if (done) return;
    const id = setTimeout(() => {
      if (phase < PHASES.length - 1) setPhase(phase + 1);
      else if (round < ROUNDS) {
        setRound(round + 1);
        setPhase(0);
      } else setDone(true);
    }, PHASES[phase].seconds * 1000);
    return () => clearTimeout(id);
  }, [phase, round, done]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const current = PHASES[phase];
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Breathing exercise">
      <div className="modal breathing">
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        {done ? (
          <>
            <div className="breath-circle small" />
            <h2>Nicely done</h2>
            <p className="muted">Notice how you feel now. You can check in again whenever you like.</p>
            <button className="btn primary" onClick={onClose}>Done</button>
          </>
        ) : (
          <>
            <div
              className={`breath-circle ${started && current.grow ? "grow" : "shrink"}`}
              style={{ transitionDuration: `${current.seconds}s` }}
            />
            <h2 aria-live="polite">{current.label}</h2>
            <p className="muted">Round {round} of {ROUNDS}</p>
          </>
        )}
      </div>
    </div>
  );
}
