import { useEffect, useState } from "react";
import { LEVEL_COPY, levelOf } from "../levels";

// Half-circle gauge from 0 to 100, with faint Low / Medium / High zones.
export function Gauge({ score }: { score: number }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(score));
    return () => cancelAnimationFrame(id);
  }, [score]);

  const level = levelOf(score);
  const r = 80;
  const cx = 100;
  const cy = 100;
  const half = Math.PI * r;
  const point = (frac: number, radius: number) => {
    const a = Math.PI * (1 - frac);
    return [cx + radius * Math.cos(a), cy - radius * Math.sin(a)];
  };
  const arc = (from: number, to: number, radius = r) => {
    const [x1, y1] = point(from, radius);
    const [x2, y2] = point(to, radius);
    return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${radius} ${radius} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
  };
  const zone = 95; // thin outer ring showing the Low / Medium / High bands
  const gap = 0.012;

  return (
    <figure className="gauge" aria-label={`Stress ${Math.round(score * 100)} out of 100, ${LEVEL_COPY[level].short}`}>
      <svg viewBox="0 0 200 118">
        <path d={arc(0, 1)} className="gauge-track" />
        <path d={arc(0, 0.4 - gap, zone)} className="gauge-zone low" />
        <path d={arc(0.4 + gap, 0.7 - gap, zone)} className="gauge-zone medium" />
        <path d={arc(0.7 + gap, 1, zone)} className="gauge-zone high" />
        <path
          d={arc(0, 1)}
          className={`gauge-value ${level.toLowerCase()}`}
          strokeDasharray={half}
          strokeDashoffset={half * (1 - shown)}
        />
      </svg>
      <div className="gauge-readout">
        <span className="gauge-number">{Math.round(score * 100)}</span>
        <span className={`gauge-level ${level.toLowerCase()}`}>{LEVEL_COPY[level].short}</span>
      </div>
      <div className="gauge-scale" aria-hidden>
        <span>0</span>
        <span>100</span>
      </div>
    </figure>
  );
}
