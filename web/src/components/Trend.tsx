import { levelOf } from "../levels";

// Small line of this session's check-ins, oldest to newest, on a fixed 0-100 scale.
export function Trend({ scores }: { scores: number[] }) {
  const w = 320;
  const h = 84;
  const pad = 10;
  const x = (i: number) => pad + (scores.length === 1 ? 0 : (i / (scores.length - 1)) * (w - 2 * pad));
  const y = (v: number) => pad + (1 - v) * (h - 2 * pad);
  const line = scores.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");

  return (
    <figure className="trend">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Stress across ${scores.length} check-ins: ${scores.map((s) => Math.round(s * 100)).join(", ")}`}>
        <line x1={pad} x2={w - pad} y1={y(0.4)} y2={y(0.4)} className="trend-guide" />
        <line x1={pad} x2={w - pad} y1={y(0.7)} y2={y(0.7)} className="trend-guide" />
        <path d={line} className="trend-line" />
        {scores.map((v, i) => (
          <circle key={i} cx={x(i)} cy={y(v)} r={i === scores.length - 1 ? 5.5 : 4} className={`trend-dot ${levelOf(v).toLowerCase()}`} />
        ))}
      </svg>
      <figcaption>
        {scores.map((s, i) => (
          <span key={i}>{Math.round(s * 100)}</span>
        ))}
      </figcaption>
    </figure>
  );
}
