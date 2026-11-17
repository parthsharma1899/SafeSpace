// A small inline-SVG line chart: one y-axis fixed to [0, 1], a legend, thin lines.

export interface Series {
  name: string;
  color: string;
  points: (number | null)[];
}

export function LineChart({
  series,
  window: windowSize = 60,
  height = 240,
}: {
  series: Series[];
  window?: number;
  height?: number;
}) {
  const w = 640;
  const h = height;
  const padL = 34;
  const padB = 22;
  const padT = 10;
  const padR = 10;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const n = windowSize;

  const x = (i: number) => padL + (n <= 1 ? 0 : (i / (n - 1)) * plotW);
  const y = (v: number) => padT + (1 - v) * plotH;

  const path = (points: (number | null)[]) => {
    const tail = points.slice(-n);
    let d = "";
    tail.forEach((v, i) => {
      if (v === null || Number.isNaN(v)) return;
      d += `${d && !d.endsWith("M") ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)} `;
    });
    return d.trim();
  };

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="fusion comparison">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line className="grid" x1={padL} x2={w - padR} y1={y(g)} y2={y(g)} />
            <text className="axis" x={padL - 6} y={y(g) + 3} textAnchor="end">
              {g}
            </text>
          </g>
        ))}
        {series.map((s) => (
          <path key={s.name} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={s.name === "v2" ? 2.4 : 1.6} />
        ))}
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.color }} /> {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}
