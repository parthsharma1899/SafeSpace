import { useEffect, useRef, useState } from "react";
import { GuardedPaperWMSS } from "./fusion/v1";
import { ResidualPersistenceV2, TUNED } from "./fusion/v2";
import { SmoothedMean } from "./fusion/mean";
import { LineChart } from "./Chart";

const MODALITIES = ["wrist", "text", "voice", "face"];
const COLORS = { v2: "#2a78d6", v1: "#eb6834", mean: "#1baf7a" };

interface Row {
  v1: number | null;
  v2: number | null;
  mean: number | null;
}

export function Playground() {
  const [values, setValues] = useState<Record<string, number>>({ wrist: 0.5, text: 0.5, voice: 0.5, face: 0.5 });
  const [missing, setMissing] = useState<Record<string, boolean>>({ wrist: false, text: false, voice: false, face: false });
  const [rows, setRows] = useState<Row[]>([]);
  const [auto, setAuto] = useState(false);

  const engines = useRef({
    v1: new GuardedPaperWMSS(MODALITIES),
    v2: new ResidualPersistenceV2(MODALITIES, TUNED),
    mean: new SmoothedMean(MODALITIES),
  });
  const latest = useRef({ values, missing });
  latest.current = { values, missing };

  const step = () => {
    const { values: v, missing: m } = latest.current;
    const scores: Record<string, number> = {};
    for (const mod of MODALITIES) if (!m[mod]) scores[mod] = v[mod];
    const r1 = engines.current.v1.step({ ...scores });
    const r2 = engines.current.v2.step({ ...scores });
    const rm = engines.current.mean.step({ ...scores });
    setRows((prev) => [...prev.slice(-120), { v1: r1.score, v2: r2.score, mean: rm.score }]);
  };

  const reset = () => {
    engines.current = {
      v1: new GuardedPaperWMSS(MODALITIES),
      v2: new ResidualPersistenceV2(MODALITIES, TUNED),
      mean: new SmoothedMean(MODALITIES),
    };
    setRows([]);
  };

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(step, 400);
    return () => clearInterval(id);
  }, [auto]);

  const last = rows[rows.length - 1];

  return (
    <section className="playground">
      <p className="lede">
        Set each model's stress score, mark any as missing, and step the clock. Compare the paper's
        fusion (<b style={{ color: COLORS.v1 }}>v1</b>), the redesign (<b style={{ color: COLORS.v2 }}>v2</b>),
        and a smoothed mean (<b style={{ color: COLORS.mean }}>mean</b>). All three run in your browser.
      </p>

      <div className="sliders">
        {MODALITIES.map((m) => (
          <div className="slider-row" key={m}>
            <label>
              <span className="mod">{m}</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={values[m]}
                disabled={missing[m]}
                onChange={(e) => setValues({ ...values, [m]: parseFloat(e.target.value) })}
              />
              <span className="val">{missing[m] ? "—" : values[m].toFixed(2)}</span>
            </label>
            <label className="miss">
              <input type="checkbox" checked={missing[m]} onChange={(e) => setMissing({ ...missing, [m]: e.target.checked })} />
              missing
            </label>
          </div>
        ))}
      </div>

      <div className="controls">
        <button onClick={step} disabled={auto}>Step</button>
        <button className={auto ? "on" : ""} onClick={() => setAuto((a) => !a)}>
          {auto ? "Stop" : "Auto-run"}
        </button>
        <button onClick={reset}>Reset</button>
      </div>

      {last && (
        <div className="readout">
          <span style={{ color: COLORS.v1 }}>v1 {fmt(last.v1)}</span>
          <span style={{ color: COLORS.v2 }}>v2 {fmt(last.v2)}</span>
          <span style={{ color: COLORS.mean }}>mean {fmt(last.mean)}</span>
        </div>
      )}

      <LineChart
        series={[
          { name: "v1", color: COLORS.v1, points: rows.map((r) => r.v1) },
          { name: "mean", color: COLORS.mean, points: rows.map((r) => r.mean) },
          { name: "v2", color: COLORS.v2, points: rows.map((r) => r.v2) },
        ]}
      />

      <p className="note">
        Try: push one slider to 1.0 and the rest low — v1 jumps toward the loud one, v2 holds with the
        agreeing majority. Mark a modality missing — it is dropped, never guessed.
      </p>
    </section>
  );
}

function fmt(v: number | null) {
  return v === null ? "—" : v.toFixed(3);
}
