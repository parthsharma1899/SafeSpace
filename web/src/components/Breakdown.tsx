import type { CheckIn, Signal } from "../api";
import { friendlyReason, LEVEL_COPY, levelOf, SIGNAL_NAMES } from "../levels";
import { FaceIcon, PulseIcon, VoiceIcon, WordsIcon } from "./Icons";

const ICONS: Record<Signal, () => JSX.Element> = {
  text: WordsIcon,
  voice: VoiceIcon,
  face: FaceIcon,
  wrist: PulseIcon,
};

// What each signal read, and how much it counted toward the overall score.
// `earlier` holds this session's previous check-ins, used to show carried-over readings.
export function Breakdown({ result, earlier = [] }: { result: CheckIn; earlier?: CheckIn[] }) {
  const shown: Signal[] = ["text", "voice", "face"];
  if (result.modalities.wrist || result.carried.includes("wrist")) shown.push("wrist");

  return (
    <ul className="breakdown">
      {shown.map((s) => {
        const Icon = ICONS[s];
        const fresh = result.modalities[s];
        const carried = result.carried.includes(s);
        const prior = carried ? findPrior(earlier, s) : undefined;
        const score = fresh?.available ? fresh.score : prior;
        const used = score !== null && score !== undefined;
        const level = levelOf(score ?? 0);
        const weight = result.weights[s];
        return (
          <li key={s} className={used ? (carried ? "carried" : "") : "off"}>
            <span className="b-icon"><Icon /></span>
            <div className="b-body">
              <div className="b-head">
                <span className="b-name">{SIGNAL_NAMES[s]}</span>
                {used && (
                  <span className="b-value">
                    {Math.round(score * 100)} <small>· {LEVEL_COPY[level].short.toLowerCase()}</small>
                  </span>
                )}
              </div>
              {used ? (
                <>
                  <div className="b-bar" aria-hidden>
                    <span className={level.toLowerCase()} style={{ width: `${Math.max(3, score * 100)}%` }} />
                  </div>
                  {weight !== undefined && (
                    <span className="b-weight">
                      {carried ? "From your last check-in · " : ""}Counted for {Math.round(weight * 100)}% of your result
                    </span>
                  )}
                </>
              ) : (
                <span className="b-reason">{fresh ? friendlyReason(fresh.reason) : "Skipped"}</span>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

// The most recent earlier reading of this signal.
function findPrior(earlier: CheckIn[], s: Signal): number | undefined {
  for (let i = earlier.length - 1; i >= 0; i--) {
    const m = earlier[i].modalities[s];
    if (m?.available && m.score !== null) return m.score;
  }
  return undefined;
}
