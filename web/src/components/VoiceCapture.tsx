import { useState } from "react";
import { mediaErrorMessage, recordVoice } from "../media";
import { CheckIcon } from "./Icons";

const SECONDS = 5;
export const VOICE_PROMPT = "Kids are talking by the door.";

type State = "idle" | "waiting" | "recording" | "processing" | "done";

export function VoiceCapture({ voice, onVoice }: { voice: Blob | null; onVoice: (b: Blob | null) => void }) {
  const [state, setState] = useState<State>(voice ? "done" : "idle");
  const [level, setLevel] = useState(0);
  const [remaining, setRemaining] = useState(SECONDS);
  const [error, setError] = useState("");
  const [hint, setHint] = useState("");

  const record = async () => {
    setError("");
    setHint("");
    onVoice(null);
    setRemaining(SECONDS);
    setState("waiting");
    let timer: ReturnType<typeof setInterval> | undefined;
    try {
      const rec = await recordVoice(SECONDS, setLevel, () => {
        setState("recording");
        timer = setInterval(() => setRemaining((r) => Math.max(0, r - 1)), 1000);
      });
      clearInterval(timer);
      setState("processing");
      if (rec.peak < 0.02) {
        setHint("We could barely hear you. Try again, a little closer to the microphone.");
        setState("idle");
        return;
      }
      onVoice(rec.wav);
      setState("done");
    } catch (e) {
      clearInterval(timer);
      setError(mediaErrorMessage(e, "microphone"));
      setState("idle");
    }
  };

  return (
    <div className="capture">
      <p className="prompt-label">Read this sentence aloud, the way you would say it right now:</p>
      <blockquote className="prompt">“{VOICE_PROMPT}”</blockquote>

      {state === "recording" || state === "waiting" ? (
        <div className="recording" role="status" aria-live="polite">
          <span className="rec-dot" />
          <span>{state === "waiting" ? "Starting microphone…" : `Recording… ${remaining}s`}</span>
          <span className="meter" aria-hidden>
            <span style={{ width: `${Math.round(level * 100)}%` }} />
          </span>
        </div>
      ) : state === "processing" ? (
        <div className="recording" role="status">Saving your recording…</div>
      ) : state === "done" ? (
        <div className="captured-row">
          <span className="ok"><CheckIcon /> Voice recorded</span>
          <button className="btn link" onClick={record}>Record again</button>
        </div>
      ) : (
        <button className="btn secondary" onClick={record}>
          <span className="rec-dot idle" /> Record {SECONDS} seconds
        </button>
      )}

      {hint && <p className="hint">{hint}</p>}
      {error && <p className="field-error">{error}</p>}
    </div>
  );
}
