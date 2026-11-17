import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { ApiError, type CheckIn, createSession, submitCheckIn } from "./api";
import { Breakdown } from "./components/Breakdown";
import { Breathing } from "./components/Breathing";
import { CameraCapture } from "./components/CameraCapture";
import { Gauge } from "./components/Gauge";
import { ClockIcon, FaceIcon, LockIcon, Logo, SkipIcon, VoiceIcon, WordsIcon } from "./components/Icons";
import { Trend } from "./components/Trend";
import { VoiceCapture } from "./components/VoiceCapture";
import { LEVEL_COPY, levelOf } from "./levels";
import "./styles.css";

type Screen = "welcome" | "checkin" | "result";

function App() {
  const [screen, setScreen] = useState<Screen>("welcome");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [history, setHistory] = useState<CheckIn[]>([]);
  const [text, setText] = useState("");
  const [voice, setVoice] = useState<Blob | null>(null);
  const [photo, setPhoto] = useState<Blob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [breathing, setBreathing] = useState(false);

  const added = [text.trim() !== "", voice !== null, photo !== null].filter(Boolean).length;
  const latest = history[history.length - 1];

  const begin = async () => {
    setError("");
    setBusy(true);
    try {
      const sid = sessionId ?? (await createSession());
      setSessionId(sid);
      setScreen("checkin");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const submit = async () => {
    if (!sessionId || added === 0) return;
    setError("");
    setBusy(true);
    try {
      const { result, sessionId: sid } = await submitCheckIn(sessionId, { text, voice, photo });
      setSessionId(sid);
      if (result.fused === null) {
        setError("We couldn't read any of your signals this time. Please check the notes below each one and try again.");
        return;
      }
      setHistory((h) => [...h, result]);
      setScreen("result");
      window.scrollTo({ top: 0 });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const again = () => {
    setText("");
    setVoice(null);
    setPhoto(null);
    setError("");
    setScreen("checkin");
    window.scrollTo({ top: 0 });
  };

  return (
    <div className="shell">
      <header className="topbar">
        <button className="brand" onClick={() => setScreen("welcome")} aria-label="SafeSpace home">
          <Logo /> <span>SafeSpace</span>
        </button>
      </header>

      <main>
        {screen === "welcome" && (
          <section className="welcome">
            <h1>How are you, really?</h1>
            <p className="lede">
              A one-minute stress check-in. Tell us how you feel, read one sentence aloud and take a quick
              selfie. SafeSpace combines what it picks up into a gentle reading of your stress, and suggests
              something that might help.
            </p>
            <ul className="features">
              <li><ClockIcon /><div><b>About a minute</b><span>Three quick steps.</span></div></li>
              <li><SkipIcon /><div><b>Skip anything</b><span>Use one, two or all three.</span></div></li>
              <li><LockIcon /><div><b>Nothing is saved</b><span>Your words, voice and photo are deleted right after analysis.</span></div></li>
            </ul>
            <button className="btn primary big" onClick={begin} disabled={busy}>
              {busy ? "Starting…" : "Start check-in"}
            </button>
            {error && <p className="banner error">{error}</p>}
          </section>
        )}

        {screen === "checkin" && (
          <section className="checkin">
            <div className="section-head">
              <h1>Your check-in</h1>
              <p className="muted">Add any or all of these. More signals give a more reliable reading.</p>
            </div>

            <article className="card">
              <header><span className="card-icon"><WordsIcon /></span><h2>How are you feeling?</h2><span className="optional">Optional</span></header>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                maxLength={1000}
                placeholder="A sentence or two is plenty. For example: busy week, a bit on edge about tomorrow."
                aria-label="How are you feeling?"
              />
            </article>

            <article className="card">
              <header><span className="card-icon"><VoiceIcon /></span><h2>Your voice</h2><span className="optional">Optional</span></header>
              <VoiceCapture voice={voice} onVoice={setVoice} />
            </article>

            <article className="card">
              <header><span className="card-icon"><FaceIcon /></span><h2>Your expression</h2><span className="optional">Optional</span></header>
              <CameraCapture photo={photo} onPhoto={setPhoto} />
            </article>

            {error && <p className="banner error">{error}</p>}

            <div className="submit-bar">
              <span className="muted">{added} of 3 added</span>
              <button className="btn primary" onClick={submit} disabled={busy || added === 0}>
                {busy ? "Reading your check-in…" : "See my results"}
              </button>
            </div>
          </section>
        )}

        {screen === "result" && latest && latest.fused !== null && (
          <section className="result">
            <p className="eyebrow">Your check-in{history.length > 1 ? ` · ${history.length} this session` : ""}</p>
            <h1>{LEVEL_COPY[levelOf(latest.fused)].title}</h1>
            <Gauge score={latest.fused} />

            {latest.message && <div className={`message ${levelOf(latest.fused).toLowerCase()}`}>{latest.message}</div>}

            <div className="actions">
              <button className={`btn ${levelOf(latest.fused) === "Low" ? "secondary" : "primary"}`} onClick={() => setBreathing(true)}>
                Try a 1-minute breathing exercise
              </button>
            </div>

            <h2 className="sub">What SafeSpace picked up</h2>
            <Breakdown result={latest} earlier={history.slice(0, -1)} />
            {latest.carried.length > 0 && (
              <p className="note">
                Your result also counts readings from your last check-in. They fade out over a few minutes, so
                each new check-in moves your result gradually.
              </p>
            )}
            {latest.spread !== null && latest.spread > 0.25 && (
              <p className="note">Your signals disagreed with each other, so this reading is less certain than usual.</p>
            )}

            {history.length > 1 && (
              <>
                <h2 className="sub">This session</h2>
                <Trend scores={history.map((h) => h.fused as number)} />
              </>
            )}

            <div className="actions">
              <button className="btn primary" onClick={again}>Check in again</button>
              <button className="btn ghost" onClick={() => setScreen("welcome")}>Finish</button>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>
          SafeSpace is a research prototype, not a medical device, and can be wrong. If you feel unsafe or
          in crisis, contact your local emergency number or a crisis line.
        </p>
      </footer>

      {breathing && <Breathing onClose={() => setBreathing(false)} />}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
