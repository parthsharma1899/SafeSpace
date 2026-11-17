import type { Session } from "@supabase/supabase-js";
import { StrictMode, useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ApiError,
  type CheckIn,
  type Config,
  createSession,
  deleteMyData,
  getConfig,
  getHistory,
  getMe,
  giveConsent,
  type HistoryItem,
  type Me,
  submitAccountCheckIn,
  submitGuestCheckIn,
} from "./api";
import { accessToken, initAuth, onAuthChange, signOut } from "./auth";
import { BaselineProgress, ChangeFromNormal } from "./components/Baseline";
import { Breakdown } from "./components/Breakdown";
import { Breathing } from "./components/Breathing";
import { CameraCapture } from "./components/CameraCapture";
import { Gauge } from "./components/Gauge";
import { ChartIcon, ClockIcon, FaceIcon, LockIcon, Logo, SkipIcon, UserIcon, VoiceIcon, WordsIcon } from "./components/Icons";
import { Trend } from "./components/Trend";
import { VoiceCapture } from "./components/VoiceCapture";
import { LEVEL_COPY, levelOf } from "./levels";
import { Account, Consent } from "./screens/Account";
import { NewPassword, SignIn } from "./screens/SignIn";
import "./styles.css";

type Screen = "loading" | "welcome" | "signin" | "newpassword" | "consent" | "home" | "checkin" | "result" | "account";
type Mode = "account" | "guest";

const errorText = (e: unknown) => (e instanceof ApiError || e instanceof Error ? e.message : "Something went wrong. Please try again.");

function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [screen, setScreen] = useState<Screen>("loading");
  const [mode, setMode] = useState<Mode>("guest");
  const [session, setSession] = useState<Session | null>(null);
  const [me, setMe] = useState<Me | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [guestId, setGuestId] = useState<string | null>(null);
  const [guestHistory, setGuestHistory] = useState<CheckIn[]>([]);
  const [result, setResult] = useState<CheckIn | null>(null);
  const [text, setText] = useState("");
  const [voice, setVoice] = useState<Blob | null>(null);
  const [photo, setPhoto] = useState<Blob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [breathing, setBreathing] = useState(false);
  // The signed-in user id last routed for; undefined until the first auth event.
  const userId = useRef<string | null | undefined>(undefined);

  const go = (s: Screen) => {
    setError("");
    setScreen(s);
    window.scrollTo({ top: 0 });
  };

  const token = async () => {
    const t = await accessToken();
    if (!t) throw new ApiError("Your sign-in has expired. Please sign in again.", 401);
    return t;
  };

  const loadAccount = useCallback(async () => {
    const t = await token();
    const [m, h] = await Promise.all([getMe(t), getHistory(t)]);
    setMe(m);
    setHistory(h);
    return m;
  }, []);

  // Start-up: read the server's config, then restore any saved sign-in.
  useEffect(() => {
    let unsubscribe = () => {};
    getConfig()
      .then((cfg) => {
        setConfig(cfg);
        if (!cfg.auth) return setScreen("welcome");
        initAuth(cfg.auth.url, cfg.auth.publishable_key);
        unsubscribe = onAuthChange((s, recovering) => {
          setSession(s);
          if (recovering) return setScreen("newpassword");
          const id = s?.user.id ?? null;
          if (id === userId.current) return; // token refresh, nothing to route
          userId.current = id;
          if (!s) {
            setMe(null);
            setHistory([]);
            setMode("guest");
            return setScreen("welcome");
          }
          setMode("account");
          setScreen("loading");
          loadAccount()
            .then((m) => setScreen(m.consented ? "home" : "consent"))
            .catch((e) => {
              setError(errorText(e));
              setScreen("welcome");
            });
        });
      })
      .catch(() => {
        setConfig({ auth: null, baseline_readings: 6, demo: false });
        setError("We couldn't reach the SafeSpace service. Please check it is running and reload.");
        setScreen("welcome");
      });
    return () => unsubscribe();
  }, [loadAccount]);

  const added = [text.trim() !== "", voice !== null, photo !== null].filter(Boolean).length;
  const needed = config?.baseline_readings ?? 6;

  const startGuest = async () => {
    setError("");
    setBusy(true);
    try {
      setGuestId(guestId ?? (await createSession()));
      setMode("guest");
      go("checkin");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const agree = async () => {
    setBusy(true);
    setError("");
    try {
      await giveConsent(await token());
      await loadAccount();
      go("home");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const submit = async () => {
    if (added === 0) return;
    setError("");
    setBusy(true);
    try {
      let r: CheckIn;
      if (mode === "account") {
        r = await submitAccountCheckIn(await token(), { text, voice, photo });
      } else {
        const out = await submitGuestCheckIn(guestId ?? (await createSession()), { text, voice, photo });
        setGuestId(out.sessionId);
        r = out.result;
      }
      if (r.fused === null) {
        setError("We couldn't read any of your signals this time. Please check the notes below each one and try again.");
        return;
      }
      setResult(r);
      if (mode === "account") await loadAccount();
      else setGuestHistory((h) => [...h, r]);
      setText("");
      setVoice(null);
      setPhoto(null);
      go("result");
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const leave = async () => {
    await signOut();
    setResult(null);
  };

  const removeData = async () => {
    await deleteMyData(await token());
    setResult(null);
    await loadAccount();
    go("consent");
  };

  const homeScreen: Screen = mode === "account" && session ? "home" : "welcome";
  const scores = (mode === "account" ? history.map((h) => h.fused) : guestHistory.map((h) => h.fused)).filter((s): s is number => s !== null);

  return (
    <div className="shell">
      <header className="topbar">
        <button className="brand" onClick={() => go(homeScreen)} aria-label="SafeSpace home">
          <Logo /> <span>SafeSpace</span>
        </button>
        {session && mode === "account" && me?.consented && (
          <button className="btn ghost small" onClick={() => go("account")}><UserIcon /> Account</button>
        )}
      </header>

      {config?.demo && (
        <p className="demo-banner" role="status">
          Demo mode: the models aren't installed on this server, so every check-in returns the same sample scores.
        </p>
      )}

      <main>
        {screen === "loading" && <p className="muted center">Loading…</p>}

        {screen === "welcome" && (
          <section className="welcome">
            <h1>How are you, really?</h1>
            <p className="lede">
              A one-minute stress check-in. Tell us how you feel, read one sentence aloud and take a quick selfie.
              SafeSpace combines what it picks up into a gentle reading of your stress, and suggests something that
              might help.
            </p>
            <ul className="features">
              <li><ClockIcon /><div><b>About a minute</b><span>Three quick steps. Skip any of them.</span></div></li>
              <li><ChartIcon /><div><b>Learns your normal</b><span>With an account, each check-in is compared with your own usual level.</span></div></li>
              <li><LockIcon /><div><b>Your words, voice and photo are never kept</b><span>They're deleted right after analysis. Only scores are saved, and only if you sign in.</span></div></li>
            </ul>
            <div className="actions">
              {config?.auth ? (
                <>
                  <button className="btn primary big" onClick={() => go("signin")}>Sign in or create an account</button>
                  <button className="btn ghost big" onClick={startGuest} disabled={busy}><SkipIcon /> Try once without an account</button>
                </>
              ) : (
                <button className="btn primary big" onClick={startGuest} disabled={busy || !config}>{busy ? "Starting…" : "Start check-in"}</button>
              )}
            </div>
            {error && <p className="banner error">{error}</p>}
          </section>
        )}

        {screen === "signin" && <SignIn onBack={() => go("welcome")} />}
        {screen === "newpassword" && <NewPassword onDone={() => go(me?.consented ? "home" : "consent")} />}

        {screen === "consent" && <Consent onAgree={agree} onSignOut={leave} busy={busy} error={error} />}

        {screen === "home" && me && (
          <section className="home">
            <div className="section-head">
              <h1>{me.checkins === 0 ? "Your first check-in" : "Welcome back"}</h1>
              <p className="muted">
                {me.checkins === 0
                  ? "Each check-in teaches SafeSpace a little more about what's normal for you."
                  : `You've checked in ${me.checkins} time${me.checkins === 1 ? "" : "s"}.`}
              </p>
            </div>

            <button className="btn primary big" onClick={() => go("checkin")}>Start check-in</button>

            <article className="card">
              {me.latest?.baseline_ready ? (
                <>
                  <header><h2>Your last check-in</h2></header>
                  {me.latest.fused !== null && (
                    <p>
                      <b className="big-number">{Math.round(me.latest.fused * 100)}</b>{" "}
                      <span className="muted">· {LEVEL_COPY[levelOf(me.latest.fused)].short} · {when(me.latest.t)}</span>
                    </p>
                  )}
                  {me.latest.change !== null && <ChangeFromNormal change={me.latest.change} />}
                </>
              ) : (
                <BaselineProgress counts={me.latest?.baseline_counts ?? {}} needed={me.baseline_readings} />
              )}
            </article>

            {scores.length > 1 && (
              <>
                <h2 className="sub">Your recent check-ins</h2>
                <Trend scores={scores.slice(-12)} />
              </>
            )}
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

            {mode === "guest" && <p className="note">You're checking in as a guest: nothing is saved, and results aren't compared with your normal.</p>}
            {error && <p className="banner error">{error}</p>}

            <div className="submit-bar">
              <span className="muted">{added} of 3 added</span>
              <button className="btn primary" onClick={submit} disabled={busy || added === 0}>
                {busy ? "Reading your check-in…" : "See my results"}
              </button>
            </div>
          </section>
        )}

        {screen === "result" && result && result.fused !== null && (
          <section className="result">
            <p className="eyebrow">
              {mode === "account" ? `Check-in ${result.checkin_number}` : `Guest check-in${guestHistory.length > 1 ? ` · ${guestHistory.length} this visit` : ""}`}
            </p>
            <h1>{LEVEL_COPY[levelOf(result.fused)].title}</h1>
            <Gauge score={result.fused} />

            {mode === "account" &&
              (result.baseline_ready && result.change !== null ? (
                <ChangeFromNormal change={result.change} />
              ) : (
                <article className="card">
                  <BaselineProgress counts={result.baseline_counts} needed={needed} />
                </article>
              ))}

            {result.message && <div className={`message ${levelOf(result.fused).toLowerCase()}`}>{result.message}</div>}

            <div className="actions">
              <button className={`btn ${levelOf(result.fused) === "Low" ? "secondary" : "primary"}`} onClick={() => setBreathing(true)}>
                Try a 1-minute breathing exercise
              </button>
            </div>

            <h2 className="sub">What SafeSpace picked up</h2>
            <Breakdown result={result} earlier={guestHistory.slice(0, -1)} />
            {result.spread !== null && result.spread > 0.25 && (
              <p className="note">Your signals disagreed with each other, so this reading is less certain than usual.</p>
            )}

            {scores.length > 1 && (
              <>
                <h2 className="sub">{mode === "account" ? "Your recent check-ins" : "This visit"}</h2>
                <Trend scores={scores.slice(-12)} />
              </>
            )}

            {mode === "guest" && config?.auth && (
              <p className="note">
                Want SafeSpace to learn your normal? <button className="btn link inline" onClick={() => go("signin")}>Create a free account</button>
              </p>
            )}

            <div className="actions">
              <button className="btn primary" onClick={() => go("checkin")}>Check in again</button>
              <button className="btn ghost" onClick={() => go(homeScreen)}>Done</button>
            </div>
          </section>
        )}

        {screen === "account" && session && me && (
          <Account email={session.user.email ?? ""} checkins={me.checkins} onDelete={removeData} onSignOut={leave} onBack={() => go("home")} />
        )}
      </main>

      <footer className="footer">
        <p>
          SafeSpace is a research prototype, not a medical device, and can be wrong. If you feel unsafe or in crisis,
          contact your local emergency number or a crisis line.
        </p>
      </footer>

      {breathing && <Breathing onClose={() => setBreathing(false)} />}
    </div>
  );
}

function when(t: number): string {
  const d = new Date(t * 1000);
  const days = Math.floor((Date.now() - d.getTime()) / 86_400_000);
  if (days < 1) return `today, ${d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
  if (days < 2) return "yesterday";
  return d.toLocaleDateString([], { day: "numeric", month: "short" });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
