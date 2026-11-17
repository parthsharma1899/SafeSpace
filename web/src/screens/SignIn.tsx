import { type FormEvent, useEffect, useState } from "react";
import { googleEnabled, sendPasswordReset, setNewPassword, signIn, signInWithGoogle, signUp } from "../auth";

// Google's "G" mark, as its sign-in branding asks.
const GoogleMark = () => (
  <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden>
    <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5z" />
    <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z" />
    <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-7.9l-6.5 5C9.5 39.6 16.2 44 24 44z" />
    <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C37 39.2 44 34 44 24c0-1.3-.1-2.4-.4-3.5z" />
  </svg>
);

type Mode = "signin" | "signup" | "reset";

export function SignIn({ onBack }: { onBack: () => void }) {
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [google, setGoogle] = useState(false);
  const [emailOpen, setEmailOpen] = useState(false);

  useEffect(() => {
    googleEnabled().then(setGoogle);
  }, []);

  const withGoogle = async () => {
    setError("");
    setBusy(true);
    try {
      await signInWithGoogle(); // the page navigates away to Google
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
      setBusy(false);
    }
  };

  const switchTo = (m: Mode) => {
    setMode(m);
    setError("");
    setNotice("");
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setNotice("");
    setBusy(true);
    try {
      if (mode === "signin") {
        await signIn(email.trim(), password);
      } else if (mode === "signup") {
        const needsConfirm = await signUp(email.trim(), password);
        if (needsConfirm) setNotice(`We've sent a confirmation link to ${email.trim()}. Open it, then sign in here.`);
      } else {
        await sendPasswordReset(email.trim());
        setNotice(`If there's an account for ${email.trim()}, we've sent it a link to set a new password.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const title = { signin: "Welcome back", signup: "Create your account", reset: "Reset your password" }[mode];
  const action = { signin: "Sign in", signup: "Create account", reset: "Send reset link" }[mode];
  const intro = "An account lets SafeSpace learn what's normal for you, so each check-in is compared with your own baseline.";

  // With Google on, it is the main way in; email stays available behind a link.
  if (google && !emailOpen) {
    return (
      <section className="auth">
        <div className="section-head">
          <h1>Sign in to SafeSpace</h1>
          <p className="muted">{intro}</p>
        </div>
        <div className="card form">
          <button className="btn google" onClick={withGoogle} disabled={busy}>
            <GoogleMark /> {busy ? "Opening Google…" : "Continue with Google"}
          </button>
          <p className="note">New here? Continuing with Google creates your account.</p>
          {error && <p className="banner error" role="alert">{error}</p>}
        </div>
        <div className="auth-links">
          <button className="btn link" onClick={() => setEmailOpen(true)}>Use email instead</button>
          <button className="btn link" onClick={onBack}>Back</button>
        </div>
      </section>
    );
  }

  return (
    <section className="auth">
      <div className="section-head">
        <h1>{title}</h1>
        <p className="muted">{mode === "reset" ? "Enter your email and we'll send you a link." : intro}</p>
      </div>

      <form className="card form" onSubmit={submit}>
        <label>
          <span>Email</span>
          <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        {mode !== "reset" && (
          <label>
            <span>Password</span>
            <input
              type="password"
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              required
              minLength={mode === "signup" ? 8 : undefined}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            {mode === "signup" && <small className="muted">At least 8 characters.</small>}
          </label>
        )}
        {error && <p className="banner error" role="alert">{error}</p>}
        {notice && <p className="banner info" role="status">{notice}</p>}
        <button className="btn primary" type="submit" disabled={busy}>
          {busy ? "Please wait…" : action}
        </button>
      </form>

      <div className="auth-links">
        {mode === "signin" && (
          <>
            <button className="btn link" onClick={() => switchTo("signup")}>New here? Create an account</button>
            <button className="btn link" onClick={() => switchTo("reset")}>Forgot password?</button>
          </>
        )}
        {mode !== "signin" && <button className="btn link" onClick={() => switchTo("signin")}>I already have an account</button>}
        {google && <button className="btn link" onClick={() => setEmailOpen(false)}>Continue with Google instead</button>}
        <button className="btn link" onClick={onBack}>Back</button>
      </div>
    </section>
  );
}

export function NewPassword({ onDone }: { onDone: () => void }) {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await setNewPassword(password);
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="auth">
      <div className="section-head">
        <h1>Choose a new password</h1>
      </div>
      <form className="card form" onSubmit={submit}>
        <label>
          <span>New password</span>
          <input type="password" autoComplete="new-password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} />
          <small className="muted">At least 8 characters.</small>
        </label>
        {error && <p className="banner error" role="alert">{error}</p>}
        <button className="btn primary" type="submit" disabled={busy}>{busy ? "Saving…" : "Save password"}</button>
      </form>
    </section>
  );
}
