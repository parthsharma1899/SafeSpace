import { type FormEvent, useState } from "react";
import { sendPasswordReset, setNewPassword, signIn, signUp } from "../auth";

type Mode = "signin" | "signup" | "reset";

export function SignIn({ onBack }: { onBack: () => void }) {
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

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

  return (
    <section className="auth">
      <div className="section-head">
        <h1>{title}</h1>
        <p className="muted">
          {mode === "reset"
            ? "Enter your email and we'll send you a link."
            : "An account lets SafeSpace learn what's normal for you, so each check-in is compared with your own baseline."}
        </p>
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
