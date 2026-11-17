import { useState } from "react";
import { CheckIcon, LockIcon } from "../components/Icons";

export function Consent({ onAgree, onSignOut, busy, error }: { onAgree: () => void; onSignOut: () => void; busy: boolean; error: string }) {
  return (
    <section className="consent">
      <div className="section-head">
        <h1>Before your first check-in</h1>
        <p className="muted">SafeSpace learns what's normal for you by keeping your results over time.</p>
      </div>
      <article className="card">
        <ul className="plain-list">
          <li><CheckIcon /><span><b>What we keep:</b> the scores from each check-in (a number per signal) and when you checked in.</span></li>
          <li><LockIcon /><span><b>What we never keep:</b> your words, your voice recording or your photo. They are analysed and deleted straight away.</span></li>
          <li><CheckIcon /><span><b>Your control:</b> you can delete everything we've kept at any time from your account page.</span></li>
        </ul>
        <p className="note">
          SafeSpace is a research prototype, not a medical device. Its readings can be wrong and should not be used to
          make decisions about your health.
        </p>
      </article>
      {error && <p className="banner error" role="alert">{error}</p>}
      <div className="actions">
        <button className="btn primary" onClick={onAgree} disabled={busy}>{busy ? "Saving…" : "I agree, continue"}</button>
        <button className="btn ghost" onClick={onSignOut}>Sign out</button>
      </div>
    </section>
  );
}

export function Account({
  email,
  checkins,
  onDelete,
  onSignOut,
  onBack,
}: {
  email: string;
  checkins: number;
  onDelete: () => Promise<void>;
  onSignOut: () => void;
  onBack: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const remove = async () => {
    setBusy(true);
    setError("");
    try {
      await onDelete();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong. Please try again.");
      setBusy(false);
    }
  };

  return (
    <section className="account">
      <div className="section-head">
        <h1>Your account</h1>
        <p className="muted">Signed in as {email}</p>
      </div>

      <article className="card">
        <header><h2>Your data</h2></header>
        <p>
          SafeSpace has kept the scores from <b>{checkins}</b> check-in{checkins === 1 ? "" : "s"}. Never your words,
          voice or photos.
        </p>
        {!confirming ? (
          <button className="btn danger" onClick={() => setConfirming(true)}>Delete my data</button>
        ) : (
          <div className="confirm">
            <p><b>Delete all {checkins} saved check-ins?</b> SafeSpace will forget your baseline and start learning it again. This can't be undone.</p>
            <div className="actions">
              <button className="btn danger" onClick={remove} disabled={busy}>{busy ? "Deleting…" : "Yes, delete everything"}</button>
              <button className="btn ghost" onClick={() => setConfirming(false)} disabled={busy}>Cancel</button>
            </div>
          </div>
        )}
        {error && <p className="banner error" role="alert">{error}</p>}
      </article>

      <div className="actions">
        <button className="btn ghost" onClick={onBack}>Back</button>
        <button className="btn ghost" onClick={onSignOut}>Sign out</button>
      </div>
    </section>
  );
}
