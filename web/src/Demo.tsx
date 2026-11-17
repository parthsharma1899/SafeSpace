import { useRef, useState } from "react";

const DEFAULT_API = "http://127.0.0.1:8080";
const LEVEL_COLOR: Record<string, string> = { Low: "#0ca30c", Medium: "#eda100", High: "#d03b3b" };

interface Obs {
  t: number;
  fused: number | null;
  level: string;
  message: string;
  modalities: Record<string, { available: boolean; score: number | null; reason: string }>;
  weights: Record<string, number>;
}

export function Demo() {
  const [api, setApi] = useState(DEFAULT_API);
  const [session, setSession] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [history, setHistory] = useState<Obs[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [imageBlob, setImageBlob] = useState<Blob | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const start = async () => {
    setError("");
    try {
      const r = await fetch(`${api}/sessions`, { method: "POST" });
      const body = await r.json();
      setSession(body.session_id);
      setHistory([]);
    } catch (e) {
      setError("Could not reach the API. Start it with: uvicorn safespace.api.app:app");
    }
  };

  const enableCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      setError("Camera unavailable (permission denied or no device).");
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 320;
    canvas.height = video.videoHeight || 240;
    canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((b) => setImageBlob(b), "image/jpeg");
  };

  const recordAudio = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = () => {
        setAudioBlob(new Blob(chunks, { type: "audio/webm" }));
        stream.getTracks().forEach((t) => t.stop());
      };
      rec.start();
      setTimeout(() => rec.stop(), 4000);
    } catch {
      setError("Microphone unavailable (permission denied or no device).");
    }
  };

  const submit = async () => {
    if (!session) return;
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      if (text.trim()) form.append("text", text);
      if (imageBlob) form.append("image", imageBlob, "face.jpg");
      if (audioBlob) form.append("audio", audioBlob, "voice.webm");
      const r = await fetch(`${api}/sessions/${session}/observations`, { method: "POST", body: form });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const obs: Obs = await r.json();
      setHistory((h) => [...h, obs]);
      setImageBlob(null);
      setAudioBlob(null);
    } catch (e) {
      setError(`Check-in failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="demo">
      <div className="api-row">
        <label>
          API
          <input value={api} onChange={(e) => setApi(e.target.value)} />
        </label>
        {session ? <span className="pill">session {session.slice(0, 8)}…</span> : <button onClick={start}>Start session</button>}
      </div>

      {error && <p className="error">{error}</p>}

      {session && (
        <>
          <div className="checkin">
            <label className="field">
              <span>How are you feeling?</span>
              <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="A sentence or two…" />
            </label>

            <div className="media">
              <div className="cam">
                <video ref={videoRef} muted playsInline />
                <div className="btns">
                  <button onClick={enableCamera}>Enable camera</button>
                  <button onClick={capturePhoto}>Capture photo {imageBlob ? "✓" : ""}</button>
                </div>
              </div>
              <div className="mic">
                <button onClick={recordAudio}>Record 4s voice {audioBlob ? "✓" : ""}</button>
              </div>
            </div>

            <button className="submit" onClick={submit} disabled={busy}>
              {busy ? "Analysing…" : "Check in"}
            </button>
          </div>

          <div className="timeline">
            {history.length === 0 && <p className="muted">No check-ins yet.</p>}
            {history.map((obs, i) => (
              <div className="obs" key={i}>
                <div className="obs-head">
                  <span className="level" style={{ background: LEVEL_COLOR[obs.level] }}>{obs.level}</span>
                  <span className="score">{obs.fused === null ? "—" : obs.fused.toFixed(3)}</span>
                  <span className="t">t+{obs.t}s</span>
                </div>
                <div className="mods">
                  {Object.entries(obs.modalities).map(([m, d]) => (
                    <span key={m} className={d.available ? "mod on" : "mod off"}>
                      {m}: {d.available ? d.score?.toFixed(2) : d.reason}
                      {obs.weights[m] !== undefined && d.available ? ` (${(obs.weights[m] * 100).toFixed(0)}%)` : ""}
                    </span>
                  ))}
                </div>
                {obs.message && <p className="message">{obs.message}</p>}
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
