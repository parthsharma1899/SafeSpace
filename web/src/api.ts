// Client for the SafeSpace API. In development Vite proxies /api to the backend,
// so the app and the API share an origin. Set VITE_API_URL to point elsewhere.

const BASE = import.meta.env.VITE_API_URL ?? "/api";

export type Signal = "text" | "voice" | "face" | "wrist";

export interface SignalResult {
  available: boolean;
  score: number | null;
  reason: string;
}

export interface CheckIn {
  t: number;
  modalities: Partial<Record<Signal, SignalResult>>;
  fused: number | null;
  weights: Partial<Record<Signal, number>>;
  /** Signals not sent this time that still counted, carried from a recent check-in. */
  carried: Signal[];
  spread: number | null;
  /** Level of the current estimate (what to show now). */
  estimate_level: "Low" | "Medium" | "High" | null;
  /** Slow-to-clear alert level, for notifications. */
  level: "Low" | "Medium" | "High";
  message: string;
}

export interface CheckInInput {
  text?: string;
  voice?: Blob | null;
  photo?: Blob | null;
}

export class ApiError extends Error {}

export async function createSession(): Promise<string> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/sessions`, { method: "POST" });
  } catch {
    throw new ApiError("We couldn't reach the SafeSpace service. Please check it is running and try again.");
  }
  if (!res.ok) throw new ApiError("The SafeSpace service isn't responding right now. Please try again.");
  return (await res.json()).session_id;
}

async function post(sessionId: string, input: CheckInInput): Promise<Response> {
  const form = new FormData();
  if (input.text?.trim()) form.append("text", input.text.trim());
  if (input.voice) form.append("audio", input.voice, "voice.wav");
  if (input.photo) form.append("image", input.photo, "photo.jpg");
  return fetch(`${BASE}/sessions/${sessionId}/observations`, { method: "POST", body: form });
}

// Submits a check-in. If the session has expired (30 minutes idle), starts a new
// one and retries once; returns the session id that was actually used.
export async function submitCheckIn(sessionId: string, input: CheckInInput): Promise<{ result: CheckIn; sessionId: string }> {
  let sid = sessionId;
  let res: Response;
  try {
    res = await post(sid, input);
    if (res.status === 404) {
      sid = await createSession();
      res = await post(sid, input);
    }
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError("We couldn't reach the SafeSpace service. Please check your connection and try again.");
  }
  if (res.status === 413) throw new ApiError("That recording or photo is too large. Please try a shorter clip.");
  if (!res.ok) throw new ApiError("Something went wrong while analysing your check-in. Please try again.");
  return { result: await res.json(), sessionId: sid };
}
