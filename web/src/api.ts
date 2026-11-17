// Client for the SafeSpace API. In development Vite proxies /api to the backend,
// so the app and the API share an origin. Set VITE_API_URL to point elsewhere.
//
// Two kinds of check-in:
// - signed in (`/me/...`): the request carries the Supabase access token and the
//   scores are saved, so the person's baseline builds across visits;
// - guest (`/sessions/...`): nothing is saved.

const BASE = import.meta.env.VITE_API_URL ?? "/api";

export type Signal = "text" | "voice" | "face" | "wrist";
export type Level = "Low" | "Medium" | "High";

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
  estimate_level: Level | null;
  /** Alert level: follows the change from the person's normal once it is learned. */
  level: Level;
  /** The person's normal level (average of their baselines for the signals used). */
  normal: number | null;
  /** How far this reading is from the person's normal. */
  change: number | null;
  baseline_ready: boolean;
  /** Readings so far toward a learned baseline, out of Config.baseline_readings. */
  baseline_progress: number;
  /** Readings so far for each signal ever used, capped at Config.baseline_readings. */
  baseline_counts: Partial<Record<Signal, number>>;
  checkin_number: number;
  message: string;
}

export interface HistoryItem {
  t: number;
  fused: number | null;
  change: number | null;
  baseline_ready: boolean;
  baseline_progress: number;
  baseline_counts: Partial<Record<Signal, number>>;
  estimate_level: Level | null;
}

export interface Config {
  auth: { url: string; publishable_key: string } | null;
  baseline_readings: number;
  demo: boolean;
}

export interface Me {
  consented: boolean;
  checkins: number;
  baseline_readings: number;
  latest: HistoryItem | null;
}

export interface CheckInInput {
  text?: string;
  voice?: Blob | null;
  photo?: Blob | null;
}

export class ApiError extends Error {
  constructor(message: string, readonly status = 0) {
    super(message);
  }
}

const UNREACHABLE = "We couldn't reach the SafeSpace service. Please check your connection and try again.";

async function call(path: string, init: RequestInit = {}, token?: string): Promise<Response> {
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(UNREACHABLE);
  }
  if (res.ok) return res;
  if (res.status === 401) throw new ApiError("Your sign-in has expired. Please sign in again.", 401);
  if (res.status === 413) throw new ApiError("That recording or photo is too large. Please try a shorter clip.", 413);
  if (res.status === 503) throw new ApiError("Sign-in can't be checked right now. Please try again in a moment.", 503);
  throw new ApiError("Something went wrong. Please try again.", res.status);
}

function form(input: CheckInInput): FormData {
  const f = new FormData();
  if (input.text?.trim()) f.append("text", input.text.trim());
  if (input.voice) f.append("audio", input.voice, "voice.wav");
  if (input.photo) f.append("image", input.photo, "photo.jpg");
  return f;
}

export async function getConfig(): Promise<Config> {
  return (await call("/config")).json();
}

// ---- signed in ------------------------------------------------------------------

export async function getMe(token: string): Promise<Me> {
  return (await call("/me", {}, token)).json();
}

export async function giveConsent(token: string): Promise<void> {
  await call("/me/consent", { method: "POST" }, token);
}

export async function submitAccountCheckIn(token: string, input: CheckInInput): Promise<CheckIn> {
  return (await call("/me/checkins", { method: "POST", body: form(input) }, token)).json();
}

export async function getHistory(token: string): Promise<HistoryItem[]> {
  return (await (await call("/me/history", {}, token)).json()).checkins;
}

export async function deleteMyData(token: string): Promise<number> {
  return (await (await call("/me", { method: "DELETE" }, token)).json()).deleted_checkins;
}

// ---- guest ------------------------------------------------------------------------

export async function createSession(): Promise<string> {
  return (await (await call("/sessions", { method: "POST" })).json()).session_id;
}

// Submits a guest check-in. If the session has expired (30 minutes idle), starts a
// new one and retries once; returns the session id that was actually used.
export async function submitGuestCheckIn(sessionId: string, input: CheckInInput): Promise<{ result: CheckIn; sessionId: string }> {
  try {
    return { result: await (await call(`/sessions/${sessionId}/observations`, { method: "POST", body: form(input) })).json(), sessionId };
  } catch (e) {
    if (!(e instanceof ApiError) || e.status !== 404) throw e;
  }
  const sid = await createSession();
  return { result: await (await call(`/sessions/${sid}/observations`, { method: "POST", body: form(input) })).json(), sessionId: sid };
}
