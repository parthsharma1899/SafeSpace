import type { Signal } from "./api";

export type Level = "Low" | "Medium" | "High";

// The paper's thresholds: below 0.4 is Low, up to 0.7 is Medium, above is High.
export function levelOf(score: number): Level {
  if (score < 0.4) return "Low";
  if (score <= 0.7) return "Medium";
  return "High";
}

export const LEVEL_COPY: Record<Level, { title: string; short: string }> = {
  Low: { title: "You seem fairly calm right now", short: "Calm" },
  Medium: { title: "You seem a little tense", short: "Some tension" },
  High: { title: "You seem to be under a lot of stress", short: "High stress" },
};

export const SIGNAL_NAMES: Record<Signal, string> = {
  text: "Your words",
  voice: "Your voice",
  face: "Your expression",
  wrist: "Your body (wearable)",
};

const REASONS: Record<string, string> = {
  "no text": "Skipped",
  "no face detected": "No face found. Try facing the camera in better light.",
  "clip shorter than 1 s": "The recording was too short.",
  "clip is silent": "We couldn't hear anything in the recording.",
  "face model disabled": "Not available right now.",
  "could not read the recording": "We couldn't process the recording.",
  "could not read the photo": "We couldn't process the photo.",
  "could not process the text": "We couldn't process the text.",
  "no resting baseline": "Needs a 2-minute resting reading first.",
  "window shorter than 60 s": "Needs at least a minute of readings.",
  "signal too noisy": "The reading was too noisy.",
};

export function friendlyReason(reason: string): string {
  return REASONS[reason] ?? "Not available.";
}
