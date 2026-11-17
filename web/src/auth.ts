// Sign-in with Supabase Auth (Google, or email and password). The project URL and its
// publishable key come from the API's /config, so the web app has no build-time
// settings. Supabase keeps the session in the browser and refreshes it.

import { createClient, type Session, type SupabaseClient } from "@supabase/supabase-js";

let client: SupabaseClient | null = null;
let settings: { url: string; key: string } | null = null;

export function initAuth(url: string, publishableKey: string): SupabaseClient {
  settings = { url, key: publishableKey };
  client ??= createClient(url, publishableKey, {
    auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
  });
  return client;
}

/** Whether Google sign-in is switched on in the Supabase project. */
export async function googleEnabled(): Promise<boolean> {
  if (!settings) return false;
  try {
    const res = await fetch(`${settings.url}/auth/v1/settings`, { headers: { apikey: settings.key } });
    return res.ok && Boolean((await res.json()).external?.google);
  } catch {
    return false;
  }
}

/** True when the app runs inside another page's frame (e.g. the Hugging Face Space page). */
export function isFramed(): boolean {
  try {
    return window.self !== window.top;
  } catch {
    return true; // a cross-origin parent hides window.top
  }
}

/** Sends the browser to Google; Supabase brings it back here signed in. Google
 * refuses to load inside a frame, so a framed app opens sign-in in a new tab. */
export async function signInWithGoogle(): Promise<void> {
  if (isFramed()) {
    window.open(window.location.origin, "_blank", "noopener");
    return;
  }
  const { error } = await auth().signInWithOAuth({
    provider: "google",
    options: { redirectTo: window.location.origin },
  });
  if (error) throw new Error(friendly(error.message));
}

function auth() {
  if (!client) throw new Error("sign-in is not configured");
  return client.auth;
}

/** A current access token, refreshed by Supabase when needed. */
export async function accessToken(): Promise<string | null> {
  const { data } = await auth().getSession();
  return data.session?.access_token ?? null;
}

export async function signIn(email: string, password: string): Promise<void> {
  const { error } = await auth().signInWithPassword({ email, password });
  if (error) throw new Error(friendly(error.message));
}

/** Returns true when the account still needs its email confirmed. */
export async function signUp(email: string, password: string): Promise<boolean> {
  const { data, error } = await auth().signUp({
    email,
    password,
    options: { emailRedirectTo: window.location.origin },
  });
  if (error) throw new Error(friendly(error.message));
  return data.session === null;
}

export async function sendPasswordReset(email: string): Promise<void> {
  const { error } = await auth().resetPasswordForEmail(email, { redirectTo: window.location.origin });
  if (error) throw new Error(friendly(error.message));
}

export async function setNewPassword(password: string): Promise<void> {
  const { error } = await auth().updateUser({ password });
  if (error) throw new Error(friendly(error.message));
}

export async function signOut(): Promise<void> {
  await auth().signOut();
}

export function onAuthChange(handler: (session: Session | null, recovering: boolean) => void): () => void {
  const { data } = auth().onAuthStateChange((event, session) => handler(session, event === "PASSWORD_RECOVERY"));
  return () => data.subscription.unsubscribe();
}

function friendly(message: string): string {
  if (/invalid login credentials/i.test(message)) return "That email and password don't match. Please try again.";
  if (/email not confirmed/i.test(message)) return "Please confirm your email first. Check your inbox for the link.";
  if (/rate limit/i.test(message)) return "Too many attempts. Please wait a few minutes and try again.";
  if (/already registered/i.test(message)) return "There's already an account with this email. Try signing in.";
  return message;
}
