// Browser media helpers: microphone recording encoded as WAV (so the server needs
// no audio codecs), and friendly messages for camera/microphone failures.

/** Encode mono float samples in [-1, 1] as a 16-bit PCM WAV file. */
export function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

export function peakLevel(samples: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]));
  return peak;
}

/** Turn a getUserMedia failure into a sentence a person can act on. */
export function mediaErrorMessage(err: unknown, device: "camera" | "microphone"): string {
  if (typeof window !== "undefined" && (!window.isSecureContext || !navigator.mediaDevices)) {
    return `Your browser only allows the ${device} on a secure page. Open SafeSpace at http://localhost:5173 instead.`;
  }
  const name = err instanceof DOMException ? err.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return `Access to your ${device} was blocked. Click the icon in your browser's address bar to allow it. On a Mac, also check System Settings → Privacy & Security → ${device === "camera" ? "Camera" : "Microphone"}. Then try again.`;
    case "NotFoundError":
    case "OverconstrainedError":
      return `We couldn't find a ${device} on this device.`;
    case "NotReadableError":
    case "AbortError":
      return `Your ${device} is being used by another app. Close that app and try again.`;
    default:
      return `We couldn't start your ${device}. Please try again.`;
  }
}

export interface Recording {
  wav: Blob;
  peak: number;
  seconds: number;
}

/**
 * Record `seconds` of microphone audio. `onLevel` receives a 0..1 loudness value
 * roughly every animation frame so the UI can show that the mic is hearing you.
 */
export async function recordVoice(
  seconds: number,
  onLevel: (level: number) => void,
  onStart?: () => void,
): Promise<Recording> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  const frame = new Float32Array(analyser.fftSize);
  let raf = 0;
  const tick = () => {
    analyser.getFloatTimeDomainData(frame);
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    onLevel(Math.min(1, Math.sqrt(sum / frame.length) * 6));
    raf = requestAnimationFrame(tick);
  };
  tick();

  const recorder = new MediaRecorder(stream);
  const chunks: BlobPart[] = [];
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const stopped = new Promise<void>((resolve) => (recorder.onstop = () => resolve()));
  recorder.start();
  onStart?.();
  await new Promise((r) => setTimeout(r, seconds * 1000));
  recorder.stop();
  await stopped;

  cancelAnimationFrame(raf);
  onLevel(0);
  stream.getTracks().forEach((t) => t.stop());

  // Decode whatever the browser recorded (WebM, Ogg, MP4) and re-encode as WAV.
  const encoded = await new Blob(chunks).arrayBuffer();
  const audio = await ctx.decodeAudioData(encoded);
  await ctx.close();
  const mono = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const ch = audio.getChannelData(c);
    for (let i = 0; i < ch.length; i++) mono[i] += ch[i] / audio.numberOfChannels;
  }
  return {
    wav: new Blob([encodeWav(mono, audio.sampleRate)], { type: "audio/wav" }),
    peak: peakLevel(mono),
    seconds: audio.duration,
  };
}
