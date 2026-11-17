import { describe, expect, it } from "vitest";
import { encodeWav, mediaErrorMessage, peakLevel } from "./media";

const text = (view: DataView, offset: number, length: number) =>
  String.fromCharCode(...Array.from({ length }, (_, i) => view.getUint8(offset + i)));

describe("encodeWav", () => {
  it("writes a valid 16-bit mono PCM header and data", () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1, -1, 2]);
    const view = new DataView(encodeWav(samples, 48000));
    expect(text(view, 0, 4)).toBe("RIFF");
    expect(text(view, 8, 4)).toBe("WAVE");
    expect(text(view, 36, 4)).toBe("data");
    expect(view.getUint16(20, true)).toBe(1); // PCM
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(48000);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(samples.length * 2);
    expect(view.byteLength).toBe(44 + samples.length * 2);
    expect(view.getInt16(44 + 3 * 2, true)).toBe(0x7fff); // 1.0
    expect(view.getInt16(44 + 4 * 2, true)).toBe(-0x8000); // -1.0
    expect(view.getInt16(44 + 5 * 2, true)).toBe(0x7fff); // clipped from 2.0
  });
});

describe("peakLevel", () => {
  it("finds the loudest sample", () => {
    expect(peakLevel(new Float32Array([0.1, -0.7, 0.3]))).toBeCloseTo(0.7);
  });
});

describe("mediaErrorMessage", () => {
  it("explains a blocked camera in plain language", () => {
    const msg = mediaErrorMessage(new DOMException("denied", "NotAllowedError"), "camera");
    expect(msg).toMatch(/blocked/);
  });
  it("explains a busy microphone", () => {
    const msg = mediaErrorMessage(new DOMException("busy", "NotReadableError"), "microphone");
    expect(msg).toMatch(/another app/);
  });
});
