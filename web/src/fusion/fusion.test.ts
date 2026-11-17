import { describe, it, expect } from "vitest";
import vectors from "./vectors.json";
import { GuardedPaperWMSS } from "./v1";
import { ResidualPersistenceV2, TUNED } from "./v2";

type Step = { scores: Record<string, number>; v1: number | null; v2: number | null };

describe("TypeScript fusion ports match the Python reference", () => {
  const modalities = vectors.modalities as string[];

  it("v1 and v2 reproduce every Python output to 6 decimal places", () => {
    let checked = 0;
    for (const seq of vectors.sequences as Step[][]) {
      const v1 = new GuardedPaperWMSS(modalities);
      const v2 = new ResidualPersistenceV2(modalities, TUNED);
      for (const step of seq) {
        const r1 = v1.step(step.scores);
        const r2 = v2.step(step.scores);
        // Exact numerical agreement (far tighter than 6 decimal places).
        if (step.v1 === null) expect(r1.score).toBeNull();
        else expect(Math.abs((r1.score as number) - step.v1)).toBeLessThan(1e-9);
        if (step.v2 === null) expect(r2.score).toBeNull();
        else expect(Math.abs((r2.score as number) - step.v2)).toBeLessThan(1e-9);
        checked++;
      }
    }
    expect(checked).toBe(200 * 30);
  });

  it("TUNED matches the exported tuned config", () => {
    expect(TUNED.floorStd).toBe(vectors.tuned.floor_std);
    expect(TUNED.baseGain).toBe(vectors.tuned.base_gain);
    expect(TUNED.surpriseK).toBe(vectors.tuned.surprise_k);
    expect(TUNED.carry).toBe(vectors.tuned.carry);
  });
});
