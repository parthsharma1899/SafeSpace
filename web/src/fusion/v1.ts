// Port of the paper's fusion rule (GuardedPaperWMSS) from src/safespace/fusion/v1.py.
// Verified against the Python output in vectors.json to 6 decimal places.

export type Scores = Record<string, number | null | undefined>;

export interface FusionResult {
  score: number | null;
  weights: Record<string, number>;
  available: string[];
}

function cleanScores(scores: Scores): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [name, value] of Object.entries(scores)) {
    if (value === null || value === undefined) continue;
    if (Number.isNaN(value)) continue;
    if (value < 0 || value > 1) throw new Error(`score for ${name} out of range: ${value}`);
    out[name] = value;
  }
  return out;
}

// Inner engine: the paper's exact arithmetic. Throws when all weights are zero.
class PaperWMSS {
  private ema: Record<string, number> | null = null;
  constructor(
    private modalities: string[],
    private lam = 0.3,
    private alpha = 0.7,
    private beta = 0.3,
  ) {}

  step(scores: Record<string, number>): FusionResult {
    if (this.ema === null) {
      this.ema = {};
      for (const m of this.modalities) this.ema[m] = scores[m];
    }
    const newEma: Record<string, number> = {};
    const raw: Record<string, number> = {};
    for (const m of this.modalities) {
      const s = scores[m];
      const ema = this.lam * s + (1 - this.lam) * this.ema[m];
      const residual = s - ema;
      raw[m] = Math.max(0, this.alpha * residual + this.beta * ema);
      newEma[m] = ema;
    }
    const total = this.modalities.reduce((t, m) => t + raw[m], 0);
    if (total === 0) throw new Error("ZeroTotalWeight");
    const weights: Record<string, number> = {};
    let score = 0;
    for (const m of this.modalities) {
      weights[m] = raw[m] / total;
      score += scores[m] * weights[m];
    }
    this.ema = newEma;
    return { score, weights, available: this.modalities };
  }
}

// Guarded wrapper: fills a missing score with the last value, falls back to the mean
// when every weight is zero. Mirrors GuardedPaperWMSS.
export class GuardedPaperWMSS {
  private inner: PaperWMSS;
  private last: Record<string, number> = {};
  private lastScore: number | null = null;
  constructor(private modalities: string[]) {
    this.inner = new PaperWMSS(modalities);
  }

  step(scores: Scores): FusionResult {
    const x = cleanScores(scores);
    const present: Record<string, number> = {};
    for (const m of this.modalities) if (m in x) present[m] = x[m];
    const presentKeys = Object.keys(present);
    if (presentKeys.length === 0 && Object.keys(this.last).length === 0) {
      return { score: null, weights: {}, available: [] };
    }

    const fallback =
      presentKeys.length > 0
        ? presentKeys.reduce((t, m) => t + present[m], 0) / presentKeys.length
        : null;
    const filled: Record<string, number> = { ...present };
    for (const m of this.modalities) {
      if (!(m in filled)) {
        filled[m] = m in this.last ? this.last[m] : fallback !== null ? fallback : (this.lastScore as number);
      }
    }
    for (const m of presentKeys) this.last[m] = present[m];

    let result: FusionResult;
    try {
      result = this.inner.step(filled);
    } catch {
      const keys = Object.keys(filled);
      const mean = keys.reduce((t, m) => t + filled[m], 0) / keys.length;
      const weights: Record<string, number> = {};
      for (const m of keys) weights[m] = 1 / keys.length;
      result = { score: mean, weights, available: presentKeys };
    }
    this.lastScore = result.score;
    return result;
  }
}
