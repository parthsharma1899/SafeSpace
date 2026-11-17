// Port of the redesigned fusion (ResidualPersistenceV2) from src/safespace/fusion/v2.py.
// Verified against the Python output in vectors.json to 6 decimal places.

import type { FusionResult, Scores } from "./v1";

export interface V2Config {
  lam: number;
  varRate: number;
  priorStd: number;
  floorStd: number;
  baseGain: number;
  surpriseK: number;
  carry: number;
  reliability: Record<string, number>;
  useVariance: boolean;
  useOffset: boolean;
  adaptiveGain: boolean;
}

// The tuned defaults (must match TUNED in v2.py / results/fusion_tuning.json).
export const TUNED: V2Config = {
  lam: 0.3,
  varRate: 0.05,
  priorStd: 0.15,
  floorStd: 0.05,
  baseGain: 0.3,
  surpriseK: 2.5,
  carry: 6.0,
  reliability: {},
  useVariance: true,
  useOffset: true,
  adaptiveGain: true,
};

function cleanScores(scores: Scores, known: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of known) {
    const v = scores[m];
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    if (v < 0 || v > 1) throw new Error(`score for ${m} out of range: ${v}`);
    out[m] = v;
  }
  return out;
}

function perStepRate(rate: number, dt: number): number {
  if (dt <= 0) return 0;
  return 1 - Math.pow(1 - rate, dt);
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  const mid = Math.floor(n / 2);
  return n % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

interface State {
  ema: number | null;
  var: number;
  last: number | null;
  age: number;
}

export class ResidualPersistenceV2 {
  private cfg: V2Config;
  private state: Record<string, State> = {};
  private fused: number | null = null;
  private fusedVar: number;

  constructor(private modalities: string[], config: V2Config = TUNED) {
    this.cfg = config;
    const priorVar = config.priorStd * config.priorStd;
    for (const m of modalities) this.state[m] = { ema: null, var: priorVar, last: null, age: 0 };
    this.fusedVar = priorVar / Math.max(1, modalities.length);
  }

  step(scores: Scores, dt = 1.0): FusionResult {
    const cfg = this.cfg;
    const fresh = cleanScores(scores, this.modalities);
    for (const m of this.modalities) if (this.state[m].last !== null) this.state[m].age += dt;

    const used: Record<string, number> = { ...fresh };
    const freshness: Record<string, number> = {};
    for (const m of Object.keys(fresh)) freshness[m] = 1.0;
    if (cfg.carry > 0) {
      for (const m of this.modalities) {
        const st = this.state[m];
        if (!(m in fresh) && st.last !== null && st.age < cfg.carry) {
          used[m] = st.last;
          freshness[m] = 1.0 - st.age / cfg.carry;
        }
      }
    }
    const usedKeys = Object.keys(used);
    if (usedKeys.length === 0) {
      return { score: this.fused, weights: {}, available: [] };
    }

    const known: Record<string, number> = {};
    for (const m of usedKeys) if (this.state[m].ema !== null) known[m] = this.state[m].ema as number;
    const floorVar = cfg.floorStd * cfg.floorStd;

    const trust: Record<string, number> = {};
    for (const m of usedKeys) {
      const st = this.state[m];
      const variance = cfg.useVariance ? st.var : cfg.priorStd * cfg.priorStd;
      let offset = 0;
      if (cfg.useOffset && st.ema !== null) {
        const others = Object.keys(known).filter((k) => k !== m).map((k) => known[k]);
        if (others.length) offset = (st.ema as number) - median(others);
      }
      const reliability = cfg.reliability[m] ?? 1.0;
      trust[m] = (reliability * freshness[m]) / (floorVar + variance + offset * offset);
    }

    const total = usedKeys.reduce((t, m) => t + trust[m], 0);
    const weights: Record<string, number> = {};
    for (const m of usedKeys) weights[m] = trust[m] / total;
    const consensus = usedKeys.reduce((t, m) => t + weights[m] * used[m], 0);

    if (this.fused === null) {
      this.fused = consensus;
    } else {
      const residual = consensus - this.fused;
      let gain = perStepRate(cfg.baseGain, dt);
      if (cfg.adaptiveGain) {
        const z2 = (residual * residual) / (this.fusedVar + 1e-12);
        gain += (1 - gain) * (z2 / (z2 + cfg.surpriseK * cfg.surpriseK));
      }
      this.fused += gain * residual;
      this.fusedVar += perStepRate(cfg.varRate, dt) * (residual * residual - this.fusedVar);
    }
    this.fused = Math.min(1, Math.max(0, this.fused));

    for (const m of Object.keys(fresh)) {
      const st = this.state[m];
      const s = fresh[m];
      if (st.ema === null) {
        st.ema = s;
      } else {
        const gap = st.age;
        const residual = s - st.ema;
        st.var += perStepRate(cfg.varRate, gap) * (residual * residual - st.var);
        st.ema += perStepRate(cfg.lam, gap) * residual;
      }
      st.last = s;
      st.age = 0;
    }

    return { score: this.fused, weights, available: Object.keys(fresh) };
  }
}
