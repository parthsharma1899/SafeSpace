import type { Scores } from "./v1";

// Plain mean of the available scores, smoothed over time (the simple baseline).
export class SmoothedMean {
  private level: number | null = null;
  constructor(private modalities: string[], private rate = 0.5) {}

  step(scores: Scores): { score: number | null } {
    const present: number[] = [];
    for (const m of this.modalities) {
      const v = scores[m];
      if (v !== null && v !== undefined && !Number.isNaN(v)) present.push(v);
    }
    if (present.length === 0) return { score: this.level };
    const mean = present.reduce((a, b) => a + b, 0) / present.length;
    this.level = this.level === null ? mean : this.level + this.rate * (mean - this.level);
    return { score: this.level };
  }
}
