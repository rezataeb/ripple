// Pareto-smoothed importance sampling. Line-by-line port of model/ripple_model/psis.py
// (Vehtari, Simpson, Gelman, Yao, Gabry; Zhang & Stephens 2009 for the GPD fit).
export const K_HAT_THRESHOLD = 0.7;
const EPS = 2.220446049250313e-16;
const MAX_EXP = 700;

/** Empirical-Bayes generalized-Pareto fit. x: ascending, positive. Returns [k, sigma]. */
export function gpdfit(x: Float64Array): [number, number] {
  const n = x.length;
  const priorBs = 3;
  const priorK = 10;
  const mEst = 30 + Math.floor(Math.sqrt(n));
  const anchor = x[Math.floor(n / 4 + 0.5) - 1];
  const last = x[n - 1];
  const b = new Float64Array(mEst);
  for (let j = 0; j < mEst; j++) {
    b[j] = (1 - Math.sqrt(mEst / (j + 1 - 0.5))) / (priorBs * anchor) + 1 / last;
  }
  const logLik = new Float64Array(mEst);
  for (let j = 0; j < mEst; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.log1p(-b[j] * x[i]);
    const kGrid = s / n;
    logLik[j] = n * (Math.log(-b[j] / kGrid) - kGrid - 1);
  }
  let mx = -Infinity;
  for (const v of logLik) if (v > mx) mx = v;
  let w: number[] = [];
  let bs: number[] = [];
  let tot = 0;
  const raw = Array.from(logLik, (v) => Math.exp(v - mx));
  raw.forEach((v) => (tot += v));
  const normed = raw.map((v) => v / tot);
  normed.forEach((v, j) => {
    if (v >= 10 * EPS) {
      w.push(v);
      bs.push(b[j]);
    }
  });
  if (w.length === 0) {
    w = normed;
    bs = Array.from(b);
  }
  const ws = w.reduce((a, v) => a + v, 0);
  w = w.map((v) => v / ws);
  let bPost = 0;
  for (let j = 0; j < w.length; j++) bPost += bs[j] * w[j];
  let kPost = 0;
  for (let i = 0; i < n; i++) kPost += Math.log1p(-bPost * x[i]);
  kPost /= n;
  const sigma = -kPost / bPost;
  kPost = (n * kPost + priorK * 0.5) / (n + priorK);
  return [kPost, sigma];
}

function gpdQuantile(p: number, k: number, sigma: number): number {
  if (Math.abs(k) < 1e-10) return sigma * -Math.log1p(-p);
  const e = Math.min(Math.max(-k * Math.log1p(-p), -MAX_EXP), MAX_EXP);
  return (sigma / k) * Math.expm1(e);
}

/** Stable argsort ascending. */
function argsort(a: ArrayLike<number>): Uint32Array {
  const idx = new Uint32Array(a.length);
  for (let i = 0; i < idx.length; i++) idx[i] = i;
  return idx.sort((i, j) => a[i] - a[j] || i - j);
}

/** Returns the smoothed log weights (max shifted to 0) and k-hat (NaN / -Infinity when not estimable). */
export function psisSmooth(logWeights: Float64Array): { logw: Float64Array; kHat: number } {
  const n = logWeights.length;
  const lw = Float64Array.from(logWeights);
  let mx = -Infinity;
  for (const v of lw) if (v > mx) mx = v;
  for (let i = 0; i < n; i++) lw[i] -= mx;

  const nTail = Math.trunc(Math.min(n / 5, 3 * Math.sqrt(n)));
  if (nTail < 5 || n < 20) return { logw: lw, kHat: NaN };

  const order = argsort(lw);
  const tailIdx = Array.from(order.subarray(n - nTail));
  const cutoff = lw[order[n - nTail - 1]];
  const expCut = Math.exp(cutoff);
  const exceed = Float64Array.from(tailIdx, (i) => Math.exp(lw[i]) - expCut);
  const exSorted = Float64Array.from(exceed).sort();
  const top = exSorted[exSorted.length - 1];
  if (top <= 0 || !exSorted.every(Number.isFinite)) {
    return { logw: lw, kHat: top <= 0 ? -Infinity : NaN };
  }
  const positive = exSorted.filter((v) => v > 0);
  if (positive.length < 5) return { logw: lw, kHat: -Infinity };
  const [kHat, sigma] = gpdfit(Float64Array.from(positive));

  if (Number.isFinite(kHat) && sigma > 0) {
    const m = nTail;
    const smoothed = Float64Array.from({ length: m }, (_, i) =>
      Math.log(gpdQuantile((i + 1 - 0.5) / m, kHat, sigma) + expCut),
    );
    if (smoothed.every(Number.isFinite)) {
      const ranks = new Uint32Array(m);
      argsort(exceed).forEach((src, r) => (ranks[src] = r));
      tailIdx.forEach((li, t) => (lw[li] = smoothed[ranks[t]]));
      for (let i = 0; i < n; i++) if (lw[i] > 0) lw[i] = 0;
    }
  }
  return { logw: lw, kHat };
}
