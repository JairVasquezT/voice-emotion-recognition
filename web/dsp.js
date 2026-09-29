/*
 * Audio preprocessing in the browser — a line-by-line port of the notebook:
 *   librosa.effects.trim(top_db=30) -> peak normalization -> central 3 s window
 *   -> melspectrogram(n_fft=512, win=400, hop=160, 64 mels) -> power_to_db(ref=1, top_db=80)
 * The mel filterbank comes from librosa itself (manifest.json), so results match.
 */
(function (root) {
  "use strict";

  // In-place iterative radix-2 FFT (n must be a power of two)
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
          const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
        }
      }
    }
  }

  // librosa.effects.trim(y, top_db, frame_length=2048, hop_length=512)
  function trim(y, topDb, frameLength = 2048, hop = 512) {
    const n = y.length, half = frameLength / 2, nFrames = 1 + Math.floor(n / hop);
    const ms = new Float64Array(nFrames);
    let maxMs = 0;
    for (let f = 0; f < nFrames; f++) {
      let s = 0;
      const start = f * hop - half;
      const a = Math.max(0, start), b = Math.min(n, start + frameLength);
      for (let i = a; i < b; i++) s += y[i] * y[i];
      ms[f] = s / frameLength;
      if (ms[f] > maxMs) maxMs = ms[f];
    }
    const ref = 10 * Math.log10(Math.max(1e-10, maxMs));
    let first = -1, last = -1;
    for (let f = 0; f < nFrames; f++) {
      if (10 * Math.log10(Math.max(1e-10, ms[f])) - ref > -topDb) {
        if (first < 0) first = f;
        last = f;
      }
    }
    if (first < 0) return y.subarray(0, 0);
    return y.subarray(first * hop, Math.min(n, (last + 1) * hop));
  }

  function peakNormalize(x) {
    let m = 0;
    for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] / (m + 1e-9);
    return out;
  }

  // notebook fix_length: zero-pad at the end, or keep the central n samples
  function fixLength(x, n) {
    if (x.length < n) {
      const out = new Float32Array(n);
      out.set(x);
      return { y: out, start: 0 };
    }
    const s = Math.floor((x.length - n) / 2);
    return { y: x.slice(s, s + n), start: s };
  }

  // log-mel (n_mels x n_frames), row-major: out[m * nFrames + t]
  function logmel(x, fb, cfg) {
    const { n_fft: nFft, win, hop } = cfg;
    const nBins = nFft / 2 + 1, nMels = fb.length;
    const pad = nFft / 2;
    const ypad = new Float64Array(x.length + 2 * pad);
    ypad.set(x, pad);
    const nFrames = 1 + Math.floor((ypad.length - nFft) / hop);
    const w = new Float64Array(nFft), off = Math.floor((nFft - win) / 2);
    for (let i = 0; i < win; i++) w[off + i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / win);

    const out = new Float32Array(nMels * nFrames);
    const re = new Float64Array(nFft), im = new Float64Array(nFft), pw = new Float64Array(nBins);
    let maxDb = -Infinity;
    for (let t = 0; t < nFrames; t++) {
      for (let k = 0; k < nFft; k++) { re[k] = ypad[t * hop + k] * w[k]; im[k] = 0; }
      fft(re, im);
      for (let k = 0; k < nBins; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
      for (let m = 0; m < nMels; m++) {
        const row = fb[m];
        let s = 0;
        for (let k = 0; k < nBins; k++) s += row[k] * pw[k];
        const db = 10 * Math.log10(Math.max(1e-10, s));
        out[m * nFrames + t] = db;
        if (db > maxDb) maxDb = db;
      }
    }
    const floor = maxDb - 80;
    for (let i = 0; i < out.length; i++) if (out[i] < floor) out[i] = floor;
    return { data: out, nMels, nFrames };
  }

  /**
   * Full pipeline from 16 kHz mono samples.
   * Returns { tooShort, signal (trimmed + normalized, for display), windowStart, spec }
   */
  function preprocess(y, manifest) {
    const trimmed = trim(y, manifest.trim_db);
    let peak = 0;
    for (let i = 0; i < trimmed.length; i++) peak = Math.max(peak, Math.abs(trimmed[i]));
    if (trimmed.length < 0.5 * manifest.sr || peak < 1e-3) return { tooShort: true };
    const signal = peakNormalize(trimmed);
    const { y: win3s, start } = fixLength(signal, manifest.n_samples);
    const spec = logmel(win3s, manifest.mel_fb, manifest);
    return { tooShort: false, signal, windowStart: start, spec };
  }

  const DSP = { fft, trim, peakNormalize, fixLength, logmel, preprocess };
  if (typeof module !== "undefined" && module.exports) module.exports = DSP;
  else root.DSP = DSP;
})(typeof self !== "undefined" ? self : this);
