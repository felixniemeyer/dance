/**
 * Compute per-pixel peak amplitudes from a decoded AudioBuffer.
 * Returns { peaks: Float32Array, step: number, sr: number }
 * where peaks[i] is the max absolute sample in a block of `step` samples.
 *
 * pps = peaks per second  (200 gives good waveform resolution at typical widths)
 */
export function computePeaks(audioBuffer, pps = 200) {
  const data = audioBuffer.getChannelData(0)
  const step = Math.round(audioBuffer.sampleRate / pps)
  const n    = Math.ceil(data.length / step)
  const out  = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const s = i * step
    const e = Math.min(s + step, data.length)
    let max = 0
    for (let j = s; j < e; j++) { const v = Math.abs(data[j]); if (v > max) max = v }
    out[i] = max
  }
  return { peaks: out, step, sr: audioBuffer.sampleRate }
}
