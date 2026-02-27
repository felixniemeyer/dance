<template>
  <div class="app">
    <!-- File-level status (server-driven) -->
    <div class="status-file">{{ statusFile }}</div>

    <!-- Waveform canvas: main view + minimap in one element -->
    <canvas ref="canvasEl" class="waveform" @click="onCanvasClick" />

    <!-- Live annotation status (instant, local state) -->
    <div class="status-live">{{ statusLive }}</div>

    <!-- Help -->
    <pre class="help">h/l shift onset ±1   j/k next/prev bar   digits = beats/bar (fast-type for multi-digit)   Shift+digit = note value
Space loop bar   Enter save+next   Esc skip</pre>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

// ── Constants ────────────────────────────────────────────────────────────────

const PPS  = 200   // peaks per second stored in the peaks array
const MINI = 50    // minimap height in CSS px

// ── Server state ─────────────────────────────────────────────────────────────

const serverDone    = ref(false)
const currentFile   = ref('')
const remaining     = ref(0)
const beatTimes     = ref([])
const tempo         = ref(120)
const loading       = ref(false)

// ── Local annotation state ────────────────────────────────────────────────────

const beatIdx  = ref(0)
const bpb      = ref(4)   // beats per bar  (numerator)
const bpd      = ref(4)   // note value      (denominator, display only)
const digitBuf = ref('')  // multi-digit accumulation buffer
let   digitTimer = null

// ── Audio ────────────────────────────────────────────────────────────────────

let audioCtx      = null    // AudioContext (lazy-created on first gesture)
let decodedBuffer = null    // AudioBuffer from decodeAudioData
let peaks         = null    // Float32Array, ~PPS samples/sec
let peakStep      = 0       // actual samples per peak bucket
let peakSR        = 1       // sampleRate of the decoded buffer
let source        = null    // current AudioBufferSourceNode

// Playback tracking (needed for animated playhead)
let srcStartCtx   = 0       // audioCtx.currentTime when source.start() was called
let loopStart     = 0       // seconds
let loopEnd       = 4       // seconds

const isPlaying     = ref(false)
const audioDuration = ref(0)

// ── View ─────────────────────────────────────────────────────────────────────

const viewStart = ref(0)
const viewEnd   = ref(30)
const canvasEl  = ref(null)
let   rafId     = null

// ── Computed ─────────────────────────────────────────────────────────────────

const barExtents = computed(() => {
  const bt = beatTimes.value, bi = beatIdx.value, n = bpb.value
  if (!bt.length || bi >= bt.length) return [0, 4]
  const s  = bt[bi]
  const ei = bi + n
  if (ei < bt.length) {
    // Use the actual next downbeat — each bar can have a slightly different
    // duration now that the DP tracker follows real onset positions.
    return [s, bt[ei]]
  }
  // Near end of song: extrapolate using the median of the last few local IBIs
  // (more accurate than global tempo for tracks with slight drift).
  const ibis = []
  for (let i = Math.max(0, bt.length - 9); i < bt.length - 1; i++) {
    ibis.push(bt[i + 1] - bt[i])
  }
  ibis.sort((a, b) => a - b)
  const localBeat = ibis[Math.floor(ibis.length / 2)] ?? (60 / tempo.value)
  return [s, s + localBeat * n]
})

const statusFile = computed(() => {
  if (serverDone.value) return 'All files annotated — you can close this window.'
  if (loading.value)    return 'Loading…'
  return `${currentFile.value}   |   ${tempo.value.toFixed(1)} BPM   |   ${remaining.value} files remaining`
})

const statusLive = computed(() => {
  const bt  = beatTimes.value, bi = beatIdx.value, n = bpb.value, d = bpd.value
  const nB  = bt.length > bi ? Math.ceil((bt.length - bi) / n) : 0
  const [bs] = barExtents.value
  const buf  = digitBuf.value
  return (
    `beat_idx=${bi}   ${n}/${d}   ~${nB} bars   bar_start=${bs.toFixed(2)}s` +
    `   ${isPlaying.value ? '▶ PLAYING' : '⏸ paused'}` +
    (buf ? `   [typing: ${buf}…]` : '')
  )
})

// ── Peak computation ──────────────────────────────────────────────────────────

function computePeaks(ab) {
  const data = ab.getChannelData(0)
  const step = Math.round(ab.sampleRate / PPS)
  peakStep   = step
  peakSR     = ab.sampleRate
  const out  = new Float32Array(Math.ceil(data.length / step))
  for (let i = 0; i < out.length; i++) {
    const s = i * step, e = Math.min(s + step, data.length)
    let max = 0
    for (let j = s; j < e; j++) { const v = Math.abs(data[j]); if (v > max) max = v }
    out[i] = max
  }
  return out
}

// ── Canvas draw ───────────────────────────────────────────────────────────────

function draw() {
  const el = canvasEl.value
  if (!el) return

  const dpr = window.devicePixelRatio || 1
  const W   = el.offsetWidth
  const H   = el.offsetHeight
  if (!W || !H) return

  // Resize backing store to match CSS size × DPR
  const bw = Math.round(W * dpr), bh = Math.round(H * dpr)
  if (el.width !== bw || el.height !== bh) { el.width = bw; el.height = bh }

  const ctx  = el.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const MAIN = H - MINI - 2   // main waveform area height
  const MID  = MAIN / 2

  // ── Background ────────────────────────────────────────────────────────────
  ctx.fillStyle = '#0f0f1e'
  ctx.fillRect(0, 0, W, H)

  if (!peaks) {
    ctx.fillStyle = '#444'
    ctx.font = '13px monospace'
    ctx.fillText(loading.value ? 'Decoding audio…' : 'No audio loaded', 16, H / 2)
    return
  }

  const bt   = beatTimes.value, bi = beatIdx.value, bpbV = bpb.value
  const vs   = viewStart.value, ve = viewEnd.value,  vd   = ve - vs
  const [barS, barE] = barExtents.value
  const total = audioDuration.value

  // ── Bar highlight ─────────────────────────────────────────────────────────
  ctx.fillStyle = 'rgba(0,200,80,0.08)'
  const bx0 = ((barS - vs) / vd) * W
  const bx1 = ((barE - vs) / vd) * W
  ctx.fillRect(bx0, 0, bx1 - bx0, MAIN)

  // ── Waveform ──────────────────────────────────────────────────────────────
  // Map screen pixel → time → peak index. Pixels outside [0, duration] are void.
  ctx.fillStyle = '#3a8aef'
  for (let px = 0; px < W; px++) {
    const t0 = vs + (px / W) * vd
    const t1 = vs + ((px + 1) / W) * vd
    const a  = Math.floor(t0 * peakSR / peakStep)
    const b  = Math.ceil(t1 * peakSR / peakStep)
    let max = 0
    for (let p = Math.max(0, a); p < Math.min(peaks.length, b); p++) {
      if (peaks[p] > max) max = peaks[p]
    }
    const h = max * MID * 0.92
    ctx.fillRect(px, MID - h, 1, h * 2)
  }

  // ── Beat lines ────────────────────────────────────────────────────────────
  ctx.save()
  ctx.beginPath(); ctx.rect(0, 0, W, MAIN); ctx.clip()
  for (let i = 0; i < bt.length; i++) {
    const t = bt[i]
    if (t < vs - 0.5 || t > ve + 0.5) continue
    const x = ((t - vs) / vd) * W
    if (i === bi) {
      ctx.strokeStyle = 'rgba(255,80,80,0.95)'; ctx.lineWidth = 2
    } else if (((i - bi) % bpbV + bpbV) % bpbV === 0) {
      ctx.strokeStyle = 'rgba(255,220,80,0.75)'; ctx.lineWidth = 1
    } else {
      ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1
    }
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, MAIN); ctx.stroke()
  }
  ctx.restore()

  // ── Playhead ──────────────────────────────────────────────────────────────
  if (isPlaying.value && audioCtx && source) {
    const elapsed  = audioCtx.currentTime - srcStartCtx
    const loopDur  = loopEnd - loopStart
    const pos      = loopDur > 0 ? loopStart + (elapsed % loopDur) : loopStart
    const px       = ((pos - vs) / vd) * W
    if (px >= -2 && px <= W + 2) {
      ctx.strokeStyle = 'rgba(255,255,255,0.8)'
      ctx.lineWidth   = 1.5
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, MAIN); ctx.stroke()
    }
  }

  // ── Separator ─────────────────────────────────────────────────────────────
  ctx.fillStyle = '#1e1e3a'
  ctx.fillRect(0, MAIN, W, 2)

  // ── Minimap ───────────────────────────────────────────────────────────────
  const MY = MAIN + 2

  ctx.fillStyle = '#080814'
  ctx.fillRect(0, MY, W, MINI)

  // Full-song waveform (downsampled)
  ctx.fillStyle = '#1a4a8a'
  for (let px = 0; px < W; px++) {
    const a = Math.floor((px / W) * peaks.length)
    const b = Math.ceil(((px + 1) / W) * peaks.length)
    let max = 0
    for (let p = a; p < b && p < peaks.length; p++) {
      if (peaks[p] > max) max = peaks[p]
    }
    const h = max * (MINI / 2) * 0.85
    ctx.fillRect(px, MY + MINI / 2 - h, 1, h * 2)
  }

  // Bar markers in minimap (show globally: first bar at bi % bpbV, then every bpbV)
  for (let i = bi % bpbV; i < bt.length; i += bpbV) {
    const x = (bt[i] / total) * W
    ctx.strokeStyle = i === bi ? 'rgba(255,80,80,0.7)' : 'rgba(255,220,80,0.3)'
    ctx.lineWidth   = i === bi ? 1.5 : 1
    ctx.beginPath(); ctx.moveTo(x, MY); ctx.lineTo(x, MY + MINI); ctx.stroke()
  }

  // View window indicator
  const vx0 = (vs / total) * W
  const vx1 = (ve / total) * W
  ctx.fillStyle   = 'rgba(255,255,255,0.10)'
  ctx.fillRect(vx0, MY, vx1 - vx0, MINI)
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.lineWidth   = 1
  ctx.strokeRect(vx0, MY, vx1 - vx0, MINI)
}

// ── View management ───────────────────────────────────────────────────────────

function scrollToBar() {
  const bt = beatTimes.value, bi = beatIdx.value
  if (!bt.length || bi >= bt.length) return
  const [barS, barE] = barExtents.value
  const barDur  = barE - barS          // actual duration of this specific bar
  const barMid  = barS + barDur / 2
  const viewDur = barDur * 16
  viewStart.value = barMid - viewDur / 2
  viewEnd.value   = barMid + viewDur / 2
}

// ── WebAudio playback ────────────────────────────────────────────────────────

function stopSource() {
  if (!source) return
  try { source.stop() } catch (_) {}
  source.disconnect()
  source = null
}

function startSource(barStart, barEnd) {
  if (!audioCtx || !decodedBuffer) return
  stopSource()
  source            = audioCtx.createBufferSource()
  source.buffer     = decodedBuffer
  source.loop       = true
  source.loopStart  = barStart
  source.loopEnd    = barEnd
  source.connect(audioCtx.destination)
  loopStart         = barStart
  loopEnd           = barEnd
  srcStartCtx       = audioCtx.currentTime
  source.start(0, barStart)
}

// Restart loop at updated bar extents while playing
function updateLoop() {
  if (!isPlaying.value) return
  const [bs, be] = barExtents.value
  startSource(bs, be)
}

async function togglePlay() {
  if (!decodedBuffer) return
  if (!audioCtx) audioCtx = new AudioContext()
  if (audioCtx.state === 'suspended') await audioCtx.resume()
  if (isPlaying.value) {
    stopSource()
    isPlaying.value = false
  } else {
    const [bs, be] = barExtents.value
    startSource(bs, be)
    isPlaying.value = true
  }
}

// RAF loop while playing (animates playhead)
watch(isPlaying, val => {
  if (val) {
    const loop = () => { draw(); rafId = requestAnimationFrame(loop) }
    rafId = requestAnimationFrame(loop)
  } else {
    if (rafId) { cancelAnimationFrame(rafId); rafId = null }
    draw()
  }
})

// ── Load audio ────────────────────────────────────────────────────────────────

async function loadAudio(token) {
  stopSource(); isPlaying.value = false
  decodedBuffer = null; peaks = null; audioDuration.value = 0
  draw()
  const resp      = await fetch(`/audio/${token}`)
  const arrayBuf  = await resp.arrayBuffer()
  if (!audioCtx) audioCtx = new AudioContext()
  decodedBuffer   = await audioCtx.decodeAudioData(arrayBuf)
  audioDuration.value = decodedBuffer.duration
  peaks           = computePeaks(decodedBuffer)
  draw()
}

// ── State management ──────────────────────────────────────────────────────────

function applyState(state) {
  if (state.done) { serverDone.value = true; draw(); return }
  serverDone.value  = false
  currentFile.value = state.filename  || ''
  remaining.value   = state.remaining || 0
  beatTimes.value   = state.beat_times || []
  tempo.value       = state.tempo || 120
  beatIdx.value     = 0; bpb.value = 4; bpd.value = 4
  digitBuf.value    = ''; clearTimeout(digitTimer); digitTimer = null
  scrollToBar()
  loadAudio(state.token)
}

async function apiAction(type, extra = {}) {
  if (loading.value) return
  loading.value = true; stopSource(); isPlaying.value = false
  try {
    const resp = await fetch('/api/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, ...extra }),
    })
    applyState(await resp.json())
  } finally {
    loading.value = false
  }
}

// ── Multi-digit bpb input ─────────────────────────────────────────────────────

function accumDigit(d) {
  clearTimeout(digitTimer)
  digitBuf.value += d
  digitTimer = setTimeout(commitDigit, 1000)
}

function commitDigit() {
  clearTimeout(digitTimer); digitTimer = null
  const n = parseInt(digitBuf.value, 10)
  if (n > 0) { bpb.value = n; updateLoop() }
  digitBuf.value = ''
  draw()
}

// ── Keyboard ─────────────────────────────────────────────────────────────────

function onKeyDown(e) {
  if (serverDone.value || loading.value) return

  const code    = e.code || ''
  const isDigit = code.startsWith('Digit') || code.startsWith('Numpad')
  const digit   = isDigit ? code.replace('Digit', '').replace('Numpad', '') : null
  const bt      = beatTimes.value

  if (e.key === 'h') {
    e.preventDefault()
    beatIdx.value = Math.max(0, beatIdx.value - 1)
    scrollToBar(); updateLoop(); draw()

  } else if (e.key === 'l') {
    e.preventDefault()
    beatIdx.value = Math.min(bt.length - 1, beatIdx.value + 1)
    scrollToBar(); updateLoop(); draw()

  } else if (e.key === 'j') {
    e.preventDefault()
    beatIdx.value = Math.min(bt.length - 1, beatIdx.value + bpb.value)
    scrollToBar(); updateLoop(); draw()

  } else if (e.key === 'k') {
    e.preventDefault()
    beatIdx.value = Math.max(0, beatIdx.value - bpb.value)
    scrollToBar(); updateLoop(); draw()

  } else if (e.key === ' ') {
    e.preventDefault()
    togglePlay()

  } else if (e.key === 'Enter') {
    e.preventDefault()
    commitDigit()
    apiAction('save', { beat_idx: beatIdx.value, bpb: bpb.value })

  } else if (e.key === 'Escape') {
    e.preventDefault()
    digitBuf.value = ''; clearTimeout(digitTimer)
    apiAction('skip')

  } else if (isDigit && digit !== null) {
    e.preventDefault()
    if (e.shiftKey) {
      // Shift+digit → denominator (note value: 4, 8, 16…)
      const d = parseInt(digit, 10)
      bpd.value = d === 0 ? 16 : d
      draw()
    } else {
      // Plain digit → accumulate beats-per-bar counter
      accumDigit(digit)
    }
  }
}

// ── Minimap click → seek to nearest bar ──────────────────────────────────────

function onCanvasClick(e) {
  const el = canvasEl.value
  if (!el || !peaks) return
  const rect = el.getBoundingClientRect()
  const y    = e.clientY - rect.top
  const MAIN = rect.height - MINI - 2
  if (y <= MAIN) return   // click in main waveform area — ignore for now

  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
  const pos  = frac * audioDuration.value
  const bt   = beatTimes.value, bpbV = bpb.value

  // Find nearest bar position
  let bestIdx = beatIdx.value, bestDist = Infinity
  for (let i = 0; i < bt.length; i += bpbV) {
    const d = Math.abs(bt[i] - pos)
    if (d < bestDist) { bestDist = d; bestIdx = i }
  }
  beatIdx.value = bestIdx
  scrollToBar(); updateLoop(); draw()
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

onMounted(async () => {
  window.addEventListener('keydown', onKeyDown)
  const ro = new ResizeObserver(() => draw())
  ro.observe(canvasEl.value)

  const resp = await fetch('/api/state')
  applyState(await resp.json())
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDown)
  if (rafId) cancelAnimationFrame(rafId)
  stopSource()
})
</script>

<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0 }

html, body {
  height: 100%;
  background: #0f0f1e;
  color: #eee;
  font-family: 'Courier New', monospace;
}

#app { height: 100% }

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 8px;
  gap: 5px;
}

.status-file {
  font-size: 11px;
  color: #555;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 0;
}

canvas.waveform {
  flex: 1;
  width: 100%;
  min-height: 0;
  display: block;
  cursor: pointer;
}

.status-live {
  font-size: 13px;
  color: #ccc;
  flex-shrink: 0;
}

.help {
  font-size: 10px;
  color: #383850;
  line-height: 1.6;
  flex-shrink: 0;
}
</style>
