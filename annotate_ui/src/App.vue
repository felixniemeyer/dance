<template>
  <div class="app">
    <div class="status-file">
      <template v-if="serverError">
        <span class="badge badge-error">⚠ server offline</span>
        <span class="meta"> — start  python annotate.py  then refresh</span>
      </template>
      <template v-else-if="serverDone">
        All files annotated — you can close this window.
      </template>
      <template v-else>
        <span class="filename">{{ currentFile }}</span>
        <span class="meta">
          &nbsp;|&nbsp; {{ tempo.toFixed(1) }} BPM
          &nbsp;|&nbsp; {{ remaining }} remaining
          <span v-if="preloadingNext" class="badge badge-preload" title="next song is being analysed">⟳ next</span>
        </span>
      </template>
    </div>

    <canvas ref="canvasEl" class="waveform" @click="onCanvasClick" />

    <div class="status-live">
      <template v-if="!serverDone && !serverError">
        <span class="timesig">
          <span class="ts-num">{{ bpb }}</span>
          <span class="ts-den">{{ bpd }}</span>
        </span>
        <span v-if="subdivision > 1" class="badge badge-subdiv">×{{ subdivision }}</span>
        <span class="status-info">{{ statusInfo }}</span>
        <span v-if="digitBuf" class="badge badge-typing">{{ digitBuf }}…</span>
      </template>
    </div>

    <pre class="help">h/l ±sub-beat · j/k ±bar · s/d subdivide · 0-9 bpb · Shift+0-9 note · Space loop · Enter save+next · Esc skip</pre>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

// ── Constants ────────────────────────────────────────────────────────────────

const PPS         = 200   // peaks per second
const MINI        = 50    // minimap height px
const BAR_FRAC    = 0.15  // current bar occupies this fraction of view width
const MAX_SUBDIV  = 8

// ── Server state ─────────────────────────────────────────────────────────────

const serverDone     = ref(false)
const serverError    = ref(false)
const currentFile    = ref('')
const remaining      = ref(0)
const beatTimes      = ref([])
const tempo          = ref(120)
const loading        = ref(false)
const preloadingNext = ref(false)

// ── Local annotation state ────────────────────────────────────────────────────

const beatIdx    = ref(0)   // index into subdividedBeats
const bpb        = ref(4)   // beats per bar (musical, before subdivision)
const bpd        = ref(4)   // note value denominator (display only)
const subdivision = ref(1)  // sub-beat grid: 1=off, 2=halves, 3=triplets, …
const digitBuf   = ref('')
let   digitTimer  = null

// ── Audio ────────────────────────────────────────────────────────────────────

let audioCtx      = null
let decodedBuffer = null
let peaks         = null
let peakStep      = 0
let peakSR        = 1
let source        = null
let srcStartCtx   = 0
let loopStart     = 0
let loopEnd       = 4

const isPlaying     = ref(false)
const audioDuration = ref(0)

// ── View ─────────────────────────────────────────────────────────────────────

const viewStart = ref(0)
const viewEnd   = ref(30)
const canvasEl  = ref(null)
let   rafId     = null

// ── Derived beat grid ─────────────────────────────────────────────────────────

// Full grid with sub-beats interpolated between real detected beats.
const subdividedBeats = computed(() => {
  const bt = beatTimes.value, k = subdivision.value
  if (k <= 1 || bt.length < 2) return bt
  const out = []
  for (let i = 0; i < bt.length - 1; i++) {
    out.push(bt[i])
    for (let j = 1; j < k; j++) out.push(bt[i] + (bt[i + 1] - bt[i]) * (j / k))
  }
  out.push(bt[bt.length - 1])
  return out
})

// Effective beats-per-bar in the sub-beat grid (= musical bpb × subdivision)
const effectiveBpb = computed(() => bpb.value * subdivision.value)

// ── Computed bar extents (respects actual beat positions + subdivision) ────────

const barExtents = computed(() => {
  const sb = subdividedBeats.value, bi = beatIdx.value, n = effectiveBpb.value
  if (!sb.length) return [0, 4]
  // Local sub-beat interval for extrapolation beyond array bounds
  const ibis = []
  for (let i = Math.max(0, sb.length - 9); i < sb.length - 1; i++) ibis.push(sb[i + 1] - sb[i])
  ibis.sort((a, b) => a - b)
  const lbi = ibis[Math.floor(ibis.length / 2)] ?? (60 / tempo.value / subdivision.value)
  const getT = i => i >= 0 && i < sb.length ? sb[i]
                  : i < 0                   ? sb[0] + i * lbi
                  :                           sb[sb.length - 1] + (i - sb.length + 1) * lbi
  return [getT(bi), getT(bi + n)]
})

// ── Status ────────────────────────────────────────────────────────────────────

const statusInfo = computed(() => {
  const sb   = subdividedBeats.value, bi = beatIdx.value
  const effN = effectiveBpb.value
  const nB   = sb.length > bi ? Math.ceil((sb.length - bi) / effN) : 0
  const [bs] = barExtents.value
  const k    = subdivision.value
  const subStr = k > 1 ? ` ×${k} = ${bpb.value * k}/${bpd.value}` : ''
  return `${subStr}   ~${nB} bars   bar @ ${bs.toFixed(3)}s   ${isPlaying.value ? '▶' : '⏸'}`
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
  const W = el.offsetWidth, H = el.offsetHeight
  if (!W || !H) return
  const bw = Math.round(W * dpr), bh = Math.round(H * dpr)
  if (el.width !== bw || el.height !== bh) { el.width = bw; el.height = bh }
  const ctx = el.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const MAIN = H - MINI - 2
  const MID  = MAIN / 2

  ctx.fillStyle = '#0f0f1e'
  ctx.fillRect(0, 0, W, H)

  if (!peaks) {
    ctx.font = '14px monospace'
    if (serverError.value) {
      ctx.fillStyle = '#ff4c4c'
      ctx.fillText('Cannot reach server — is annotate.py running?', 16, H / 2)
    } else if (loading.value) {
      ctx.fillStyle = '#00bcd4'
      ctx.fillText('Analysing audio… please wait', 16, H / 2)
    } else if (serverDone.value) {
      ctx.fillStyle = '#aaa'
      ctx.fillText('All done!', 16, H / 2)
    } else {
      ctx.fillStyle = '#555'
      ctx.fillText('No audio loaded', 16, H / 2)
    }
    return
  }

  const sb    = subdividedBeats.value
  const bi    = beatIdx.value
  const effN  = effectiveBpb.value
  const k     = subdivision.value
  const vs    = viewStart.value, ve = viewEnd.value, vd = ve - vs
  const [barS, barE] = barExtents.value
  const total = audioDuration.value

  // ── Bar highlight ──────────────────────────────────────────────────────────
  ctx.fillStyle = 'rgba(0,200,80,0.08)'
  const bx0 = ((barS - vs) / vd) * W
  const bx1 = ((barE - vs) / vd) * W
  ctx.fillRect(bx0, 0, bx1 - bx0, MAIN)

  // ── Waveform ───────────────────────────────────────────────────────────────
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

  // ── Beat & sub-beat lines ──────────────────────────────────────────────────
  ctx.save()
  ctx.beginPath(); ctx.rect(0, 0, W, MAIN); ctx.clip()
  for (let i = 0; i < sb.length; i++) {
    const t = sb[i]
    if (t < vs - 0.5 || t > ve + 0.5) continue
    const x         = ((t - vs) / vd) * W
    const isRealBeat = (i % k) === 0
    const isBarStart = ((i - bi) % effN + effN) % effN === 0

    if (i === bi) {
      // Current downbeat — always full height, red
      ctx.strokeStyle = 'rgba(255,80,80,0.95)'; ctx.lineWidth = 2
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, MAIN); ctx.stroke()
    } else if (isBarStart) {
      // Other bar starts: full height if real beat, half height if sub-beat
      ctx.strokeStyle = isRealBeat ? 'rgba(255,220,80,0.75)' : 'rgba(255,200,60,0.45)'
      ctx.lineWidth   = 1
      const top = isRealBeat ? 0 : MAIN * 0.25
      const bot = isRealBeat ? MAIN : MAIN * 0.75
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bot); ctx.stroke()
    } else if (isRealBeat) {
      // Plain beat, not a bar start
      ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, MAIN); ctx.stroke()
    } else {
      // Pure sub-beat marker — short tick in the middle
      ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1
      ctx.beginPath(); ctx.moveTo(x, MAIN * 0.35); ctx.lineTo(x, MAIN * 0.65); ctx.stroke()
    }
  }
  ctx.restore()

  // ── Playhead ───────────────────────────────────────────────────────────────
  if (isPlaying.value && audioCtx && source) {
    const elapsed = audioCtx.currentTime - srcStartCtx
    const loopDur = loopEnd - loopStart
    const pos     = loopDur > 0 ? loopStart + (elapsed % loopDur) : loopStart
    const px      = ((pos - vs) / vd) * W
    if (px >= -2 && px <= W + 2) {
      ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1.5
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

  // Full-song waveform
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

  // Bar markers in minimap (globally, using full grid from first bar)
  const firstBarI = ((bi % effN) + effN) % effN
  for (let i = firstBarI; i < sb.length; i += effN) {
    const x = (sb[i] / total) * W
    ctx.strokeStyle = i === bi ? 'rgba(255,80,80,0.7)' : 'rgba(255,220,80,0.3)'
    ctx.lineWidth   = i === bi ? 1.5 : 1
    ctx.beginPath(); ctx.moveTo(x, MY); ctx.lineTo(x, MY + MINI); ctx.stroke()
  }

  // View window indicator
  const vx0 = (vs / total) * W
  const vx1 = (ve / total) * W
  ctx.fillStyle   = 'rgba(255,255,255,0.10)'
  ctx.fillRect(vx0, MY, vx1 - vx0, MINI)
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1
  ctx.strokeRect(vx0, MY, vx1 - vx0, MINI)
}

// ── View management ───────────────────────────────────────────────────────────

function scrollToBar() {
  const sb = subdividedBeats.value, bi = beatIdx.value
  if (!sb.length || bi >= sb.length) return
  const [barS, barE] = barExtents.value
  const barDur  = barE - barS
  const barMid  = barS + barDur / 2
  // Scale view so the current bar always occupies BAR_FRAC of the canvas width
  const viewDur = barDur / BAR_FRAC
  viewStart.value = barMid - viewDur / 2
  viewEnd.value   = barMid + viewDur / 2
}

// ── WebAudio playback ─────────────────────────────────────────────────────────

function stopSource() {
  if (!source) return
  try { source.stop() } catch (_) {}
  source.disconnect(); source = null
}

function startSource(barStart, barEnd) {
  if (!audioCtx || !decodedBuffer) return
  stopSource()
  source           = audioCtx.createBufferSource()
  source.buffer    = decodedBuffer
  source.loop      = true
  source.loopStart = barStart
  source.loopEnd   = barEnd
  source.connect(audioCtx.destination)
  loopStart    = barStart
  loopEnd      = barEnd
  srcStartCtx  = audioCtx.currentTime
  source.start(0, barStart)
}

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
    stopSource(); isPlaying.value = false
  } else {
    const [bs, be] = barExtents.value
    startSource(bs, be); isPlaying.value = true
  }
}

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
  const resp     = await fetch(`/audio/${token}`)
  const arrayBuf = await resp.arrayBuffer()
  if (!audioCtx) audioCtx = new AudioContext()
  decodedBuffer       = await audioCtx.decodeAudioData(arrayBuf)
  audioDuration.value = decodedBuffer.duration
  peaks               = computePeaks(decodedBuffer)
  draw()
}

// ── State management ──────────────────────────────────────────────────────────

function applyState(state) {
  if (state.done) { serverDone.value = true; draw(); return }
  serverDone.value     = false
  currentFile.value    = state.filename    || ''
  remaining.value      = state.remaining   || 0
  beatTimes.value      = state.beat_times  || []
  tempo.value          = state.tempo       || 120
  preloadingNext.value = state.preloading  || false
  beatIdx.value     = 0; bpb.value = 4; bpd.value = 4; subdivision.value = 1
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
    if (!resp.ok) { console.error('api/action error', resp.status); return }
    serverError.value = false
    applyState(await resp.json())
  } catch (err) {
    console.error('api/action fetch failed:', err)
    serverError.value = true
    draw()
  } finally {
    loading.value = false
  }
}

// ── Subdivision ───────────────────────────────────────────────────────────────

function setSubdivision(newK) {
  newK = Math.max(1, Math.min(MAX_SUBDIV, newK))
  if (newK === subdivision.value) return
  // Remember current time position before the grid changes
  const currentTime = subdividedBeats.value[beatIdx.value] ?? 0
  subdivision.value = newK
  // Find nearest index in new subdivided grid
  const newSB = subdividedBeats.value  // recomputed synchronously
  let best = 0, bestDist = Infinity
  for (let i = 0; i < newSB.length; i++) {
    const d = Math.abs(newSB[i] - currentTime)
    if (d < bestDist) { bestDist = d; best = i }
  }
  beatIdx.value = best
  scrollToBar(); updateLoop(); draw()
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
  if (n > 0) { bpb.value = n; scrollToBar(); updateLoop() }
  digitBuf.value = ''
  draw()
}

// ── Keyboard ──────────────────────────────────────────────────────────────────

function onKeyDown(e) {
  if (serverDone.value || loading.value) return

  const code    = e.code || ''
  const isDigit = code.startsWith('Digit') || code.startsWith('Numpad')
  const digit   = isDigit ? code.replace('Digit', '').replace('Numpad', '') : null
  const sb      = subdividedBeats.value
  const effN    = effectiveBpb.value

  if (e.key === 'h') {
    e.preventDefault()
    beatIdx.value = Math.max(0, beatIdx.value - 1)
    scrollToBar(); updateLoop(); draw()

  } else if (e.key === 'l') {
    e.preventDefault()
    beatIdx.value = Math.min(sb.length - 1, beatIdx.value + 1)
    scrollToBar(); updateLoop(); draw()

  } else if (e.key === 'j') {
    e.preventDefault()
    if (beatIdx.value < sb.length - 1) beatIdx.value = Math.min(sb.length - 1, beatIdx.value + effN)
    scrollToBar(); updateLoop(); draw()

  } else if (e.key === 'k') {
    e.preventDefault()
    if (beatIdx.value > 0) beatIdx.value -= effN   // allow one step into negative
    scrollToBar(); updateLoop(); draw()

  } else if (e.key === 's') {
    e.preventDefault()
    setSubdivision(subdivision.value + 1)

  } else if (e.key === 'd') {
    e.preventDefault()
    setSubdivision(subdivision.value - 1)

  } else if (e.key === ' ') {
    e.preventDefault()
    togglePlay()

  } else if (e.key === 'Enter') {
    e.preventDefault()
    commitDigit()
    const bi = beatIdx.value
    // Local sub-beat interval for extrapolating bars before sb[0]
    const ibis = []
    for (let i = Math.max(0, sb.length - 9); i < sb.length - 1; i++) ibis.push(sb[i + 1] - sb[i])
    ibis.sort((a, b) => a - b)
    const lbi = ibis[Math.floor(ibis.length / 2)] ?? (60 / tempo.value / subdivision.value)
    const getT = i => i >= 0 && i < sb.length ? sb[i] : i < 0 ? sb[0] + i * lbi : sb[sb.length - 1] + (i - sb.length + 1) * lbi
    // JS % can be negative, so force into [0, effN)
    const firstI = ((bi % effN) + effN) % effN
    const barStarts = []
    // Extrapolated bars before sb[0] (when bi < 0)
    for (let i = bi; i < firstI; i += effN) barStarts.push(getT(i))
    // Detected bars
    for (let i = firstI; i < sb.length; i += effN) barStarts.push(sb[i])
    apiAction('save', { bar_starts: barStarts })

  } else if (e.key === 'Escape') {
    e.preventDefault()
    digitBuf.value = ''; clearTimeout(digitTimer)
    apiAction('skip')

  } else if (isDigit && digit !== null) {
    e.preventDefault()
    if (e.shiftKey) {
      const d = parseInt(digit, 10)
      bpd.value = d === 0 ? 16 : d
      draw()
    } else {
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
  if (y <= MAIN) return

  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
  const pos  = frac * audioDuration.value
  const sb   = subdividedBeats.value
  const effN = effectiveBpb.value

  let bestIdx = beatIdx.value, bestDist = Infinity
  for (let i = 0; i < sb.length; i += effN) {
    const d = Math.abs(sb[i] - pos)
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
  try {
    const resp = await fetch('/api/state')
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    serverError.value = false
    applyState(await resp.json())
  } catch (err) {
    console.error('Could not reach server:', err)
    serverError.value = true
    draw()
  }
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
  color: #fff;
  font-family: 'Courier New', monospace;
  font-size: 1rem;
}

#app { height: 100% }

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 8px;
  gap: 5px;
}

/* ── Top status bar ─────────────────────────────────────────────────────────── */

.status-file {
  font-size: 1rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 0;
}

.status-file .filename { color: #fff; font-weight: bold }
.status-file .meta     { color: #888 }

/* ── Canvas ──────────────────────────────────────────────────────────────────── */

canvas.waveform {
  flex: 1;
  width: 100%;
  min-height: 0;
  display: block;
  cursor: pointer;
}

/* ── Bottom status bar ──────────────────────────────────────────────────────── */

.status-live {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1rem;
  color: #fff;
  flex-shrink: 0;
}

.status-info { color: #ccc }

/* ── Time signature ─────────────────────────────────────────────────────────── */

.timesig {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  line-height: 1.1;
  font-size: 1rem;
  font-weight: bold;
  gap: 1px;
}

.ts-num {
  color: #00e5ff;
  background: rgba(0, 229, 255, 0.10);
  padding: 0 6px;
  border-radius: 2px;
}

.ts-den {
  color: #ff69b4;
  background: rgba(255, 105, 180, 0.10);
  padding: 0 6px;
  border-radius: 2px;
}

/* ── Badges ──────────────────────────────────────────────────────────────────── */

.badge {
  font-size: 0.8rem;
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: bold;
  white-space: nowrap;
}

.badge-error   { color: #ff4c4c; background: rgba(255,76,76,0.15) }
.badge-preload { color: #00bcd4; background: rgba(0,188,212,0.12) }
.badge-subdiv  { color: #ffd740; background: rgba(255,215,64,0.12) }
.badge-typing  { color: #ff9800; background: rgba(255,152,0,0.12) }

/* ── Help text ───────────────────────────────────────────────────────────────── */

.help {
  font-size: 1rem;
  color: #888;
  line-height: 1.4;
  flex-shrink: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
