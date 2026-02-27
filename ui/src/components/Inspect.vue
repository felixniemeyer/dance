<template>
  <div class="inspect">

    <!-- ── Controls bar ──────────────────────────────────────────────────── -->
    <div class="controls">

      <!-- Source toggle -->
      <div class="ctrl-group">
        <label class="lbl">Source</label>
        <div class="radio-row">
          <label class="radio-lbl">
            <input type="radio" v-model="source" value="chunk" />
            Chunks ({{ chunks.length }})
          </label>
          <label class="radio-lbl">
            <input type="radio" v-model="source" value="test"
                   :disabled="testFiles.length === 0" />
            Real-world ({{ testFiles.length }})
          </label>
        </div>
      </div>

      <span class="sep">│</span>

      <!-- File picker -->
      <div class="ctrl-group" style="flex:1; min-width:0">
        <label class="lbl">File</label>
        <div class="row-inline">
          <select class="sel" v-model="selectedFile">
            <option v-for="f in fileList" :key="f" :value="f">{{ f }}</option>
          </select>
          <button class="btn" @click="randomFile">Random</button>
        </div>
      </div>

      <span class="sep">│</span>

      <!-- Checkpoint -->
      <div class="ctrl-group">
        <label class="lbl">Model</label>
        <select class="sel sel-model" v-model="selectedTag">
          <option value="">(none)</option>
          <option v-for="t in tags" :key="t" :value="t">{{ t }}</option>
        </select>
      </div>

      <!-- Epoch -->
      <div class="ctrl-group">
        <label class="lbl">Epoch</label>
        <select class="sel sel-epoch" v-model="selectedEpoch">
          <option v-for="e in epochsForTag" :key="e" :value="e">{{ e }}</option>
        </select>
      </div>

      <button class="btn btn-infer" @click="runInfer" :disabled="inferRunning || !selectedTag">
        {{ inferRunning ? 'Running…' : 'Run inference' }}
      </button>

      <span class="status-msg" :class="inferError ? 'status-err' : 'status-ok'">
        {{ statusMsg }}
      </span>

    </div>

    <!-- ── Waveform canvas ───────────────────────────────────────────────── -->
    <div class="canvas-wrap canvas-wave">
      <canvas ref="waveCanvas" class="cv" />
      <div class="cv-label">Waveform</div>
    </div>

    <!-- ── Phase canvas ─────────────────────────────────────────────────── -->
    <div class="canvas-wrap canvas-phase">
      <canvas ref="phaseCanvas" class="cv" @click="onPhaseClick" />
      <div class="cv-label">Bar phase</div>
    </div>

    <!-- ── Transport ────────────────────────────────────────────────────── -->
    <div class="transport">
      <button class="btn btn-play" @click="togglePlay" :disabled="!audioReady">
        {{ isPlaying ? '⏸ Pause' : '▶ Play' }}
      </button>
      <span class="time-display">{{ formatTime(playPos) }} / {{ formatTime(duration) }}</span>
      <span v-if="!audioReady && selectedFile" class="loading-badge">loading audio…</span>
    </div>

  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { computePeaks } from '../utils/audio.js'

// ── State ─────────────────────────────────────────────────────────────────────

const source       = ref('chunk')
const chunks       = ref([])
const testFiles    = ref([])
const tags         = ref([])
const allEpochs    = ref({})        // { tag: [epochs…] }
const selectedTag  = ref('')
const selectedEpoch = ref(null)
const selectedFile = ref('')
const statusMsg    = ref('')
const inferError   = ref(false)
const inferRunning = ref(false)

// label data
const barTimes    = ref([])         // from .bars
const phaseTimes  = ref([])         // label phase timeline
const phaseVals   = ref([])         // label phase values [0,1)
const duration    = ref(0)

// prediction data
const predTimes  = ref([])
const predPhases = ref([])

// audio + peaks
const audioReady    = ref(false)
let   audioCtx      = null
let   decodedBuffer = null
let   peakData      = null           // { peaks, step, sr }

// playback
const isPlaying = ref(false)
const playPos   = ref(0)
let   source_   = null              // AudioBufferSourceNode (named _ to avoid shadowing)
let   playStart = 0                 // audioCtx.currentTime when play started
let   playOffset = 0                // audio time when play started
let   rafId     = null

// canvas refs
const waveCanvas  = ref(null)
const phaseCanvas = ref(null)

// ── Derived ───────────────────────────────────────────────────────────────────

const fileList = computed(() => source.value === 'chunk' ? chunks.value : testFiles.value)

const epochsForTag = computed(() => {
  if (!selectedTag.value) return []
  return allEpochs.value[selectedTag.value] ?? []
})

// ── Watchers ──────────────────────────────────────────────────────────────────

watch(source, () => {
  selectedFile.value = fileList.value[0] ?? ''
})

watch(selectedTag, tag => {
  const epochs = allEpochs.value[tag] ?? []
  selectedEpoch.value = epochs[0] ?? null
})

watch(selectedFile, file => {
  if (file) loadFile(file)
})

// ── API helpers ───────────────────────────────────────────────────────────────

async function apiFetch(path) {
  const r = await fetch('/inspect' + path)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

// ── Data loading ──────────────────────────────────────────────────────────────

async function loadCatalogue() {
  try {
    const [c, t, cp] = await Promise.all([
      apiFetch('/chunks'),
      apiFetch('/test-files'),
      apiFetch('/checkpoints'),
    ])
    chunks.value    = c.chunks    ?? []
    testFiles.value = t.files     ?? []
    tags.value      = cp.tags     ?? []
    allEpochs.value = cp.epochs   ?? {}
    selectedTag.value   = tags.value[0]    ?? ''
    selectedFile.value  = fileList.value[0] ?? ''
  } catch (e) {
    statusMsg.value = `Backend error: ${e.message}`
    inferError.value = true
  }
}

async function loadFile(file) {
  // Clear previous data
  barTimes.value  = []
  phaseTimes.value = []
  phaseVals.value  = []
  predTimes.value  = []
  predPhases.value = []
  duration.value   = 0
  audioReady.value = false
  peakData = null
  stopAudio()
  drawAll()

  // Load label data (chunk only) + audio in parallel
  const labelPromise = source.value === 'chunk'
    ? apiFetch(`/chunk-data/${encodeURIComponent(file)}`)
    : Promise.resolve(null)

  const audioUrl = source.value === 'chunk'
    ? `/inspect/chunk-audio/${encodeURIComponent(file)}.ogg`
    : `/inspect/test-audio/${encodeURIComponent(file)}`

  const audioPromise = fetchAndDecodeAudio(audioUrl)

  const [label] = await Promise.all([labelPromise, audioPromise])

  if (label) {
    barTimes.value   = label.bar_times   ?? []
    phaseTimes.value = label.phase_times ?? []
    phaseVals.value  = label.phase       ?? []
    duration.value   = label.duration    ?? 0
  }
  drawAll()
}

async function fetchAndDecodeAudio(url) {
  try {
    if (!audioCtx) audioCtx = new AudioContext()
    const resp = await fetch(url)
    const buf  = await resp.arrayBuffer()
    decodedBuffer = await audioCtx.decodeAudioData(buf)
    duration.value = decodedBuffer.duration
    peakData = computePeaks(decodedBuffer)
    audioReady.value = true
  } catch (e) {
    console.error('Audio load failed:', e)
  }
}

// ── Inference ─────────────────────────────────────────────────────────────────

async function runInfer() {
  if (!selectedTag.value || selectedEpoch.value == null || !selectedFile.value) return
  inferRunning.value = true
  inferError.value   = false
  statusMsg.value    = 'Running…'
  predTimes.value    = []
  predPhases.value   = []
  try {
    const r = await fetch('/inspect/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: source.value,
        file:   selectedFile.value,
        tag:    selectedTag.value,
        epoch:  selectedEpoch.value,
      }),
    })
    const data = await r.json()
    if (data.error) throw new Error(data.error)
    predTimes.value  = data.times  ?? []
    predPhases.value = data.phases ?? []
    statusMsg.value  = `Done — ${selectedTag.value}/${selectedEpoch.value}`
  } catch (e) {
    statusMsg.value  = `Error: ${e.message}`
    inferError.value = true
  } finally {
    inferRunning.value = false
    drawAll()
  }
}

// ── Audio playback ────────────────────────────────────────────────────────────

function stopAudio() {
  if (source_) { try { source_.stop() } catch (_) {}; source_.disconnect(); source_ = null }
  isPlaying.value = false
  if (rafId) { cancelAnimationFrame(rafId); rafId = null }
}

async function togglePlay() {
  if (!decodedBuffer) return
  if (!audioCtx) audioCtx = new AudioContext()
  if (audioCtx.state === 'suspended') await audioCtx.resume()
  if (isPlaying.value) {
    playOffset = currentPlayPos()
    stopAudio()
    drawAll()
    return
  }
  source_ = audioCtx.createBufferSource()
  source_.buffer = decodedBuffer
  source_.connect(audioCtx.destination)
  source_.onended = () => { if (isPlaying.value) { stopAudio(); playOffset = 0; drawAll() } }
  playStart  = audioCtx.currentTime
  source_.start(0, playOffset)
  isPlaying.value = true
  const loop = () => { drawAll(); rafId = requestAnimationFrame(loop) }
  rafId = requestAnimationFrame(loop)
}

function currentPlayPos() {
  if (!isPlaying.value || !audioCtx) return playOffset
  return Math.min(playOffset + (audioCtx.currentTime - playStart), duration.value)
}

watch(isPlaying, val => { if (!val) playPos.value = playOffset })

// ── Drawing ───────────────────────────────────────────────────────────────────

function drawAll() {
  const pos = currentPlayPos()
  playPos.value = pos
  drawWave(pos)
  drawPhase(pos)
}

function setupCanvas(el) {
  const dpr = window.devicePixelRatio || 1
  const W = el.offsetWidth, H = el.offsetHeight
  if (!W || !H) return null
  const bw = Math.round(W * dpr), bh = Math.round(H * dpr)
  if (el.width !== bw || el.height !== bh) { el.width = bw; el.height = bh }
  const ctx = el.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return { ctx, W, H }
}

function drawWave(pos) {
  const el = waveCanvas.value
  if (!el) return
  const cv = setupCanvas(el)
  if (!cv) return
  const { ctx, W, H } = cv
  const dur = duration.value

  ctx.fillStyle = '#0a0a18'
  ctx.fillRect(0, 0, W, H)

  if (!peakData || !dur) {
    ctx.fillStyle = '#333'
    ctx.font = '13px monospace'
    ctx.fillText(selectedFile.value ? 'Loading…' : 'Select a file', 12, H / 2)
    return
  }

  const { peaks, step, sr } = peakData
  const MID = H / 2

  // Waveform bars
  ctx.fillStyle = '#2a6ab8'
  for (let px = 0; px < W; px++) {
    const t0 = (px / W) * dur
    const t1 = ((px + 1) / W) * dur
    const a  = Math.floor(t0 * sr / step)
    const b  = Math.ceil(t1  * sr / step)
    let max = 0
    for (let p = Math.max(0, a); p < Math.min(peaks.length, b); p++) if (peaks[p] > max) max = peaks[p]
    const h = max * MID * 0.92
    ctx.fillRect(px, MID - h, 1, h * 2)
  }

  // Label bar markers (orange)
  ctx.strokeStyle = 'rgba(255,140,0,0.55)'
  ctx.lineWidth = 1
  for (const t of barTimes.value) {
    const x = (t / dur) * W
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
  }

  // Predicted bar markers — detect wrap-arounds in prediction phase
  if (predTimes.value.length > 0) {
    ctx.strokeStyle = 'rgba(100,255,100,0.5)'
    ctx.lineWidth = 1
    for (let i = 1; i < predPhases.value.length; i++) {
      if (predPhases.value[i - 1] > 0.8 && predPhases.value[i] < 0.2) {
        const frac = (1 - predPhases.value[i - 1]) / (1 - predPhases.value[i - 1] + predPhases.value[i])
        const t    = predTimes.value[i - 1] + (predTimes.value[i] - predTimes.value[i - 1]) * frac
        const x    = (t / dur) * W
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
      }
    }
  }

  // Playhead
  if (dur > 0) {
    const px = (pos / dur) * W
    ctx.strokeStyle = 'rgba(255,255,255,0.85)'
    ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke()
  }
}

function drawPhase(pos) {
  const el = phaseCanvas.value
  if (!el) return
  const cv = setupCanvas(el)
  if (!cv) return
  const { ctx, W, H } = cv
  const dur = duration.value

  ctx.fillStyle = '#070712'
  ctx.fillRect(0, 0, W, H)

  if (!dur) return

  // Axes grid lines at 0.25, 0.5, 0.75
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'
  ctx.lineWidth = 1
  for (const frac of [0.25, 0.5, 0.75]) {
    const y = H - frac * H
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
  }

  // Label phase (orange sawtooth)
  if (phaseTimes.value.length > 0) {
    drawPhaseSignal(ctx, phaseTimes.value, phaseVals.value, 'rgba(255,140,0,0.85)', W, H, dur)
  }

  // Predicted phase (green)
  if (predTimes.value.length > 0) {
    drawPhaseSignal(ctx, predTimes.value, predPhases.value, 'rgba(80,255,80,0.85)', W, H, dur)
  }

  // Playhead
  if (dur > 0) {
    const px = (pos / dur) * W
    ctx.strokeStyle = 'rgba(255,255,255,0.85)'
    ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke()
  }
}

function drawPhaseSignal(ctx, times, phases, color, W, H, dur) {
  ctx.strokeStyle = color
  ctx.lineWidth = 1.4
  ctx.beginPath()
  let started = false
  for (let i = 0; i < times.length; i++) {
    const x = (times[i] / dur) * W
    const y = H - phases[i] * H     // phase 0 → bottom, 1 → top
    // Break line at sawtooth wrap-around
    if (i > 0 && phases[i] < phases[i - 1] - 0.3) {
      ctx.stroke()
      ctx.beginPath()
      started = false
    }
    if (!started) { ctx.moveTo(x, y); started = true }
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
}

// ── Seek by clicking phase canvas ─────────────────────────────────────────────

function onPhaseClick(e) {
  if (!duration.value) return
  const el   = phaseCanvas.value
  const rect = el.getBoundingClientRect()
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
  const t    = frac * duration.value
  playOffset = t
  if (isPlaying.value) {
    stopAudio()
    togglePlay()
  } else {
    drawAll()
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function randomFile() {
  const list = fileList.value
  if (!list.length) return
  selectedFile.value = list[Math.floor(Math.random() * list.length)]
}

function formatTime(s) {
  if (!s) return '0:00'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60).toString().padStart(2, '0')
  return `${m}:${sec}`
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

let ro = null
onMounted(async () => {
  await loadCatalogue()
  ro = new ResizeObserver(() => drawAll())
  if (waveCanvas.value)  ro.observe(waveCanvas.value.parentElement)
  if (phaseCanvas.value) ro.observe(phaseCanvas.value.parentElement)
  drawAll()
})

onUnmounted(() => {
  stopAudio()
  if (ro) ro.disconnect()
})
</script>

<style scoped>
.inspect {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 6px 8px;
  gap: 5px;
  background: #0f0f1e;
  color: #fff;
  font-family: 'Courier New', monospace;
  font-size: 0.85rem;
}

/* ── Controls ───────────────────────────────────────────────────────────────── */

.controls {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  padding: 6px 10px;
  background: #0a0a18;
  border: 1px solid #1e1e3a;
  border-radius: 4px;
  flex-shrink: 0;
}

.ctrl-group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.lbl {
  font-size: 0.72rem;
  color: #555;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.row-inline {
  display: flex;
  align-items: center;
  gap: 5px;
}

.radio-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.radio-lbl {
  display: flex;
  align-items: center;
  gap: 4px;
  color: #ccc;
  cursor: pointer;
  white-space: nowrap;
}
.radio-lbl input { accent-color: #00e5ff; cursor: pointer }

.sep { color: #333; font-size: 1.2rem; align-self: center }

.sel {
  background: #111128;
  color: #ddd;
  border: 1px solid #2a2a4a;
  border-radius: 3px;
  padding: 4px 6px;
  font-family: inherit;
  font-size: 0.82rem;
  max-width: 340px;
}
.sel-model { max-width: 200px }
.sel-epoch { max-width: 80px }

.btn {
  background: #1a2a5a;
  color: #ccc;
  border: 1px solid #2a3a7a;
  border-radius: 3px;
  padding: 4px 10px;
  cursor: pointer;
  font-family: inherit;
  font-size: 0.82rem;
  white-space: nowrap;
}
.btn:hover:not(:disabled) { background: #253580; color: #fff }
.btn:disabled { opacity: 0.4; cursor: default }

.btn-infer { background: #0d3d22; border-color: #1a6b3c; color: #7fff7f }
.btn-infer:hover:not(:disabled) { background: #145230 }

.btn-play  { background: #1a2a5a; min-width: 90px }

.status-msg  { font-size: 0.8rem; min-width: 10px }
.status-ok   { color: #7fff7f }
.status-err  { color: #ff5252 }

/* ── Canvases ────────────────────────────────────────────────────────────────── */

.canvas-wrap {
  position: relative;
  border: 1px solid #1e1e3a;
  border-radius: 3px;
  overflow: hidden;
  min-height: 0;
}

.canvas-wave  { flex: 2 }
.canvas-phase { flex: 3 }

.cv {
  display: block;
  width: 100%;
  height: 100%;
}

.cv-label {
  position: absolute;
  top: 4px;
  left: 8px;
  font-size: 0.7rem;
  color: #333;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  pointer-events: none;
}

/* ── Transport ──────────────────────────────────────────────────────────────── */

.transport {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  padding: 2px 4px;
}

.time-display {
  color: #666;
  font-size: 0.82rem;
  min-width: 80px;
}

.loading-badge {
  color: #00bcd4;
  font-size: 0.8rem;
}
</style>
