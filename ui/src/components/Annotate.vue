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
          &nbsp;|&nbsp; seg {{ currentSegmentIdx + 1 }}/{{ segments.length }}
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
        <span class="seg-status" :class="`seg-${currentSegment?.status ?? 'pending'}`">
          {{ currentSegment?.status ?? 'pending' }}
        </span>
        <span class="status-info">{{ statusInfo }}</span>
        <span v-if="digitBuf" class="badge badge-typing">{{ digitBuf }}…</span>
      </template>
    </div>

    <pre class="help">h/l ±sub-beat · j/k ±bar · s/d subdivide · 0-9 bpb · Shift+0-9 note · Space loop · c cut · x skip · Enter accept · Esc skip song</pre>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, onActivated, onDeactivated, watch } from 'vue'
import { computePeaks as _computePeaks } from '../utils/audio.js'

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
const loading        = ref(false)
const preloadingNext = ref(false)

// ── Segment state ─────────────────────────────────────────────────────────────
// Each segment: {id, startTime, endTime, beat_times, tempo, status, bpb}
// status: 'pending' | 'accepted' | 'skipped'

const segments = ref([])

// ── Local annotation state ────────────────────────────────────────────────────

const beatIdx    = ref(0)   // index into subdividedBeats
const bpb        = ref(4)   // beats per bar for current segment
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

// ── Per-segment subdivided beat grid builder ──────────────────────────────────
// Pure helper — applies one segment's own subdivision and returns its beat array.

function buildSegSubdividedBeats(seg) {
  const bt = seg.beat_times, k = seg.subdivision
  if (k <= 1 || bt.length < 2) return bt.slice()
  const out = []
  for (let i = 0; i < bt.length - 1; i++) {
    out.push(bt[i])
    for (let j = 1; j < k; j++) out.push(bt[i] + (bt[i + 1] - bt[i]) * j / k)
  }
  out.push(bt[bt.length - 1])
  return out
}

// ── Global subdivided beat grid (segments concatenated, already in time order) ─

const subdividedBeats = computed(() => {
  const out = []
  for (const seg of segments.value) out.push(...buildSegSubdividedBeats(seg))
  return out
})

// Effective beats-per-bar in the sub-beat grid (= musical bpb × subdivision)
const effectiveBpb = computed(() => bpb.value * subdivision.value)

// ── Derived: which segment contains the current beat cursor ──────────────────

const currentSegmentIdx = computed(() => {
  const sb   = subdividedBeats.value
  const bi   = beatIdx.value
  const segs = segments.value
  if (!sb.length || !segs.length) return 0
  const t = sb[Math.max(0, Math.min(bi, sb.length - 1))]
  for (let i = 0; i < segs.length; i++) {
    if (t >= segs[i].startTime && t <= segs[i].endTime) return i
  }
  // Fallback: find nearest segment by midpoint
  let best = 0, bestD = Infinity
  for (let i = 0; i < segs.length; i++) {
    const mid = (segs[i].startTime + segs[i].endTime) / 2
    const d   = Math.abs(t - mid)
    if (d < bestD) { bestD = d; best = i }
  }
  return best
})

const currentSegment = computed(() => segments.value[currentSegmentIdx.value] ?? null)

// Current segment's tempo (for display + IBI fallback)
const tempo = computed(() => currentSegment.value?.tempo ?? 120)

// ── Sync bpb ↔ current segment ────────────────────────────────────────────────

watch(currentSegmentIdx, (idx) => {
  const seg = segments.value[idx]
  if (!seg) return
  if (seg.bpb       !== bpb.value)       bpb.value       = seg.bpb
  if (seg.subdivision !== subdivision.value) subdivision.value = seg.subdivision
})

watch(bpb, (val) => {
  const seg = segments.value[currentSegmentIdx.value]
  if (seg) seg.bpb = val
})

// ── Computed bar extents ──────────────────────────────────────────────────────

const barExtents = computed(() => {
  const seg = currentSegment.value
  if (!seg) return [0, 4]
  const segSb = buildSegSubdividedBeats(seg)
  const n     = seg.bpb * seg.subdivision
  const off   = seg.beatOffset ?? 0
  if (!segSb.length) return [0, 4]
  const ibis = []
  for (let i = Math.max(0, segSb.length - 9); i < segSb.length - 1; i++) ibis.push(segSb[i + 1] - segSb[i])
  ibis.sort((a, b) => a - b)
  const lbi = ibis[Math.floor(ibis.length / 2)] ?? (60 / (seg.tempo || 120) / seg.subdivision)
  const getT = i => i >= 0 && i < segSb.length ? segSb[i]
                  : i < 0                       ? segSb[0] + i * lbi
                  :                               segSb[segSb.length - 1] + (i - segSb.length + 1) * lbi
  return [getT(off), getT(off + n)]
})

// ── Status ────────────────────────────────────────────────────────────────────

const statusInfo = computed(() => {
  const seg = currentSegment.value
  if (!seg) return ''
  const segSb = buildSegSubdividedBeats(seg)
  const effN  = seg.bpb * seg.subdivision
  const off   = seg.beatOffset ?? 0
  const nB    = segSb.length > off ? Math.ceil((segSb.length - off) / effN) : 0
  const [bs]  = barExtents.value
  const k     = seg.subdivision
  const subStr = k > 1 ? ` ×${k} = ${seg.bpb * k}/${bpd.value}` : ''
  return `${subStr}   ~${nB} bars   bar @ ${bs.toFixed(3)}s   ${isPlaying.value ? '▶' : '⏸'}`
})

// ── Beat offset tracking ──────────────────────────────────────────────────────
// Called after every navigation action to keep seg.beatOffset pointing at the
// position in the segment's own subdivided grid that matches the global cursor.

function updateBeatOffset() {
  const seg = currentSegment.value
  if (!seg) return
  const sb = subdividedBeats.value
  const bi = beatIdx.value
  if (bi < 0 || bi >= sb.length) return
  const t     = sb[bi]
  const segSb = buildSegSubdividedBeats(seg)
  let best = 0, bestDist = Infinity
  for (let i = 0; i < segSb.length; i++) {
    const d = Math.abs(segSb[i] - t)
    if (d < bestDist) { bestDist = d; best = i }
  }
  seg.beatOffset = best
}

// ── Peak computation ──────────────────────────────────────────────────────────

function computePeaks(ab) {
  const result = _computePeaks(ab, PPS)
  peakStep = result.step
  peakSR   = result.sr
  return result.peaks
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

  // ── Non-active segment overlay ────────────────────────────────────────────
  const cseg = currentSegment.value
  if (cseg) {
    ctx.fillStyle = 'rgba(0,0,0,0.27)'
    const segX0 = ((cseg.startTime - vs) / vd) * W
    const segX1 = ((cseg.endTime   - vs) / vd) * W
    if (segX0 > 0) ctx.fillRect(0,             0, Math.min(segX0, W),       MAIN)
    if (segX1 < W) ctx.fillRect(Math.max(segX1, 0), 0, W - Math.max(segX1, 0), MAIN)
  }

  // ── Beat & sub-beat lines (per segment, each with its own bpb/subdivision) ──
  const cursorTime = bi >= 0 && bi < sb.length ? sb[bi] : -1
  ctx.save()
  ctx.beginPath(); ctx.rect(0, 0, W, MAIN); ctx.clip()
  for (const seg of segments.value) {
    if (seg.endTime < vs - 0.5 || seg.startTime > ve + 0.5) continue
    const segSb   = buildSegSubdividedBeats(seg)
    const segEffN = seg.bpb * seg.subdivision
    const segOff  = seg.beatOffset ?? 0
    const segK    = seg.subdivision
    for (let i = 0; i < segSb.length; i++) {
      const t          = segSb[i]
      if (t < vs - 0.5 || t > ve + 0.5) continue
      const x          = ((t - vs) / vd) * W
      const isRealBeat = (i % segK) === 0
      const isBarStart = ((i - segOff) % segEffN + segEffN) % segEffN === 0
      const isCursor   = Math.abs(t - cursorTime) < 1e-6
      if (isCursor) {
        ctx.strokeStyle = 'rgba(255,80,80,0.95)'; ctx.lineWidth = 2
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, MAIN); ctx.stroke()
      } else if (isBarStart) {
        ctx.strokeStyle = isRealBeat ? 'rgba(255,220,80,0.75)' : 'rgba(255,200,60,0.45)'
        ctx.lineWidth   = 1
        const top = isRealBeat ? 0 : MAIN * 0.25
        const bot = isRealBeat ? MAIN : MAIN * 0.75
        ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bot); ctx.stroke()
      } else if (isRealBeat) {
        ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, MAIN); ctx.stroke()
      } else {
        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(x, MAIN * 0.35); ctx.lineTo(x, MAIN * 0.65); ctx.stroke()
      }
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

  // ── Segment overlays in minimap ────────────────────────────────────────────
  for (const seg of segments.value) {
    const sx0 = (seg.startTime / total) * W
    const sx1 = (seg.endTime   / total) * W
    const sw  = Math.max(1, sx1 - sx0)
    if (seg.status === 'accepted') {
      ctx.fillStyle   = 'rgba(0,200,80,0.12)'
      ctx.fillRect(sx0, MY, sw, MINI)
      ctx.strokeStyle = 'rgba(0,200,80,0.5)'; ctx.lineWidth = 1
      ctx.strokeRect(sx0 + 0.5, MY + 0.5, sw - 1, MINI - 1)
    } else if (seg.status === 'skipped') {
      ctx.fillStyle   = 'rgba(255,60,60,0.30)'
      ctx.fillRect(sx0, MY, sw, MINI)
      ctx.strokeStyle = 'rgba(255,80,80,0.6)'; ctx.lineWidth = 1
      ctx.strokeRect(sx0 + 0.5, MY + 0.5, sw - 1, MINI - 1)
    } else {
      // pending — subtle outline only
      ctx.strokeStyle = 'rgba(255,255,255,0.20)'; ctx.lineWidth = 1
      ctx.strokeRect(sx0 + 0.5, MY + 0.5, sw - 1, MINI - 1)
    }
  }

  // Bar markers in minimap (per segment, each with its own bpb/subdivision)
  for (const seg of segments.value) {
    const segSb   = buildSegSubdividedBeats(seg)
    const segEffN = seg.bpb * seg.subdivision
    const firstI  = (seg.beatOffset ?? 0) % segEffN
    for (let i = firstI; i < segSb.length; i += segEffN) {
      const t        = segSb[i]
      const x        = (t / total) * W
      const isCursor = Math.abs(t - cursorTime) < 1e-6
      ctx.strokeStyle = isCursor ? 'rgba(255,80,80,0.7)' : 'rgba(255,220,80,0.3)'
      ctx.lineWidth   = isCursor ? 1.5 : 1
      ctx.beginPath(); ctx.moveTo(x, MY); ctx.lineTo(x, MY + MINI); ctx.stroke()
    }
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

// ── Bar start computation ─────────────────────────────────────────────────────

/**
 * Compute bar starts for a single segment using its own beatOffset, bpb, subdivision.
 *
 * beatOffset IS a bar-start position in the segment's local subdivided grid.
 * Bar starts repeat every effN steps, so the first bar start in [0, n) is
 * beatOffset % effN, giving bar starts at [firstI, firstI+effN, firstI+2*effN, ...].
 */
function computeBarStartsFor(seg) {
  const segSb  = buildSegSubdividedBeats(seg)
  const effN   = seg.bpb * seg.subdivision
  const firstI = (seg.beatOffset ?? 0) % effN
  const barStarts = []
  for (let i = firstI; i < segSb.length; i += effN) barStarts.push(segSb[i])
  return barStarts.length >= 2 ? barStarts : []
}

// ── State management ──────────────────────────────────────────────────────────

function applyState(state) {
  if (state.done) { serverDone.value = true; draw(); return }
  serverDone.value     = false
  currentFile.value    = state.filename    || ''
  remaining.value      = state.remaining   || 0
  preloadingNext.value = state.preloading  || false

  segments.value = (state.segments || []).map(s => ({
    id:          s.id,
    startTime:   s.start_time,
    endTime:     s.end_time,
    beat_times:  s.beat_times,
    tempo:       s.tempo,
    status:      'pending',
    bpb:         4,
    subdivision: 1,
    beatOffset:  0,
  }))

  beatIdx.value = 0; bpb.value = 4; bpd.value = 4; subdivision.value = 1
  digitBuf.value = ''; clearTimeout(digitTimer); digitTimer = null
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
  const currentTime = subdividedBeats.value[beatIdx.value] ?? 0
  subdivision.value = newK
  const seg = segments.value[currentSegmentIdx.value]
  if (seg) seg.subdivision = newK
  const newSB = subdividedBeats.value
  let best = 0, bestDist = Infinity
  for (let i = 0; i < newSB.length; i++) {
    const d = Math.abs(newSB[i] - currentTime)
    if (d < bestDist) { bestDist = d; best = i }
  }
  beatIdx.value = best
  updateBeatOffset()
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

// ── Save trigger ──────────────────────────────────────────────────────────────

function fireSave() {
  const segsPayload = segments.value.map(s => ({
    start:      s.startTime,
    end:        s.endTime,
    bar_starts: computeBarStartsFor(s),
    status:     s.status,
  }))
  apiAction('save', { segments: segsPayload })
}

// Scroll to first beat of the next pending segment (returns true if found)
function goToNextPending() {
  const nextPending = segments.value.find(s => s.status === 'pending')
  if (!nextPending) return false
  const firstBeat = nextPending.beat_times[0]
  if (firstBeat === undefined) return false
  const sb = subdividedBeats.value
  let best = 0, bestDist = Infinity
  for (let i = 0; i < sb.length; i++) {
    const d = Math.abs(sb[i] - firstBeat)
    if (d < bestDist) { bestDist = d; best = i }
  }
  beatIdx.value = best
  updateBeatOffset()
  scrollToBar()
  return true
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
    updateBeatOffset(); scrollToBar(); updateLoop(); draw()

  } else if (e.key === 'l') {
    e.preventDefault()
    beatIdx.value = Math.min(sb.length - 1, beatIdx.value + 1)
    updateBeatOffset(); scrollToBar(); updateLoop(); draw()

  } else if (e.key === 'j') {
    e.preventDefault()
    if (beatIdx.value < sb.length - 1) beatIdx.value = Math.min(sb.length - 1, beatIdx.value + effN)
    updateBeatOffset(); scrollToBar(); updateLoop(); draw()

  } else if (e.key === 'k') {
    e.preventDefault()
    if (beatIdx.value > 0) beatIdx.value -= effN
    updateBeatOffset(); scrollToBar(); updateLoop(); draw()

  } else if (e.key === 's') {
    e.preventDefault()
    setSubdivision(subdivision.value + 1)

  } else if (e.key === 'd') {
    e.preventDefault()
    setSubdivision(subdivision.value - 1)

  } else if (e.key === ' ') {
    e.preventDefault()
    togglePlay()

  } else if (e.key === 'c') {
    // Cut current segment at the current beat position
    e.preventDefault()
    const idx = currentSegmentIdx.value
    const seg = segments.value[idx]
    if (!seg || beatIdx.value >= sb.length) return
    const cutTime = sb[beatIdx.value]
    if (cutTime <= seg.startTime || cutTime >= seg.endTime) return

    const beforeBeats = seg.beat_times.filter(t => t < cutTime)
    const afterBeats  = seg.beat_times.filter(t => t >= cutTime)
    if (beforeBeats.length < 4 || afterBeats.length < 4) return   // too short

    const leftSeg  = { ...seg, endTime:   cutTime, beat_times: beforeBeats, status: 'pending', bpb: bpb.value, subdivision: subdivision.value, beatOffset: seg.beatOffset ?? 0 }
    const rightSeg = { ...seg, startTime: cutTime, beat_times: afterBeats,  status: 'pending', bpb: bpb.value, subdivision: subdivision.value, beatOffset: 0 }

    const newSegs = [...segments.value]
    newSegs.splice(idx, 1, leftSeg, rightSeg)
    newSegs.forEach((s, i) => { s.id = i })
    segments.value = newSegs
    draw()

  } else if (e.key === 'x') {
    // Toggle current segment: pending→skipped, skipped→pending, accepted→skipped
    e.preventDefault()
    const seg = segments.value[currentSegmentIdx.value]
    if (!seg) return
    const transitions = { pending: 'skipped', skipped: 'pending', accepted: 'skipped' }
    seg.status = transitions[seg.status] ?? 'skipped'
    draw()

  } else if (e.key === 'Enter') {
    e.preventDefault()
    commitDigit()

    const idx = currentSegmentIdx.value
    const seg = segments.value[idx]
    if (seg) seg.status = 'accepted'

    if (segments.value.every(s => s.status !== 'pending')) {
      fireSave()
    } else {
      goToNextPending()
      draw()
    }

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
  updateBeatOffset(); scrollToBar(); updateLoop(); draw()
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

// With <KeepAlive> the component stays alive when switching tabs.
// Re-attach the keyboard listener when the tab becomes visible again,
// and detach it when hidden so keys on the Inspect tab don't trigger actions.
onActivated(()   => window.addEventListener('keydown', onKeyDown))
onDeactivated(() => window.removeEventListener('keydown', onKeyDown))

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

/* ── Segment status pill ────────────────────────────────────────────────────── */

.seg-status {
  font-size: 0.8rem;
  padding: 1px 7px;
  border-radius: 3px;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.seg-pending  { color: #888;    background: rgba(255,255,255,0.08) }
.seg-accepted { color: #00e676; background: rgba(0,200,80,0.15)    }
.seg-skipped  { color: #ff5252; background: rgba(255,60,60,0.15)   }

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
