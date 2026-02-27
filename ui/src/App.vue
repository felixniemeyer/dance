<template>
  <div class="shell">
    <nav class="tabs">
      <RouterLink to="/annotate" class="tab" :class="{ disabled: !annotateUp }">
        <span class="dot" :class="annotateUp ? 'dot-on' : 'dot-off'" />
        Annotate
      </RouterLink>
      <RouterLink to="/inspect" class="tab" :class="{ disabled: !inspectUp }">
        <span class="dot" :class="inspectUp ? 'dot-on' : 'dot-off'" />
        Inspect
      </RouterLink>
    </nav>
    <div class="view">
      <RouterView />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const annotateUp = ref(false)
const inspectUp  = ref(false)

async function checkHealth() {
  annotateUp.value = await ping('/health')
  inspectUp.value  = await ping('/inspect/health')
}

async function ping(url) {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(2000) })
    return r.ok
  } catch {
    return false
  }
}

let timer = null
onMounted(() => {
  checkHealth()
  timer = setInterval(checkHealth, 4000)
})
onUnmounted(() => clearInterval(timer))
</script>

<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0 }

html, body, #app {
  height: 100%;
  background: #0f0f1e;
  color: #fff;
  font-family: 'Courier New', monospace;
  font-size: 1rem;
}
</style>

<style scoped>
.shell {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.tabs {
  display: flex;
  gap: 2px;
  padding: 4px 8px 0;
  background: #0a0a18;
  border-bottom: 1px solid #1e1e3a;
  flex-shrink: 0;
}

.tab {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 16px 6px;
  border-radius: 4px 4px 0 0;
  font-size: 0.85rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  text-decoration: none;
  color: #666;
  background: #111128;
  border: 1px solid #1e1e3a;
  border-bottom: none;
  transition: color 0.15s, background 0.15s;
  user-select: none;
}

.tab:hover         { color: #aaa; background: #181830 }
.tab.router-link-active { color: #fff; background: #0f0f1e }

.tab.disabled      { pointer-events: none; opacity: 0.45 }

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.dot-on  { background: #00e676; box-shadow: 0 0 4px #00e676 }
.dot-off { background: #444 }

.view {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
</style>
