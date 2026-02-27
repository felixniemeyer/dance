import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    host: true,         // bind to 0.0.0.0 so LAN machines can reach it
    allowedHosts: true, // allow any hostname (e.g. felix-pc.local)
    port: 5173,
    proxy: {
      '/api':    'http://localhost:8050',
      '/audio':  'http://localhost:8050',
      '/health': 'http://localhost:8050',
      '/inspect': {
        target:  'http://localhost:8051',
        rewrite: path => path.replace(/^\/inspect/, ''),
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
