import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Адрес бэкенда пайплайна: фронт ходит только на относительные пути /api/*,
// а vite.dev-сервер проксирует их на Python-ядро (WP2/WP3).
// Переопределяется переменной окружения VITE_BACKEND_ORIGIN — например,
// для проверки пакетного SSE-потока на мок-сервере.
const BACKEND_ORIGIN = process.env.VITE_BACKEND_ORIGIN ?? 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: BACKEND_ORIGIN,
        changeOrigin: true,
      },
    },
  },
})
