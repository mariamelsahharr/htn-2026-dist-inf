import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Dev: the router runs on :8000; everything under /v1, /stats, /healthz is proxied to it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    proxy: {
      "/v1": "http://localhost:8000",
      "/stats": "http://localhost:8000",
      "/healthz": "http://localhost:8000",
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
})
