import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// React Compiler (the native oxc port) memoizes components and hooks at build time, so
// the source carries no memo/useMemo/useCallback. Dev without VITE_ROUTER_URL proxies
// the router's paths to :8000.
export default defineConfig({
  plugins: [react({ compiler: true }), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  server: {
    proxy: {
      "/v1": "http://localhost:8000",
      "/stats": "http://localhost:8000",
      "/healthz": "http://localhost:8000",
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
})
