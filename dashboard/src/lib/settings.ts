import { createContext, useContext } from "react"

// Where the dashboard points and how it asks. Kept in this browser only; the router
// ignores the key unless a proxy in front of it checks one.
export interface Settings {
  routerUrl: string
  apiKey: string
  provider: string // "auto" or an upstream name, sent as X-Force-Upstream
  model: string // "auto" or a model id from /v1/models
  theme: "system" | "light" | "dark"
}

export const PROVIDERS = ["auto", "cluster", "baseten", "openai", "gemini", "snowflake"] as const

const KEY = "pi-router-dashboard"

export const DEFAULTS: Settings = {
  routerUrl: "",
  apiKey: "",
  provider: "auto",
  model: "auto",
  theme: "system",
}

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? { ...DEFAULTS, ...(JSON.parse(raw) as Partial<Settings>) } : DEFAULTS
  } catch {
    return DEFAULTS
  }
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s))
  } catch {
    // private window or blocked storage: the session still works, it just forgets
  }
}

export const baseUrl = (s: Settings): string => (s.routerUrl || window.location.origin).replace(/\/+$/, "")

export function headersFor(s: Settings): Record<string, string> {
  const h: Record<string, string> = {}
  if (s.apiKey) h.Authorization = `Bearer ${s.apiKey}`
  if (s.provider !== "auto") h["X-Force-Upstream"] = s.provider
  return h
}

export const SettingsContext = createContext<{ settings: Settings; update: (patch: Partial<Settings>) => void }>({
  settings: DEFAULTS,
  update: () => {},
})

export const useSettings = () => useContext(SettingsContext)
