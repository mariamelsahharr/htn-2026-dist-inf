import { createContext, useContext } from "react"

// What this browser remembers: which model to ask for and the theme. The router
// serves this page, so there is no address to configure and no key to hold.
export interface Settings {
  model: string // "auto" lets the router decide; a tier's model id pins that tier
  theme: "system" | "light" | "dark"
}

const KEY = "pi-router-dashboard"

export const DEFAULTS: Settings = { model: "auto", theme: "system" }

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY)
    const saved = raw ? (JSON.parse(raw) as Partial<Settings>) : {}
    return { model: saved.model ?? DEFAULTS.model, theme: saved.theme ?? DEFAULTS.theme }
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

export const SettingsContext = createContext<{ settings: Settings; update: (patch: Partial<Settings>) => void }>({
  settings: DEFAULTS,
  update: () => {},
})

export const useSettings = () => useContext(SettingsContext)
