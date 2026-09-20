import { useSyncExternalStore } from "react"
import { applyTheme, type Theme } from "@/lib/theme"

// What this browser remembers: which model to ask for and the theme. A small external
// store: handlers call updateSettings, components read it with useSettings. No context,
// no effects, no stale closures.
export interface Settings {
  model: string // "auto" lets the router decide; a tier's model id pins that tier
  theme: Theme
}

const KEY = "pi-router-dashboard" // index.html reads the same key before React paints
const THEMES: Theme[] = ["system", "light", "dark"]

export const DEFAULTS: Settings = { model: "auto", theme: "system" }

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY)
    const saved = raw ? (JSON.parse(raw) as Partial<Settings>) : {}
    return {
      model: typeof saved.model === "string" && saved.model ? saved.model : DEFAULTS.model,
      theme: THEMES.includes(saved.theme as Theme) ? (saved.theme as Theme) : DEFAULTS.theme,
    }
  } catch {
    return DEFAULTS
  }
}

let settings: Settings = load()
applyTheme(settings.theme) // tells theme.ts the saved choice, so an OS flip cannot override it

const listeners = new Set<() => void>()

function subscribe(fn: () => void): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export const getSettings = (): Settings => settings

export function updateSettings(patch: Partial<Settings>): void {
  settings = { ...settings, ...patch }
  try {
    localStorage.setItem(KEY, JSON.stringify(settings))
  } catch {
    // private window or blocked storage: the session still works, it just forgets
  }
  if (patch.theme) applyTheme(patch.theme)
  for (const fn of listeners) fn()
}

export const useSettings = (): Settings => useSyncExternalStore(subscribe, getSettings, () => DEFAULTS)
