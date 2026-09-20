// The theme lives on <html>, so portals (menus, tooltips) and the body ground follow it.
// Applied from click handlers and the boot script in index.html, never from an effect.
export type Theme = "system" | "light" | "dark"

const query = () => window.matchMedia("(prefers-color-scheme: dark)")
let current: Theme = "system"

export function applyTheme(theme: Theme): void {
  current = theme
  const dark = theme === "dark" || (theme === "system" && query().matches)
  document.documentElement.classList.toggle("dark", dark)
  document.documentElement.style.colorScheme = dark ? "dark" : "light"
}

// For useSyncExternalStore: re-applies a "system" theme when the OS flips, and lets
// React re-render anything that reads the resolved value.
export function subscribeTheme(onChange: () => void): () => void {
  const mq = query()
  const handler = () => {
    if (current === "system") applyTheme("system")
    onChange()
  }
  mq.addEventListener("change", handler)
  return () => mq.removeEventListener("change", handler)
}

export const systemPrefersDark = () => query().matches
