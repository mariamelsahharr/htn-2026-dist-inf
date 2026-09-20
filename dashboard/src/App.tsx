import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Moon, Sun, SunMoon } from "lucide-react"
import { useCallback, useMemo, useState, useSyncExternalStore } from "react"
import { Chat } from "@/components/Chat"
import { NodesPanel } from "@/components/NodesPanel"
import { RatesPanel } from "@/components/RatesPanel"
import { RecentChart } from "@/components/RecentChart"
import { ModelPicker } from "@/components/ModelPicker"
import { StatusStrip } from "@/components/StatusStrip"
import { Button } from "@/components/ui/button"
import { useStats } from "@/hooks/useStats"
import { loadSettings, saveSettings, SettingsContext, type Settings } from "@/lib/settings"

const queryClient = new QueryClient()

const darkQuery = () => window.matchMedia("(prefers-color-scheme: dark)")
const subscribeSystemTheme = (cb: () => void) => {
  const mq = darkQuery()
  mq.addEventListener("change", cb)
  return () => mq.removeEventListener("change", cb)
}
const systemPrefersDark = () => darkQuery().matches

function ThemeButton({ theme, onChange }: { theme: Settings["theme"]; onChange: (t: Settings["theme"]) => void }) {
  const next: Record<Settings["theme"], Settings["theme"]> = { system: "light", light: "dark", dark: "system" }
  const Icon = theme === "light" ? Sun : theme === "dark" ? Moon : SunMoon
  return (
    <Button variant="ghost" size="sm" onClick={() => onChange(next[theme])} aria-label={`Theme: ${theme}`}>
      <Icon className="size-4" /> {theme}
    </Button>
  )
}

function Board() {
  const { data, error, isPending } = useStats()
  return (
    <>
      {isPending && <p className="text-muted-foreground text-sm">Connecting to the router.</p>}
      {error && (
        <p className="text-critical max-w-prose text-sm">
          The router did not answer ({error.message}). Start it with <code>python app.py</code> in{" "}
          <code>router/</code> and reload.
        </p>
      )}
      {data && (
        <>
          <StatusStrip stats={data} />
          <div className="grid gap-6 lg:grid-cols-[2fr_3fr]">
            <Chat />
            <div className="flex min-w-0 flex-col gap-6">
              <RatesPanel stats={data} />
              <RecentChart recent={data.recent} />
              <NodesPanel cluster={data.cluster} />
            </div>
          </div>
        </>
      )}
    </>
  )
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings)
  const update = useCallback(
    (patch: Partial<Settings>) => {
      const next = { ...settings, ...patch }
      saveSettings(next)   // in the handler, not an effect
      setSettings(next)
    },
    [settings],
  )
  const prefersDark = useSyncExternalStore(subscribeSystemTheme, systemPrefersDark, () => false)
  const dark = settings.theme === "dark" || (settings.theme === "system" && prefersDark)
  const ctx = useMemo(() => ({ settings, update }), [settings, update])

  return (
    <QueryClientProvider client={queryClient}>
      <SettingsContext.Provider value={ctx}>
        <div className={`${dark ? "dark" : ""} bg-background text-foreground min-h-full`}>
          <main className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-5 sm:px-6">
            <header className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h1 className="text-xl font-semibold">PiHive</h1>
                <p className="text-muted-foreground text-sm">Local inference on Raspberry Pis, cloud when it has to be.</p>
              </div>
              <ThemeButton theme={settings.theme} onChange={(theme) => update({ theme })} />
            </header>
            <ModelPicker />
            <Board />
          </main>
        </div>
      </SettingsContext.Provider>
    </QueryClientProvider>
  )
}
