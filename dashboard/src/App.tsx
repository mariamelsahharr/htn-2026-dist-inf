import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Moon, Sun, SunMoon } from "lucide-react"
import { lazy, Suspense, useSyncExternalStore } from "react"
import { Chat } from "@/components/Chat"
import { ModelPicker } from "@/components/ModelPicker"
import { NodesPanel } from "@/components/NodesPanel"
import { RatesPanel } from "@/components/RatesPanel"
import { SolanaPanel } from "@/components/SolanaPanel"
import { StatusStrip } from "@/components/StatusStrip"
import { Button } from "@/components/ui/button"
import { useStats } from "@/hooks/useStats"
import { ROUTER_URL } from "@/lib/config"
import { updateSettings, useSettings } from "@/lib/settings"
import { subscribeTheme, systemPrefersDark, type Theme } from "@/lib/theme"

// Recharts is the heaviest thing on the page and only the timeline needs it.
const AnswersTimeline = lazy(() => import("@/components/AnswersTimeline"))

const queryClient = new QueryClient()

// Seven Pis packed like cells: the router in copper at the centre, the nodes around it.
function HiveMark() {
  const ring = [0, 60, 120, 180, 240, 300].map((deg) => {
    const a = (deg * Math.PI) / 180
    return [14 + 8.2 * Math.cos(a), 14 + 8.2 * Math.sin(a)] as const
  })
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true" className="shrink-0">
      {ring.map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r="3.1" fill="var(--good)" />
      ))}
      <circle cx="14" cy="14" r="3.4" fill="var(--primary)" />
    </svg>
  )
}

function ThemeButton() {
  const { theme } = useSettings()
  const prefersDark = useSyncExternalStore(subscribeTheme, systemPrefersDark, () => false)
  const resolved = theme === "system" ? (prefersDark ? "dark" : "light") : theme
  const next: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" }
  const Icon = theme === "light" ? Sun : theme === "dark" ? Moon : SunMoon
  const label = theme === "system" ? `system, ${resolved}` : theme
  return (
    <Button variant="ghost" size="sm" onClick={() => updateSettings({ theme: next[theme] })} aria-label={`Theme: ${label}`}>
      <Icon className="size-4" /> <span className="capitalize">{label}</span>
    </Button>
  )
}

function Board() {
  const { data, error, isPending } = useStats()
  return (
    <>
      {isPending && <p className="text-muted-foreground text-sm">Connecting to the router.</p>}
      {error && (
        <p className="text-critical max-w-prose text-sm" role="alert">
          The router did not answer ({error.message}). It should be running at{" "}
          <code className="font-mono">{ROUTER_URL || window.location.origin}</code>.
          {data ? " Showing the last board it sent." : ""}
        </p>
      )}
      {data && (
        <>
          <StatusStrip stats={data} stale={error != null} />
          <div className="grid items-start gap-6 lg:grid-cols-[2fr_3fr]">
            <Chat />
            <div className="flex min-w-0 flex-col gap-6">
              <RatesPanel stats={data} />
              <Suspense fallback={<div className="border-border h-64 border-t" aria-hidden="true" />}>
                <AnswersTimeline recent={data.recent} cluster={data.cluster} />
              </Suspense>
              <NodesPanel cluster={data.cluster} />
              {data.solana && <SolanaPanel solana={data.solana} />}
            </div>
          </div>
        </>
      )}
    </>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="bg-background text-foreground min-h-full">
        <main className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-5 sm:px-6">
          <header className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <HiveMark />
              <div>
                <h1 className="text-lg leading-tight font-semibold">PiHive</h1>
                <p className="text-muted-foreground text-sm">Local inference on Raspberry Pis, cloud when it has to be.</p>
              </div>
            </div>
            <ThemeButton />
          </header>
          <ModelPicker />
          <Board />
        </main>
      </div>
    </QueryClientProvider>
  )
}
