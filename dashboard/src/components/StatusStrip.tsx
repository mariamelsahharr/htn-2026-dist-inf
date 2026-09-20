import type { CSSProperties } from "react"
import { STATE_TONE, fmt, type Stats } from "@/lib/api"

// The front panel: one link light per node, the cluster state, and the figure the
// room asks about first. Everything below it is detail.
export function StatusStrip({ stats }: { stats: Stats }) {
  const c = stats.cluster
  const state = c.state ?? stats.cluster_status
  const tone = STATE_TONE[state] ?? STATE_TONE.unknown
  const workers = c.workers ?? []
  const rootUp = state !== "down" && state !== "unreachable" && state !== "unknown"
  const local = stats.rates.cluster
  const cloud = Object.entries(stats.rates).find(([k]) => k !== "cluster" && k !== "cache")
  const stale = stats.status_age_s != null && stats.status_age_s > 10

  return (
    <section aria-label="Cluster front panel" className="border-border grid gap-4 border-y py-4 md:grid-cols-[1fr_auto] md:items-center">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <span className="flex items-center gap-2">
          <span className="led" style={{ "--led": tone } as CSSProperties} />
          <span className="font-medium capitalize">{state}</span>
          {c.nodes_active != null && (
            <span className="text-muted-foreground">
              {c.nodes_active} of {c.nodes_total} nodes
            </span>
          )}
          {stale && <span className="text-warn">status {Math.round(stats.status_age_s ?? 0)}s old</span>}
        </span>
        <ul className="flex flex-wrap items-center gap-x-5 gap-y-2" aria-label="Nodes">
          <li className="flex items-center gap-2">
            <span
              className={`led ${rootUp ? "led-live" : ""}`}
              style={{ "--led": rootUp ? "var(--good)" : "var(--critical)" } as CSSProperties}
            />
            <span className="text-sm">root</span>
          </li>
          {workers.map((w) => {
            const led = w.alive ? (w.in_set ? "var(--good)" : "var(--warn)") : "var(--critical)"
            return (
              <li key={w.host} className="flex items-center gap-2">
                <span className={`led ${w.in_set ? "led-live" : ""}`} style={{ "--led": led } as CSSProperties} />
                <span className="text-sm">{w.host.split(".").pop()}</span>
              </li>
            )
          })}
          {workers.length === 0 && <li className="text-muted-foreground text-sm">no supervisor in reach</li>}
        </ul>
      </div>
      <div className="flex items-baseline gap-6">
        <div>
          <div className="text-4xl leading-none font-semibold" style={{ color: "#199e70" }}>
            {fmt(local?.decode_tps)}
          </div>
          <div className="text-muted-foreground mt-1 text-sm">tok/s on the Pis</div>
        </div>
        {cloud && (
          <div>
            <div className="text-2xl leading-none font-medium" style={{ color: "var(--foreground)" }}>
              {fmt(cloud[1].decode_tps)}
            </div>
            <div className="text-muted-foreground mt-1 text-sm">tok/s on {cloud[0]}</div>
          </div>
        )}
      </div>
    </section>
  )
}
