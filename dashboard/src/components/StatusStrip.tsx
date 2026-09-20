import type { CSSProperties } from "react"
import { Sparkline } from "@/components/Sparkline"
import { STATE_TONE, colorFor, fmt, tierLabel, tierRank, tierVars, type Stats, type WorkerStatus } from "@/lib/api"

// The front panel: one link light per node, the cluster state, and the figure the
// room asks about first. Everything below it is detail.

function Led({ tone, live, label }: { tone: string; live: boolean; label: string }) {
  return (
    <span className={`led ${live ? "led-live" : ""}`} style={{ "--led": tone } as CSSProperties} role="img" aria-label={label} />
  )
}

function workerLed(w: WorkerStatus): { tone: string; live: boolean; label: string } {
  if (!w.alive) return { tone: "var(--critical)", live: false, label: "unreachable" }
  if (w.in_set) return { tone: "var(--good)", live: true, label: "serving" }
  return { tone: "var(--warn)", live: false, label: "standing by" }
}

export function StatusStrip({ stats, stale }: { stats: Stats; stale: boolean }) {
  const c = stats.cluster
  const state = c.state ?? stats.cluster_status
  const tone = STATE_TONE[state] ?? STATE_TONE.unknown
  const workers = c.workers ?? []
  const rootUp = state !== "down" && state !== "unreachable" && state !== "unknown"
  const local = stats.rates.cluster
  const cloud = Object.entries(stats.rates)
    .filter(([k]) => k !== "cluster" && k !== "cache")
    .sort(([a], [b]) => tierRank(a) - tierRank(b))[0]
  const old = stats.status_age_s != null && stats.status_age_s > 10
  const trend = stats.recent
    .filter((r) => r.served_by === "cluster" && r.decode_tps != null)
    .slice(-20)
    .map((r) => r.decode_tps as number)
  const pct = stats.pct_local ?? 0

  return (
    <section
      aria-label="Cluster front panel"
      className={`border-border grid gap-5 border-y py-4 transition-opacity md:grid-cols-[1fr_auto] md:items-center ${stale ? "opacity-60" : ""}`}
    >
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <span className="flex items-center gap-2">
          <Led tone={tone} live={false} label={`cluster ${state}`} />
          <span className="font-medium capitalize">{state}</span>
          {c.nodes_active != null && (
            <span className="text-muted-foreground">
              {c.model_max_nodes != null && c.nodes_total != null && c.model_max_nodes < c.nodes_total
                ? `${c.nodes_active} serving, ${c.nodes_total} in the cluster`
                : `${c.nodes_active} of ${c.nodes_total} nodes`}
            </span>
          )}
          {stale && <span className="text-critical text-sm">router not answering</span>}
          {!stale && old && <span className="text-warn text-sm">status {Math.round(stats.status_age_s ?? 0)} s old</span>}
        </span>
        <ul className="flex flex-wrap items-center gap-x-5 gap-y-2" aria-label="Nodes">
          <li className="flex items-center gap-2">
            <Led tone={rootUp ? "var(--good)" : "var(--critical)"} live={rootUp && !stale} label={rootUp ? "serving" : "down"} />
            <span className="text-sm">root</span>
          </li>
          {workers.map((w) => {
            const led = workerLed(w)
            return (
              <li key={w.host} className="flex items-center gap-2">
                <Led tone={led.tone} live={led.live && !stale} label={led.label} />
                <span className="text-sm">{w.host.split(".").pop()}</span>
              </li>
            )
          })}
          {workers.length === 0 && <li className="text-muted-foreground text-sm">no supervisor in reach</li>}
        </ul>
      </div>
      <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
        <div className="flex items-end gap-3">
          <div>
            <div className="figure tier-text text-5xl" style={tierVars("cluster")}>
              {fmt(local?.decode_tps)}
            </div>
            <div className="text-muted-foreground mt-1.5 text-sm">tok/s on the Pis</div>
          </div>
          <Sparkline values={trend} color={colorFor("cluster")} />
        </div>
        <div className="min-w-36">
          <div className="figure text-2xl">{pct}%</div>
          <div
            className="bg-muted mt-2 h-1.5 w-full overflow-hidden rounded-full"
            role="meter"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Share of answers served on the Pis"
          >
            <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: colorFor("cluster") }} />
          </div>
          <div className="text-muted-foreground mt-1.5 text-sm">served on the Pis</div>
        </div>
        {cloud && (
          <div>
            <div className="figure text-2xl">{fmt(cloud[1].decode_tps)}</div>
            <div className="text-muted-foreground mt-1.5 text-sm">tok/s on {tierLabel(cloud[0])}</div>
          </div>
        )}
      </div>
    </section>
  )
}
