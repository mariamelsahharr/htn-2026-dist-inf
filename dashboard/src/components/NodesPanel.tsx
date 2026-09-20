import { fmt, type ClusterStatus, type Telemetry, type WorkerStatus } from "@/lib/api"

// One grid for every node so temperatures sit under temperatures. Columns: name, state,
// temperature, clock, free memory, load, flags.
const GRID = "grid grid-cols-[minmax(7rem,1.2fr)_minmax(5rem,1fr)_8rem_5.5rem_6.5rem_4rem_minmax(0,1fr)] items-center gap-x-4"

function Readings({ t }: { t: Telemetry | null | undefined }) {
  if (!t) {
    return (
      <>
        <span className="text-muted-foreground col-span-5 text-sm">no telemetry</span>
      </>
    )
  }
  const hot = (t.temp_c ?? 0) >= 80
  const throttled = t.flags.some((f) => !f.endsWith("_since_boot"))
  return (
    <>
      <span className="flex items-center gap-2">
        <span className={`text-sm ${hot ? "text-critical" : ""}`}>{fmt(t.temp_c)} °C</span>
        <span className="bg-muted relative h-1.5 w-12 overflow-hidden rounded-full" aria-hidden="true">
          <span
            className="absolute inset-y-0 left-0 rounded-full"
            style={{ width: `${Math.min(100, ((t.temp_c ?? 0) / 100) * 100)}%`, backgroundColor: hot ? "var(--critical)" : "#199e70" }}
          />
          <span className="absolute inset-y-0 w-px" style={{ left: "80%", backgroundColor: "var(--warn)" }} />
        </span>
      </span>
      <span className="text-sm">{t.cpu_mhz ?? "–"} MHz</span>
      <span className="text-muted-foreground text-sm">{t.mem_available_mb ?? "–"} MB free</span>
      <span className="text-muted-foreground text-sm">load {fmt(t.load1)}</span>
      <span className={`truncate text-sm ${throttled ? "text-warn" : "text-muted-foreground"}`}>
        {throttled ? "throttling now" : t.flags.length > 0 ? "throttled earlier" : ""}
      </span>
    </>
  )
}

function stateOf(w: WorkerStatus): string {
  if (!w.alive) return `unreachable, ${w.consecutive_fails} misses`
  return w.in_set ? "serving" : "standing by"
}

export function NodesPanel({ cluster }: { cluster: ClusterStatus }) {
  const workers = cluster.workers ?? []
  const model = cluster.root?.model?.split("/").pop()?.replace(/^dllama_model_/, "").replace(/\.m$/, "")
  return (
    <section aria-label="Nodes">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-medium">Nodes</h2>
        {model && (
          <span className="text-muted-foreground text-sm">
            {model}
            {cluster.root?.load_seconds != null ? `, loaded in ${fmt(cluster.root.load_seconds, 0)} s` : ""}
          </span>
        )}
      </div>
      {cluster.reason && <p className="text-muted-foreground mb-2 text-sm">{cluster.reason}</p>}
      <div className="overflow-x-auto">
        <ul className="divide-border min-w-[50rem] divide-y">
          <li className={`${GRID} py-2`}>
            <span className="text-sm font-medium">root</span>
            <span className="text-muted-foreground text-xs">serving</span>
            <Readings t={cluster.root?.telemetry} />
          </li>
          {workers.map((w) => (
            <li key={w.host} className={`${GRID} py-2`}>
              <span className="text-sm font-medium">{w.host}</span>
              <span className="text-muted-foreground text-xs">{stateOf(w)}</span>
              <Readings t={w.telemetry} />
            </li>
          ))}
          {workers.length === 0 && (
            <li className="text-muted-foreground py-2 text-sm">
              The supervisor on the root Pi is not answering, so there is nothing to show per node. The router still
              serves from the cloud.
            </li>
          )}
        </ul>
      </div>
      {cluster.events && cluster.events.length > 0 && (
        <ol className="text-muted-foreground mt-3 space-y-0.5 text-xs" aria-label="Recent supervisor events">
          {cluster.events
            .slice(-4)
            .reverse()
            .map((e) => (
              <li key={`${e.t}-${e.msg}`}>
                <time dateTime={new Date(e.t * 1000).toISOString()}>{new Date(e.t * 1000).toLocaleTimeString()}</time>{" "}
                {e.msg}
              </li>
            ))}
        </ol>
      )}
    </section>
  )
}
