import { fmt, type ClusterStatus, type Telemetry } from "@/lib/api"

function Reading({ t }: { t: Telemetry | null | undefined }) {
  if (!t) return <span className="text-muted-foreground text-sm">no telemetry</span>
  const hot = (t.temp_c ?? 0) >= 80
  const throttled = t.flags.some((f) => !f.endsWith("_since_boot"))
  return (
    <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
      <span className={hot ? "text-critical" : ""}>{fmt(t.temp_c)} °C</span>
      <span>{t.cpu_mhz ?? "–"} MHz</span>
      <span className="text-muted-foreground">{t.mem_available_mb ?? "–"} MB free</span>
      <span className="text-muted-foreground">load {fmt(t.load1)}</span>
      {throttled && <span className="text-warn">throttling now</span>}
      {!throttled && t.flags.length > 0 && <span className="text-muted-foreground">throttled earlier</span>}
    </span>
  )
}

export function NodesPanel({ cluster }: { cluster: ClusterStatus }) {
  const workers = cluster.workers ?? []
  return (
    <section aria-label="Nodes">
      <h2 className="mb-2 text-base font-medium">Nodes</h2>
      {cluster.reason && <p className="text-muted-foreground mb-2 text-sm">{cluster.reason}</p>}
      <ul className="divide-border divide-y">
        <li className="flex flex-wrap items-center justify-between gap-2 py-2">
          <span className="flex items-baseline gap-2">
            <span className="text-sm font-medium">root</span>
            <span className="text-muted-foreground text-xs">{cluster.root?.model?.split("/").pop()}</span>
            {cluster.root?.load_seconds != null && (
              <span className="text-muted-foreground text-xs">loaded in {fmt(cluster.root.load_seconds, 0)} s</span>
            )}
          </span>
          <Reading t={cluster.root?.telemetry} />
        </li>
        {workers.map((w) => (
          <li key={w.host} className="flex flex-wrap items-center justify-between gap-2 py-2">
            <span className="flex items-baseline gap-2">
              <span className="text-sm font-medium">{w.host}</span>
              <span className="text-muted-foreground text-xs">
                {w.alive ? (w.in_set ? "serving" : "standing by") : `unreachable, ${w.consecutive_fails} misses`}
              </span>
            </span>
            <Reading t={w.telemetry} />
          </li>
        ))}
        {workers.length === 0 && (
          <li className="text-muted-foreground py-2 text-sm">
            The supervisor on the root Pi is not answering, so there is nothing to show per node. The router still
            serves from the cloud.
          </li>
        )}
      </ul>
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
