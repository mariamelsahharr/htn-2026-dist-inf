import { Panel } from "@/components/Panel"
import { colorFor, fmt, type ClusterStatus, type Telemetry, type WorkerStatus } from "@/lib/api"
import { gigabytes, gigahertz } from "@/lib/words"

// One grid for every node so temperatures sit under temperatures.
// Columns: name, state, temperature, clock, free memory, load.
const GRID = "grid grid-cols-[minmax(7rem,1.4fr)_7.5rem_minmax(7.5rem,1fr)_4.5rem_5rem_4.5rem] items-center gap-x-4"

function Thermometer({ c }: { c: number }) {
  const hot = c >= 80
  return (
    <span className="flex items-center gap-2">
      <span className={`text-sm ${hot ? "text-critical font-medium" : ""}`}>{fmt(c)} °C</span>
      <span className="bg-muted relative h-1.5 w-14 overflow-hidden rounded-full" aria-hidden="true">
        <span
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${Math.min(100, c)}%`, backgroundColor: hot ? "var(--critical)" : colorFor("cluster") }}
        />
        <span className="absolute inset-y-0 w-px" style={{ left: "80%", backgroundColor: "var(--warn)" }} />
      </span>
    </span>
  )
}

function Readings({ t }: { t: Telemetry | null | undefined }) {
  if (!t) return <span className="text-muted-foreground col-span-4 text-sm">no telemetry</span>
  return (
    <>
      {t.temp_c == null ? <span className="text-muted-foreground text-sm">–</span> : <Thermometer c={t.temp_c} />}
      <span className="text-sm">{gigahertz(t.cpu_mhz)}</span>
      <span className="text-muted-foreground text-sm">{gigabytes(t.mem_available_mb)} free</span>
      <span className="text-muted-foreground text-sm">load {fmt(t.load1)}</span>
    </>
  )
}

// A worker process is present when it is either waiting for the root (listening) or attached
// to it (an established connection); neither means the process is gone.
const workerGone = (t: Telemetry | null | undefined): boolean =>
  t?.worker_listening === false && !(t.worker_connections ?? 0)

// State plus the one flag that matters: a node throttling right now, or a worker whose
// process is gone. The root runs dllama-api, not a worker, so it is never checked for one.
function State({
  text,
  tone,
  t,
  worker = true,
}: {
  text: string
  tone: "good" | "warn" | "critical" | "muted"
  t?: Telemetry | null
  worker?: boolean
}) {
  const throttling = t?.flags.some((f) => !f.endsWith("_since_boot")) ?? false
  const color = { good: "text-foreground", warn: "text-warn", critical: "text-critical", muted: "text-muted-foreground" }[tone]
  return (
    <span className="flex flex-col text-xs leading-tight">
      <span className={color}>{text}</span>
      {throttling && <span className="text-warn">throttling</span>}
      {!throttling && worker && workerGone(t) && <span className="text-critical">no worker process</span>}
    </span>
  )
}

function workerState(w: WorkerStatus): { text: string; tone: "good" | "warn" | "critical" | "muted" } {
  if (!w.alive) return { text: `unreachable, ${w.consecutive_fails} misses`, tone: "critical" }
  return w.in_set ? { text: "serving", tone: "good" } : { text: "standing by", tone: "warn" }
}

export function NodesPanel({ cluster }: { cluster: ClusterStatus }) {
  const workers = cluster.workers ?? []
  const model = cluster.root?.model?.split("/").pop()?.replace(/^dllama_model_/, "").replace(/\.m$/, "")
  const rootDown = cluster.state === "down" || cluster.state === "unreachable"
  return (
    <Panel
      label="Nodes"
      aside={
        model && (
          <>
            {model}
            {cluster.root?.load_seconds != null ? `, loaded in ${fmt(cluster.root.load_seconds, 0)} s` : ""}
          </>
        )
      }
    >
      {cluster.reason && <p className="text-muted-foreground mb-2 text-sm">{cluster.reason}</p>}
      <div className="overflow-x-auto" tabIndex={0}>
        <ul className="divide-border min-w-[38rem] divide-y">
          <li className={`${GRID} py-2`}>
            <span className="text-sm font-medium">root</span>
            <State text={rootDown ? "down" : "serving"} tone={rootDown ? "critical" : "good"} t={cluster.root?.telemetry} worker={false} />
            <Readings t={cluster.root?.telemetry} />
          </li>
          {workers.map((w) => (
            <li key={w.host} className={`${GRID} py-2`}>
              <span className="text-sm font-medium">{w.host}</span>
              <State {...workerState(w)} t={w.telemetry} />
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
            .map((e, i) => (
              <li key={`${e.t}-${i}`}>
                <time dateTime={new Date(e.t * 1000).toISOString()}>{new Date(e.t * 1000).toLocaleTimeString()}</time>{" "}
                {e.msg}
              </li>
            ))}
        </ol>
      )}
    </Panel>
  )
}
