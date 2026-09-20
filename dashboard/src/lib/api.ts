// Types for what the router serves. Field names mirror router/app.py (`stats()`) and
// cluster/supervisor/status.example.json; everything optional is optional there too.

export interface Telemetry {
  temp_c: number | null
  throttled: string | null
  flags: string[]
  mem_available_mb: number | null
  mem_total_mb: number | null
  load1: number | null
  cpu_mhz: number | null
  uptime_s: number | null
}

export interface WorkerStatus {
  host: string
  port: number
  alive: boolean | null
  in_set: boolean
  consecutive_fails: number
  telemetry: Telemetry | null
}

export interface ClusterStatus {
  state?: string
  reason?: string
  nodes_total?: number
  nodes_active?: number
  min_nodes?: number
  valid_node_counts?: number[]
  active_workers?: string[]
  workers?: WorkerStatus[]
  root?: { model?: string; load_seconds?: number | null; telemetry?: Telemetry | null }
  generation?: number
  restarts?: number
  events?: { t: number; msg: string }[]
}

export interface RateSummary {
  n: number
  inflight: number
  decode_tps: number | null
  decode_tps_p50: number | null
  decode_tps_p95: number | null
  prefill_tps: number | null
  tps: number | null
  ttft_ms: number | null
  ttft_ms_p50: number | null
  ttft_ms_p95: number | null
  latency_ms: number | null
  latency_ms_p50: number | null
  latency_ms_p95: number | null
}

export interface RecentAnswer {
  request_id?: string
  served_by: string
  routed_to: string
  reason: string
  fallback: boolean
  stream: boolean
  latency_ms: number
  ttft_ms?: number
  prompt_tokens: number
  gen_tokens?: number
  tokens_source?: "usage" | "chunks" | "chars"
  prefill_tps?: number
  decode_tps?: number
  tps?: number
  nodes_active?: number | null
  cluster_state?: string
  ts: number
}

export interface SolanaSummary {
  cluster: string
  explorer: string
  initialized: boolean
  sent: number
  errors: number
  pending_jobs: number
  payer: string
  last: Record<string, unknown> | null
  recent: ({ t: number; kind: string; signature: string; explorer: string } & Record<string, unknown>)[]
}

export interface Stats {
  total_requests: number
  served: Record<string, number>
  pct_local: number | null
  pct_by_upstream: Record<string, number>
  tiers: string[]
  by_reason: Record<string, number>
  fallbacks: Record<string, number>
  cluster_status: string
  status_age_s: number | null
  breakers_open_s: Record<string, number>
  rates: Record<string, RateSummary>
  inflight: Record<string, number>
  recent: RecentAnswer[]
  cluster: ClusterStatus
  solana: SolanaSummary | null
}

import { ROUTER_URL } from "@/lib/config"

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${ROUTER_URL}${path}`, { cache: "no-store" })
  if (!r.ok) throw new Error(`${path} answered ${r.status}`)
  return (await r.json()) as T
}

export const getStats = () => getJson<Stats>("/stats")

export interface Model {
  id: string
  owned_by: string // the tier that serves it: cluster, baseten, openai, gemini, snowflake
}

export async function getModels(): Promise<Model[]> {
  const body = await getJson<{ data: Model[] }>("/v1/models")
  return body.data.filter((m) => !m.id.endsWith("-heavy"))
}

// One color per upstream everywhere on the page; checked for contrast on both grounds.
export const UPSTREAM_COLOR: Record<string, string> = {
  cluster: "#199e70",
  baseten: "#3987e5",
  openai: "#d95926",
  gemini: "#c98500",
  snowflake: "#d55181",
  cache: "#8a8a8a",
}

export const colorFor = (upstream: string): string => UPSTREAM_COLOR[upstream] ?? "#8a8a8a"

export const STATE_TONE: Record<string, string> = {
  healthy: "var(--good)",
  degraded: "var(--warn)",
  restarting: "var(--chart-2)",
  down: "var(--critical)",
  unreachable: "var(--muted-foreground)",
  unknown: "var(--muted-foreground)",
}

export const fmt = (n: number | null | undefined, digits = 1): string =>
  n == null || Number.isNaN(n) ? "–" : n.toFixed(digits)
