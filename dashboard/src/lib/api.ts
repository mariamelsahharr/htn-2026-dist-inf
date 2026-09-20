// Types for what the router serves. Field names mirror router/service.py (`stats()`) and
// cluster/supervisor/status.example.json; everything optional is optional there too.

import type { CSSProperties } from "react"
import { ROUTER_URL } from "@/lib/config"

export interface Telemetry {
  temp_c: number | null
  throttled: string | null
  flags: string[]
  mem_available_mb: number | null
  mem_total_mb: number | null
  load1: number | null
  cpu_mhz: number | null
  uptime_s: number | null
  worker_listening?: boolean | null
  ts?: number
}

export interface WorkerStatus {
  host: string
  port: number
  alive: boolean | null
  in_set: boolean
  consecutive_fails: number
  last_seen?: number | null
  last_probe?: number | null
  telemetry: Telemetry | null
}

export interface ClusterStatus {
  state?: string
  reason?: string
  since?: number
  state_age_s?: number
  nodes_total?: number
  nodes_active?: number
  min_nodes?: number
  valid_node_counts?: number[]
  node_counts_source?: string
  model_header?: Record<string, number>
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

export interface ChainEvent {
  t: number
  kind: string
  signature: string
  explorer: string
  host?: string
  state?: string
  active?: string[]
  served_by?: string
  result_sha256?: string
}

export interface SolanaSummary {
  cluster: string
  explorer: string
  initialized: boolean
  alive?: boolean
  sent: number
  errors: number
  pending_jobs: number
  payer: string
  last: ChainEvent | null
  recent: ChainEvent[]
}

export interface Stats {
  total_requests: number
  served: Record<string, number>
  pct_local: number | null
  pct_by_upstream: Record<string, number>
  tiers: string[]
  tier_health: Record<string, string>
  by_reason: Record<string, number>
  fallbacks: Record<string, number>
  cluster_status: string
  status_age_s: number | null
  cluster_waiting?: number
  breakers_open_s: Record<string, number>
  rates: Record<string, RateSummary>
  inflight: Record<string, number>
  local_prefill_tps_estimate?: number | null
  recent: RecentAnswer[]
  cluster: ClusterStatus
  solana: SolanaSummary | null
}

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

// One color and one name per upstream, everywhere on the page. Text built from the
// color goes through .tier-text so it keeps contrast on both grounds.
export const UPSTREAM_COLOR = {
  cluster: "#199e70",
  baseten: "#3987e5",
  openai: "#d95926",
  gemini: "#c98500",
  snowflake: "#d55181",
  cache: "#8a8a8a",
} as const

export type Upstream = keyof typeof UPSTREAM_COLOR

export const TIER_LABEL: Record<Upstream, string> = {
  cluster: "Pis",
  baseten: "Baseten",
  openai: "OpenAI",
  gemini: "Gemini",
  snowflake: "Snowflake",
  cache: "cache",
}

export const TIER_ORDER: Upstream[] = ["cluster", "baseten", "openai", "gemini", "snowflake", "cache"]

const isUpstream = (up: string): up is Upstream => up in UPSTREAM_COLOR

export const colorFor = (upstream: string): string => (isUpstream(upstream) ? UPSTREAM_COLOR[upstream] : "#8a8a8a")
export const tierLabel = (upstream: string): string => (isUpstream(upstream) ? TIER_LABEL[upstream] : upstream)
export const tierVars = (upstream: string): CSSProperties => ({ "--tier": colorFor(upstream) }) as CSSProperties
export const tierRank = (upstream: string): number => (isUpstream(upstream) ? TIER_ORDER.indexOf(upstream) : 99)

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
