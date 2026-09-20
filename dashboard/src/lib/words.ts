// The router speaks in log keys; the page speaks in sentences. One place for the mapping.

const REASONS: Record<string, string> = {
  default_local: "short question",
  over_size_threshold: "long prompt",
  complex_task_code: "pasted code",
  complex_task_turns: "long conversation",
  tools_attached: "tool call",
  model_pinned: "your pick",
  forced_by_header: "forced",
  heavy_model_tag: "heavy model asked for",
  escalate_header: "escalation asked for",
  answer_cache: "same question just now",
  demo_cache: "canned answer, nothing was up",
  cluster_unreachable: "Pis unreachable",
  cluster_down: "Pis down",
  cluster_restarting: "Pis restarting",
  cluster_unknown: "Pis not checked yet",
  cluster_degraded_below_min: "too few Pis",
}

export const TIER_NAME: Record<string, string> = {
  cluster: "Pis",
  baseten: "Baseten",
  openai: "OpenAI",
  gemini: "Gemini",
  snowflake: "Snowflake",
  cache: "cache",
}

export function whyRouted(reason: string): string {
  const [base, suffix] = reason.split("+")
  const core = base.replace(/_no_cloud$/, "")
  let text = REASONS[core] ?? core.replaceAll("_", " ")
  if (base.endsWith("_no_cloud")) text += ", no cloud configured"
  if (suffix === "fallback") text += ", after a fallback"
  return text
}

// "pre_commit:UpstreamError: HTTP 403: {\"error\":\"please..." -> "before the first token: HTTP 403"
export function whyFellBack(key: string): string {
  const [stage, ...rest] = key.split(":")
  const detail = rest.join(":").replace(/^\s*\w*Error:\s*/, "").trim()
  const STAGE: Record<string, string> = {
    pre_commit: "before the first token",
    blocking: "on a blocking call",
    mid_stream: "mid-answer",
    breaker_open: "paused",
  }
  const short = detail.split(/[{(]/)[0].trim().slice(0, 48)
  return `${STAGE[stage] ?? stage}${short ? `: ${short}` : ""}`
}
