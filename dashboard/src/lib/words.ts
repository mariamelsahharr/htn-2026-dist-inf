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
  const detail = rest.join(":").replace(/^\s*[A-Z][A-Za-z]*:\s*/, "").trim() // drop the exception class
  const STAGE: Record<string, string> = {
    pre_commit: "before the first token",
    blocking: "on a blocking call",
    mid_stream: "mid-answer",
    breaker_open: "paused",
  }
  const short = detail.split(/[{(]/)[0].replace(/[:\s]+$/, "").slice(0, 48)
  return `${STAGE[stage] ?? stage}${short ? `: ${short}` : ""}`
}

// Reasoning models (Qwen3 on the Pis) stream a <think>…</think> block before the answer.
// It is kept in the transcript for the next turn but not shown.
export function visibleText(text: string): string {
  const closed = text.replace(/<think>[\s\S]*?<\/think>\s*/g, "")
  const open = closed.indexOf("<think>")
  return open === -1 ? closed : closed.slice(0, open)
}

// A pasted wall of text is shown by its head; the full text still goes to the router.
export function shownText(text: string): string {
  return text.length > 320 ? `${text.slice(0, 280).trimEnd()}… (${text.length.toLocaleString()} characters)` : text
}

export const gigahertz = (mhz: number | null | undefined): string => (mhz == null ? "–" : `${(mhz / 1000).toFixed(1)} GHz`)
export const gigabytes = (mb: number | null | undefined): string => (mb == null ? "–" : `${(mb / 1024).toFixed(1)} GB`)
