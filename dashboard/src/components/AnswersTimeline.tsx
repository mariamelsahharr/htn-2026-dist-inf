import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { colorFor, type ClusterStatus, type RecentAnswer } from "@/lib/api"
import { whyRouted } from "@/lib/words"

const SLOTS = 30
const TIER_LABEL: Record<string, string> = {
  cluster: "Pis",
  baseten: "Baseten",
  openai: "OpenAI",
  gemini: "Gemini",
  snowflake: "Snowflake",
  cache: "cache",
}

interface Slot {
  i: number
  tps: number | null
  ttft: number | null
  answer: RecentAnswer | null
}

interface Mark {
  slot: number
  msg: string
}

// Fixed slots, newest at the right: one answer is one narrow bar and the chart fills up
// over time instead of stretching one answer across the width.
function toSlots(recent: RecentAnswer[]): Slot[] {
  const tail = recent.slice(-SLOTS)
  const pad = SLOTS - tail.length
  return Array.from({ length: SLOTS }, (_, i) => {
    const answer = i >= pad ? tail[i - pad] : null
    return {
      i,
      tps: answer ? (answer.decode_tps ?? answer.tps ?? 0) : null,
      ttft: answer?.ttft_ms ?? null,
      answer,
    }
  })
}

// Supervisor events land on the first answer that came after them: one mark per slot,
// the latest event wins, and the label is the state word ("degraded"), not the sentence.
function toMarks(slots: Slot[], events: ClusterStatus["events"]): Mark[] {
  const bySlot = new Map<number, string>()
  for (const e of events ?? []) {
    const slot = slots.find((s) => s.answer && s.answer.ts >= e.t)
    if (slot && slots.some((s) => s.answer && s.answer.ts < e.t)) {
      const state = /state=(\w+)/.exec(e.msg)?.[1]
      bySlot.set(slot.i, state ?? e.msg.split(":")[0].slice(0, 24))
    }
  }
  return [...bySlot.entries()].map(([slot, msg]) => ({ slot, msg }))
}

function AnswerTip({ active, payload }: { active?: boolean; payload?: { payload: Slot }[] }) {
  const a = payload?.[0]?.payload.answer
  if (!active || !a) return null
  return (
    <div className="bg-popover text-popover-foreground border-border max-w-64 rounded-md border px-3 py-2 text-xs shadow-md">
      <div className="font-medium" style={{ color: colorFor(a.served_by) }}>
        {TIER_LABEL[a.served_by] ?? a.served_by}, {whyRouted(a.reason)}
        {a.fallback && !a.reason.includes("fallback") ? ", after a fallback" : ""}
      </div>
      <div>
        {a.decode_tps ?? a.tps ?? "–"} tok/s, first token {a.ttft_ms ?? "–"} ms, {a.latency_ms} ms in all
      </div>
      <div className="text-muted-foreground">
        {a.gen_tokens ?? "?"} tokens ({a.tokens_source}), {a.nodes_active ?? "?"} nodes, cluster {a.cluster_state}
      </div>
    </div>
  )
}

function Legend({ recent }: { recent: RecentAnswer[] }) {
  const present = [...new Set(recent.map((r) => r.served_by))]
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm" aria-label="Who served">
      {present.map((up) => (
        <li key={up} className="flex items-center gap-1.5">
          <span className="inline-block size-2 rounded-full" style={{ backgroundColor: colorFor(up) }} />
          {TIER_LABEL[up] ?? up}
        </li>
      ))}
    </ul>
  )
}

function Chart({
  slots,
  marks,
  dataKey,
  unit,
  height,
  withMarks,
}: {
  slots: Slot[]
  marks: Mark[]
  dataKey: "tps" | "ttft"
  unit: string
  height: number
  withMarks: boolean
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={slots} margin={{ top: withMarks ? 14 : 4, right: 4, bottom: 0, left: 0 }} barCategoryGap={2}>
        <XAxis dataKey="i" hide />
        <YAxis
          width={48}
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => `${v}${unit}`}
        />
        <Tooltip content={<AnswerTip />} cursor={{ fill: "color-mix(in oklab, var(--foreground) 8%, transparent)" }} />
        {marks.map((m) => (
          <ReferenceLine
            key={`${m.slot}-${m.msg}`}
            x={m.slot}
            stroke="var(--warn)"
            strokeDasharray="3 3"
            label={
              withMarks
                ? { value: m.msg, position: "top", fontSize: 10, fill: "var(--warn)" }
                : undefined
            }
          />
        ))}
        <Bar dataKey={dataKey} radius={[2, 2, 0, 0]} isAnimationActive={false} maxBarSize={12}>
          {slots.map((s) => (
            <Cell key={s.i} fill={s.answer ? colorFor(s.answer.served_by) : "transparent"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function AnswersTimeline({ recent, cluster }: { recent: RecentAnswer[]; cluster: ClusterStatus }) {
  const slots = toSlots(recent)
  const marks = toMarks(slots, cluster.events)
  return (
    <section aria-label="Answers over time">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-medium">Last {Math.min(recent.length, SLOTS) || ""} answers</h2>
        <Legend recent={recent} />
      </div>
      {recent.length === 0 ? (
        <p className="text-muted-foreground text-sm">Nothing answered yet. Ask something on the left and it lands here.</p>
      ) : (
        <div className="space-y-1">
          <div className="text-muted-foreground text-xs">decode tok/s</div>
          <Chart slots={slots} marks={marks} dataKey="tps" unit="" height={140} withMarks />
          <div className="text-muted-foreground text-xs">first token, ms</div>
          <Chart slots={slots} marks={marks} dataKey="ttft" unit="" height={90} withMarks={false} />
        </div>
      )}
    </section>
  )
}
