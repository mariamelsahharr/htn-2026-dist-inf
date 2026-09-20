import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Panel } from "@/components/Panel"
import { colorFor, tierLabel, type ClusterStatus, type RecentAnswer } from "@/lib/api"
import { SLOTS, toMarks, toSlots, type Mark, type Slot } from "@/lib/timeline"
import { whyRouted } from "@/lib/words"

function AnswerTip({ active, payload }: { active?: boolean; payload?: { payload: Slot }[] }) {
  const a = payload?.[0]?.payload.answer
  if (!active || !a) return null
  return (
    <div className="bg-popover text-popover-foreground border-border max-w-64 rounded-md border px-3 py-2 text-xs shadow-md">
      <div className="font-medium" style={{ color: colorFor(a.served_by) }}>
        {tierLabel(a.served_by)}, {whyRouted(a.reason)}
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
          {tierLabel(up)}
        </li>
      ))}
    </ul>
  )
}

const EMPTY_SLOT = "color-mix(in oklab, var(--foreground) 6%, transparent)"

function Chart({
  slots,
  marks,
  dataKey,
  height,
  withMarks,
}: {
  slots: Slot[]
  marks: Mark[]
  dataKey: "tps" | "ttft"
  height: number
  withMarks: boolean
}) {
  // Empty slots draw as a faint 2% stub so the 24 positions read as an instrument.
  const max = Math.max(1, ...slots.map((s) => s[dataKey] ?? 0))
  const data = slots.map((s) => ({ ...s, value: s.answer ? (s[dataKey] ?? 0) : max * 0.02 }))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: withMarks ? 16 : 4, right: 4, bottom: 0, left: 0 }} barCategoryGap={3}>
        <XAxis dataKey="i" hide />
        <YAxis
          width={44}
          domain={[0, max]}
          tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => `${Math.round(v)}`}
        />
        <Tooltip content={<AnswerTip />} cursor={{ fill: "color-mix(in oklab, var(--foreground) 8%, transparent)" }} />
        {marks.map((m) => (
          <ReferenceLine
            key={`${m.slot}-${m.msg}`}
            x={m.slot}
            stroke="var(--warn)"
            strokeDasharray="3 3"
            label={withMarks ? { value: m.msg, position: "top", fontSize: 10, fill: "var(--warn)" } : undefined}
          />
        ))}
        <Bar dataKey="value" radius={[2, 2, 0, 0]} isAnimationActive={false} maxBarSize={18}>
          {data.map((s) => (
            <Cell key={s.i} fill={s.answer ? colorFor(s.answer.served_by) : EMPTY_SLOT} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function AnswersTimeline({ recent, cluster }: { recent: RecentAnswer[]; cluster: ClusterStatus }) {
  const slots = toSlots(recent)
  const marks = toMarks(slots, cluster.events)
  const shown = Math.min(recent.length, SLOTS)
  return (
    <Panel label={shown ? `Last ${shown} answers` : "Answers"} aside={<Legend recent={recent} />}>
      {recent.length === 0 ? (
        <p className="text-muted-foreground text-sm">Nothing answered yet. Ask something on the left and it lands here.</p>
      ) : (
        <div className="space-y-1">
          <div className="text-muted-foreground text-xs">decode tok/s</div>
          <Chart slots={slots} marks={marks} dataKey="tps" height={150} withMarks />
          <div className="text-muted-foreground text-xs">first token, ms</div>
          <Chart slots={slots} marks={marks} dataKey="ttft" height={96} withMarks={false} />
        </div>
      )}
    </Panel>
  )
}

export default AnswersTimeline
