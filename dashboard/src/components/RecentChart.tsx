import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { colorFor, type RecentAnswer } from "@/lib/api"

interface Point {
  i: number
  tps: number
  answer: RecentAnswer
}

function TooltipBody({ active, payload }: { active?: boolean; payload?: { payload: Point }[] }) {
  if (!active || !payload?.[0]) return null
  const a = payload[0].payload.answer
  return (
    <div className="bg-popover text-popover-foreground border-border rounded-md border px-3 py-2 text-xs shadow-md">
      <div className="font-medium" style={{ color: colorFor(a.served_by) }}>
        {a.served_by}, {a.reason}
        {a.fallback ? ", after a fallback" : ""}
      </div>
      <div>
        {a.decode_tps ?? a.tps ?? "–"} tok/s, first token in {a.ttft_ms ?? "–"} ms, {a.latency_ms} ms in all
      </div>
      <div className="text-muted-foreground">
        {a.gen_tokens ?? "?"} tokens ({a.tokens_source}), {a.nodes_active ?? "?"} nodes, cluster {a.cluster_state}
      </div>
    </div>
  )
}

export function RecentChart({ recent }: { recent: RecentAnswer[] }) {
  const data: Point[] = recent.slice(-30).map((answer, i) => ({ i, tps: answer.decode_tps ?? answer.tps ?? 0, answer }))
  return (
    <section aria-label="Recent answers">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-base font-medium">Last {data.length || ""} answers</h2>
        <span className="text-muted-foreground text-sm">decode tok/s, colored by who served it</span>
      </div>
      <div className="h-40">
        {data.length === 0 ? (
          <p className="text-muted-foreground text-sm">Each answer becomes a bar here.</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <XAxis dataKey="i" hide />
              <YAxis tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} width={44} axisLine={false} tickLine={false} />
              <Tooltip content={<TooltipBody />} cursor={{ fill: "color-mix(in oklab, var(--foreground) 8%, transparent)" }} />
              <Bar dataKey="tps" radius={[2, 2, 0, 0]} isAnimationActive={false}>
                {data.map((p) => (
                  <Cell key={p.i} fill={colorFor(p.answer.served_by)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  )
}
