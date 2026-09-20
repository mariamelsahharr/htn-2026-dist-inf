import { Panel } from "@/components/Panel"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { UpstreamBadge } from "@/components/UpstreamBadge"
import { fmt, tierRank, type Stats } from "@/lib/api"
import { whyFellBack } from "@/lib/words"

export function RatesPanel({ stats }: { stats: Stats }) {
  const rows = Object.entries(stats.rates).sort(([a], [b]) => tierRank(a) - tierRank(b))
  const fallbacks = Object.entries(stats.fallbacks)
  return (
    <Panel
      label="Throughput"
      aside={
        <>
          {stats.total_requests} answered, {stats.pct_local ?? 0}% on the Pis
        </>
      }
    >
      {rows.length === 0 ? (
        <p className="text-muted-foreground text-sm">No answers yet. The first message fills this in.</p>
      ) : (
        <div className="overflow-x-auto" tabIndex={0}>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Upstream</TableHead>
                <TableHead className="text-right">Decode tok/s</TableHead>
                <TableHead className="text-right">Prefill tok/s</TableHead>
                <TableHead className="text-right">First token</TableHead>
                <TableHead className="text-right">Slowest 5%</TableHead>
                <TableHead className="text-right">Answers</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(([up, r]) => (
                <TableRow key={up}>
                  <TableCell>
                    <span className="flex flex-wrap items-center gap-2">
                      <UpstreamBadge upstream={up} />
                      {r.inflight > 0 && <span className="text-muted-foreground text-xs">{r.inflight} in flight</span>}
                      {stats.breakers_open_s[up] != null && (
                        <span className="text-critical text-xs">paused {Math.round(stats.breakers_open_s[up])} s</span>
                      )}
                    </span>
                  </TableCell>
                  <TableCell className="text-right font-medium">
                    {fmt(r.decode_tps)}
                    <span className="text-muted-foreground text-xs font-normal"> median {fmt(r.decode_tps_p50)}</span>
                  </TableCell>
                  <TableCell className="text-right">{fmt(r.prefill_tps, 0)}</TableCell>
                  <TableCell className="text-right">{fmt(r.ttft_ms_p50, 0)} ms</TableCell>
                  <TableCell className="text-right">{fmt(r.latency_ms_p95, 0)} ms</TableCell>
                  <TableCell className="text-muted-foreground text-right">{r.n}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
      {fallbacks.length > 0 && (
        <p className="text-muted-foreground mt-2 text-xs">
          Fallbacks: {fallbacks.map(([k, v]) => `${whyFellBack(k)} (${v})`).join("; ")}
        </p>
      )}
    </Panel>
  )
}
