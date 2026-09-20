import { ExternalLink } from "lucide-react"
import type { SolanaSummary } from "@/lib/api"

const KIND: Record<string, string> = {
  initialize: "cluster account created",
  register_node: "node registered",
  set_worker_set: "worker set changed",
  commit_job: "answer committed",
}

function describe(row: SolanaSummary["recent"][number]): string {
  switch (row.kind) {
    case "register_node":
      return String(row.host ?? "")
    case "set_worker_set":
      return `${row.state ?? ""}, ${(row.active as string[] | undefined)?.length ?? 0} active`
    case "commit_job":
      return `by ${row.served_by ?? "?"}, sha ${String(row.result_sha256 ?? "").slice(0, 10)}`
    default:
      return ""
  }
}

// What the cluster has written to Solana: the account every judge can open, and the last
// events with their signatures. Devnet only; the Pis never touch the chain.
export function SolanaPanel({ solana }: { solana: SolanaSummary }) {
  return (
    <section aria-label="On chain">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-medium">On chain</h2>
        <a
          href={solana.explorer}
          target="_blank"
          rel="noreferrer"
          className="text-muted-foreground hover:text-foreground flex items-center gap-1 text-sm underline-offset-2 hover:underline"
        >
          cluster account on Solana Devnet <ExternalLink className="size-3.5" />
        </a>
      </div>
      <p className="text-muted-foreground mb-2 text-sm">
        {solana.initialized ? `${solana.sent} transactions this session` : "waiting for the first supervisor status"}
        {solana.pending_jobs > 0 ? `, ${solana.pending_jobs} waiting to send` : ""}
        {solana.errors > 0 ? `, ${solana.errors} retried` : ""}
      </p>
      {solana.recent.length === 0 ? (
        <p className="text-muted-foreground text-sm">Nothing written yet. The first worker-set change or answer lands here.</p>
      ) : (
        <ul className="divide-border divide-y">
          {[...solana.recent].reverse().map((row) => (
            <li key={row.signature} className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 py-1.5 text-sm">
              <span className="flex min-w-0 items-baseline gap-2">
                <time className="text-muted-foreground shrink-0 text-xs" dateTime={new Date(row.t * 1000).toISOString()}>
                  {new Date(row.t * 1000).toLocaleTimeString()}
                </time>
                <span>{KIND[row.kind] ?? row.kind}</span>
                <span className="text-muted-foreground truncate text-xs">{describe(row)}</span>
              </span>
              <a
                href={row.explorer}
                target="_blank"
                rel="noreferrer"
                className="text-muted-foreground hover:text-foreground flex shrink-0 items-center gap-1 text-xs underline-offset-2 hover:underline"
              >
                {row.signature.slice(0, 8)}… <ExternalLink className="size-3" />
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
