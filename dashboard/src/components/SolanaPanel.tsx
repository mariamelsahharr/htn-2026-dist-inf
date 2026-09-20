import { ExternalLink } from "lucide-react"
import { Panel } from "@/components/Panel"
import { tierLabel, type ChainEvent, type SolanaSummary } from "@/lib/api"

const KIND: Record<string, string> = {
  initialize: "cluster account created",
  register_node: "node registered",
  set_worker_set: "worker set changed",
  commit_job: "answer committed",
}

function describe(row: ChainEvent): string {
  switch (row.kind) {
    case "register_node":
      return row.host ?? ""
    case "set_worker_set":
      return `${row.state ?? ""}, ${row.active?.length ?? 0} active`
    case "commit_job":
      return `by ${tierLabel(row.served_by ?? "?")}, sha ${(row.result_sha256 ?? "").slice(0, 10)}`
    default:
      return ""
  }
}

// What the cluster has written to Solana: the account every judge can open, and the last
// events with their signatures. Devnet only; the Pis never touch the chain.
export function SolanaPanel({ solana }: { solana: SolanaSummary }) {
  const status = !solana.initialized
    ? "waiting for the first supervisor status"
    : `${solana.sent} transactions this session` +
      (solana.pending_jobs > 0 ? `, ${solana.pending_jobs} waiting to send` : "") +
      (solana.errors > 0 ? `, ${solana.errors} retried` : "")
  return (
    <Panel
      label="On chain"
      aside={
        <a
          href={solana.explorer}
          target="_blank"
          rel="noreferrer"
          className="hover:text-foreground flex items-center gap-1 underline-offset-2 hover:underline"
        >
          cluster account on Solana Devnet <ExternalLink className="size-3.5" />
        </a>
      }
    >
      <p className="text-muted-foreground mb-2 text-sm">
        {status}
        {solana.alive === false && <span className="text-critical"> · attestor stopped</span>}
      </p>
      {solana.recent.length === 0 ? (
        <p className="text-muted-foreground text-sm">Nothing written yet. The first worker-set change or answer lands here.</p>
      ) : (
        <ul className="divide-border divide-y">
          {[...solana.recent].reverse().map((row) => (
            <li key={row.signature} className="flex items-baseline gap-x-3 py-1.5 text-sm">
              <time className="text-muted-foreground shrink-0 text-xs" dateTime={new Date(row.t * 1000).toISOString()}>
                {new Date(row.t * 1000).toLocaleTimeString()}
              </time>
              <span className="min-w-0 flex-1 truncate">
                <span className="whitespace-nowrap">{KIND[row.kind] ?? row.kind}</span>
                <span className="text-muted-foreground ml-2 text-xs">{describe(row)}</span>
              </span>
              <a
                href={row.explorer}
                target="_blank"
                rel="noreferrer"
                className="text-muted-foreground hover:text-foreground flex shrink-0 items-center gap-1 font-mono text-xs underline-offset-2 hover:underline"
              >
                {row.signature.slice(0, 8)}… <ExternalLink className="size-3" />
              </a>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  )
}
