import type { ReactNode } from "react"

// One rhythm for every panel on the board: a hairline, a silkscreen label, an aside
// with the figure or link that matters, then the content. Nothing is a card except
// the chat, which is the one thing you operate.
export function Panel({ label, aside, children }: { label: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section aria-label={label} className="border-border border-t pt-3">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 className="silk">{label}</h2>
        {aside && <div className="text-muted-foreground text-sm">{aside}</div>}
      </div>
      {children}
    </section>
  )
}
