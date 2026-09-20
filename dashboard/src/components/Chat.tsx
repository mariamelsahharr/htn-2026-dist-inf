import { useQueryClient } from "@tanstack/react-query"
import { RotateCcw, SendHorizontal, Square } from "lucide-react"
import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { UpstreamBadge } from "@/components/UpstreamBadge"
import { STATS_KEY } from "@/hooks/useStats"
import { colorFor } from "@/lib/api"
import { isAbort, streamChat } from "@/lib/chat"
import { recentHistory } from "@/lib/history"
import { DEMO_PROMPTS } from "@/lib/demo"
import { useSettings } from "@/lib/settings"
import { appendMessages, patchMessage, setTranscript, useTranscript, type Message } from "@/lib/transcript"
import { shownText, visibleText, whyRouted } from "@/lib/words"

const clock = () => performance.now() // read in event handlers only, never during render

function Empty() {
  return (
    <div className="m-auto max-w-xs text-center">
      <p className="text-sm">Ask anything. The router decides who answers.</p>
      <ul className="text-muted-foreground mt-4 space-y-2 text-left text-sm">
        <li className="flex items-start gap-2">
          <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ backgroundColor: colorFor("cluster") }} />
          Anything that fits the Pis&apos; context stays on the Pis.
        </li>
        <li className="flex items-start gap-2">
          <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ backgroundColor: colorFor("openai") }} />
          Prompts too long for it, hundreds of lines of code, or a cluster that is down go to the cloud.
        </li>
        <li className="flex items-start gap-2">
          <span className="text-warn mt-0.5 shrink-0 text-xs">↳</span>
          If the Pis die mid-answer, the cloud finishes it and the badge says so.
        </li>
      </ul>
    </div>
  )
}

function Bubble({ m }: { m: Message }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="bg-primary text-primary-foreground max-w-[85%] rounded-lg px-3 py-2 text-sm break-words whitespace-pre-wrap">
          {shownText(m.text)}
        </div>
      </div>
    )
  }
  const text = visibleText(m.text)
  return (
    <div className="flex justify-start">
      <div className="bg-muted max-w-[85%] rounded-lg px-3 py-2 text-sm break-words">
        <div className="mb-1 flex flex-wrap items-center gap-2" role={m.ms != null ? "status" : undefined}>
          {m.served ? (
            <span className="flex items-center gap-1">
              <UpstreamBadge upstream={m.served.servedBy} />
              {m.served.continuedBy && (
                <>
                  <span className="text-warn text-xs">died, finished by</span>
                  <UpstreamBadge upstream={m.served.continuedBy} />
                </>
              )}
            </span>
          ) : (
            <span className="text-muted-foreground text-xs">routing</span>
          )}
          {m.served?.reason && <span className="text-muted-foreground text-xs">{whyRouted(m.served.reason)}</span>}
          {m.ms != null && <span className="text-muted-foreground text-xs">{m.ms} ms</span>}
        </div>
        <div className="whitespace-pre-wrap">{text || (m.text ? "thinking" : "")}</div>
        {m.error && <div className="text-critical mt-1 text-xs">{m.error}</div>}
      </div>
    </div>
  )
}

export function Chat() {
  const { model } = useSettings()
  const messages = useTranscript()
  const queryClient = useQueryClient()
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const abort = useRef<AbortController | null>(null)

  function reset() {
    abort.current?.abort()
    setTranscript([])
  }

  async function send(preset?: string) {
    const text = (preset ?? input).trim()
    if (!text || busy) return
    setInput("")
    const history = recentHistory(messages, text)
    const botId = crypto.randomUUID()
    appendMessages({ id: crypto.randomUUID(), role: "user", text }, { id: botId, role: "assistant", text: "" })
    setBusy(true)
    abort.current = new AbortController()
    const t0 = clock()

    // Tokens arrive faster than frames are worth painting: collect, flush once per frame.
    let pending = ""
    let frame = 0
    const flush = () => {
      frame = 0
      const piece = pending
      pending = ""
      if (piece) patchMessage(botId, (m) => ({ ...m, text: m.text + piece }))
    }
    try {
      await streamChat(
        model,
        history,
        (piece) => {
          pending += piece
          if (!frame) frame = requestAnimationFrame(flush)
        },
        (served) => patchMessage(botId, (m) => ({ ...m, served })),
        abort.current.signal,
      )
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      patchMessage(botId, (m) => ({ ...m, error: isAbort(e) ? "Stopped." : `The router could not answer: ${msg}` }))
    } finally {
      if (frame) cancelAnimationFrame(frame)
      flush()
      patchMessage(botId, (m) => ({ ...m, ms: Math.round(clock() - t0) }))
      setBusy(false)
      abort.current = null
      void queryClient.invalidateQueries({ queryKey: STATS_KEY }) // the board shows this answer now, not next poll
    }
  }

  const target = model === "auto" ? "wherever the router decides" : model

  return (
    <section
      aria-label="Chat"
      className="bg-card border-border flex h-[min(34rem,70dvh)] min-w-0 flex-col rounded-lg border shadow-sm lg:sticky lg:top-4 lg:h-[max(30rem,calc(100dvh-17rem))] lg:max-h-[52rem]"
    >
      <div className="border-border flex items-center justify-between gap-3 border-b px-4 py-2">
        <div className="flex min-w-0 items-baseline gap-3">
          <h2 className="text-base font-medium">Ask</h2>
          <span className="text-muted-foreground truncate text-xs">{target}</span>
        </div>
        {messages.length > 0 && (
          <Button type="button" variant="ghost" size="sm" onClick={reset}>
            <RotateCcw className="size-4" /> New chat
          </Button>
        )}
      </div>
      {messages.length === 0 ? (
        <div className="flex min-h-0 flex-1 px-4 py-3">
          <Empty />
        </div>
      ) : (
        // column-reverse pins the newest message to the bottom while streaming, no scroll code
        <div className="flex min-h-0 flex-1 flex-col-reverse overflow-y-auto px-4 py-3" role="log" aria-label="Transcript">
          <div className="space-y-3">
            {messages.map((m) => (
              <Bubble key={m.id} m={m} />
            ))}
          </div>
        </div>
      )}
      <div className="border-border flex flex-wrap items-center gap-2 border-t px-3 pt-2 text-xs">
        <span className="text-muted-foreground">Try</span>
        {DEMO_PROMPTS.map((d) => (
          <Button
            key={d.label}
            type="button"
            variant="secondary"
            size="sm"
            disabled={busy}
            aria-description={d.hint}
            title={d.hint}
            onClick={() => void send(d.text)}
          >
            {d.label}
          </Button>
        ))}
      </div>
      <form
        className="flex gap-2 p-3"
        onSubmit={(e) => {
          e.preventDefault()
          void send()
        }}
      >
        <Input
          id="chat-input"
          aria-label="Message"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something"
          disabled={busy}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              void send()
            }
          }}
        />
        {busy ? (
          <Button type="button" variant="secondary" onClick={() => abort.current?.abort()}>
            <Square className="size-4" /> Stop
          </Button>
        ) : (
          <Button type="submit">
            <SendHorizontal className="size-4" /> Send
          </Button>
        )}
      </form>
    </section>
  )
}
