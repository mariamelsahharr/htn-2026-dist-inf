import { SendHorizontal, Square } from "lucide-react"
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions"
import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { UpstreamBadge } from "@/components/UpstreamBadge"
import { streamChat, type Served } from "@/lib/chat"
import { useSettings } from "@/lib/settings"

interface Message {
  id: string
  role: "user" | "assistant"
  text: string
  served?: Served
  ms?: number
  error?: string
}


export function Chat() {
  const { settings } = useSettings()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const abort = useRef<AbortController | null>(null)

  const patch = (id: string, f: (m: Message) => Message) =>
    setMessages((ms) => ms.map((m) => (m.id === id ? f(m) : m)))

  async function send() {
    const text = input.trim()
    if (!text || busy) return
    setInput("")
    const history: ChatCompletionMessageParam[] = [
      ...messages.filter((m) => !m.error && m.text).map((m) => ({ role: m.role, content: m.text })),
      { role: "user", content: text },
    ]
    const userId = crypto.randomUUID()
    const botId = crypto.randomUUID()
    setMessages((ms) => [...ms, { id: userId, role: "user", text }, { id: botId, role: "assistant", text: "" }])
    setBusy(true)
    abort.current = new AbortController()
    const t0 = performance.now()
    try {
      await streamChat(
        settings.model,
        history,
        (piece) => patch(botId, (m) => ({ ...m, text: m.text + piece })),
        (served) => patch(botId, (m) => ({ ...m, served })),
        abort.current.signal,
      )
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      patch(botId, (m) => ({ ...m, error: /abort/i.test(msg) ? "Stopped." : `The router could not answer: ${msg}` }))
    } finally {
      patch(botId, (m) => ({ ...m, ms: Math.round(performance.now() - t0) }))
      setBusy(false)
      abort.current = null
    }
  }

  const target = settings.model === "auto" ? "wherever the router decides" : settings.model

  return (
    <section aria-label="Chat" className="bg-card border-border flex h-full min-h-[30rem] min-w-0 flex-col rounded-lg border shadow-sm">
      <div className="border-border flex items-baseline justify-between border-b px-4 py-3">
        <h2 className="text-base font-medium">Ask</h2>
        <span className="text-muted-foreground truncate text-xs">{target}</span>
      </div>
      {/* column-reverse pins the newest message to the bottom while streaming, no scroll code */}
      <div className="flex min-h-0 flex-1 flex-col-reverse overflow-y-auto px-4 py-3">
        <div className="space-y-3">
          {messages.length === 0 && (
            <p className="text-muted-foreground max-w-prose text-sm">
              Short, plain questions stay on the Pis. Long prompts, pasted code, questions that ask for deep
              analysis, or a cluster that is down go to the cloud. Each answer says who served it and why.
            </p>
          )}
          {messages.map((m) => (
            <div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
              <div
                className={
                  m.role === "user"
                    ? "bg-primary text-primary-foreground max-w-[85%] rounded-lg px-3 py-2 text-sm break-words"
                    : "bg-muted max-w-[85%] rounded-lg px-3 py-2 text-sm break-words"
                }
              >
                {m.role === "assistant" && (
                  <div className="mb-1 flex flex-wrap items-center gap-2">
                    {m.served ? (
                      <UpstreamBadge upstream={m.served.servedBy} />
                    ) : (
                      <span className="text-muted-foreground text-xs">routing</span>
                    )}
                    {m.served?.reason && <span className="text-muted-foreground text-xs">{m.served.reason}</span>}
                    {m.ms != null && <span className="text-muted-foreground text-xs">{m.ms} ms</span>}
                  </div>
                )}
                <div className="whitespace-pre-wrap">{m.text}</div>
                {m.error && <div className="text-critical mt-1 text-xs">{m.error}</div>}
              </div>
            </div>
          ))}
        </div>
      </div>
      <form
        className="border-border flex gap-2 border-t p-3"
        onSubmit={(e) => {
          e.preventDefault()
          void send()
        }}
      >
        <Input
          id="chat-input"
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
