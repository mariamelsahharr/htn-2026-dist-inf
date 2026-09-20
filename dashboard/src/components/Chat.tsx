import { RotateCcw, SendHorizontal, Square } from "lucide-react"
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions"
import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { UpstreamBadge } from "@/components/UpstreamBadge"
import { streamChat, type Served } from "@/lib/chat"
import { useSettings } from "@/lib/settings"
import { whyRouted } from "@/lib/words"
import { DEMO_PROMPTS } from "@/lib/demo"

interface Message {
  id: string
  role: "user" | "assistant"
  text: string
  served?: Served
  ms?: number
  error?: string
}


const TRANSCRIPT_KEY = "pi-router-chat"
const clock = () => performance.now() // read in event handlers only, never during render

// A pasted wall of text is shown by its head; the full text still goes to the router.
function shown(text: string): string {
  return text.length > 320 ? `${text.slice(0, 280).trimEnd()}… (${text.length.toLocaleString()} characters)` : text
}

// Reasoning models (Qwen3 on the Pis) stream a <think>…</think> block before the answer.
// It is kept in the transcript for the next turn but not shown.
function visible(text: string): string {
  const closed = text.replace(/<think>[\s\S]*?<\/think>\s*/g, "")
  const open = closed.indexOf("<think>")
  return open === -1 ? closed : closed.slice(0, open)
}

// The transcript survives a reload during the demo; "New chat" clears it.
function loadTranscript(): Message[] {
  try {
    return JSON.parse(sessionStorage.getItem(TRANSCRIPT_KEY) ?? "[]") as Message[]
  } catch {
    return []
  }
}

function saveTranscript(ms: Message[]): Message[] {
  try {
    sessionStorage.setItem(TRANSCRIPT_KEY, JSON.stringify(ms))
  } catch {
    // storage blocked: the chat still works for this page view
  }
  return ms
}

export function Chat() {
  const { settings } = useSettings()
  const [messages, setMessages] = useState<Message[]>(loadTranscript)
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const abort = useRef<AbortController | null>(null)

  const patch = (id: string, f: (m: Message) => Message) =>
    setMessages((ms) => saveTranscript(ms.map((m) => (m.id === id ? f(m) : m))))

  function reset() {
    abort.current?.abort()
    setMessages(saveTranscript([]))
  }

  async function send(preset?: string) {
    const text = (preset ?? input).trim()
    if (!text || busy) return
    setInput("")
    const history: ChatCompletionMessageParam[] = [
      ...messages.filter((m) => !m.error && m.text).map((m) => ({ role: m.role, content: m.text })),
      { role: "user", content: text },
    ]
    const userId = crypto.randomUUID()
    const botId = crypto.randomUUID()
    setMessages((ms) => saveTranscript([...ms, { id: userId, role: "user", text }, { id: botId, role: "assistant", text: "" }]))
    setBusy(true)
    abort.current = new AbortController()
    const t0 = clock()
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
      patch(botId, (m) => ({ ...m, ms: Math.round(clock() - t0) }))
      setBusy(false)
      abort.current = null
    }
  }

  const target = settings.model === "auto" ? "wherever the router decides" : settings.model

  return (
    <section aria-label="Chat" className="bg-card border-border flex h-[34rem] min-w-0 flex-col rounded-lg border shadow-sm lg:sticky lg:top-4 lg:h-[calc(100vh-2rem)] lg:max-h-[52rem]">
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
      {/* column-reverse pins the newest message to the bottom while streaming, no scroll code */}
      <div className="flex min-h-0 flex-1 flex-col-reverse overflow-y-auto px-4 py-3">
        <div className="space-y-3">
          {messages.length === 0 && (
            <p className="text-muted-foreground max-w-prose text-sm">
              Short questions stay on the Pis. Long prompts, pasted code, or a cluster that is down go to the
              cloud. Each answer says who served it and why.
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
                )}
                <div className="whitespace-pre-wrap">
                  {m.role === "assistant" ? visible(m.text) || (m.text ? "thinking" : "") : shown(m.text)}
                </div>
                {m.error && <div className="text-critical mt-1 text-xs">{m.error}</div>}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="border-border flex flex-wrap items-center gap-2 border-t px-3 pt-2 text-xs">
        <span className="text-muted-foreground">Try</span>
        {DEMO_PROMPTS.map((d) => (
          <Button key={d.label} type="button" variant="secondary" size="sm" disabled={busy} title={d.hint} onClick={() => void send(d.text)}>
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
